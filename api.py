"""
api.py — FastAPI gateway for the Multi-Agent Customer Support System.

All JSON/SSE routes live under /api/* (auth + rate-limited). The chat frontend
is served as static files at "/" and "/web/*". /health and /metrics are open,
unauthenticated infra endpoints.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

import conversation_service
import db.crud as crud
from auth import require_api_key
from db.session import get_db, init_db
from logging_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Multi-Agent Customer Support System",
    description="LangGraph-powered multi-agent support chat with explicit ReAct traces",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


# ---------------------------------------------------------------------------
# Rate limiting — keyed by X-API-Key, falls back to remote address
# ---------------------------------------------------------------------------

def _rate_limit_key(request: Request) -> str:
    return request.headers.get("x-api-key") or get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _gemini_api_key(x_gemini_api_key: str = Header(default="", alias="X-Gemini-Api-Key")) -> str | None:
    """
    BYOK: optional caller-supplied Gemini key. Never logged, never persisted —
    read once here and threaded straight into conversation_service, which
    threads it into the graph's `config` (see agents/llm_utils.py).
    """
    return x_gemini_api_key or None


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class CreateConversationRequest(BaseModel):
    user_id: str = "u_anonymous"


class SendMessageRequest(BaseModel):
    text: str


class TicketRequest(BaseModel):
    ticket_text: str
    user_id: str = "u_anonymous"


# ---------------------------------------------------------------------------
# Open, unauthenticated infra endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        crud.list_conversations(db, limit=1)  # touch the DB to confirm connectivity
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "db_ok": db_ok}


app.mount("/metrics", make_asgi_app())

# ---------------------------------------------------------------------------
# Static chat frontend
# ---------------------------------------------------------------------------

app.mount("/web", StaticFiles(directory="frontend/web"), name="web")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("frontend/web/index.html")


# ---------------------------------------------------------------------------
# Authenticated API router
# ---------------------------------------------------------------------------

api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


@api.post("/conversations")
@limiter.limit("20/minute")
def create_conversation(request: Request, body: CreateConversationRequest, db: Session = Depends(get_db)):
    conv = crud.create_conversation(db, user_id=body.user_id)
    return {"conversation_id": conv.id, "user_id": conv.user_id, "status": conv.status}


@api.get("/conversations")
@limiter.limit("60/minute")
def list_conversations_endpoint(
    request: Request,
    user_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    convs = crud.list_conversations(db, user_id=user_id, limit=limit, offset=offset)
    return [
        {"conversation_id": c.id, "user_id": c.user_id, "status": c.status,
         "awaiting_customer_input": c.awaiting_customer_input, "updated_at": c.updated_at.isoformat()}
        for c in convs
    ]


@api.get("/conversations/{conversation_id}")
@limiter.limit("60/minute")
def get_conversation_endpoint(
    request: Request,
    conversation_id: str,
    include_trace: bool = False,
    db: Session = Depends(get_db),
):
    conv = crud.get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")
    messages = crud.list_messages(db, conversation_id)
    result_messages = []
    for m in messages:
        entry = {"id": m.id, "role": m.role, "content": m.content, "ticket_id": m.ticket_id,
                  "created_at": m.created_at.isoformat()}
        if include_trace and m.ticket_id:
            run = crud.get_ticket_run(db, m.ticket_id)
            if run:
                entry["state"] = run.state_json
        result_messages.append(entry)
    return {
        "conversation_id": conv.id,
        "user_id": conv.user_id,
        "status": conv.status,
        "awaiting_customer_input": conv.awaiting_customer_input,
        "messages": result_messages,
    }


@api.post("/conversations/{conversation_id}/messages")
@limiter.limit("10/minute")
def send_message(
    request: Request,
    conversation_id: str,
    body: SendMessageRequest,
    db: Session = Depends(get_db),
    gemini_api_key: str | None = Depends(_gemini_api_key),
):
    conv = crud.get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")
    try:
        run, assistant_msg = conversation_service.run_turn(db, conv, body.text, gemini_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ticket_id": run.ticket_id,
        "assistant_message": {"id": assistant_msg.id, "content": assistant_msg.content},
        "escalation_decision": run.escalation_decision,
        "processing_time_s": run.processing_time_s,
    }


@api.post("/conversations/{conversation_id}/messages/stream")
@limiter.limit("10/minute")
async def send_message_stream(
    request: Request,
    conversation_id: str,
    body: SendMessageRequest,
    db: Session = Depends(get_db),
    gemini_api_key: str | None = Depends(_gemini_api_key),
):
    conv = crud.get_conversation(db, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found")

    async def event_stream():
        async for frame in conversation_service.stream_turn(db, conv, body.text, gemini_api_key):
            yield f"data: {json.dumps(frame, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Back-compat single-shot ticket endpoints (Streamlit's old contract)
# ---------------------------------------------------------------------------

@api.post("/ticket", deprecated=True)
@limiter.limit("10/minute")
def submit_ticket(request: Request, body: TicketRequest, db: Session = Depends(get_db),
                   gemini_api_key: str | None = Depends(_gemini_api_key)):
    """Deprecated: creates a single-turn conversation under the hood."""
    conv = crud.create_conversation(db, user_id=body.user_id)
    try:
        run, assistant_msg = conversation_service.run_turn(db, conv, body.ticket_text, gemini_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ticket_id": run.ticket_id,
        "conversation_id": conv.id,
        "status": "completed",
        "escalation_decision": run.escalation_decision,
        "final_response": assistant_msg.content,
        "processing_time_s": run.processing_time_s,
    }


@api.get("/ticket/{ticket_id}")
@limiter.limit("60/minute")
def get_ticket_summary(request: Request, ticket_id: str, db: Session = Depends(get_db)):
    run = crud.get_ticket_run(db, ticket_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found")
    return {k: v for k, v in run.state_json.items() if k != "react_trace"} | {
        "ticket_id": run.ticket_id,
        "conversation_id": run.conversation_id,
        "status": run.status,
        "processing_time_s": run.processing_time_s,
    }


@api.get("/ticket/{ticket_id}/trace")
@limiter.limit("60/minute")
def get_ticket_trace(request: Request, ticket_id: str, db: Session = Depends(get_db)):
    run = crud.get_ticket_run(db, ticket_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found")
    return run.state_json


@api.get("/tickets")
@limiter.limit("60/minute")
def list_tickets(request: Request, user_id: str | None = None, limit: int = 20,
                  offset: int = 0, db: Session = Depends(get_db)):
    runs = crud.list_ticket_runs(db, user_id=user_id, limit=limit, offset=offset)
    return [
        {"ticket_id": r.ticket_id, "conversation_id": r.conversation_id,
         "decision": r.escalation_decision, "classification": r.classification,
         "status": r.status, "created_at": r.created_at.isoformat()}
        for r in runs
    ]


app.include_router(api)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
