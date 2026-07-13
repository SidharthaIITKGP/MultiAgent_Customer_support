"""
api.py — FastAPI gateway for the Multi-Agent Customer Support System.

Single endpoint:
  POST /ticket — runs the LangGraph, returns full reasoning trace as JSON

Storage: in-memory dict (no DB, no auth, keeps it lean).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.graph import run_graph
from agents.state import SupportTicketState, state_to_dict

app = FastAPI(
    title="Multi-Agent Customer Support System",
    description="LangGraph-powered multi-agent support system with explicit ReAct traces",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory ticket store
_ticket_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    ticket_text: str
    user_id: str = "u_anonymous"


class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    escalation_decision: str | None = None
    final_response: str | None = None
    react_trace_length: int = 0
    processing_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "stored_tickets": len(_ticket_store)}


@app.post("/ticket", response_model=TicketResponse)
def submit_ticket(req: TicketRequest):
    """
    Run a support ticket through the full LangGraph pipeline.
    Returns the ticket ID and summary — use GET /ticket/{id}/trace for the full trace.
    """
    import time

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    start = time.time()

    initial_state: SupportTicketState = {
        "ticket_id": ticket_id,
        "user_id": req.user_id,
        "ticket_text": req.ticket_text,
        "react_trace": [],
        "tool_calls_made": [],
        "knowledge_retry_count": 0,
        "tool_retry_count": 0,
        "insufficient_context": False,
    }

    try:
        final_state = run_graph(initial_state)
        elapsed = time.time() - start
        serialized = state_to_dict(final_state)
        serialized["processing_time_s"] = round(elapsed, 2)
        serialized["submitted_at"] = datetime.now(timezone.utc).isoformat()
        _ticket_store[ticket_id] = serialized

        return TicketResponse(
            ticket_id=ticket_id,
            status="completed",
            escalation_decision=serialized.get("escalation_decision"),
            final_response=serialized.get("final_response"),
            react_trace_length=len(serialized.get("react_trace", [])),
            processing_time_s=round(elapsed, 2),
        )
    except Exception as e:
        elapsed = time.time() - start
        _ticket_store[ticket_id] = {
            "ticket_id": ticket_id,
            "error": str(e),
            "status": "error",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ticket/{ticket_id}")
def get_ticket_summary(ticket_id: str):
    """Get ticket summary without the full trace."""
    if ticket_id not in _ticket_store:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found")
    data = _ticket_store[ticket_id]
    # Return without the full trace for brevity
    return {k: v for k, v in data.items() if k != "react_trace"}


@app.get("/ticket/{ticket_id}/trace")
def get_ticket_trace(ticket_id: str):
    """Get the FULL reasoning trace for a ticket — every Thought/Action/Observation."""
    if ticket_id not in _ticket_store:
        raise HTTPException(status_code=404, detail=f"Ticket '{ticket_id}' not found")
    return _ticket_store[ticket_id]


@app.get("/tickets")
def list_tickets():
    """List all processed tickets (summary only)."""
    return [
        {
            "ticket_id": tid,
            "submitted_at": data.get("submitted_at"),
            "decision": data.get("escalation_decision"),
            "classification": data.get("classification"),
            "status": data.get("status", "completed"),
        }
        for tid, data in _ticket_store.items()
    ]


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
