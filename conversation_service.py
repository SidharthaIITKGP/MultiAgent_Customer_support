"""
conversation_service.py

Owns the business logic for turning a chat message into a graph run:
- composing ticket_text across a request_info round-trip
- assembling SupportTicketState for the next turn
- persisting the turn's Message/TicketRun rows and updating Conversation state

Kept separate from api.py so the HTTP layer stays thin wiring.
"""

from __future__ import annotations

import os
import time

import db.crud as crud
from agents.graph import astream_graph, run_graph
from agents.state import state_to_dict
from logging_config import get_logger

logger = get_logger(__name__)

MAX_COMPOSED_TICKET_TEXT = 4000
CONVERSATION_HISTORY_WINDOW = int(os.getenv("CONVERSATION_HISTORY_WINDOW", "10"))


def _compose_ticket_text(db_session, conversation, new_text: str) -> str:
    """
    If the previous turn ended with request_info, chain the new message onto the
    prior ticket_text so downstream agents see the full thread of the issue —
    this is why knowledge_agent/action_agent/escalation_agent need no changes at
    all: they already just consume ticket_text opaquely. Self-chaining: if the
    prior ticket_text was itself already composed, this naturally carries the
    whole history forward across multiple clarification rounds.
    """
    if not conversation.awaiting_customer_input:
        return new_text

    last_run = crud.get_last_ticket_run_for_conversation(db_session, conversation.id)
    if last_run is None:
        return new_text

    messages = crud.list_messages(db_session, conversation.id)
    last_assistant_msg = next(
        (m for m in reversed(messages) if m.role == "assistant"), None
    )
    assistant_text = last_assistant_msg.content if last_assistant_msg else "(no prior response)"

    composed = (
        f"{last_run.ticket_text}\n\n"
        f"[Assistant asked]: {assistant_text}\n"
        f"[Customer replied]: {new_text}"
    )
    if len(composed) > MAX_COMPOSED_TICKET_TEXT:
        # Truncate the oldest content first, keep the most recent exchange intact.
        overflow = len(composed) - MAX_COMPOSED_TICKET_TEXT
        composed = "...[earlier context truncated]...\n" + composed[overflow:]
    return composed


def build_next_turn_state(db_session, conversation, ticket_id: str, new_text: str) -> dict:
    ticket_text = _compose_ticket_text(db_session, conversation, new_text)
    history_msgs = crud.get_last_n_messages(
        db_session, conversation.id, CONVERSATION_HISTORY_WINDOW
    )
    conversation_history = [{"role": m.role, "content": m.content} for m in history_msgs]

    return {
        "ticket_id": ticket_id,
        "user_id": conversation.user_id,
        "ticket_text": ticket_text,
        "conversation_id": conversation.id,
        "conversation_history": conversation_history,
        "previous_turn_requested_info": conversation.awaiting_customer_input,
        "react_trace": [],
        "tool_calls_made": [],
        "knowledge_retry_count": 0,
        "tool_retry_count": 0,
        "insufficient_context": False,
    }


def run_turn(db_session, conversation, user_text: str, gemini_api_key: str | None = None):
    """
    Synchronous turn: persist the user message, run the graph, persist the
    TicketRun + assistant message, update conversation.awaiting_customer_input.
    Returns (ticket_run, assistant_message).
    """
    from db.models import new_ticket_id

    ticket_id = new_ticket_id()
    user_msg = crud.create_message(db_session, conversation.id, "user", user_text)

    initial_state = build_next_turn_state(db_session, conversation, ticket_id, user_text)

    start = time.time()
    try:
        final_state = run_graph(initial_state, api_key=gemini_api_key)
        elapsed = time.time() - start
        serialized = state_to_dict(final_state)
        serialized["ticket_id"] = ticket_id

        run = crud.create_ticket_run(
            db_session,
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            ticket_text=initial_state["ticket_text"],
            status="completed",
            state_json=serialized,
            escalation_decision=serialized.get("escalation_decision"),
            classification=serialized.get("classification"),
            processing_time_s=round(elapsed, 2),
            ticket_id=ticket_id,
        )

        crud.set_message_ticket_id(db_session, user_msg.id, ticket_id)
        assistant_msg = crud.create_message(
            db_session,
            conversation.id,
            "assistant",
            serialized.get("final_response", "(no response generated)"),
            ticket_id=ticket_id,
        )
        crud.set_conversation_awaiting(
            db_session, conversation.id, serialized.get("escalation_decision") == "request_info"
        )
        return run, assistant_msg

    except Exception as e:
        elapsed = time.time() - start
        logger.error("graph run failed for ticket %s: %s", ticket_id, e)
        crud.create_ticket_run(
            db_session,
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            ticket_text=initial_state["ticket_text"],
            status="error",
            state_json={},
            processing_time_s=round(elapsed, 2),
            error_message=str(e),
            ticket_id=ticket_id,
        )
        crud.set_message_ticket_id(db_session, user_msg.id, ticket_id)
        raise


async def stream_turn(db_session, conversation, user_text: str, gemini_api_key: str | None = None):
    """
    Async generator yielding SSE-shaped dict frames as the graph runs, then
    persisting the turn exactly like run_turn once it completes.

    Frame shapes: {"type": "started", ...}, {"type": "node_update", ...} (once
    per completed node, repeats for knowledge retries), {"type": "final", ...},
    {"type": "error", ...}.
    """
    from db.models import new_ticket_id

    ticket_id = new_ticket_id()
    user_msg = crud.create_message(db_session, conversation.id, "user", user_text)
    initial_state = build_next_turn_state(db_session, conversation, ticket_id, user_text)

    yield {"type": "started", "conversation_id": conversation.id, "ticket_id": ticket_id}

    start = time.time()
    final_state: dict = {}
    try:
        async for chunk in astream_graph(initial_state, api_key=gemini_api_key):
            for node_name, node_output in chunk.items():
                serialized_update = state_to_dict(node_output)
                final_state.update(serialized_update)
                yield {
                    "type": "node_update",
                    "node": node_name,
                    "react_trace_delta": serialized_update.get("react_trace", []),
                    "state_delta": {
                        k: v for k, v in serialized_update.items() if k != "react_trace"
                    },
                }

        elapsed = time.time() - start
        final_state["ticket_id"] = ticket_id

        crud.create_ticket_run(
            db_session,
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            ticket_text=initial_state["ticket_text"],
            status="completed",
            state_json=final_state,
            escalation_decision=final_state.get("escalation_decision"),
            classification=final_state.get("classification"),
            processing_time_s=round(elapsed, 2),
            ticket_id=ticket_id,
        )

        crud.set_message_ticket_id(db_session, user_msg.id, ticket_id)
        crud.create_message(
            db_session,
            conversation.id,
            "assistant",
            final_state.get("final_response", "(no response generated)"),
            ticket_id=ticket_id,
        )
        crud.set_conversation_awaiting(
            db_session, conversation.id, final_state.get("escalation_decision") == "request_info"
        )

        yield {
            "type": "final",
            "ticket_id": ticket_id,
            "final_response": final_state.get("final_response"),
            "escalation_decision": final_state.get("escalation_decision"),
            "confidence_breakdown": final_state.get("confidence_breakdown"),
            "classification": final_state.get("classification"),
            "urgency": final_state.get("urgency"),
            "sentiment": final_state.get("sentiment"),
        }

    except Exception as e:
        elapsed = time.time() - start
        logger.error("graph stream failed for ticket %s: %s", ticket_id, e)
        crud.create_ticket_run(
            db_session,
            conversation_id=conversation.id,
            user_id=conversation.user_id,
            ticket_text=initial_state["ticket_text"],
            status="error",
            state_json=final_state,
            processing_time_s=round(elapsed, 2),
            error_message=str(e),
            ticket_id=ticket_id,
        )
        crud.set_message_ticket_id(db_session, user_msg.id, ticket_id)
        yield {"type": "error", "message": str(e)}
