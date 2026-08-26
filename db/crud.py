"""
db/crud.py

Thin data-access functions used by conversation_service.py and api.py.
No business logic here (composed ticket_text, state assembly, etc. lives in
conversation_service.py) — just persistence operations.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Conversation, Message, TicketRun


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(db: Session, user_id: str) -> Conversation:
    conv = Conversation(user_id=user_id)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_conversation(db: Session, conversation_id: str) -> Conversation | None:
    return db.get(Conversation, conversation_id)


def list_conversations(
    db: Session, user_id: str | None = None, limit: int = 20, offset: int = 0
) -> list[Conversation]:
    stmt = select(Conversation).order_by(Conversation.updated_at.desc())
    if user_id:
        stmt = stmt.where(Conversation.user_id == user_id)
    stmt = stmt.limit(limit).offset(offset)
    return list(db.scalars(stmt))


def set_conversation_awaiting(db: Session, conversation_id: str, awaiting: bool) -> None:
    conv = db.get(Conversation, conversation_id)
    if conv is not None:
        conv.awaiting_customer_input = awaiting
        db.commit()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def create_message(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
    ticket_id: str | None = None,
) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content, ticket_id=ticket_id)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def set_message_ticket_id(db: Session, message_id: str, ticket_id: str) -> None:
    msg = db.get(Message, message_id)
    if msg is not None:
        msg.ticket_id = ticket_id
        db.commit()


def list_messages(db: Session, conversation_id: str) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return list(db.scalars(stmt))


def get_last_n_messages(db: Session, conversation_id: str, n: int) -> list[Message]:
    """Most-recent-last, for feeding conversation_history into SupportTicketState."""
    messages = list_messages(db, conversation_id)
    return messages[-n:] if n > 0 else messages


# ---------------------------------------------------------------------------
# Ticket runs
# ---------------------------------------------------------------------------

def create_ticket_run(
    db: Session,
    conversation_id: str,
    user_id: str,
    ticket_text: str,
    status: str,
    state_json: dict,
    escalation_decision: str | None = None,
    classification: str | None = None,
    processing_time_s: float | None = None,
    error_message: str | None = None,
    ticket_id: str | None = None,
) -> TicketRun:
    kwargs = dict(
        conversation_id=conversation_id,
        user_id=user_id,
        ticket_text=ticket_text,
        status=status,
        state_json=state_json,
        escalation_decision=escalation_decision,
        classification=classification,
        processing_time_s=processing_time_s,
        error_message=error_message,
    )
    if ticket_id is not None:
        kwargs["ticket_id"] = ticket_id
    run = TicketRun(**kwargs)
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_ticket_run(db: Session, ticket_id: str) -> TicketRun | None:
    return db.get(TicketRun, ticket_id)


def list_ticket_runs(
    db: Session, user_id: str | None = None, limit: int = 20, offset: int = 0
) -> list[TicketRun]:
    stmt = select(TicketRun).order_by(TicketRun.created_at.desc())
    if user_id:
        stmt = stmt.where(TicketRun.user_id == user_id)
    stmt = stmt.limit(limit).offset(offset)
    return list(db.scalars(stmt))


def get_last_ticket_run_for_conversation(db: Session, conversation_id: str) -> TicketRun | None:
    stmt = (
        select(TicketRun)
        .where(TicketRun.conversation_id == conversation_id)
        .order_by(TicketRun.created_at.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()
