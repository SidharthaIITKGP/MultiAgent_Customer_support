"""
db/models.py

Persistence schema for multi-turn conversations.

Conversation  — one chat thread for one user.
Message       — one turn's worth of chat content (user or assistant), in order.
TicketRun     — one execution of the LangGraph pipeline (one graph.invoke() call),
                storing the full serialized state (including the react_trace) as
                a JSON blob. Named TicketRun at the class level to signal "a single
                run", but its primary key column is `ticket_id` — matching the
                pre-existing vocabulary used throughout agents/state.py, api.py,
                and eval/ — so /ticket/{ticket_id} URLs and eval scripts stay valid.

No FK from TicketRun -> Message (would be circular); Message.ticket_id points at
TicketRun.ticket_id instead, set on both the triggering user message and the
resulting assistant message once a turn completes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, Boolean, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_conversation_id() -> str:
    return f"CONV-{uuid.uuid4().hex[:8].upper()}"


def new_message_id() -> str:
    return f"MSG-{uuid.uuid4().hex[:8].upper()}"


def new_ticket_id() -> str:
    return f"TKT-{uuid.uuid4().hex[:8].upper()}"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_conversation_id)
    user_id: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="active")  # active | closed
    awaiting_customer_input: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_message_id)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(String)  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    ticket_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("tickets.ticket_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)


class TicketRun(Base):
    __tablename__ = "tickets"

    ticket_id: Mapped[str] = mapped_column(String, primary_key=True, default=new_ticket_id)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id"), index=True
    )
    user_id: Mapped[str] = mapped_column(String, index=True)
    ticket_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, default="completed")  # completed | error
    state_json: Mapped[dict] = mapped_column(JSON, default=dict)
    escalation_decision: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    classification: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    processing_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
