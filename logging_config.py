"""
logging_config.py

Structured logging for the Multi-Agent Customer Support System.
Replaces scattered print() calls across agents/, memory/, and graph.py with
proper logger calls that carry a correlation id (ticket_id / conversation_id)
and can be redirected/filtered by LOG_LEVEL.

Usage:
    from logging_config import get_logger, set_correlation_id
    logger = get_logger(__name__)
    set_correlation_id(ticket_id="TKT-ABC123", conversation_id="CONV-XYZ")
    logger.info("classification complete", extra={"classification": "refund"})

Never log secrets. REDACTED_HEADERS lists header names that must never be
logged verbatim anywhere in the codebase (auth.py and api.py request logging
must consult this list).
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

REDACTED_HEADERS = {"x-api-key", "x-gemini-api-key", "authorization"}

_ticket_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("ticket_id", default="-")
_conversation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("conversation_id", default="-")


def set_correlation_id(ticket_id: str | None = None, conversation_id: str | None = None) -> None:
    """Bind ticket_id/conversation_id to the current execution context so every
    subsequent log record (in this thread/task) carries them automatically."""
    if ticket_id is not None:
        _ticket_id_var.set(ticket_id)
    if conversation_id is not None:
        _conversation_id_var.set(conversation_id)


def clear_correlation_id() -> None:
    _ticket_id_var.set("-")
    _conversation_id_var.set("-")


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ticket_id = _ticket_id_var.get()
        record.conversation_id = _conversation_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "ticket_id": getattr(record, "ticket_id", "-"),
            "conversation_id": getattr(record, "conversation_id", "-"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure_logging() -> None:
    """Idempotent — safe to call from every module that wants a logger."""
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_CorrelationFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers unless the operator explicitly wants DEBUG everywhere.
    if level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
