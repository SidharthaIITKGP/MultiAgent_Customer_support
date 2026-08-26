"""
db/session.py

Engine/session setup. SQLite by default (zero external infra for a demo),
swappable to Postgres purely via the DATABASE_URL env var — no code change.

Migration story: Base.metadata.create_all() on startup, no Alembic. Proportionate
for 3 tables and a single-process demo with no live data needing a zero-downtime
schema migration; if this ever needs that, Alembic is the documented next step.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/support.db")

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    # Allow use across FastAPI's threadpool-dispatched sync routes.
    _connect_args["check_same_thread"] = False
    # ./data/support.db needs its parent dir to exist before sqlite3 can open it.
    db_path = DATABASE_URL.removeprefix("sqlite:///")
    if db_path and db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a Session, closes it after the request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
