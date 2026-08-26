"""
memory/ticket_memory.py

Per-user ticket memory using a separate ChromaDB collection.

Purpose: intake_agent retrieves similar past tickets for THIS user before classifying.
This is retrieval-augmented REASONING (not just RAG over docs) — the past resolution
history directly informs the intake agent's urgency decision and intake_reasoning.

After each ticket resolves (in escalation_agent), the ticket+resolution pair is stored.

TC-16 depends on this: user u_repeat with 3 stored unresolved tickets should cause
the intake_agent to escalate instead of repeating the same resolution.
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "ticket_memory"


def _get_memory_collection():
    """Return the ticket_memory ChromaDB collection with local embeddings."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_similar_tickets(user_id: str, ticket_text: str, k: int = 5) -> list[dict]:
    """
    Retrieve the k most similar past tickets for this user.
    Used by intake_agent before classifying.

    Returns a list of dicts: {text, resolution, resolved, similarity, timestamp}
    """
    collection = _get_memory_collection()

    # Filter to this user's tickets only
    try:
        results = collection.query(
            query_texts=[ticket_text],
            n_results=min(k, max(1, collection.count())),
            where={"user_id": {"$eq": user_id}},
            include=["documents", "distances", "metadatas"],
        )
    except Exception:
        # If no results for this user or collection is empty
        return []

    docs      = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]

    tickets = []
    for doc, dist, meta in zip(docs, distances, metas):
        similarity = max(0.0, 1.0 - dist)
        tickets.append({
            "text":       doc,
            "resolution": meta.get("resolution", "no resolution recorded"),
            "resolved":   meta.get("resolved", False),
            "similarity": round(similarity, 3),
            "timestamp":  meta.get("timestamp", ""),
        })

    # Sort by similarity descending
    tickets.sort(key=lambda x: x["similarity"], reverse=True)
    return tickets


def store_ticket_resolution(
    user_id: str,
    ticket_text: str,
    resolution: str,
    resolved: bool,
) -> str:
    """
    Store a ticket+resolution pair in memory after the ticket is handled.
    Called by escalation_agent after deciding the outcome.
    Returns the stored document ID.
    """
    collection = _get_memory_collection()

    doc_id = f"mem_{user_id}_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    collection.upsert(
        ids=[doc_id],
        documents=[ticket_text],
        metadatas=[{
            "user_id":    user_id,
            "resolution": resolution[:500],  # cap length
            "resolved":   resolved,
            "timestamp":  timestamp,
        }],
    )
    return doc_id


def seed_past_tickets(user_id: str, tickets: list[dict]) -> None:
    """
    Seed historical tickets for a user (used by eval to set up TC-16).
    Each ticket: {text, resolution, resolved}
    """
    collection = _get_memory_collection()
    timestamp = datetime.now(timezone.utc).isoformat()

    for i, ticket in enumerate(tickets):
        doc_id = f"seed_{user_id}_{i}_{uuid.uuid4().hex[:6]}"
        collection.upsert(
            ids=[doc_id],
            documents=[ticket["text"]],
            metadatas=[{
                "user_id":    user_id,
                "resolution": ticket.get("resolution", "")[:500],
                "resolved":   ticket.get("resolved", False),
                "timestamp":  timestamp,
            }],
        )
    logger.info("seeded %d past tickets for user '%s'", len(tickets), user_id)


def clear_user_memory(user_id: str) -> None:
    """
    Remove all stored tickets for a user. Used between eval test cases to avoid bleed.
    """
    collection = _get_memory_collection()
    try:
        # Get all IDs for this user
        results = collection.get(where={"user_id": {"$eq": user_id}})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info("cleared %d memories for user '%s'", len(ids_to_delete), user_id)
    except Exception as e:
        logger.warning("clear_user_memory error: %s", e)


def get_user_ticket_count(user_id: str) -> int:
    """Return total number of stored tickets for a user."""
    collection = _get_memory_collection()
    try:
        results = collection.get(where={"user_id": {"$eq": user_id}})
        return len(results.get("ids", []))
    except Exception:
        return 0
