"""
agents/knowledge_agent.py

RAG agent with an explicit ReAct loop (max 3 iterations):
  Thought (what query best captures this?) → Action (vector search) →
  Observation (is retrieved context relevant?) → loop if relevance < 0.6

Key behaviors:
- Uses cosine similarity to evaluate retrieval quality at each iteration
- Reformulates the query if relevance is low (citing the intake_reasoning for guidance)
- Sets knowledge_reasoning: WHY this context was selected (passed to action_agent)
- knowledge_reasoning explicitly states whether context is sufficient — action_agent
  uses this to evaluate insufficient_context at the end of its own loop
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.llm_utils import build_llm, extract_text
from agents.state import ReActStep, SupportTicketState
from logging_config import get_logger

load_dotenv()
logger = get_logger(__name__)

MODEL_NAME   = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
CHROMA_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION   = "knowledge_base"
MAX_RAG_ITER = 3
RELEVANCE_THRESHOLD = 0.60   # cosine similarity; below this → reformulate and retry
TOP_K = 4                    # chunks to retrieve per query


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_collection():
    """Return the ChromaDB knowledge_base collection with local embeddings."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(COLLECTION, embedding_function=embedding_fn)


def _search(collection, query: str, n: int = TOP_K) -> tuple[str, float, list[str]]:
    """
    Run a vector search query.
    Returns: (concatenated_context, best_similarity_score, source_list)
    """
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "distances", "metadatas"],
    )
    docs      = results["documents"][0]
    distances = results["distances"][0]   # cosine distance (lower = more similar)
    metas     = results["metadatas"][0]

    # Convert distance → similarity (ChromaDB cosine distance: 0=identical, 2=opposite)
    similarities = [max(0.0, 1.0 - d) for d in distances]
    best_similarity = max(similarities) if similarities else 0.0

    sources = [m.get("source", "unknown") for m in metas]
    context_parts = []
    for doc, sim, src in zip(docs, similarities, sources):
        context_parts.append(f"[Source: {src} | similarity: {sim:.3f}]\n{doc}")

    return "\n\n---\n\n".join(context_parts), best_similarity, sources


QUERY_REFINEMENT_PROMPT = """You are helping reformulate a search query to better retrieve customer support documentation.

Original ticket: {ticket_text}
Classification: {classification}
Intake reasoning: {intake_reasoning}

Previous query: {prev_query}
Previous retrieval relevance score: {prev_score:.3f} (below threshold of 0.60)
Previous retrieved context summary: {prev_context_summary}

The previous query did not retrieve sufficiently relevant documents.
Formulate a DIFFERENT, more specific query that would better find:
- The specific policy or procedure the customer needs
- Technical troubleshooting steps for this issue type
- Relevant FAQ content

Respond with ONLY the new search query (no explanation, just the query string)."""


def run_knowledge_agent(state: SupportTicketState, config=None) -> dict:
    """LangGraph node function for the knowledge agent.

    `config` intentionally has no type annotation: LangGraph only injects the
    runtime RunnableConfig into a node if this param is unannotated or typed
    exactly RunnableConfig/Optional[RunnableConfig] (checked by literal string
    match, since this module uses `from __future__ import annotations`).
    """

    collection = _get_collection()

    ticket_text    = state.get("ticket_text", "")
    classification = state.get("classification", "general")
    urgency        = state.get("urgency", "medium")
    intake_reasoning = state.get("intake_reasoning", "")

    new_trace_steps: list[ReActStep] = []
    existing_trace = state.get("react_trace", [])

    # Build initial query from the ticket and classification
    initial_query = f"{classification} {ticket_text}"
    # Combine the raw request with its category to target the appropriate policy
    # domain without discarding customer-specific details.

    current_query = initial_query
    best_context  = ""
    best_score    = 0.0
    best_sources: list[str] = []
    rag_iterations = 0

    logger.info("knowledge_agent start classification=%s urgency=%s", classification, urgency)

    for iteration in range(1, MAX_RAG_ITER + 1):
        rag_iterations = iteration
        thought = (
            f"Iteration {iteration}: Searching for context about '{current_query}'. "
            f"Relevance threshold: {RELEVANCE_THRESHOLD}. "
            + (f"Previous score was {best_score:.3f} — reformulating." if iteration > 1 else "Initial query.")
        )
        logger.debug("RAG iteration %d/%d THOUGHT: %s", iteration, MAX_RAG_ITER, thought)
        logger.debug("ACTION: vector_search(query=%r, k=%d)", current_query, TOP_K)

        context, score, sources = _search(collection, current_query, n=TOP_K)

        observation = (
            f"Retrieved {TOP_K} chunks from {set(sources)}. "
            f"Best similarity: {score:.3f}. "
            f"{'Relevance SUFFICIENT — stopping search.' if score >= RELEVANCE_THRESHOLD else 'Relevance LOW — will reformulate.'}"
        )
        logger.debug("OBSERVATION: %s", observation)

        step = ReActStep(
            agent="knowledge",
            iteration=iteration,
            thought=thought,
            action=f"vector_search(query='{current_query}', k={TOP_K})",
            observation=observation,
            timestamp=_now(),
        )
        new_trace_steps.append(step)

        # Keep best result across iterations
        if score > best_score:
            best_score   = score
            best_context = context
            best_sources = sources

        if score >= RELEVANCE_THRESHOLD:
            logger.debug("relevance threshold met (%.3f >= %.2f), stopping", score, RELEVANCE_THRESHOLD)
            break

        # Need to reformulate — use LLM to generate a better query
        if iteration < MAX_RAG_ITER:
            refinement_response = build_llm(config, model=MODEL_NAME, temperature=0, max_output_tokens=512).invoke([
                SystemMessage(content="You are a search query refinement specialist."),
                HumanMessage(content=QUERY_REFINEMENT_PROMPT.format(
                    ticket_text=ticket_text,
                    classification=classification,
                    intake_reasoning=intake_reasoning,
                    prev_query=current_query,
                    prev_score=score,
                    prev_context_summary=context[:300] + "..." if len(context) > 300 else context,
                )),
            ])
            new_query = extract_text(refinement_response.content)
            current_query = new_query[:200]  # cap query length
            logger.debug("reformulated query: %r", current_query)

    # ---- Generate knowledge_reasoning ----
    # The LLM synthesizes what was retrieved and whether it's sufficient
    # Skip this LLM round-trip when the very first search already cleared the
    # relevance bar cleanly: action_agent already receives best_context (the
    # raw retrieved text) directly, so this call is a "reflect on what we just
    # found" nicety, not something the pipeline functionally depends on. Kept
    # for the cases where it earns its cost: multi-iteration or weak retrieval,
    # where an actual synthesis over ambiguous/partial context has real value.
    if rag_iterations == 1 and best_score >= RELEVANCE_THRESHOLD:
        knowledge_reasoning = (
            f"Strong match found on the first search (similarity {best_score:.3f} "
            f"from {set(best_sources)}) — no reformulation needed. See retrieved_context "
            f"for the full policy text."
        )
    else:
        reasoning_prompt = f"""You retrieved the following context for this support ticket:

Ticket: {ticket_text}
Classification: {classification}

Retrieved context (best {TOP_K} chunks, best similarity: {best_score:.3f}):
{best_context[:1500]}

Write a 2-3 sentence summary explaining:
1. What relevant information was retrieved (be specific about policies or procedures found)
2. Whether this context is sufficient for an action agent to resolve the ticket, or what information is missing
3. Any key facts from the context the action agent should be aware of

Be concise and factual. This will be passed to the action agent."""

        reasoning_response = build_llm(config, model=MODEL_NAME, temperature=0, max_output_tokens=768).invoke([
            SystemMessage(content="You are a knowledge synthesis agent."),
            HumanMessage(content=reasoning_prompt),
        ])
        knowledge_reasoning = extract_text(reasoning_response.content)

    logger.info(
        "knowledge_agent done relevance_score=%.3f iterations=%d",
        best_score, rag_iterations,
    )
    logger.debug("knowledge_reasoning: %s", knowledge_reasoning[:300])

    return {
        "retrieved_context": best_context,
        "retrieval_relevance_score": best_score,
        "knowledge_reasoning": knowledge_reasoning,
        "rag_iterations": rag_iterations,
        "react_trace": existing_trace + new_trace_steps,
        # Increment the graph-level knowledge retry counter
        "knowledge_retry_count": state.get("knowledge_retry_count", 0) + 1,
    }
