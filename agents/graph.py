"""
agents/graph.py

LangGraph StateGraph wiring all four agents together with conditional edges.

Graph structure:
  START → intake → [conditional] → knowledge → action → [conditional] → escalation → END
                        |              ↑                        |
                        └── skip if ───┘──── retry_knowledge ───┘ (max 2 retries)
                        needs_clarification

Retry logic:
- Tool failure: handled INSIDE action_agent's own ReAct loop (not a graph-level concern)
- Insufficient context: graph routes action → knowledge (up to 2 times), then forces escalation

Latency shortcut:
- If intake determines the ticket is too ambiguous to act on (needs_clarification),
  route straight to escalation — there's nothing for knowledge/action to usefully
  do yet, and running them anyway would burn several sequential LLM calls just to
  arrive at the same "please clarify" outcome. See route_after_intake.

recursion_limit=15 is passed at invocation time to fail predictably, not hang.
"""

from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.escalation_agent import run_escalation_agent
from agents.intake_agent import run_intake_agent
from agents.knowledge_agent import run_knowledge_agent
from agents.action_agent import run_action_agent
from agents.state import SupportTicketState
from logging_config import get_logger, set_correlation_id
from metrics import agent_latency_seconds, record_run_metrics

load_dotenv()
logger = get_logger(__name__)

GRAPH_RECURSION_LIMIT = 15


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_intake(state: SupportTicketState) -> str:
    """
    When the ticket itself is too ambiguous to act on (e.g. no order ID),
    intake sets needs_clarification=True. There's nothing for knowledge_agent
    to usefully retrieve or action_agent to act on yet, so skip straight to
    escalation instead of burning a full RAG search + ReAct tool-calling
    attempt (each several sequential LLM calls) just to end up asking the
    same clarifying question anyway. This is a latency optimization, not a
    behavior change: those two agents would very likely have landed on
    request_info regardless, just after several times the round-trips.
    """
    if state.get("needs_clarification", False):
        logger.info("route_after_intake -> escalation (needs_clarification, skipping knowledge+action)")
        return "escalation"
    return "knowledge"


def route_after_action(state: SupportTicketState) -> str:
    """
    Two branches only. Tool-call failure/retry is NOT a graph-level concern —
    it is handled inside action_agent's own ReAct loop.

    Branch 1: retry_knowledge — action_agent needs better context (knowledge gap)
    Branch 2: escalation     — everything else (complete, failed, retry limit hit)
    """
    insufficient = state.get("insufficient_context", False)
    knowledge_retries = state.get("knowledge_retry_count", 0)

    if insufficient and knowledge_retries < 2:
        logger.info("route_after_action -> retry_knowledge (retry #%d)", knowledge_retries + 1)
        return "retry_knowledge"
    # Only knowledge gaps re-enter RAG; tool failures stay inside the Action
    # Agent's local ReAct loop to avoid duplicate retries.

    if insufficient and knowledge_retries >= 2:
        logger.info("route_after_action -> escalation (knowledge retry limit reached)")
    else:
        logger.info("route_after_action -> escalation (action complete)")

    return "escalation"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def _instrument(func, agent_name: str):
    """Wrap a node function with per-agent latency observation.
    Uses functools.wraps so inspect.signature (which LangGraph uses to decide
    whether to inject `config`) follows __wrapped__ back to the original
    function's signature — the wrapper's own `(state, config=None)` shape is
    identical anyway, but this keeps it robust if that ever changes.
    """
    @functools.wraps(func)
    def wrapper(state, config=None):
        start = time.time()
        try:
            return func(state, config)
        finally:
            agent_latency_seconds.labels(agent=agent_name).observe(time.time() - start)
    return wrapper


def build_graph() -> StateGraph:
    graph = StateGraph(SupportTicketState)

    graph.add_node("intake", _instrument(run_intake_agent, "intake"))
    graph.add_node("knowledge", _instrument(run_knowledge_agent, "knowledge"))
    graph.add_node("action", _instrument(run_action_agent, "action"))
    graph.add_node("escalation", _instrument(run_escalation_agent, "escalation"))

    # Fixed edges
    graph.add_edge(START, "intake")
    graph.add_edge("knowledge", "action")
    graph.add_edge("escalation", END)

    # Conditional: intake → escalation directly (needs_clarification, skips
    #              knowledge+action entirely — see route_after_intake)
    #              intake → knowledge (normal path)
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "knowledge": "knowledge",
            "escalation": "escalation",
        },
    )

    # Conditional: action → knowledge (insufficient context, max 2 retries)
    #              action → escalation (complete or retry limit hit)
    graph.add_conditional_edges(
        "action",
        route_after_action,
        {
            "retry_knowledge": "knowledge",
            "escalation": "escalation",
        },
    )

    return graph


# Compile once — reuse across requests
_compiled_app = None


def get_compiled_app():
    global _compiled_app
    if _compiled_app is None:
        graph = build_graph()
        _compiled_app = graph.compile()
    return _compiled_app


def _build_config(api_key: str | None) -> dict:
    return {
        "recursion_limit": GRAPH_RECURSION_LIMIT,
        "configurable": {"gemini_api_key": api_key},
    }


def run_graph(initial_state: SupportTicketState, api_key: str | None = None) -> SupportTicketState:
    """
    Synchronous invocation. Returns the final state.
    Uses recursion_limit=15 to prevent infinite loops.

    api_key: optional caller-supplied Gemini API key (BYOK). Threaded through
    LangGraph's `config` — never stored in state — so it never ends up persisted
    or serialized alongside the ticket trace. Falls back to GOOGLE_API_KEY env
    var when omitted.
    """
    set_correlation_id(
        ticket_id=initial_state.get("ticket_id"),
        conversation_id=initial_state.get("conversation_id"),
    )
    app = get_compiled_app()
    start = time.time()
    try:
        result = app.invoke(initial_state, config=_build_config(api_key))
        record_run_metrics(result, time.time() - start, status="completed")
        return result
    except Exception:
        record_run_metrics({}, time.time() - start, status="error")
        raise


async def astream_graph(initial_state: SupportTicketState, api_key: str | None = None):
    """
    Async generator that yields graph chunks for streaming (used by the chat frontend).
    Each chunk is a dict: {node_name: state_updates}

    api_key: see run_graph().
    """
    set_correlation_id(
        ticket_id=initial_state.get("ticket_id"),
        conversation_id=initial_state.get("conversation_id"),
    )
    app = get_compiled_app()
    start = time.time()
    final_state: dict = {}
    try:
        async for chunk in app.astream(initial_state, config=_build_config(api_key)):
            for update in chunk.values():
                final_state.update(update)
            yield chunk
        record_run_metrics(final_state, time.time() - start, status="completed")
    except Exception:
        record_run_metrics(final_state, time.time() - start, status="error")
        raise
