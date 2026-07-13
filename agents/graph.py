"""
agents/graph.py

LangGraph StateGraph wiring all four agents together with conditional edges.

Graph structure:
  START → intake → knowledge → action → [conditional] → escalation → END
                                   ↑                        |
                                   └──── retry_knowledge ───┘ (max 2 retries)

Retry logic:
- Tool failure: handled INSIDE action_agent's own ReAct loop (not a graph-level concern)
- Insufficient context: graph routes action → knowledge (up to 2 times), then forces escalation

recursion_limit=15 is passed at invocation time to fail predictably, not hang.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.escalation_agent import run_escalation_agent
from agents.intake_agent import run_intake_agent
from agents.knowledge_agent import run_knowledge_agent
from agents.action_agent import run_action_agent
from agents.state import SupportTicketState

load_dotenv()

GRAPH_RECURSION_LIMIT = 15


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

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
        print(f"[graph] route_after_action → retry_knowledge (retry #{knowledge_retries + 1})")
        return "retry_knowledge"

    if insufficient and knowledge_retries >= 2:
        print(f"[graph] route_after_action → escalation (knowledge retry limit reached)")
    else:
        print(f"[graph] route_after_action → escalation (action complete)")

    return "escalation"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(SupportTicketState)

    graph.add_node("intake", run_intake_agent)
    graph.add_node("knowledge", run_knowledge_agent)
    graph.add_node("action", run_action_agent)
    graph.add_node("escalation", run_escalation_agent)

    # Fixed edges
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "knowledge")
    graph.add_edge("knowledge", "action")
    graph.add_edge("escalation", END)

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


def run_graph(initial_state: SupportTicketState) -> SupportTicketState:
    """
    Synchronous invocation. Returns the final state.
    Uses recursion_limit=15 to prevent infinite loops.
    """
    app = get_compiled_app()
    result = app.invoke(
        initial_state,
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )
    return result


async def astream_graph(initial_state: SupportTicketState):
    """
    Async generator that yields graph chunks for streaming (used by Streamlit frontend).
    Each chunk is a dict: {node_name: state_updates}
    """
    app = get_compiled_app()
    async for chunk in app.astream(
        initial_state,
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    ):
        yield chunk
