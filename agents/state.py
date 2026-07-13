"""
agents/state.py

Shared LangGraph state schema for the Multi-Agent Customer Support System.
All agents read from and write to this single state object as it flows through the graph.

Key design decisions:
- ReActStep stores every Thought/Action/Observation across ALL agents, ordered.
  This is the primary deliverable for demos — the interviewer can see literal reasoning.
- knowledge_retry_count is separate from tool_retry_count to keep two concerns distinct:
    - knowledge_retry_count: graph-level loops back to knowledge_agent (max 2)
    - tool_retry_count: intra-loop retries within action_agent (counted inside the node)
- ConfidenceBreakdown is a named struct so escalation_agent's score is auditable,
  not a single opaque float.
"""

from __future__ import annotations

from typing import Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class ReActStep(BaseModel):
    """One Thought → Action → Observation cycle from any agent."""
    agent: str                   # "intake" | "knowledge" | "action" | "escalation"
    iteration: int               # 1-indexed within that agent's loop
    thought: str                 # LLM's reasoning before acting
    action: str                  # What the agent did (tool call, query, classification, etc.)
    observation: str             # Result of the action
    timestamp: Optional[str] = None  # ISO timestamp, set automatically in each agent


class ToolCallRecord(BaseModel):
    """Detailed record of a single tool call made by action_agent."""
    tool_name: str
    arguments: dict[str, Any]
    result: Any                  # Raw result from the tool
    success: bool
    error_message: Optional[str] = None


class ConfidenceBreakdown(BaseModel):
    """
    Named components of the escalation confidence score.
    Each component is 0.0–1.0. Final confidence is a weighted average.
    Stored in state so the trace is auditable.
    """
    retrieval_relevance: float = Field(
        description="How well knowledge_agent's retrieved context matched the ticket. "
                    "From retrieval_relevance_score."
    )
    tool_call_success: float = Field(
        description="Fraction of action_agent's tool calls that succeeded. "
                    "0.0 = all failed, 1.0 = all succeeded."
    )
    classification_confidence: float = Field(
        description="intake_agent's self-reported confidence in its classification."
    )
    sentiment_score: float = Field(
        description="Normalized sentiment: 1.0 = positive, 0.5 = neutral, 0.0 = angry/negative."
    )
    overall: float = Field(
        description="Weighted average of all components. "
                    "High overall → auto_resolve. Low → escalate."
    )
    weights_used: dict[str, float] = Field(
        default_factory=lambda: {
            "retrieval_relevance": 0.20,
            "tool_call_success":   0.40,
            "classification_confidence": 0.20,
            "sentiment_score":     0.20,
        },
        description="Weights used in the weighted average computation."
    )


# ---------------------------------------------------------------------------
# Main shared state
# ---------------------------------------------------------------------------

class SupportTicketState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph StateGraph.
    total=False means all fields are optional at state creation time;
    each agent populates its own fields.
    """

    # ---- Input (set once at graph entry) ----
    ticket_id: str
    user_id: str
    ticket_text: str

    # ---- Intake Agent outputs ----
    classification: str           # billing | technical | refund | account | general
    urgency: str                  # low | medium | high | critical
    sentiment: str                # positive | neutral | negative | angry
    classification_confidence: float   # 0.0–1.0, agent's self-reported confidence
    intake_reasoning: str         # WHY the agent classified this way (passed to next agents)
    # Past tickets retrieved from memory — informs classification
    similar_past_tickets: List[dict]   # [{text, resolution, resolved}, ...]

    # ---- Knowledge Agent outputs ----
    retrieved_context: str        # Concatenated top-k chunks
    retrieval_relevance_score: float   # Cosine similarity of best chunk (0.0–1.0)
    knowledge_reasoning: str      # WHY this context was selected and whether it's sufficient
    rag_iterations: int           # How many RAG search iterations knowledge_agent ran

    # ---- Action Agent outputs ----
    tool_calls_made: List[ToolCallRecord]   # Full record of every tool call
    action_result: str            # Final natural-language outcome description
    action_success: bool          # True if the primary task was completed
    tool_retry_count: int         # Intra-loop retries within action_agent (NOT graph routing)
    insufficient_context: bool    # True if action_agent determined knowledge gap → re-route

    # ---- Escalation Agent outputs ----
    escalation_decision: str      # auto_resolve | escalate | request_info
    escalation_justification: str # Natural language citing specific trace evidence
    confidence_breakdown: ConfidenceBreakdown  # Auditable named components
    final_response: str           # The response to send to the customer

    # ---- Full reasoning trace (all agents, all steps, in order) ----
    react_trace: List[ReActStep]  # Append-only; each agent adds its steps here

    # ---- Graph control flow ----
    knowledge_retry_count: int    # How many times graph has looped back to knowledge_agent (max 2)
    error_message: str            # Set if a node encounters an unrecoverable error


# ---------------------------------------------------------------------------
# Helper: serialize state for API response
# ---------------------------------------------------------------------------

def state_to_dict(state: SupportTicketState) -> dict:
    """Convert state to a JSON-serializable dict for API responses."""
    result = dict(state)

    # Convert Pydantic models to dicts
    if "react_trace" in result and result["react_trace"]:
        result["react_trace"] = [
            step.model_dump() if isinstance(step, ReActStep) else step
            for step in result["react_trace"]
        ]

    if "tool_calls_made" in result and result["tool_calls_made"]:
        result["tool_calls_made"] = [
            tc.model_dump() if isinstance(tc, ToolCallRecord) else tc
            for tc in result["tool_calls_made"]
        ]

    if "confidence_breakdown" in result and result["confidence_breakdown"]:
        cb = result["confidence_breakdown"]
        result["confidence_breakdown"] = (
            cb.model_dump() if isinstance(cb, ConfidenceBreakdown) else cb
        )

    if "similar_past_tickets" not in result:
        result["similar_past_tickets"] = []

    return result
