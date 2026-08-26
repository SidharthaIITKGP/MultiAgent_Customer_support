"""
metrics.py — Prometheus metrics for the support system.

Skips OpenTelemetry/Jaeger deliberately: this is one process talking to Gemini,
a local mock backend, and local ChromaDB — no service mesh to trace across.
react_trace + these metrics + correlated JSON logs cover "is it fast / did it
work / what happened on ticket X" without that operational cost.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

tickets_processed_total = Counter(
    "tickets_processed_total", "Tickets processed by the graph", ["status"]
)
escalation_decisions_total = Counter(
    "escalation_decisions_total", "Escalation decisions made", ["decision"]
)
tool_calls_total = Counter(
    "tool_calls_total", "Tool calls made by the action agent", ["tool_name", "success"]
)
agent_latency_seconds = Histogram(
    "agent_latency_seconds", "Per-agent-node latency", ["agent"]
)
graph_latency_seconds = Histogram(
    "graph_latency_seconds", "End-to-end graph run latency"
)
retrieval_relevance = Histogram(
    "retrieval_relevance", "Knowledge agent retrieval relevance score"
)


def record_run_metrics(final_state: dict, elapsed_s: float, status: str) -> None:
    """Call once after a graph run (success or error) completes."""
    tickets_processed_total.labels(status=status).inc()
    graph_latency_seconds.observe(elapsed_s)

    decision = final_state.get("escalation_decision")
    if decision:
        escalation_decisions_total.labels(decision=decision).inc()

    score = final_state.get("retrieval_relevance_score")
    if isinstance(score, (int, float)):
        retrieval_relevance.observe(score)

    for tc in final_state.get("tool_calls_made", []) or []:
        name = tc.tool_name if hasattr(tc, "tool_name") else tc.get("tool_name", "unknown")
        success = tc.success if hasattr(tc, "success") else tc.get("success", False)
        tool_calls_total.labels(tool_name=name, success=str(bool(success))).inc()
