"""
eval/run_eval.py

Runs all test cases through the LangGraph, measures:
  - Decision accuracy (escalation precision)
  - Tool-call success rate
  - Average ReAct iterations per agent
  - Retrieval relevance distribution
  - Memory-escalation assertion (TC-16)
  - End-to-end latency per ticket

Usage:
    # Start mock_backend first: uvicorn mock_backend:app --port 8000
    # Run: python eval/run_eval.py [--cases tc_001 tc_006 tc_016] [--no-reset]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.graph import run_graph  # noqa: E402
from agents.state import state_to_dict  # noqa: E402
from memory.ticket_memory import clear_user_memory, seed_past_tickets  # noqa: E402


# ---------------------------------------------------------------------------
# Load test cases
# ---------------------------------------------------------------------------

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"


def load_test_cases(filter_ids: list[str] | None = None) -> list[dict]:
    cases = json.loads(TEST_CASES_PATH.read_text())
    # Filtering supports full benchmarks and focused regression runs using the
    # same fixture file.
    if filter_ids:
        cases = [c for c in cases if c["id"] in filter_ids]
    return cases


# ---------------------------------------------------------------------------
# Run a single test case
# ---------------------------------------------------------------------------

def run_single(tc: dict, reset_backend: bool = True) -> dict:
    """Run one test case and return a result dict."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {tc['id']} — {tc['description']}")
    print(f"{'='*70}")

    user_id = tc["user_id"]

    # ---- Seed past tickets for TC-16 (memory test) ----
    past_tickets = tc.get("seed_past_tickets", [])
    if past_tickets:
        clear_user_memory(user_id)   # reset first to avoid bleed from previous runs
        seed_past_tickets(user_id, past_tickets)
    else:
        # Don't clear memory for users without explicit seeding (let natural memory accumulate)
        pass

    # ---- Build initial state ----
    ticket_id = tc["id"] + "_" + uuid.uuid4().hex[:6]
    initial_state = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "ticket_text": tc["ticket_text"],
        "react_trace": [],
        "tool_calls_made": [],
        "knowledge_retry_count": 0,
        "tool_retry_count": 0,
        "insufficient_context": False,
    }

    # ---- Run the graph ----
    start_time = time.time()
    try:
        final_state = run_graph(initial_state)
        elapsed = time.time() - start_time
        error = None
    except Exception as e:
        elapsed = time.time() - start_time
        final_state = initial_state
        error = str(e)
        print(f"ERROR: {error}")

    serialized = state_to_dict(final_state)

    # ---- Extract metrics ----
    tool_calls = serialized.get("tool_calls_made", [])
    n_tool_calls = len(tool_calls)
    n_successful = sum(1 for tc_r in tool_calls if tc_r.get("success", False))
    tool_success_rate = n_successful / n_tool_calls if n_tool_calls > 0 else None

    react_trace = serialized.get("react_trace", [])
    iter_counts = {}
    for step in react_trace:
        agent = step.get("agent", "?")
        iter_counts[agent] = max(iter_counts.get(agent, 0), step.get("iteration", 0))

    decision = serialized.get("escalation_decision", "MISSING")
    expected = tc.get("expected_decision", "")
    decision_correct = (decision == expected)

    retrieval_score = serialized.get("retrieval_relevance_score", None)

    # ---- TC-16 specific assertion ----
    memory_escalation_triggered = None
    if tc["id"] == "tc_016":
        intake_reasoning = serialized.get("intake_reasoning", "").lower()
        keywords = ["prior", "repeat", "history", "previous", "unresolved", "past", "again", "multiple"]
        memory_escalation_triggered = any(kw in intake_reasoning for kw in keywords)
        if not memory_escalation_triggered:
            print("  ⚠️  TC-016 ASSERTION FAILED: intake_reasoning doesn't reference past ticket history!")
            print(f"     intake_reasoning: {intake_reasoning[:200]}")

    # ---- Print result summary ----
    status_icon = "✓" if decision_correct else "✗"
    print(f"\n{status_icon} Decision: {decision} (expected: {expected}) | {'PASS' if decision_correct else 'FAIL'}")
    print(f"  Latency: {elapsed:.1f}s")
    print(f"  Tool calls: {n_tool_calls} ({n_successful} succeeded)")
    print(f"  ReAct iterations: {iter_counts}")
    print(f"  Retrieval relevance: {retrieval_score:.3f}" if retrieval_score else "  Retrieval relevance: N/A")
    if serialized.get("confidence_breakdown"):
        cb = serialized["confidence_breakdown"]
        print(f"  Confidence breakdown: {json.dumps({k: v for k, v in cb.items() if k != 'weights_used'})}")
    if memory_escalation_triggered is not None:
        icon = "✓" if memory_escalation_triggered else "✗"
        print(f"  {icon} memory_escalation_triggered: {memory_escalation_triggered}")
    if error:
        print(f"  ERROR: {error}")

    return {
        "id": tc["id"],
        "description": tc["description"],
        "decision": decision,
        "expected_decision": expected,
        "decision_correct": decision_correct,
        "latency_s": round(elapsed, 2),
        "n_tool_calls": n_tool_calls,
        "n_successful_tool_calls": n_successful,
        "tool_success_rate": round(tool_success_rate, 3) if tool_success_rate is not None else None,
        "iter_counts": iter_counts,
        "retrieval_relevance": round(retrieval_score, 3) if retrieval_score else None,
        "memory_escalation_triggered": memory_escalation_triggered,
        "confidence_breakdown": serialized.get("confidence_breakdown"),
        "escalation_justification": serialized.get("escalation_justification", ""),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    print(f"\n\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")

    n_total = len(results)
    n_correct = sum(1 for r in results if r["decision_correct"])
    decision_accuracy = n_correct / n_total if n_total else 0

    all_success_rates = [r["tool_success_rate"] for r in results if r["tool_success_rate"] is not None]
    all_latencies = [r["latency_s"] for r in results]
    all_relevances = [r["retrieval_relevance"] for r in results if r["retrieval_relevance"] is not None]

    avg_iter = {}
    for r in results:
        for agent, iters in r.get("iter_counts", {}).items():
            avg_iter.setdefault(agent, []).append(iters)

    print(f"\n📊 Decision Accuracy:      {n_correct}/{n_total} = {decision_accuracy:.1%}")
    print(f"🔧 Tool-call Success Rate: {sum(all_success_rates)/len(all_success_rates):.1%}" if all_success_rates else "🔧 Tool-call Success Rate: N/A")
    print(f"⏱  Avg Latency:            {sum(all_latencies)/len(all_latencies):.1f}s")
    print(f"📚 Avg Retrieval Relevance: {sum(all_relevances)/len(all_relevances):.3f}" if all_relevances else "📚 Avg Retrieval Relevance: N/A")

    print("\n📈 Avg ReAct iterations per agent:")
    for agent, iters in sorted(avg_iter.items()):
        print(f"    {agent}: {sum(iters)/len(iters):.1f} avg (max {max(iters)})")

    # TC-16 memory assertion
    tc016 = next((r for r in results if r["id"] == "tc_016"), None)
    if tc016:
        triggered = tc016.get("memory_escalation_triggered")
        icon = "✓" if triggered else "✗"
        print(f"\n🧠 {icon} TC-016 memory_escalation_triggered: {triggered}")

    # Per-case table
    print(f"\n{'─'*70}")
    print(f"{'ID':<10} {'Decision':<15} {'Expected':<15} {'Correct':<8} {'Latency':>8} {'Tools':>6} {'Relevance':>10}")
    print(f"{'─'*70}")
    for r in results:
        correct_icon = "✓" if r["decision_correct"] else "✗"
        rel = f"{r['retrieval_relevance']:.3f}" if r["retrieval_relevance"] else "N/A"
        print(f"{r['id']:<10} {r['decision']:<15} {r['expected_decision']:<15} {correct_icon:<8} {r['latency_s']:>7.1f}s {r['n_tool_calls']:>6} {rel:>10}")

    # Save full results to file
    output_path = Path(__file__).parent / "eval_results.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n📁 Full results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation test cases through the support agent graph")
    parser.add_argument("--cases", nargs="*", help="Specific test case IDs to run (e.g. tc_001 tc_006 tc_016)")
    parser.add_argument("--no-seed", action="store_true", help="Skip seeding past tickets for TC-016")
    args = parser.parse_args()

    test_cases = load_test_cases(filter_ids=args.cases)
    print(f"Running {len(test_cases)} test case(s)...")
    print("Make sure mock_backend.py is running: uvicorn mock_backend:app --port 8000")
    print("Make sure ChromaDB is ingested: python knowledge_base/ingest.py")

    results = []
    for tc in test_cases:
        result = run_single(tc)
        results.append(result)

    print_summary(results)
