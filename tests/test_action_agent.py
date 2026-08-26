from langchain_core.messages import AIMessage

from agents.action_agent import run_action_agent


def _tool_call_message(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])


def test_clean_success_skips_context_eval_call(install_fake_llm, patched_backend):
    """No tool failures + a clean final answer -> the extra context-sufficiency
    LLM round-trip should be skipped entirely (latency optimization)."""
    fake = install_fake_llm([
        _tool_call_message("check_order_status", {"order_id": "1001"}),
        _tool_call_message("process_refund", {"order_id": "1001", "user_id": "u001", "amount": 89.99}),
        AIMessage(content="Refund processed successfully."),  # final answer, no tool calls -> breaks loop
    ])

    state = {
        "ticket_id": "t1", "user_id": "u001", "ticket_text": "refund order 1001",
        "react_trace": [], "tool_calls_made": [],
    }
    result = run_action_agent(state)

    assert result["action_success"] is True
    assert result["insufficient_context"] is False
    assert "Skipped" in [s.observation for s in result["react_trace"] if "insufficient_context" in s.observation][0]
    # 3 scripted responses consumed (2 tool-call turns + 1 final answer), no 4th
    # (eval) call was made — the fake would just return an empty AIMessage if
    # over-called, so assert the call count matches exactly.
    assert len(fake.calls) == 3


def test_tool_failure_still_runs_context_eval_call(install_fake_llm, patched_backend):
    """A failed tool call (locked account) should still trigger the real
    context-sufficiency eval call — this path must not be short-circuited."""
    fake = install_fake_llm([
        _tool_call_message("process_refund", {"order_id": "1003", "user_id": "u002", "amount": 149.0}),
        AIMessage(content="The refund failed because the account is locked."),  # final answer after failure
        AIMessage(content='{"insufficient_context": false, "reason": "policy context was fine"}'),
    ])

    state = {
        "ticket_id": "t2", "user_id": "u002", "ticket_text": "refund order 1003",
        "react_trace": [], "tool_calls_made": [],
    }
    result = run_action_agent(state)

    assert result["insufficient_context"] is False
    # All 3 scripted responses were consumed, including the eval call.
    assert len(fake.calls) == 3
    reason_step = [s.observation for s in result["react_trace"] if "insufficient_context" in s.observation][-1]
    assert "policy context was fine" in reason_step
