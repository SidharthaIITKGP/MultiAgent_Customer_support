from langchain_core.messages import AIMessage

from agents.escalation_agent import _compute_confidence_breakdown, _parse_escalation_json, run_escalation_agent
from agents.state import ToolCallRecord


def test_confidence_breakdown_weighted_average_with_pydantic_tool_calls():
    state = {
        "retrieval_relevance_score": 0.8,
        "classification_confidence": 0.9,
        "sentiment": "positive",
        "tool_calls_made": [
            ToolCallRecord(tool_name="check_order_status", arguments={}, result={}, success=True),
            ToolCallRecord(tool_name="process_refund", arguments={}, result={}, success=False),
        ],
    }
    cb = _compute_confidence_breakdown(state)
    assert cb.tool_call_success == 0.5
    assert cb.retrieval_relevance == 0.8
    assert cb.classification_confidence == 0.9
    assert cb.sentiment_score == 1.0
    expected = round(0.8 * 0.2 + 0.5 * 0.4 + 0.9 * 0.2 + 1.0 * 0.2, 3)
    assert cb.overall == expected


def test_confidence_breakdown_with_plain_dict_tool_calls():
    state = {
        "tool_calls_made": [
            {"tool_name": "x", "success": True},
            {"tool_name": "y", "success": True},
        ],
        "sentiment": "angry",
    }
    cb = _compute_confidence_breakdown(state)
    assert cb.tool_call_success == 1.0
    assert cb.sentiment_score == 0.0


def test_confidence_breakdown_no_tool_calls_is_neutral():
    cb = _compute_confidence_breakdown({})
    assert cb.tool_call_success == 0.5
    assert cb.sentiment_score == 0.5  # unknown sentiment defaults neutral


def test_parse_escalation_json_valid():
    text = '```json\n{"decision": "auto_resolve", "justification": "ok", "final_response": "done"}\n```'
    result = _parse_escalation_json(text)
    assert result["decision"] == "auto_resolve"
    assert result["final_response"] == "done"


def test_parse_escalation_json_falls_back_to_escalate_on_garbage():
    result = _parse_escalation_json("not json at all")
    assert result["decision"] == "escalate"


def test_needs_clarification_skip_note_reaches_the_llm_prompt(install_fake_llm, monkeypatch):
    """When intake skipped knowledge/action (needs_clarification), escalation's
    prompt must explicitly say so and carry the clarifying_question — otherwise
    the LLM would see empty knowledge/action results with no explanation."""
    import memory.ticket_memory as ticket_memory
    monkeypatch.setattr(ticket_memory, "store_ticket_resolution", lambda **kwargs: "mem-id")

    fake = install_fake_llm([
        AIMessage(content='{"decision": "request_info", '
                           '"justification": "ticket is missing an order ID", '
                           '"final_response": "Could you share your order ID?"}'),
    ])

    state = {
        "ticket_id": "t1", "user_id": "u001", "ticket_text": "where is my order",
        "needs_clarification": True, "clarifying_question": "Could you share your order ID?",
        "react_trace": [],
    }
    result = run_escalation_agent(state)

    assert result["escalation_decision"] == "request_info"
    sent_prompt = fake.calls[0][1].content  # [SystemMessage, HumanMessage(context)]
    assert "SKIPPED" in sent_prompt
    assert "Could you share your order ID?" in sent_prompt
