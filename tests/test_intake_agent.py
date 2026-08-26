import json

from langchain_core.messages import AIMessage

from agents.intake_agent import run_intake_agent


def test_needs_clarification_and_question_surface_into_state(install_fake_llm):
    install_fake_llm([
        AIMessage(content=json.dumps({
            "classification": "general",
            "urgency": "low",
            "sentiment": "neutral",
            "classification_confidence": 0.9,
            "intake_reasoning": "Customer gave no order ID or account detail.",
            "needs_clarification": True,
            "clarifying_question": "Could you share your order ID?",
        })),
    ])

    state = {"ticket_text": "where is my order", "user_id": "u001"}
    result = run_intake_agent(state)

    assert result["needs_clarification"] is True
    assert result["clarifying_question"] == "Could you share your order ID?"


def test_confident_classification_does_not_need_clarification(install_fake_llm):
    install_fake_llm([
        AIMessage(content=json.dumps({
            "classification": "refund",
            "urgency": "medium",
            "sentiment": "neutral",
            "classification_confidence": 0.95,
            "intake_reasoning": "Clear refund request with order ID 1001.",
            "needs_clarification": False,
            "clarifying_question": None,
        })),
    ])

    state = {"ticket_text": "refund order 1001, item damaged", "user_id": "u001"}
    result = run_intake_agent(state)

    assert result["needs_clarification"] is False
    assert result["clarifying_question"] is None
