from agents.knowledge_agent import run_knowledge_agent
import agents.knowledge_agent as knowledge_agent


class _FakeCollection:
    def __init__(self, similarity):
        self._similarity = similarity

    def query(self, query_texts, n_results, include):
        distance = 1.0 - self._similarity
        return {
            "documents": [["Refunds are eligible within 30 days of delivery."] * n_results],
            "distances": [[distance] * n_results],
            "metadatas": [[{"source": "refund_policy.md"}] * n_results],
        }


def test_strong_first_hit_skips_reasoning_llm_call(install_fake_llm, monkeypatch):
    """A clean, above-threshold match on iteration 1 should skip the
    reasoning-synthesis LLM call entirely — zero LLM calls for this node."""
    fake = install_fake_llm([])  # no responses scripted; any .invoke() call is a bug
    monkeypatch.setattr(knowledge_agent, "_get_collection", lambda: _FakeCollection(0.85))

    state = {
        "ticket_text": "refund order 1001", "classification": "refund",
        "urgency": "medium", "intake_reasoning": "clear refund request",
        "react_trace": [],
    }
    result = run_knowledge_agent(state)

    assert result["rag_iterations"] == 1
    assert result["retrieval_relevance_score"] == 0.85
    assert "no reformulation needed" in result["knowledge_reasoning"]
    assert len(fake.calls) == 0


def test_weak_match_still_calls_reasoning_llm(install_fake_llm, monkeypatch):
    """Below-threshold relevance (even after exhausting iterations) should
    still invoke the real synthesis call — this path must not be skipped."""
    from langchain_core.messages import AIMessage

    fake = install_fake_llm([
        AIMessage(content="reformulated query"),  # refinement after iter 1
        AIMessage(content="reformulated query 2"),  # refinement after iter 2
        AIMessage(content="Context is thin; may not cover this specific case."),  # synthesis
    ])
    monkeypatch.setattr(knowledge_agent, "_get_collection", lambda: _FakeCollection(0.2))

    state = {
        "ticket_text": "obscure question", "classification": "general",
        "urgency": "low", "intake_reasoning": "unclear", "react_trace": [],
    }
    result = run_knowledge_agent(state)

    assert result["rag_iterations"] == 3
    assert len(fake.calls) == 3
    assert "thin" in result["knowledge_reasoning"]
