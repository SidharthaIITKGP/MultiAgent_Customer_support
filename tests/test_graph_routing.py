from agents.graph import route_after_action


def test_insufficient_context_retries_when_under_limit():
    state = {"insufficient_context": True, "knowledge_retry_count": 0}
    assert route_after_action(state) == "retry_knowledge"

    state = {"insufficient_context": True, "knowledge_retry_count": 1}
    assert route_after_action(state) == "retry_knowledge"


def test_insufficient_context_escalates_at_retry_limit():
    state = {"insufficient_context": True, "knowledge_retry_count": 2}
    assert route_after_action(state) == "escalation"


def test_sufficient_context_escalates_regardless_of_retry_count():
    state = {"insufficient_context": False, "knowledge_retry_count": 0}
    assert route_after_action(state) == "escalation"

    state = {"insufficient_context": False, "knowledge_retry_count": 5}
    assert route_after_action(state) == "escalation"


def test_missing_fields_default_to_escalation():
    assert route_after_action({}) == "escalation"
