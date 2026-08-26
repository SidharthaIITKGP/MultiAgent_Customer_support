from fastapi.testclient import TestClient

import api
import conversation_service

HEADERS = {"X-API-Key": "test-key"}


def _fake_run_graph(recorder=None):
    def _run(initial_state, api_key=None):
        if recorder is not None:
            recorder.append(api_key)
        return {
            **initial_state,
            "classification": "refund",
            "urgency": "medium",
            "sentiment": "neutral",
            "escalation_decision": "auto_resolve",
            "final_response": "stub response",
            "confidence_breakdown": {"overall": 0.9},
            "react_trace": [],
            "tool_calls_made": [],
        }
    return _run


def test_health_is_open_no_auth_needed():
    with TestClient(api.app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_metrics_is_open_no_auth_needed():
    with TestClient(api.app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200


def test_conversations_requires_auth():
    with TestClient(api.app) as client:
        r = client.post("/api/conversations", json={"user_id": "u001"})
        assert r.status_code == 401


def test_conversations_rejects_wrong_key():
    with TestClient(api.app) as client:
        r = client.post("/api/conversations", json={"user_id": "u001"}, headers={"X-API-Key": "wrong"})
        assert r.status_code == 401


def test_full_conversation_flow_with_stubbed_graph(monkeypatch):
    monkeypatch.setattr(conversation_service, "run_graph", _fake_run_graph())
    with TestClient(api.app) as client:
        r = client.post("/api/conversations", json={"user_id": "u001"}, headers=HEADERS)
        assert r.status_code == 200
        conv_id = r.json()["conversation_id"]

        r = client.post(f"/api/conversations/{conv_id}/messages", json={"text": "refund order 1001"}, headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["escalation_decision"] == "auto_resolve"
        assert body["assistant_message"]["content"] == "stub response"

        r = client.get(f"/api/conversations/{conv_id}", headers=HEADERS)
        assert r.status_code == 200
        messages = r.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        ticket_id = messages[-1]["ticket_id"]
        r = client.get(f"/api/ticket/{ticket_id}", headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["escalation_decision"] == "auto_resolve"


def test_byok_gemini_header_reaches_run_graph(monkeypatch):
    seen_keys = []
    monkeypatch.setattr(conversation_service, "run_graph", _fake_run_graph(seen_keys))
    with TestClient(api.app) as client:
        r = client.post("/api/conversations", json={"user_id": "u001"}, headers=HEADERS)
        conv_id = r.json()["conversation_id"]

        r = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"text": "hello"},
            headers={**HEADERS, "X-Gemini-Api-Key": "user-supplied-key-123"},
        )
        assert r.status_code == 200
        assert seen_keys == ["user-supplied-key-123"]


def test_nonexistent_conversation_404():
    with TestClient(api.app) as client:
        r = client.get("/api/conversations/CONV-DOESNOTEXIST", headers=HEADERS)
        assert r.status_code == 404


def test_rate_limit_on_conversation_creation():
    rl_headers = {"X-API-Key": "test-key-ratelimit"}
    with TestClient(api.app) as client:
        codes = []
        for _ in range(25):
            r = client.post("/api/conversations", json={"user_id": "u001"}, headers=rl_headers)
            codes.append(r.status_code)
        assert 429 in codes
