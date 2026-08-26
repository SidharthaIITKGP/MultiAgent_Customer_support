"""
tests/conftest.py

Env vars are set BEFORE any project module is imported (db/session.py and
api.py both read DATABASE_URL/API_KEYS at import time), so this must happen at
conftest module load time, not inside a fixture.

These are forced (not setdefault), not merely defaulted: the suite needs a
SPECIFIC, KNOWN set of fake credentials (e.g. two distinct API keys — one
shared, one dedicated to the rate-limit test so it doesn't collide with other
tests hammering the same key+endpoint) regardless of whatever the runner's own
job-level env already exports. CI setting API_KEYS itself is exactly the case
this must override — with setdefault, that CI value would win and silently
drop the second key, which is what caused this suite to pass locally (no
API_KEYS in the shell) but fail in CI (API_KEYS=test-key already exported).
"""

import copy
import os
import tempfile

_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db.name}"
os.environ["API_KEYS"] = "test-key,test-key-ratelimit"
os.environ["GOOGLE_API_KEY"] = "test-key-not-real"

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import agents.llm_utils as llm_utils  # noqa: E402
import mock_backend  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mock_backend():
    """mock_backend.py's module-level dicts mutate (refunds flip `refunded=True`,
    etc.) — snapshot/restore around every test so order never matters."""
    snapshot = {
        "ACCOUNTS": copy.deepcopy(mock_backend.ACCOUNTS),
        "ORDERS": copy.deepcopy(mock_backend.ORDERS),
        "REFUNDS": copy.deepcopy(mock_backend.REFUNDS),
        "PASSWORD_RESETS": copy.deepcopy(mock_backend.PASSWORD_RESETS),
    }
    yield
    mock_backend.ACCOUNTS.clear()
    mock_backend.ACCOUNTS.update(snapshot["ACCOUNTS"])
    mock_backend.ORDERS.clear()
    mock_backend.ORDERS.update(snapshot["ORDERS"])
    mock_backend.REFUNDS.clear()
    mock_backend.REFUNDS.update(snapshot["REFUNDS"])
    mock_backend.PASSWORD_RESETS.clear()
    mock_backend.PASSWORD_RESETS.update(snapshot["PASSWORD_RESETS"])


class _FakeModel:
    """Stands in for a ChatGoogleGenerativeAI instance. `responses` is a list of
    either plain strings or already-built AIMessage objects, consumed in order
    across successive .invoke() calls (and shared across .bind_tools() calls,
    since bind_tools() returns `self`)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            return AIMessage(content="")
        nxt = self._responses.pop(0)
        return nxt if isinstance(nxt, AIMessage) else AIMessage(content=nxt)

    def bind_tools(self, tools):
        return self


@pytest.fixture
def install_fake_llm(monkeypatch):
    """Usage: fake = install_fake_llm(["response 1", AIMessage(content=...), ...])
    Patches the ONE seam every agent resolves through: agents.llm_utils.build_llm
    calls ChatGoogleGenerativeAI(**kwargs), imported at agents/llm_utils.py's own
    top level — patching that single name covers intake/knowledge/action/escalation
    simultaneously, since they all call build_llm() rather than constructing the
    model directly."""

    def _install(responses):
        fake = _FakeModel(responses)
        monkeypatch.setattr(llm_utils, "ChatGoogleGenerativeAI", lambda **kwargs: fake)
        return fake

    return _install


@pytest.fixture
def patched_backend(monkeypatch):
    """Routes agents.action_agent's tool calls into the real mock_backend FastAPI
    app in-process (via TestClient) instead of real HTTP — exercises real business
    logic (403 locked account, 409 already refunded, etc.) with no network."""
    from fastapi.testclient import TestClient
    import agents.action_agent as action_agent

    client = TestClient(mock_backend.app)

    def fake_call_backend(method, path, **kwargs):
        resp = client.request(method, path, **kwargs)
        if resp.status_code < 300:
            return resp.json()
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return {"success": False, "error": True, "status_code": resp.status_code, "detail": detail}

    monkeypatch.setattr(action_agent, "_call_backend", fake_call_backend)
    return client


@pytest.fixture
def tmp_chroma(monkeypatch, tmp_path):
    """CHROMA_PATH is read into a module-level constant at import time in both
    agents/knowledge_agent.py and memory/ticket_memory.py — monkeypatch.setenv
    after import does nothing, so patch the module attribute directly."""
    import agents.knowledge_agent as knowledge_agent
    import memory.ticket_memory as ticket_memory

    path = str(tmp_path / "chroma")
    monkeypatch.setattr(knowledge_agent, "CHROMA_PATH", path)
    monkeypatch.setattr(ticket_memory, "CHROMA_PATH", path)
    return path
