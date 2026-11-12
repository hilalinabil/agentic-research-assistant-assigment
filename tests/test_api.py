"""Phase 6: HTTP API contract (agent mocked; no network)."""
import threading

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.rag.generator import DISCLAIMER
from app.schemas.api import AskResponse


class StubIndex:
    documents = [object()] * 21
    chunks = [object()] * 525


class StubAgent:
    index = StubIndex()

    def __init__(self, exc=None):
        self.exc = exc
        self.questions = []

    def run(self, question):
        self.questions.append(question)
        if self.exc:
            raise self.exc
        return AskResponse(question=question, answer=f"Answer [dailymed-warfarin.md:L129-L145]\n\n*{DISCLAIMER}*",
                           citations=[{"source": "dailymed-warfarin.md", "start_line": 129, "end_line": 145}],
                           sources=["dailymed-warfarin.md"], disclaimer=DISCLAIMER, status="success")


@pytest.fixture
def client(monkeypatch):
    st = main.AgentState()
    monkeypatch.setattr(main, "state", st)
    monkeypatch.setattr(st, "start", lambda max_attempts=None: None)   # do not build a real index
    with TestClient(main.app) as c:
        yield c, st


def make_ready(st, agent=None, error=""):
    st.agent, st.error = agent, error
    st.status = "ready" if agent else "failed"


def test_health_ok_while_index_building(client):
    c, st = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["index_ready"] is False


def test_health_ready(client):
    c, st = client
    make_ready(st, StubAgent())
    body = c.get("/health").json()
    assert body == {"status": "ok", "index_ready": True, "index_status": "ready", "build_attempts": 0,
                    "documents": 21, "chunks": 525, "detail": ""}


def test_ask_success(client):
    c, st = client
    agent = StubAgent()
    make_ready(st, agent)
    r = c.post("/ask", json={"question": "  What are the main contraindications for warfarin?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success" and body["disclaimer"] == DISCLAIMER
    assert body["citations"][0]["source"] == "dailymed-warfarin.md"
    assert agent.questions == ["What are the main contraindications for warfarin?"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "  "}, {"question": 5},
                                     {"question": "x" * 2001}, {"q": "warfarin"}])
def test_ask_validation_errors(client, payload):
    c, st = client
    make_ready(st, StubAgent())
    assert c.post("/ask", json=payload).status_code == 422


def test_ask_not_ready_returns_503(client):
    c, st = client
    make_ready(st, None, error="ValueError: Missing required environment variables: OPENAI_API_KEY")
    r = c.post("/ask", json={"question": "warfarin?"})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]


def test_ask_provider_error_returns_502(client):
    c, st = client
    req = httpx.Request("POST", "http://llm/chat")
    make_ready(st, StubAgent(exc=openai.APIConnectionError(request=req)))
    assert c.post("/ask", json={"question": "warfarin?"}).status_code == 502


def test_ask_unexpected_error_is_500_without_details(client):
    c, st = client
    make_ready(st, StubAgent(exc=RuntimeError("secret detail")))
    r = c.post("/ask", json={"question": "warfarin?"})
    assert r.status_code == 500 and "secret" not in r.text


# --- startup index build: retries, backoff, rebuild on /ask (embedder mocked) ----------

from tests.conftest import fake_embed  # noqa: E402
from types import SimpleNamespace  # noqa: E402


class FlakyEmbeddingClient:
    """Embedding client whose first ``failures`` calls raise a connection error."""

    def __init__(self, failures):
        self.failures = failures
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, model, input):
        self.calls += 1
        if self.calls <= self.failures:
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://llm/embeddings"))
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=fake_embed(t)) for i, t in enumerate(input)])


@pytest.fixture
def flaky(monkeypatch):
    """Patch the client factory used by create_agent(); real loader, chunker and index builder run."""
    def install(failures):
        fake = FlakyEmbeddingClient(failures)
        monkeypatch.setattr(main, "get_openai_client", lambda _settings: fake)
        monkeypatch.setattr(main.settings, "openai_api_key", "test-key")
        monkeypatch.setattr(main.settings, "openai_model", "test-model")
        return fake
    return install


def run_build(st, max_attempts=None):
    st.start(max_attempts).join(30)


def test_build_retries_until_success(flaky):
    flaky(failures=2)
    st = main.AgentState(max_attempts=5, base_delay=0)
    run_build(st)
    assert st.status == "ready" and st.attempts == 3 and st.error == ""
    assert len(st.agent.index.chunks) > 500


def test_build_uses_exponential_backoff(flaky, monkeypatch):
    flaky(failures=3)
    sleeps = []
    monkeypatch.setattr(main.time, "sleep", sleeps.append)
    st = main.AgentState(max_attempts=5, base_delay=2)
    run_build(st)
    assert st.status == "ready" and sleeps == [2, 4, 8]


def test_build_gives_up_after_max_attempts_and_hides_details(flaky):
    flaky(failures=100)
    st = main.AgentState(max_attempts=5, base_delay=0)
    run_build(st)
    assert st.status == "failed" and st.attempts == 5 and st.agent is None
    assert st.error == "index build failed (APIConnectionError); see server logs"


def test_missing_configuration_is_not_retried(monkeypatch):
    monkeypatch.setattr(main.settings, "openai_api_key", "")
    called = []
    monkeypatch.setattr(main, "create_agent", lambda: called.append(1))
    st = main.AgentState(max_attempts=5, base_delay=0)
    run_build(st)
    assert st.status == "failed" and called == [] and "OPENAI_API_KEY" in st.error


def test_health_is_200_and_reports_failed_build(flaky, monkeypatch):
    flaky(failures=100)
    st = main.AgentState(max_attempts=2, base_delay=0)
    run_build(st)
    monkeypatch.setattr(main, "state", st)
    monkeypatch.setattr(st, "start", lambda max_attempts=None: None)
    with TestClient(main.app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["index_status"] == "failed" and r.json()["build_attempts"] == 2


def test_ask_triggers_rebuild_after_failed_startup(flaky, monkeypatch):
    fake = flaky(failures=100)
    st = main.AgentState(max_attempts=2, base_delay=0)
    run_build(st)
    assert st.status == "failed"
    fake.failures = 0                     # provider is back
    agent = StubAgent()
    real_create = main.create_agent
    monkeypatch.setattr(main, "create_agent", lambda: (real_create(), agent)[1])  # real build, stub answers
    monkeypatch.setattr(main, "state", st)
    original_start = st.start
    started = []

    def start(max_attempts=None):         # the app's lifespan start is a no-op here; record on-demand starts
        if max_attempts is None:
            return None
        started.append(max_attempts)
        return original_start(max_attempts)

    monkeypatch.setattr(st, "start", start)
    with TestClient(main.app) as c:
        r = c.post("/ask", json={"question": "warfarin contraindications?"})
    assert r.status_code == 200 and st.status == "ready"
    assert started == [1]                 # exactly one on-demand attempt
    assert agent.questions == ["warfarin contraindications?"]


def test_ask_returns_503_when_on_demand_rebuild_also_fails(flaky, monkeypatch):
    flaky(failures=100)
    st = main.AgentState(max_attempts=1, base_delay=0)
    run_build(st)
    monkeypatch.setattr(main, "state", st)
    with TestClient(main.app) as c:
        r = c.post("/ask", json={"question": "warfarin contraindications?"})
    assert r.status_code == 503 and "APIConnectionError" in r.json()["detail"]
    assert st.attempts >= 2               # startup attempt + on-demand attempt
