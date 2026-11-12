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
    monkeypatch.setattr(st, "build", lambda: None)      # do not build a real index
    with TestClient(main.app) as c:
        yield c, st


def make_ready(st, agent=None, error=""):
    st.agent, st.error = agent, error
    st.ready.set()


def test_health_ok_while_index_building(client):
    c, st = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["index_ready"] is False


def test_health_ready(client):
    c, st = client
    make_ready(st, StubAgent())
    body = c.get("/health").json()
    assert body == {"status": "ok", "index_ready": True, "documents": 21, "chunks": 525, "detail": ""}


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


def test_state_build_records_config_error(monkeypatch):
    st = main.AgentState()
    monkeypatch.setattr(main, "get_openai_client", lambda s: (_ for _ in ()).throw(ValueError("Missing OPENAI_MODEL")))
    st.build()
    assert st.ready.is_set() and st.agent is None and "OPENAI_MODEL" in st.error


def test_state_build_hides_provider_error_details(monkeypatch):
    class AuthenticationError(Exception):
        pass

    def fail(_settings):
        raise AuthenticationError("Incorrect API key provided: sk-abc***xyz, org org-123")

    st = main.AgentState()
    monkeypatch.setattr(main, "get_openai_client", fail)
    st.build()
    assert st.error == "index build failed (AuthenticationError); see server logs"
    assert "sk-" not in st.error and "org-" not in st.error
