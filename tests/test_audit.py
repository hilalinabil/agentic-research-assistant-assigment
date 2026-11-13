"""Audit log: one JSONL entry per /ask, written in the background, never affecting the response."""
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.audit import audit_log
from tests.test_api import StubAgent, make_ready


@pytest.fixture
def client(monkeypatch):
    st = main.AgentState()
    monkeypatch.setattr(main, "state", st)
    monkeypatch.setattr(st, "start", lambda max_attempts=None: None)
    monkeypatch.setattr(main.settings, "openai_model", "chat-m")
    monkeypatch.setattr(main.settings, "openai_embedding_model", "embed-m")
    monkeypatch.setattr(main.settings, "openai_api_key", "sk-secret-key")
    monkeypatch.setattr(main.settings, "openai_base_url", "https://secret.example/v1")
    with TestClient(main.app) as c:
        yield c, st


def read_entries(directory):
    path = directory / f"audit-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    text = path.read_text(encoding="utf-8")
    assert "sk-secret-key" not in text and "secret.example" not in text
    return [json.loads(line) for line in text.splitlines()]


def test_entry_written_on_success(client, _isolated_audit_log):
    c, st = client
    make_ready(st, StubAgent())
    r = c.post("/ask", json={"question": "Warfarin contraindications?"})
    assert r.status_code == 200
    [entry] = read_entries(_isolated_audit_log)
    assert entry["request"] == {"question": "Warfarin contraindications?"}
    assert entry["response"]["status"] == "success"
    assert entry["response"]["answer"] == r.json()["answer"]
    assert entry["response"]["citations"][0]["source"] == "dailymed-warfarin.md"
    assert entry["response"]["critic_verdict"] is None          # stub response has no review
    assert entry["error_type"] is None and entry["http_status"] == 200
    assert entry["chat_model"] == "chat-m" and entry["embedding_model"] == "embed-m"
    assert entry["latency_ms"] >= 0 and len(entry["request_id"]) == 36
    assert datetime.fromisoformat(entry["timestamp"]).tzinfo is not None


def test_entry_written_on_error(client, _isolated_audit_log):
    c, st = client
    make_ready(st, StubAgent(exc=RuntimeError("boom")))
    r = c.post("/ask", json={"question": "warfarin?"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal error while answering the question."}
    [entry] = read_entries(_isolated_audit_log)
    assert entry["error_type"] == "RuntimeError" and entry["http_status"] == 500
    assert entry["response"] == {"status": "error", "answer": None, "citations": [], "critic_verdict": None}


def test_entry_written_when_not_ready(client, _isolated_audit_log):
    c, st = client
    make_ready(st, None, error="index build failed")
    assert c.post("/ask", json={"question": "warfarin?"}).status_code == 503
    [entry] = read_entries(_isolated_audit_log)
    assert entry["error_type"] == "ServiceNotReady" and entry["http_status"] == 503


def test_write_failure_does_not_change_response(client, monkeypatch, tmp_path, caplog):
    c, st = client
    make_ready(st, StubAgent())
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")                                     # mkdir under a file fails
    monkeypatch.setenv("AUDIT_LOG_DIR", str(blocker / "sub"))
    baseline_dir = tmp_path / "ok"
    r = c.post("/ask", json={"question": "warfarin?"})
    assert r.status_code == 200 and r.json()["status"] == "success"
    assert "Audit log write failed" in caplog.text
    monkeypatch.setenv("AUDIT_LOG_DIR", str(baseline_dir))
    assert c.post("/ask", json={"question": "warfarin?"}).json() == r.json()


def test_record_never_raises_on_bad_input(_isolated_audit_log):
    audit_log.record(object(), object(), {})                    # malformed response object
    audit_log.record(None, None, None)


def test_critic_verdict_is_recorded(client, _isolated_audit_log):
    c, st = client

    class ReviewedAgent(StubAgent):
        def run(self, question):
            return super().run(question).model_copy(update={"review": {"verdict": "approve", "feedback": "ok"}})

    make_ready(st, ReviewedAgent())
    c.post("/ask", json={"question": "warfarin?"})
    [entry] = read_entries(_isolated_audit_log)
    assert entry["response"]["critic_verdict"] == "approve"
