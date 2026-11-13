"""
Shared test fixtures.

* Tests marked ``@pytest.mark.integration`` call the real LLM/embedding
  endpoint and only run when ``RUN_INTEGRATION=1`` is set.
* Everything else is offline: embeddings come from a deterministic hashed
  bag-of-words model and chat completions from a scripted fake client.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from app.core.config import Settings
from app.retrieval.embedder import Embedder
from app.retrieval.index import CorpusIndex, build_index


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="Integration test: set RUN_INTEGRATION=1 (needs live LLM credentials).")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Offline embeddings
# ---------------------------------------------------------------------------

FAKE_DIM = 512


def fake_embed(text: str) -> List[float]:
    vec = [0.0] * FAKE_DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) < 3:
            continue
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % FAKE_DIM] += 1.0
    if not any(vec):
        vec[0] = 1.0
    return vec


class FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = 0

    def create(self, model: str, input: List[str]):
        self.calls += 1
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=fake_embed(t)) for i, t in enumerate(input)])


class FakeEmbeddingClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsAPI()


# ---------------------------------------------------------------------------
# Scripted chat completions
# ---------------------------------------------------------------------------

def tool_call(name: str, arguments: Dict[str, Any] | str, call_id: Optional[str] = None):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id or f"call_{name}_{abs(hash(args)) % 10_000}", type="function",
                           function=SimpleNamespace(name=name, arguments=args))


def completion(content: Optional[str] = None, tool_calls: Optional[list] = None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg,
                                                    finish_reason="tool_calls" if tool_calls else "stop")])


class ScriptedChat:
    """Returns pre-defined responses in order and records every request."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))  # snapshot: callers mutate message lists
        if not self.responses:
            raise AssertionError("ScriptedChat ran out of responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeLLMClient:
    def __init__(self, responses: list):
        self.chat = SimpleNamespace(completions=ScriptedChat(responses))
        self.embeddings = FakeEmbeddingsAPI()

    @property
    def requests(self):
        return self.chat.completions.requests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_base_url="", openai_model="test-chat-model",
                    openai_embedding_model="test-embedding-model")


@pytest.fixture(scope="session")
def corpus_index() -> CorpusIndex:
    """Real corpus, real chunker, fake (offline) embeddings."""
    settings = Settings(openai_api_key="x", openai_model="m", openai_embedding_model="e")
    embedder = Embedder(client=FakeEmbeddingClient(), model="fake", batch_size=64)
    return build_index(settings, embedder)


@pytest.fixture(autouse=True)
def _isolated_audit_log(tmp_path, monkeypatch):
    """Keep /ask audit entries written during tests out of the repository's logs/data/."""
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "audit"))
    return tmp_path / "audit"
