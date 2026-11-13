"""FastAPI application: GET /health, POST /ask and the static frontend.

Route handlers only translate HTTP <-> agent calls; all logic lives in
``app.agent``. The corpus index is built in a background thread at startup so
the server (and /health) is reachable immediately. A failed build is retried
with exponential backoff; if it still failed, the next /ask triggers one more
build attempt before answering 503.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import openai
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.agent import ResearchAgent
from app.audit import audit_log
from app.core.config import settings
from app.core.llm_client import get_openai_client
from app.retrieval.embedder import Embedder
from app.retrieval.index import build_index
from app.schemas.api import AskRequest, AskResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpx2", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("app")

INDEX_WAIT_SECONDS = 180


def create_agent() -> ResearchAgent:
    """Build the corpus index (embedding calls) and the agent."""
    client = get_openai_client(settings)
    embedder = Embedder(client, settings.openai_embedding_model, settings.embedding_batch_size)
    return ResearchAgent(client, settings, build_index(settings, embedder))


class AgentState:
    """Builds the agent in a background thread, with retries, and reports progress."""

    def __init__(self, max_attempts: Optional[int] = None, base_delay: Optional[float] = None) -> None:
        self.max_attempts = max_attempts or settings.index_build_attempts
        self.base_delay = settings.index_retry_base_seconds if base_delay is None else base_delay
        self.agent: Optional[ResearchAgent] = None
        self.status = "building"   # building | ready | failed
        self.error = ""
        self.attempts = 0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self, max_attempts: Optional[int] = None) -> Optional[threading.Thread]:
        """Start a build round unless the agent exists or a build is already running."""
        with self._lock:
            if self.agent is not None:
                return None
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self.status = "building"
            self._thread = threading.Thread(target=self._run, args=(max_attempts or self.max_attempts,),
                                            name="index-builder", daemon=True)
            self._thread.start()
            return self._thread

    def _run(self, attempts: int) -> None:
        missing = settings.missing_api_config()
        if missing:  # retrying cannot fix missing configuration
            self.error = f"Missing required environment variables: {', '.join(missing)}"
            logger.error("Startup failed: %s", self.error)
            self.status = "failed"
            return
        for attempt in range(1, attempts + 1):
            self.attempts += 1
            try:
                self.agent = create_agent()
                self.error, self.status = "", "ready"
                return
            except Exception as exc:
                # Provider errors can contain account details; clients only get the error type.
                logger.exception("Index build attempt %d/%d failed", attempt, attempts)
                self.error = f"index build failed ({type(exc).__name__}); see server logs"
            if attempt < attempts:
                delay = self.base_delay * 2 ** (attempt - 1)
                logger.warning("Retrying index build in %.0fs", delay)
                time.sleep(delay)
        self.status = "failed"

    def ensure_ready(self, timeout: float) -> bool:
        """Wait for a running build; if the last build round failed, try once more now."""
        if self.agent is not None:
            return True
        thread = self.start(max_attempts=1) if self.status == "failed" else self._thread
        if thread is not None:
            thread.join(timeout)
        return self.agent is not None


state = AgentState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.start()
    yield


app = FastAPI(title="Agentic Pharmacology Research Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    agent = state.agent
    if agent:
        return HealthResponse(status="ok", index_ready=True, index_status="ready", build_attempts=state.attempts,
                              documents=len(agent.index.documents), chunks=len(agent.index.chunks))
    return HealthResponse(status="ok", index_ready=False, index_status=state.status, build_attempts=state.attempts,
                          detail=state.error or "index is being built")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, background_tasks: BackgroundTasks):
    started = time.perf_counter()
    response, error_type, http_status, detail = None, None, 200, ""
    if not state.ensure_ready(timeout=INDEX_WAIT_SECONDS):
        error_type, http_status = "ServiceNotReady", 503
        detail = f"Service not ready: {state.error or 'index still building'}"
    else:
        try:
            response = state.agent.run(request.question)
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            logger.error("LLM provider unreachable: %s", exc)
            error_type, http_status = type(exc).__name__, 502
            detail = "The language-model provider could not be reached."
        except openai.APIError as exc:
            logger.error("LLM provider error: %s", exc)
            error_type, http_status = type(exc).__name__, 502
            detail = "The language-model provider returned an error."
        except Exception as exc:
            logger.exception("Unexpected error while answering")
            error_type, http_status = type(exc).__name__, 500
            detail = "Internal error while answering the question."

    # Audit entry is written after the response is sent (background task); it never raises.
    background_tasks.add_task(audit_log.record, request, response, {
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "http_status": http_status, "error_type": error_type,
        "chat_model": settings.openai_model, "embedding_model": settings.openai_embedding_model,
    })
    if response is None:
        # Same body/status as HTTPException, but a returned response keeps the background task.
        return JSONResponse(status_code=http_status, content={"detail": detail})
    return response


# Serve the built React frontend (if present) at "/". Registered last so API routes take precedence.
if settings.frontend_dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dist_dir, html=True), name="frontend")
else:
    logger.warning("Frontend build not found at %s; only the API is served", settings.frontend_dist_dir)
