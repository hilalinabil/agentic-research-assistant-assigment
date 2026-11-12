"""FastAPI application: GET /health, POST /ask and the static frontend.

Route handlers only translate HTTP <-> agent calls; all logic lives in
``app.agent``. The corpus index is built in a background thread at startup so
the server (and /health) is reachable immediately; /ask waits for the index.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

import openai
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.agent.agent import ResearchAgent
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


class AgentState:
    """Holds the agent once the index is built (or the startup error)."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.agent: Optional[ResearchAgent] = None
        self.error: str = ""

    def build(self) -> None:
        try:
            client = get_openai_client(settings)
            embedder = Embedder(client, settings.openai_embedding_model, settings.embedding_batch_size)
            index = build_index(settings, embedder)
            self.agent = ResearchAgent(client, settings, index)
        except ValueError as exc:  # our own configuration validation: message names the missing variables only
            logger.error("Startup failed: %s", exc)
            self.error = str(exc)
        except Exception as exc:  # report through /health and /ask instead of crashing the server
            # Provider errors can contain account details; clients only get the error type.
            logger.exception("Startup failed")
            self.error = f"index build failed ({type(exc).__name__}); see server logs"
        finally:
            self.ready.set()


state = AgentState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=state.build, name="index-builder", daemon=True).start()
    yield


app = FastAPI(title="Agentic Pharmacology Research Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    agent = state.agent
    if agent:
        return HealthResponse(status="ok", index_ready=True, documents=len(agent.index.documents),
                              chunks=len(agent.index.chunks))
    detail = state.error or "index is being built"
    return HealthResponse(status="ok", index_ready=False, detail=detail)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not state.ready.wait(timeout=INDEX_WAIT_SECONDS) or state.agent is None:
        raise HTTPException(status_code=503, detail=f"Service not ready: {state.error or 'index still building'}")
    try:
        return state.agent.run(request.question)
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        logger.error("LLM provider unreachable: %s", exc)
        raise HTTPException(status_code=502, detail="The language-model provider could not be reached.")
    except openai.APIError as exc:
        logger.error("LLM provider error: %s", exc)
        raise HTTPException(status_code=502, detail="The language-model provider returned an error.")
    except Exception:
        logger.exception("Unexpected error while answering")
        raise HTTPException(status_code=500, detail="Internal error while answering the question.")


# Serve the built React frontend (if present) at "/". Registered last so API routes take precedence.
if settings.frontend_dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=settings.frontend_dist_dir, html=True), name="frontend")
else:
    logger.warning("Frontend build not found at %s; only the API is served", settings.frontend_dist_dir)
