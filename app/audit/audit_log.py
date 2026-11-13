"""Audit log: one JSON line per /ask request, in ``<AUDIT_LOG_DIR>/audit-YYYY-MM-DD.jsonl`` (UTC date).

Called from the /ask route as a FastAPI background task, after the response is
built, so it never delays or changes the HTTP response. Writes are serialised
with a lock; any failure is logged as a warning and swallowed.

Only the question, the answer summary, timing and model names are stored —
never API keys, base URLs or prompts.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DIR = "logs/data"
_lock = threading.Lock()


def _log_dir() -> Path:
    return Path(os.getenv("AUDIT_LOG_DIR") or DEFAULT_DIR)


def _build_entry(request: Any, response: Any, meta: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    error_type: Optional[str] = meta.get("error_type")
    if response is not None and error_type is None:
        review = getattr(response, "review", None) or {}
        resp = {
            "status": response.status,
            "answer": response.answer,
            "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in response.citations],
            "critic_verdict": review.get("verdict"),
        }
    else:
        resp = {"status": "error", "answer": None, "citations": [], "critic_verdict": None}
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": now.isoformat(),
        "request": {"question": getattr(request, "question", None)},
        "response": resp,
        "http_status": meta.get("http_status"),
        "latency_ms": meta.get("latency_ms"),
        "chat_model": meta.get("chat_model"),
        "embedding_model": meta.get("embedding_model"),
        "error_type": error_type,
    }


def record(request: Any, response: Any, meta: Dict[str, Any]) -> None:
    """Append one audit entry. ``response`` is None for failed requests. Never raises."""
    try:
        now = datetime.now(timezone.utc)
        line = json.dumps(_build_entry(request, response, meta or {}, now), ensure_ascii=False, default=str)
        directory = _log_dir()
        path = directory / f"audit-{now:%Y-%m-%d}.jsonl"
        with _lock:
            directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # the audit log must never break /ask
        logger.warning("Audit log write failed (%s): %s", type(exc).__name__, exc)
