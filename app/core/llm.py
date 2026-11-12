"""Small helpers around the OpenAI-compatible chat API."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMResponseError(RuntimeError):
    """The model returned something we could not use."""


def chat_text(client, model: str, messages: List[Dict[str, Any]]) -> str:
    """Single chat completion that must return text content."""
    resp = client.chat.completions.create(model=model, messages=messages)
    content = resp.choices[0].message.content
    if not content or not content.strip():
        raise LLMResponseError("Model returned empty content")
    return content.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output, tolerating ```json fences or surrounding prose."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError("No JSON object found in model output")
        try:
            obj = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Invalid JSON in model output: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMResponseError("Model output JSON is not an object")
    return obj


def chat_json(client, model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Chat completion whose output is parsed as a JSON object.

    Uses prompt instructions rather than provider-specific JSON modes so it
    works with any OpenAI-compatible endpoint.
    """
    return parse_json_object(chat_text(client, model, messages))


def as_str_list(value: Optional[Any]) -> List[str]:
    """Coerce a model-provided field into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    out = []
    for v in value if isinstance(value, list) else [value]:
        if isinstance(v, dict):
            v = "; ".join(f"{k}: {val}" for k, val in v.items())
        s = str(v).strip()
        if s:
            out.append(s)
    return out
