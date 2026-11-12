"""Local tools the agent can call, plus the registry that exposes them.

Two tools are available to the LLM via native function calling:

* ``search_corpus`` – semantic (embedding) search over all corpus chunks.
* ``read_source_document`` – read an original document: its table of
  contents, a named section, or an explicit line range.

The LLM decides which tool to call and with which arguments. Python only
validates the arguments (Pydantic), executes the function, records the
returned passages as evidence, and sends the observation back.

Security: ``read_source_document`` never builds a filesystem path from model
input. It serves documents that were loaded at startup by ``CorpusLoader``
(which is itself confined to the corpus directory), looked up by exact
filename in an allow-list. Traversal strings such as ``../../.env`` are
rejected before any lookup.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import Settings
from app.rag.evidence import Evidence, EvidenceLedger
from app.retrieval.index import CorpusIndex

logger = logging.getLogger(__name__)

MAX_RANGE_LINES = 120
MAX_SECTION_CHARS = 8_000
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:md|txt)$")


class ToolError(Exception):
    """A tool failure whose message is safe to show to the LLM."""


@dataclass
class ToolContext:
    """Per-request state shared by the tools."""
    index: CorpusIndex
    settings: Settings
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)


# ---------------------------------------------------------------------------
# Argument schemas (also used to generate the function-calling JSON schema)
# ---------------------------------------------------------------------------

class SearchCorpusArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=500,
                       description="Focused natural-language search query, e.g. 'warfarin contraindications'.")
    top_k: int = Field(5, ge=1, le=8, description="Number of passages to return (1-8).")

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("query must contain text")
        return v


class ReadSourceDocumentArgs(BaseModel):
    filename: str = Field(..., max_length=100,
                          description="Exact corpus filename, e.g. 'dailymed-warfarin.md'.")
    section: Optional[str] = Field(None, max_length=200,
                                   description="Heading text (or part of it) to read, e.g. 'CONTRAINDICATIONS' or '7.3'.")
    start_line: Optional[int] = Field(None, ge=1, description="First line to read (1-indexed).")
    end_line: Optional[int] = Field(None, ge=1, description="Last line to read (inclusive).")

    @field_validator("filename")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        v = v.strip()
        if "/" in v or "\\" in v or ".." in v or not _SAFE_FILENAME.match(v):
            raise ValueError("filename must be a bare corpus filename such as 'dailymed-warfarin.md'")
        return v


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_corpus(ctx: ToolContext, args: SearchCorpusArgs) -> Dict[str, Any]:
    results = ctx.index.retriever.retrieve(args.query, top_k=min(args.top_k, ctx.settings.max_search_top_k))
    passages = []
    for r in results:
        ch = r.chunk
        ctx.ledger.add(Evidence(
            source=ch.source_filename, start_line=ch.start_line, end_line=ch.end_line,
            text=ch.text, section=ch.section_path, origin="search_corpus", score=r.score,
        ))
        passages.append({
            "source": ch.source_filename,
            "lines": f"L{ch.start_line}-L{ch.end_line}",
            "section": ch.section_path,
            "score": round(r.score, 3),
            "text": ch.text,
        })
    return {"query": args.query, "num_results": len(passages), "passages": passages}


def _headings(lines: List[str]) -> List[Dict[str, Any]]:
    out, in_code = [], False
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            in_code = not in_code
            continue
        m = None if in_code else _HEADING.match(line)
        if m:
            out.append({"level": len(m.group(1)), "title": m.group(2), "line": i})
    return out


def read_source_document(ctx: ToolContext, args: ReadSourceDocumentArgs) -> Dict[str, Any]:
    doc = next((d for d in ctx.index.documents if d.filename == args.filename), None)
    if doc is None:
        raise ToolError(
            f"Unknown document '{args.filename}'. Available documents: {', '.join(ctx.index.filenames)}"
        )
    lines = doc.content.splitlines()
    headings = _headings(lines)

    # Mode 1: explicit line range.
    if args.start_line is not None:
        start = args.start_line
        end = args.end_line or start + 40
        if end < start:
            raise ToolError("end_line must be >= start_line")
        if start > len(lines):
            raise ToolError(f"start_line {start} is beyond the end of the document ({len(lines)} lines)")
        end = min(end, len(lines), start + MAX_RANGE_LINES - 1)
        section = next((h["title"] for h in reversed(headings) if h["line"] <= start), "")
        return _record(ctx, doc.filename, lines, start, end, section)

    # Mode 2: named section.
    if args.section:
        wanted = args.section.strip().lower()
        idx = next((i for i, h in enumerate(headings) if wanted in h["title"].lower()), None)
        if idx is None:
            raise ToolError(
                f"No section matching '{args.section}' in {doc.filename}. "
                f"Sections: {'; '.join(h['title'] for h in headings)}"
            )
        h = headings[idx]
        # The section runs until the next heading of the same or higher level.
        end = len(lines)
        for nxt in headings[idx + 1:]:
            if nxt["level"] <= h["level"]:
                end = nxt["line"] - 1
                break
        return _record(ctx, doc.filename, lines, h["line"], end, h["title"])

    # Mode 3: table of contents only (no evidence recorded).
    return {
        "source": doc.filename,
        "title": doc.title,
        "total_lines": len(lines),
        "sections": [f"{'  ' * (h['level'] - 1)}{h['title']} (L{h['line']})" for h in headings],
        "hint": "Call again with 'section' or 'start_line'/'end_line' to read text.",
    }


def _record(ctx: ToolContext, filename: str, lines: List[str], start: int, end: int, section: str) -> Dict[str, Any]:
    """Return lines start..end (truncated to a size budget) and record them as evidence."""
    picked, size, last = [], 0, start - 1
    for n in range(start, end + 1):
        line = lines[n - 1]
        if picked and size + len(line) > MAX_SECTION_CHARS:
            break
        picked.append(line)
        size += len(line) + 1
        last = n
    text = "\n".join(picked)
    ctx.ledger.add(Evidence(source=filename, start_line=start, end_line=last, text=text,
                            section=section, origin="read_source_document", line_numbered=True))
    result = {"source": filename, "lines": f"L{start}-L{last}", "section": section, "text": text}
    if last < end:
        result["truncated"] = f"Output truncated at L{last}; request start_line={last + 1} to continue."
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    func: Callable[[ToolContext, Any], Dict[str, Any]]

    def schema(self) -> Dict[str, Any]:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": params}}


TOOLS: Dict[str, Tool] = {
    t.name: t for t in [
        Tool(
            name="search_corpus",
            description=(
                "Semantic search over all corpus documents (FDA DailyMed labels, MedlinePlus consumer "
                "information and a glossary). Returns the most similar passages with source filename "
                "and line range. Use one focused query per drug/topic."
            ),
            args_model=SearchCorpusArgs,
            func=search_corpus,
        ),
        Tool(
            name="read_source_document",
            description=(
                "Read an original corpus document. Give only 'filename' to get its table of contents; "
                "add 'section' to read a whole section (e.g. '4 CONTRAINDICATIONS'); or give "
                "'start_line'/'end_line' to read exact lines. Use it to read complete sections when "
                "search results look partial or cut off."
            ),
            args_model=ReadSourceDocumentArgs,
            func=read_source_document,
        ),
    ]
}


def tool_schemas() -> List[Dict[str, Any]]:
    return [t.schema() for t in TOOLS.values()]


@dataclass
class ToolResult:
    ok: bool
    output: Dict[str, Any]

    def to_message(self) -> str:
        return json.dumps(self.output, ensure_ascii=False)


def execute_tool(name: str, raw_arguments: str | None, ctx: ToolContext) -> ToolResult:
    """Validate and execute one LLM-requested tool call. Never raises."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(False, {"error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}"})
    try:
        payload = json.loads(raw_arguments or "{}")
        if not isinstance(payload, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return ToolResult(False, {"error": f"Arguments are not a valid JSON object: {exc}"})
    try:
        args = tool.args_model.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors())
        return ToolResult(False, {"error": f"Invalid arguments for {name}: {problems}"})
    try:
        return ToolResult(True, tool.func(ctx, args))
    except ToolError as exc:
        return ToolResult(False, {"error": str(exc)})
    except Exception:  # unexpected failure: log details, give the model a safe message
        logger.exception("Tool %s failed", name)
        return ToolResult(False, {"error": f"Tool {name} failed unexpectedly. Try a different query or tool."})
