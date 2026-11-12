"""Tool registry, argument validation, safe execution and path confinement."""
import json

import pytest

from app.agent.tools import TOOLS, ToolContext, execute_tool, tool_schemas
from app.rag.evidence import EvidenceLedger


@pytest.fixture
def ctx(corpus_index, test_settings):
    return ToolContext(index=corpus_index, settings=test_settings, ledger=EvidenceLedger())


def run(ctx, name, args):
    return execute_tool(name, args if isinstance(args, str) else json.dumps(args), ctx)


# --- registry ---------------------------------------------------------------

def test_registry_exposes_two_function_schemas():
    schemas = tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"search_corpus", "read_source_document"}
    for s in schemas:
        assert s["type"] == "function"
        assert s["function"]["parameters"]["type"] == "object"
        assert s["function"]["description"]
    assert "query" in TOOLS["search_corpus"].schema()["function"]["parameters"]["required"]


# --- search_corpus ------------------------------------------------------------

def test_search_returns_passages_with_metadata_and_records_evidence(ctx):
    res = run(ctx, "search_corpus", {"query": "warfarin contraindications", "top_k": 3})
    assert res.ok
    assert res.output["num_results"] == 3
    p = res.output["passages"][0]
    assert {"source", "lines", "section", "score", "text"} <= set(p)
    assert len(ctx.ledger) >= 1
    assert all(e.origin == "search_corpus" for e in ctx.ledger.items)


@pytest.mark.parametrize("args", [
    {"query": ""},
    {"query": "   "},
    {"query": "warfarin", "top_k": 0},
    {"query": "warfarin", "top_k": 50},
    {"query": "x" * 600},
    {},
])
def test_search_rejects_invalid_arguments(ctx, args):
    res = run(ctx, "search_corpus", args)
    assert not res.ok
    assert "Invalid arguments" in res.output["error"]
    assert len(ctx.ledger) == 0


# --- read_source_document ---------------------------------------------------

@pytest.mark.parametrize("filename", [
    "../../.env",
    "../.env",
    "/etc/passwd",
    "..\\..\\.env",
    "corpus/../../../.env",
    "dailymed-warfarin.md/../../.env",
    ".env",
    "requirements.txt/..",
])
def test_read_rejects_path_traversal(ctx, filename):
    res = run(ctx, "read_source_document", {"filename": filename})
    assert not res.ok
    assert "OPENAI" not in json.dumps(res.output)
    assert len(ctx.ledger) == 0


def test_read_rejects_unknown_file_with_allowed_list(ctx):
    res = run(ctx, "read_source_document", {"filename": "dailymed-unknown.md"})
    assert not res.ok
    assert "dailymed-warfarin.md" in res.output["error"]


def test_read_table_of_contents(ctx):
    res = run(ctx, "read_source_document", {"filename": "dailymed-warfarin.md"})
    assert res.ok
    assert any("CONTRAINDICATIONS" in s for s in res.output["sections"])
    assert len(ctx.ledger) == 0  # a TOC is navigation, not evidence


def test_read_named_section_returns_whole_section_with_lines(ctx):
    res = run(ctx, "read_source_document", {"filename": "dailymed-metformin.md", "section": "4 CONTRAINDICATIONS"})
    assert res.ok
    assert res.output["section"] == "4 CONTRAINDICATIONS"
    assert "eGFR" in res.output["text"]
    ev = ctx.ledger.items[0]
    assert ev.source == "dailymed-metformin.md"
    assert ev.start_line == 44  # heading line in the source file
    assert ev.end_line < 52      # stops before "5 WARNINGS AND PRECAUTIONS"
    assert ev.line_numbered


def test_section_text_matches_source_lines_exactly(ctx, corpus_index):
    """Line numbers shown to the model must be the real line numbers of the source file."""
    run(ctx, "read_source_document", {"filename": "dailymed-warfarin.md", "section": "4 CONTRAINDICATIONS"})
    ev = ctx.ledger.items[0]
    source = (corpus_index.documents[0].file_path.parent / "dailymed-warfarin.md").read_text().splitlines()
    assert ev.text.split("\n") == source[ev.start_line - 1: ev.end_line]
    assert "Malignant hypertension" in source[ev.start_line - 1 + ev.text.split("\n").index("- Malignant hypertension")]


def test_read_line_range(ctx):
    res = run(ctx, "read_source_document", {"filename": "dailymed-warfarin.md", "start_line": 129, "end_line": 135})
    assert res.ok
    assert res.output["lines"] == "L129-L135"
    assert ctx.ledger.items[0].text.count("\n") == 6


def test_read_line_range_beyond_end_is_reported(ctx):
    res = run(ctx, "read_source_document", {"filename": "dailymed-warfarin.md", "start_line": 99999})
    assert not res.ok and "beyond the end" in res.output["error"]


def test_read_unknown_section_lists_sections(ctx):
    res = run(ctx, "read_source_document", {"filename": "dailymed-warfarin.md", "section": "nonexistent heading"})
    assert not res.ok and "Sections:" in res.output["error"]


# --- dispatcher robustness ----------------------------------------------------

def test_unknown_tool(ctx):
    res = run(ctx, "delete_files", {"path": "/"})
    assert not res.ok and "Unknown tool" in res.output["error"]


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]", '"string"'])
def test_malformed_arguments(ctx, raw):
    res = execute_tool("search_corpus", raw, ctx)
    assert not res.ok and "valid JSON object" in res.output["error"]


def test_unexpected_tool_exception_is_contained(ctx, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(ctx.index.retriever, "retrieve", boom)
    res = run(ctx, "search_corpus", {"query": "warfarin"})
    assert not res.ok
    assert "secret internal detail" not in res.output["error"]
