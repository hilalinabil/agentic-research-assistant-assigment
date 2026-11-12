"""RETRIEVE + USE TOOLS step: a bounded, LLM-driven tool-calling loop.

The model receives the question, the plan and the tool schemas, and decides
on every iteration which tool to call (or to stop). Python validates and
executes the chosen calls and feeds observations back. There is no
question-specific logic here: the research path is chosen by the model.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.agent.planner import Plan
from app.agent.tools import ToolContext, execute_tool, tool_schemas

logger = logging.getLogger(__name__)

RESEARCH_PROMPT = """You are the research step of a pharmacology question-answering system. Your job \
is to gather evidence from the local corpus with the tools provided. You must not answer from memory.

Corpus documents: {filenames}
(dailymed-*.md = FDA prescribing labels, the most detailed source; medlineplus-*.md = consumer \
information; GLOSSARY.md = term definitions.)

How to work:
- Cover every sub-question of the plan. For multi-drug questions, gather evidence for EACH drug \
separately and look for interaction information in each drug's documents.
- Use search_corpus with focused queries. When a hit is a fragment of a section you need in full \
(e.g. a contraindication list, an interaction section, a renal dosing section), use \
read_source_document to read that section.
- Avoid repeating identical calls.
- When you have evidence for every sub-question, or have established that the corpus does not \
contain it, stop calling tools and reply with a short note listing which sub-questions are covered \
and which are not. Do not write the final answer."""

NUDGE = ("You have not gathered any evidence yet. Use the tools to search the corpus before finishing.")


@dataclass
class ToolCallRecord:
    step: int
    tool: str
    arguments: str
    ok: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ResearchResult:
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = ""
    notes: str = ""


def _summarize(output: Dict[str, Any]) -> str:
    if "error" in output:
        return f"error: {output['error'][:200]}"
    if "passages" in output:
        srcs = sorted({p["source"] for p in output["passages"]})
        return f"{output['num_results']} passages from {', '.join(srcs) or 'nothing'}"
    if "sections" in output:
        return f"table of contents of {output['source']} ({len(output['sections'])} sections)"
    return f"{output.get('source')} {output.get('lines')} ({output.get('section', '')})"


def run_research(client, ctx: ToolContext, question: str, plan: Plan, coverage_note: str = "") -> ResearchResult:
    settings = ctx.settings
    result = ResearchResult()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": RESEARCH_PROMPT.format(filenames=", ".join(ctx.index.filenames))},
        {"role": "user", "content": (
            f"Question: {question}\n\nPlan (sub-questions):\n"
            + "\n".join(f"{i}. {s}" for i, s in enumerate(plan.sub_questions, 1))
            + (f"\n\nCorpus coverage:\n{coverage_note}" if coverage_note else "")
        )},
    ]
    seen_calls: set = set()
    nudged = False

    for iteration in range(1, settings.max_agent_iterations + 1):
        result.iterations = iteration
        response = client.chat.completions.create(
            model=settings.openai_model, messages=messages, tools=tool_schemas(), tool_choice="auto",
        )
        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            if len(ctx.ledger) == 0 and not nudged:
                nudged = True
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user", "content": NUDGE})
                logger.info("Research iteration %d: model stopped without evidence; nudging once", iteration)
                continue
            result.stop_reason = "model_finished"
            result.notes = (msg.content or "").strip()
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            name, raw_args = tc.function.name, tc.function.arguments or "{}"
            key = (name, json.dumps(_safe_load(raw_args), sort_keys=True))
            if len(result.tool_calls) >= settings.max_tool_calls:
                ok, output = False, {"error": "Tool-call budget exhausted. Stop and summarize your findings."}
            elif key in seen_calls:
                ok, output = False, {"error": "Duplicate call: this exact call was already made; its result is above."}
            else:
                seen_calls.add(key)
                tr = execute_tool(name, raw_args, ctx)
                ok, output = tr.ok, tr.output
            record = ToolCallRecord(iteration, name, raw_args, ok, _summarize(output))
            result.tool_calls.append(record)
            logger.info("Tool call [iter %d] %s(%s) -> %s", iteration, name, raw_args[:150], record.summary)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output, ensure_ascii=False)})
    else:
        result.stop_reason = "max_iterations"
        logger.warning("Research stopped at the iteration limit (%d)", settings.max_agent_iterations)

    return result


def _safe_load(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
