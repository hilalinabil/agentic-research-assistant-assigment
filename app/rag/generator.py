"""
Grounded answer generation (Phase 3 — basic RAG).

``generate_answer`` turns collected evidence into a cited answer. It is used
by the agent (after tool-driven research) and by ``basic_rag`` — a simple
single-retrieval baseline (question -> embed -> top-k -> answer).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.core.config import Settings
from app.core.llm import chat_text
from app.rag.evidence import Evidence, EvidenceLedger
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This information is provided for research and educational purposes only. It is not medical "
    "advice and should not replace consultation with a qualified healthcare professional."
)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "The local corpus did not return any passages relevant to this question, so no grounded "
    "answer can be given."
)

ANSWER_SYSTEM_PROMPT = """You are a pharmacology research assistant. You answer ONLY from the evidence \
passages supplied by the user message. The passages come from FDA DailyMed drug labels, MedlinePlus \
consumer information and a glossary.

Rules:
1. Every factual sentence must end with at least one citation in the exact form \
[filename:Lstart-Lend]. Use the filename and line range exactly as written in the evidence header \
(e.g. [dailymed-warfarin.md:L129-L145]). Only when a passage shows per-line numbers ("L135: ...") \
may you cite narrower lines from it (e.g. [dailymed-warfarin.md:L135-L135]). Never guess line numbers.
2. Cite a passage only if it directly supports a claim that answers the question. Passages that were \
retrieved but are about other drugs or topics must be ignored, not cited. Never attach citations to \
statements about missing information, and never cite passages just to show what the corpus contains.
3. Never state a fact, interaction, contraindication, dose or threshold that is not in the evidence. \
Reproduce numbers with their units and qualifiers exactly as written in the evidence.
4. If a value, unit or population in the question differs from how the evidence expresses it (e.g. \
a different unit, a non-indexed vs. body-surface-indexed value, adults vs. children), say so explicitly \
and state any assumption needed to apply the evidence.
5. Be complete but focused: include every point in the evidence that directly answers the question or \
qualifies the answer — including management instructions (monitoring, dose adjustment, discontinuation) \
that a label gives for the drugs or situation asked about, from each drug's own label. Do not add general \
safety information that the question does not ask about.
6. If the evidence does not cover the question or part of it, say so explicitly, without citations: \
"The provided documents do not contain information on ...". This means the local corpus lacks coverage; \
never conclude that a drug, interaction or effect does not exist, or that something is safe.
7. If sources differ or are ambiguous, point that out and cite both. Keep clinically important qualifiers \
(e.g. "unless the benefit outweighs the risk", "in patients with ...").
8. If you add reasoning that combines facts from several passages, label it "Interpretation:" and cite \
the passages it is based on.
9. Format in Markdown: a 1-3 sentence direct answer first, then short sections or bullet points, then \
an "Evidence gaps" section if anything asked is not covered. Do not add a medical disclaimer \
(the system appends one)."""


def _user_prompt(question: str, context: str, plan: Optional[List[str]], feedback: Optional[str],
                 previous_answer: Optional[str], coverage_note: Optional[str] = None) -> str:
    parts = [f"Question:\n{question}"]
    if plan:
        parts.append("Sub-questions the answer must address:\n" + "\n".join(f"- {s}" for s in plan))
    if coverage_note:
        parts.append(f"Corpus coverage:\n{coverage_note}")
    parts.append(f"Evidence passages:\n\n{context}")
    if previous_answer and feedback:
        parts.append(
            "A reviewer checked your previous draft against the evidence and found problems.\n\n"
            f"Previous draft:\n{previous_answer}\n\nReviewer feedback:\n{feedback}\n\n"
            "Write a corrected answer: remove or fix every unsupported claim, fix or remove citations that "
            "do not support their sentence, add relevant evidence that was omitted, and state explicitly "
            "what the evidence does not cover."
        )
    return "\n\n".join(parts)


def generate_answer(
    client,
    settings: Settings,
    question: str,
    ledger: EvidenceLedger,
    plan: Optional[List[str]] = None,
    feedback: Optional[str] = None,
    previous_answer: Optional[str] = None,
    coverage_note: Optional[str] = None,
) -> str:
    """Generate a cited answer from the evidence in ``ledger``."""
    if len(ledger) == 0:
        return INSUFFICIENT_EVIDENCE_ANSWER
    context = ledger.build_context(settings.max_evidence_chars)
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(question, context, plan, feedback, previous_answer, coverage_note)},
    ]
    return chat_text(client, settings.openai_model, messages)


def basic_rag(client, settings: Settings, retriever: Retriever, question: str, top_k: Optional[int] = None):
    """Non-agentic baseline: one retrieval for the raw question, then generation.

    Returns ``(answer, ledger)``.
    """
    ledger = EvidenceLedger()
    for r in retriever.retrieve(question, top_k=top_k or settings.default_top_k):
        ch = r.chunk
        ledger.add(Evidence(source=ch.source_filename, start_line=ch.start_line, end_line=ch.end_line,
                            text=ch.text, section=ch.section_path, origin="basic_rag", score=r.score))
    return generate_answer(client, settings, question, ledger), ledger
