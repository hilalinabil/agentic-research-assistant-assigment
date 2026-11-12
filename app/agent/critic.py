"""REVIEW step: check a draft answer against the evidence actually gathered.

Two layers:
1. Deterministic citation checks (``validate_citations``): every cited file
   must have been retrieved and every cited line range must overlap a
   retrieved passage.
2. An LLM critic that reads the evidence passages and the draft, and judges
   claim by claim whether each statement is supported, whether anything
   clinically ambiguous needs flagging, and whether the question is fully
   covered.

The critic's verdict cannot be "approve" while invalid citations, unsupported
claims, irrelevant citations or numbers missing from the cited passages
(``check_numbers``) exist: such a verdict is overridden to "revise". Conversely,
a "revise" verdict without any structured finding is treated as "approve".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.config import Settings
from app.core.llm import LLMResponseError, as_str_list, chat_json
from app.rag.evidence import CitationReport, EvidenceLedger, check_numbers, extract_citations

logger = logging.getLogger(__name__)

VERDICTS = ("approve", "revise", "insufficient_evidence")

CRITIC_PROMPT = """You are a strict reviewer of a pharmacology research answer. You receive the \
question, corpus coverage information, the evidence passages that were retrieved, automatic checks, and \
a draft answer.

Check the draft claim by claim against the evidence text (not against your own knowledge):
- unsupported_claims: statements not stated in, or that go beyond, the evidence (including invented \
interactions, doses, thresholds or monitoring advice, and turning "not in the documents" into "does not \
exist" or "is safe").
- citation_problems: citations whose passage does not support the sentence they are attached to.
- irrelevant_citations: citations (as "file.md:Lx-Ly") pointing to passages that do not support a claim \
answering the question, e.g. passages about other drugs or topics, or citations attached to statements \
about missing information. These will be removed from the answer.
- clinical_ambiguities: places where the answer could mislead clinically, e.g. a unit or population in \
the question that differs from the label (non-indexed vs. body-surface-indexed values, adults vs. \
children) applied without stating it, missing qualifiers, or altered numbers.
- missing_aspects: go through the evidence passages one by one and list only omissions that change or \
qualify the answer to the question asked — e.g. a monitoring, dose-adjustment or discontinuation \
instruction that a label gives for exactly the drugs/situation in the question, or a threshold or \
condition the question asks about — and parts of the question not marked as not covered. Do NOT list \
general background or safety information the question does not ask about.
- source_conflicts: disagreements between documents.

Verdict:
- "approve": every factual claim is supported, citations support their sentences, nothing listed under \
missing_aspects, and the question is covered or gaps are stated. Optional extra detail is not a reason \
to revise.
- "revise": the problems listed in the fields above can be fixed using the existing evidence. Only use \
"revise" for problems you list there; wording or style preferences are not a reason to revise.
- "insufficient_evidence": the evidence cannot support an answer to the core question (e.g. the corpus \
has no document about the drug asked about and no passage addresses it).

Return ONLY a JSON object:
{"verdict": "approve|revise|insufficient_evidence", "unsupported_claims": [], "citation_problems": [], \
"irrelevant_citations": [], "clinical_ambiguities": [], "missing_aspects": [], "source_conflicts": [], \
"feedback": "concise instructions for the writer"}"""


@dataclass
class Review:
    verdict: str
    unsupported_claims: List[str] = field(default_factory=list)
    citation_problems: List[str] = field(default_factory=list)
    irrelevant_citations: List[str] = field(default_factory=list)
    number_problems: List[str] = field(default_factory=list)
    clinical_ambiguities: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)
    source_conflicts: List[str] = field(default_factory=list)
    feedback: str = ""
    overrides: List[str] = field(default_factory=list)  # why the LLM verdict was changed

    @property
    def approved(self) -> bool:
        return self.verdict == "approve"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def feedback_text(self) -> str:
        lines = [self.feedback] if self.feedback else []
        for title, items in [("Unsupported claims", self.unsupported_claims),
                             ("Citation problems", self.citation_problems),
                             ("Citations that do not support a relevant claim (remove them)", self.irrelevant_citations),
                             ("Numbers not found in the cited passages", self.number_problems),
                             ("Clinical ambiguities", self.clinical_ambiguities),
                             ("Missing aspects", self.missing_aspects),
                             ("Source conflicts", self.source_conflicts)]:
            if items:
                lines.append(f"{title}:\n" + "\n".join(f"- {i}" for i in items))
        return "\n\n".join(lines)


def review_answer(client, settings: Settings, question: str, sub_questions: List[str], draft: str,
                  ledger: EvidenceLedger, citations: CitationReport, coverage_note: str = "") -> Review:
    number_problems = check_numbers(draft, ledger, question)
    citation_check = (
        f"Valid citations: {', '.join(c.ref for c in citations.valid) or 'none'}\n"
        f"Invalid citations: {citations.invalid or 'none'}\n"
        f"Numbers in cited sentences not found in the cited passages: {number_problems or 'none'}"
    )
    user = (
        f"Question:\n{question}\n\nSub-questions:\n" + "\n".join(f"- {s}" for s in sub_questions)
        + (f"\n\nCorpus coverage:\n{coverage_note}" if coverage_note else "")
        + f"\n\nEvidence passages:\n\n{ledger.build_context(settings.max_evidence_chars)}"
        + f"\n\nAutomatic citation check:\n{citation_check}\n\nDraft answer:\n{draft}"
    )
    messages = [{"role": "system", "content": CRITIC_PROMPT}, {"role": "user", "content": user}]
    try:
        data = chat_json(client, settings.openai_model, messages)
        verdict = str(data.get("verdict", "")).strip().lower()
        review = Review(
            verdict=verdict if verdict in VERDICTS else "revise",
            unsupported_claims=as_str_list(data.get("unsupported_claims")),
            citation_problems=as_str_list(data.get("citation_problems")),
            irrelevant_citations=[c.ref for c in extract_citations(" ".join(as_str_list(data.get("irrelevant_citations"))))],
            clinical_ambiguities=as_str_list(data.get("clinical_ambiguities")),
            missing_aspects=as_str_list(data.get("missing_aspects")),
            source_conflicts=as_str_list(data.get("source_conflicts")),
            feedback=str(data.get("feedback") or "").strip(),
        )
        if verdict not in VERDICTS:
            review.overrides.append(f"unrecognised verdict {verdict!r} treated as 'revise'")
    except LLMResponseError as exc:
        logger.warning("Critic output unusable: %s", exc)
        review = Review(verdict="revise", overrides=[f"critic output unusable ({exc})"],
                        feedback="Re-check every claim against the evidence and cite each sentence.")

    # Deterministic guards: the critic cannot approve an answer with known problems.
    for item in citations.invalid:
        review.citation_problems.append(f"{item['citation']}: {item['reason']}")
    review.number_problems = number_problems
    if review.approved:
        if citations.invalid:
            review.verdict = "revise"
            review.overrides.append("invalid citations detected by automatic check")
        elif review.unsupported_claims:
            review.verdict = "revise"
            review.overrides.append("critic listed unsupported claims but approved")
        elif not citations.valid:
            # An answer without a single valid citation can never be presented as grounded;
            # at most it is a statement that the corpus does not answer the question.
            review.verdict = "insufficient_evidence"
            review.overrides.append("approved answer has no valid citations: treated as insufficient evidence")
        elif review.irrelevant_citations:
            review.verdict = "revise"
            review.overrides.append("critic listed irrelevant citations but approved")
        elif number_problems:
            review.verdict = "revise"
            review.overrides.append("numbers not found in the cited passages")
    findings = (review.unsupported_claims + review.citation_problems + review.irrelevant_citations
                + review.number_problems + review.clinical_ambiguities + review.missing_aspects
                + review.source_conflicts)
    if review.verdict == "revise" and not findings and not review.overrides:
        # A revision must be justified by a concrete finding; free-text style advice alone is kept for
        # transparency but does not block a grounded answer.
        review.verdict = "approve"
        review.overrides.append("'revise' without any structured finding treated as 'approve'")
    logger.info("Critic verdict=%s unsupported=%d ambiguities=%d overrides=%s", review.verdict,
                len(review.unsupported_claims), len(review.clinical_ambiguities), review.overrides)
    return review
