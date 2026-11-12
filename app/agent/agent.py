"""The research agent: PLAN -> RETRIEVE/USE TOOLS -> ANSWER -> REVIEW (-> one REVISION).

LLM calls per question: 1 plan + N research iterations + 1 answer + 1 review,
plus at most 1 revision + 1 re-review when the critic does not approve.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from app.agent.critic import Review, review_answer
from app.agent.planner import make_plan
from app.agent.researcher import run_research
from app.agent.tools import ToolContext
from app.core.config import Settings
from app.rag.evidence import EvidenceLedger, remove_citations, validate_citations
from app.rag.generator import DISCLAIMER, INSUFFICIENT_EVIDENCE_ANSWER, generate_answer
from app.retrieval.index import CorpusIndex
from app.schemas.api import AskResponse, CitationOut

logger = logging.getLogger(__name__)

MAX_REVISIONS = 1


def coverage_text(coverage: Dict[str, List[str]]) -> str:
    """Describe, per drug in the plan, whether the corpus has a dedicated document."""
    lines = []
    for drug, files in coverage.items():
        if files:
            lines.append(f"- {drug}: dedicated corpus documents: {', '.join(files)}")
        else:
            lines.append(f"- {drug}: NO dedicated document in the corpus. It may at most be mentioned in passing in "
                         f"other documents. Lack of coverage is not evidence about the drug itself.")
    return "\n".join(lines)


class ResearchAgent:
    def __init__(self, client, settings: Settings, index: CorpusIndex) -> None:
        self.client = client
        self.settings = settings
        self.index = index

    def run(self, question: str) -> AskResponse:
        started = time.perf_counter()
        logger.info("Question: %s", question)

        # 1. PLAN
        plan = make_plan(self.client, self.settings, question)

        coverage = self.index.drug_coverage(plan.drugs)
        coverage_note = coverage_text(coverage)

        # 2. RETRIEVE + USE TOOLS (LLM-driven loop)
        ctx = ToolContext(index=self.index, settings=self.settings, ledger=EvidenceLedger())
        research = run_research(self.client, ctx, question, plan, coverage_note)
        ledger = ctx.ledger

        reviews: List[Review] = []
        revised = False
        removed_citations: List[str] = []
        if len(ledger) == 0:
            answer, status, final_review = INSUFFICIENT_EVIDENCE_ANSWER, "insufficient_evidence", None
        else:
            # 3. ANSWER
            answer = generate_answer(self.client, self.settings, question, ledger, plan.sub_questions,
                                     coverage_note=coverage_note)
            # 4. REVIEW, with at most one controlled revision
            for attempt in range(MAX_REVISIONS + 1):
                citation_report = validate_citations(answer, ledger)
                review = review_answer(self.client, self.settings, question, plan.sub_questions, answer,
                                       ledger, citation_report, coverage_note)
                reviews.append(review)
                # Stop when approved, when the revision budget is used, or when the answer already is a
                # citation-free "not covered by the corpus" statement that the critic agrees with.
                already_abstains = review.verdict == "insufficient_evidence" and not citation_report.valid
                if review.approved or already_abstains or attempt == MAX_REVISIONS:
                    break
                answer = generate_answer(self.client, self.settings, question, ledger, plan.sub_questions,
                                         feedback=review.feedback_text(), previous_answer=answer,
                                         coverage_note=coverage_note)
                revised = True
            final_review = reviews[-1]
            status = {"approve": "success", "insufficient_evidence": "insufficient_evidence"}.get(
                final_review.verdict, "needs_review")
            # Citations the final review judged as not supporting a relevant claim are never shown as support.
            if final_review.irrelevant_citations:
                removed_citations = final_review.irrelevant_citations
                answer = remove_citations(answer, removed_citations)
                logger.info("Removed citations judged irrelevant by the critic: %s", removed_citations)

        report = validate_citations(answer, ledger)
        citations = [
            CitationOut(source=c.source, start_line=c.start_line, end_line=c.end_line,
                        section=self._section_for(c.source, c.start_line, c.end_line, ledger) or "")
            for c in report.valid
        ]
        elapsed = round(time.perf_counter() - started, 2)
        logger.info("Done in %.1fs: status=%s tool_calls=%d evidence=%d revised=%s",
                    elapsed, status, len(research.tool_calls), len(ledger), revised)

        return AskResponse(
            question=question,
            answer=self._compose(answer, final_review, status, removed_citations),
            citations=citations,
            sources=sorted({c.source for c in citations}),
            disclaimer=DISCLAIMER,
            status=status,
            review=final_review.to_dict() if final_review else {},
            trace={
                "plan": plan.to_dict(),
                "corpus_coverage": coverage,
                "tool_calls": [t.to_dict() for t in research.tool_calls],
                "research_iterations": research.iterations,
                "research_stop_reason": research.stop_reason,
                "research_notes": research.notes,
                "evidence": [{"ref": e.ref, "section": e.section, "origin": e.origin} for e in ledger.items],
                "reviews": [r.to_dict() for r in reviews],
                "revised": revised,
                "invalid_citations": report.invalid,
                "removed_citations": removed_citations,
                "elapsed_seconds": elapsed,
            },
        )

    @staticmethod
    def _section_for(source: str, start: int, end: int, ledger: EvidenceLedger) -> Optional[str]:
        for e in ledger.items:
            if e.source == source and e.start_line <= end and start <= e.end_line:
                return e.section
        return None

    @staticmethod
    def _compose(answer: str, review: Optional[Review], status: str, removed: List[str] = ()) -> str:
        """Append unresolved reviewer flags and the mandatory disclaimer."""
        parts = [answer.strip()]
        if review and status != "success":
            flags = (review.unsupported_claims + review.citation_problems + review.number_problems
                     + review.clinical_ambiguities)
            # Do not re-introduce removed (irrelevant) citations through the notes.
            flags = [f for f in flags if not any(r.split(":")[0] in f for r in removed)]
            if flags:
                parts.append("**Reviewer notes (not fully resolved):**\n" + "\n".join(f"- {f}" for f in flags))
        elif review and review.clinical_ambiguities:
            parts.append("**Reviewer notes:**\n" + "\n".join(f"- {f}" for f in review.clinical_ambiguities))
        parts.append(f"*{DISCLAIMER}*")
        return "\n\n".join(parts)
