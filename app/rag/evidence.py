"""
Evidence collection, context construction and citation validation.

Every passage the agent sees (from either tool) is recorded as an
:class:`Evidence` item with its source file and 1-indexed line range.
Answers cite evidence as ``[filename:Lstart-Lend]``; :func:`validate_citations`
then checks each citation deterministically against what was actually
retrieved. A citation is valid only if it

* exactly matches the line range of a retrieved passage, or
* lies inside a *line-numbered* passage (an exact slice of the source file
  read with ``read_source_document``, shown to the model with per-line
  numbers, so narrower citations can be checked precisely).

Narrowing a search-result chunk to lines the model never saw numbered is
rejected, because such line numbers would be guesses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Matches "dailymed-warfarin.md:L129-L145", "file.md:L12-15" and "file.md:L12".
CITATION_PATTERN = re.compile(r"([A-Za-z0-9_.-]+\.(?:md|txt)):L(\d+)(?:\s*-\s*L?(\d+))?")


@dataclass
class Evidence:
    source: str
    start_line: int
    end_line: int
    text: str
    section: str = ""
    origin: str = ""          # which tool produced it
    score: Optional[float] = None
    line_numbered: bool = False  # text is an exact slice of lines start..end of the source

    def render(self) -> str:
        if not self.line_numbered:
            return self.text.strip()
        lines = self.text.split("\n")
        return "\n".join(f"L{self.start_line + i}: {line}" for i, line in enumerate(lines) if line.strip())

    def contains(self, other: "Evidence") -> bool:
        return (
            self.source == other.source
            and self.start_line <= other.start_line
            and other.end_line <= self.end_line
        )

    @property
    def ref(self) -> str:
        return f"{self.source}:L{self.start_line}-L{self.end_line}"


@dataclass
class Citation:
    source: str
    start_line: int
    end_line: int

    @property
    def ref(self) -> str:
        return f"{self.source}:L{self.start_line}-L{self.end_line}"


@dataclass
class CitationReport:
    valid: List[Citation] = field(default_factory=list)
    invalid: List[Dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.invalid and bool(self.valid)


class EvidenceLedger:
    """Ordered, de-duplicated collection of retrieved passages."""

    def __init__(self) -> None:
        self.items: List[Evidence] = []

    def add(self, ev: Evidence) -> bool:
        """Add a passage unless an existing passage already contains it.

        If the new passage contains existing ones, those are replaced.
        Returns ``True`` if the ledger changed.
        """
        if not ev.text.strip():
            return False
        if any(existing.contains(ev) for existing in self.items):
            return False
        self.items = [e for e in self.items if not ev.contains(e)]
        self.items.append(ev)
        return True

    def __len__(self) -> int:
        return len(self.items)

    @property
    def sources(self) -> List[str]:
        return sorted({e.source for e in self.items})

    def build_context(self, max_chars: int) -> str:
        """Render passages for the LLM, oldest first, within a character budget."""
        blocks: List[str] = []
        used = 0
        for i, ev in enumerate(self.items, start=1):
            header = f"[E{i}] {ev.ref}"
            if ev.section:
                header += f" | section: {ev.section}"
            block = f"{header}\n{ev.render()}"
            if used + len(block) > max_chars:
                blocks.append(f"[... {len(self.items) - i + 1} further passage(s) omitted to fit context limit ...]")
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)


def extract_citations(answer: str) -> List[Citation]:
    """Return unique citations found in an answer, in order of appearance."""
    seen = set()
    out: List[Citation] = []
    for m in CITATION_PATTERN.finditer(answer):
        start = int(m.group(2))
        end = int(m.group(3)) if m.group(3) else start
        if end < start:
            start, end = end, start
        key = (m.group(1), start, end)
        if key not in seen:
            seen.add(key)
            out.append(Citation(source=m.group(1), start_line=start, end_line=end))
    return out


def validate_citations(answer: str, ledger: EvidenceLedger) -> CitationReport:
    """Check every citation in ``answer`` against the retrieved evidence."""
    report = CitationReport()
    for c in extract_citations(answer):
        same_file = [e for e in ledger.items if e.source == c.source]
        if not same_file:
            report.invalid.append({"citation": c.ref, "reason": "file was never retrieved as evidence"})
        elif any(e.start_line == c.start_line and e.end_line == c.end_line for e in same_file) or any(
            e.line_numbered and e.start_line <= c.start_line and c.end_line <= e.end_line for e in same_file
        ):
            report.valid.append(c)
        elif any(e.start_line <= c.end_line and c.start_line <= e.end_line for e in same_file):
            report.invalid.append({"citation": c.ref, "reason": "line range is not a retrieved passage range "
                                   "(narrowed lines of a passage that was not shown with line numbers)"})
        else:
            report.invalid.append({"citation": c.ref, "reason": "line range does not overlap any retrieved passage"})
    return report


# ---------------------------------------------------------------------------
# Numeric fidelity check
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?")
_LIST_MARKER = re.compile(r"^\s*(?:#+\s*)?(?:\d+[.)]|[-*])\s+")
_BRACKETED_CITATION = re.compile(r"\[[^\[\]]*\.(?:md|txt):L\d+[^\[\]]*\]")


_QUANTITY = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*(%|[A-Za-zµμ]+(?:/[A-Za-z]+)?)")
_UNITS = {"mg", "mcg", "g", "kg", "ml", "l", "mmol", "meq", "iu", "unit", "%", "hour", "h", "day", "week",
          "month", "year", "min"}


def _normalize(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,000 -> 1000


def _numbers(text: str) -> set:
    return {float(n) for n in _NUMBER.findall(_normalize(text))}


def _unit(token: str) -> Optional[str]:
    """Normalise a unit token ('mL/minute' -> 'ml/min', 'weeks' -> 'week'); None if not a known unit."""
    parts = [p.lower().replace("µg", "mcg").replace("μg", "mcg") for p in token.split("/")]
    parts = ["min" if p in ("minute", "minutes") else p.rstrip("s") if p not in ("%",) and len(p) > 2 else p
             for p in parts]
    return "/".join(parts) if parts[0] in _UNITS else None


def _quantities(text: str) -> Dict[float, set]:
    """Map each number that is followed by a known unit to the set of units it appears with."""
    out: Dict[float, set] = {}
    for num, token in _QUANTITY.findall(_normalize(text)):
        unit = _unit(token)
        if unit:
            out.setdefault(float(num), set()).add(unit)
    return out


def check_numbers(answer: str, ledger: EvidenceLedger, question: str = "") -> List[str]:
    """Flag numbers (and their units) in cited lines that the cited passages do not contain.

    * a number that appears neither in the cited passages nor in the question is flagged;
    * a number written with a unit (e.g. "30 mg") is flagged if the cited passages show that number only
      with different units (e.g. "30 mL/minute").

    This is a *detector* for altered or invented doses/thresholds, not proof of grounding: a number that
    is present can still be used wrongly (that is the critic's job). Lines without citations are skipped.
    """
    from_question = _numbers(question)
    problems: List[str] = []
    for line in answer.splitlines():
        cites = extract_citations(line)
        if not cites:
            continue
        passages = [e for c in cites for e in ledger.items
                    if e.source == c.source and e.start_line <= c.end_line and c.start_line <= e.end_line]
        evidence_text = "\n".join(f"{e.section}\n{e.text}" for e in passages)
        allowed = from_question | _numbers(evidence_text)
        claim = _LIST_MARKER.sub("", _BRACKETED_CITATION.sub("", line)).strip()
        for n in sorted(_numbers(claim) - allowed):
            problems.append(f"'{_fmt(n)}' in \"{claim[:120]}\" does not appear in the cited passages")
        evidence_units = _quantities(evidence_text)
        for n, units in _quantities(claim).items():
            if n in from_question or n not in evidence_units:
                continue
            if not units & evidence_units[n]:
                problems.append(f"'{_fmt(n)} {'/'.join(sorted(units))}' in \"{claim[:120]}\": the cited passages "
                                f"give {_fmt(n)} with unit(s) {', '.join(sorted(evidence_units[n]))}")
    return problems


def _fmt(n: float):
    return int(n) if n.is_integer() else n


# ---------------------------------------------------------------------------
# Removing citations the reviewer judged non-supporting
# ---------------------------------------------------------------------------

def remove_citations(answer: str, refs: List[str]) -> str:
    """Remove the given citations (e.g. 'file.md:L1-L5') from the answer text; drop emptied brackets."""
    drop = {c.ref for r in refs for c in extract_citations(r)}
    if not drop:
        return answer

    def fix(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[;,]", match.group(0)[1:-1]) if p.strip()]
        keep = [p for p in parts if not any(c.ref in drop for c in extract_citations(p))]
        return f"[{'; '.join(keep)}]" if keep else ""

    cleaned = _BRACKETED_CITATION.sub(fix, answer)
    return re.sub(r"[ \t]+(?=[.,;:]?\s*$)", "", cleaned, flags=re.MULTILINE)
