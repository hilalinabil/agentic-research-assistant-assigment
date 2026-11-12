"""PLAN step: the LLM decomposes the user question into sub-questions."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from app.core.config import Settings
from app.core.llm import LLMResponseError, as_str_list, chat_json

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You plan research for a pharmacology question that will be answered ONLY from a \
local corpus of FDA DailyMed drug labels, MedlinePlus consumer pages and a glossary.

Break the question into the minimal set of concrete sub-questions needed to answer it completely \
(1 for a simple question, up to 5 for compound or multi-drug questions). Keep drug names and \
clinical terms exactly as written. List drugs by generic name (e.g. "apixaban" for "Eliquis"). \
Do not answer the question.

Return ONLY a JSON object:
{"drugs": ["..."], "sub_questions": ["..."], "is_compound": true|false}"""

MAX_SUB_QUESTIONS = 5


@dataclass
class Plan:
    sub_questions: List[str]
    drugs: List[str] = field(default_factory=list)
    is_compound: bool = False
    fallback: bool = False  # True when the planner failed and the raw question is used

    def to_dict(self) -> dict:
        return {"drugs": self.drugs, "sub_questions": self.sub_questions,
                "is_compound": self.is_compound, "fallback": self.fallback}


def make_plan(client, settings: Settings, question: str) -> Plan:
    messages = [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": question}]
    try:
        data = chat_json(client, settings.openai_model, messages)
        subs = as_str_list(data.get("sub_questions"))[:MAX_SUB_QUESTIONS]
        if not subs:
            raise LLMResponseError("planner returned no sub-questions")
        plan = Plan(sub_questions=subs, drugs=as_str_list(data.get("drugs")),
                    is_compound=bool(data.get("is_compound", len(subs) > 1)))
    except LLMResponseError as exc:
        logger.warning("Planner output unusable (%s); falling back to the original question", exc)
        plan = Plan(sub_questions=[question], fallback=True)
    logger.info("Plan: %s", plan.to_dict())
    return plan
