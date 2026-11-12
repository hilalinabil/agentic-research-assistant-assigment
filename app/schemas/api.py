"""Request/response models for the HTTP API."""
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["What are the main contraindications for warfarin?"])

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return v


class CitationOut(BaseModel):
    source: str
    start_line: int
    end_line: int
    section: str = ""


AnswerStatus = Literal["success", "needs_review", "insufficient_evidence"]


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationOut]
    sources: List[str]
    disclaimer: str
    status: AnswerStatus
    review: Dict[str, Any] = Field(default_factory=dict, description="Final critic review")
    trace: Dict[str, Any] = Field(default_factory=dict, description="Plan, tool calls, reviews, timing")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    index_ready: bool
    index_status: Literal["building", "ready", "failed"] = "building"
    build_attempts: int = 0
    documents: int = 0
    chunks: int = 0
    detail: str = ""
