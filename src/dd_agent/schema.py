"""Typed state model for the due diligence agent."""

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["docs", "github", "community"]


class Evidence(BaseModel):
    source_type: SourceType
    url: str
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)


class ConfidenceBreakdown(BaseModel):
    source_count_weight: float = Field(ge=0.0, le=1.0)
    agreement: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)
    aggregate: float = Field(ge=0.0, le=1.0)


class ReviewRequest(BaseModel):
    """Payload sent to the human reviewer at the approval interrupt."""

    question: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: list[Evidence] = Field(default_factory=list)


class AgentState(BaseModel):
    question: str
    sub_queries: dict[str, str] = Field(default_factory=dict)
    docs_evidence: list[Evidence] = Field(default_factory=list)
    github_evidence: list[Evidence] = Field(default_factory=list)
    community_evidence: list[Evidence] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown | None = None
    approved: bool = False
    answer: str = ""
