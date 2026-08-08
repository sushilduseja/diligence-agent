"""Confidence scoring: deterministic weighted-sum math, LLM sub-score assignment."""

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from dd_agent.schema import ConfidenceBreakdown, Evidence

logger = logging.getLogger("dd_agent")

DEFAULT_WEIGHTS = {
    "agreement": 0.4,
    "specificity": 0.3,
    "recency": 0.2,
    "source_count_weight": 0.1,
}


class ScoringError(RuntimeError):
    """Raised when the LLM produces output that cannot be parsed into SubScores."""


class SubScores(BaseModel):
    source_count_weight: float = Field(ge=0.0, le=1.0)
    agreement: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    specificity: float = Field(ge=0.0, le=1.0)


def aggregate_confidence(sub_scores: list[SubScores], weights: dict[str, float]) -> float:
    """Weighted sum of sub-scores, clipped to [0, 1]. Pure, no LLM."""
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights.values())}")
    if not sub_scores:
        return 0.0
    total = 0.0
    for s in sub_scores:
        total += weights["agreement"] * s.agreement
        total += weights["specificity"] * s.specificity
        total += weights["recency"] * s.recency
        total += weights["source_count_weight"] * s.source_count_weight
    return round(max(0.0, min(1.0, total / len(sub_scores))), 10)


def score_evidence_item(item: Evidence, llm) -> SubScores:
    """Ask the LLM to assign sub-scores to one evidence item. Returns structured SubScores."""
    prompt = (
        "Score the following evidence item on four axes, each 0-1. "
        'Respond with JSON: {"source_count_weight": float, "agreement": float, '
        '"recency": float, "specificity": float}.\n'
        f"source_type: {item.source_type}\nurl: {item.url}\nsnippet: {item.snippet}\n"
    )
    raw = llm.invoke(prompt)
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        return SubScores.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise ScoringError(f"could not parse LLM sub-scores: {e}") from e


def score(items: list[Evidence], llm, weights: dict[str, float] = None) -> ConfidenceBreakdown:
    """Score a batch of evidence into a ConfidenceBreakdown. One interface for the graph gate."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    subs = []
    for item in items:
        try:
            subs.append(score_evidence_item(item, llm))
        except ScoringError as e:
            logger.warning("dropping unparseable evidence item %s: %s", item.url, e)
    if not subs:
        return ConfidenceBreakdown(
            source_count_weight=0, agreement=0, recency=0, specificity=0, aggregate=0
        )
    aggregate = aggregate_confidence(subs, weights)
    n = len(subs)
    return ConfidenceBreakdown(
        source_count_weight=sum(s.source_count_weight for s in subs) / n,
        agreement=sum(s.agreement for s in subs) / n,
        recency=sum(s.recency for s in subs) / n,
        specificity=sum(s.specificity for s in subs) / n,
        aggregate=aggregate,
    )
