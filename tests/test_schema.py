import pytest
from pydantic import ValidationError

from dd_agent.schema import ConfidenceBreakdown, Evidence, AgentState


def test_evidence_valid():
    e = Evidence(
        source_type="docs",
        url="https://example.com/docs",
        snippet="Install with pip",
        relevance=0.9,
    )
    assert e.source_type == "docs"
    assert e.relevance == 0.9


def test_evidence_relevance_out_of_range():
    with pytest.raises(ValidationError):
        Evidence(source_type="docs", url="https://x", snippet="s", relevance=1.5)
    with pytest.raises(ValidationError):
        Evidence(source_type="docs", url="https://x", snippet="s", relevance=-0.1)


def test_evidence_source_type_literal():
    with pytest.raises(ValidationError):
        Evidence(source_type="reddit", url="https://x", snippet="s", relevance=0.5)


def test_agent_state_empty_evidence_valid():
    s = AgentState(question="Should we integrate Stripe Connect?")
    assert s.evidence == []
    assert s.confidence == 0.0


def test_serialization_round_trip():
    s = AgentState(
        question="Q",
        sub_queries={"docs": "docs q", "github": "gh q"},
        evidence=[
            Evidence(source_type="github", url="https://github.com/x", snippet="issue", relevance=0.7)
        ],
        confidence=0.8,
        confidence_breakdown=ConfidenceBreakdown(
            source_count_weight=0.1,
            agreement=0.8,
            recency=0.9,
            specificity=0.7,
            aggregate=0.75,
        ),
        answer="A",
    )
    restored = AgentState.model_validate(s.model_dump())
    assert restored == s
