import json

import pytest

from dd_agent.confidence import (
    DEFAULT_WEIGHTS,
    ScoringError,
    SubScores,
    aggregate_confidence,
    score,
    score_evidence_item,
)
from dd_agent.schema import Evidence


class FakeLLM:
    """Record calls and return canned structured output."""

    def __init__(self, response=None, raw=None):
        self.calls = 0
        self.response = response
        self.raw = raw

    def invoke(self, *args, **kwargs):
        self.calls += 1
        return self.raw


# --- aggregate_confidence: pure function, zero LLM ---


def test_aggregate_all_ones():
    subs = [SubScores(source_count_weight=1, agreement=1, recency=1, specificity=1)]
    assert aggregate_confidence(subs, DEFAULT_WEIGHTS) == 1.0


def test_aggregate_all_zero():
    subs = [SubScores(source_count_weight=0, agreement=0, recency=0, specificity=0)]
    assert aggregate_confidence(subs, DEFAULT_WEIGHTS) == 0.0


def test_aggregate_known_mixed():
    subs = [SubScores(source_count_weight=1, agreement=0.5, recency=0.5, specificity=0.5)]
    expected = 0.1 * 1 + 0.4 * 0.5 + 0.2 * 0.5 + 0.3 * 0.5
    assert aggregate_confidence(subs, DEFAULT_WEIGHTS) == pytest.approx(expected)


def test_aggregate_empty_list():
    assert aggregate_confidence([], DEFAULT_WEIGHTS) == 0.0


def test_aggregate_weights_not_summing_to_one():
    bad = {"source_count_weight": 1.0, "agreement": 1.0, "recency": 1.0, "specificity": 1.0}
    with pytest.raises(ValueError):
        aggregate_confidence([SubScores(source_count_weight=1, agreement=1, recency=1, specificity=1)], bad)


# --- score_evidence_item: mocked LLM ---


def test_score_evidence_item_parses_valid_json():
    raw = json.dumps(
        {"source_count_weight": 0.2, "agreement": 0.8, "recency": 0.6, "specificity": 0.9}
    )
    llm = FakeLLM(raw=raw)
    item = Evidence(source_type="docs", url="https://x", snippet="s", relevance=0.7)
    subs = score_evidence_item(item, llm)
    assert subs == SubScores(source_count_weight=0.2, agreement=0.8, recency=0.6, specificity=0.9)
    assert llm.calls == 1


def test_score_evidence_item_malformed_json_raises_scoring_error():
    llm = FakeLLM(raw="not json {{{")
    item = Evidence(source_type="docs", url="https://x", snippet="s", relevance=0.7)
    with pytest.raises(ScoringError):
        score_evidence_item(item, llm)


def test_score_evidence_item_parses_json_with_trailing_prose():
    raw = (
        '{"source_count_weight": 0.2, "agreement": 0.8, "recency": 0.6, "specificity": 0.9}\n\n'
        "Note: the source is authoritative and recent."
    )
    llm = FakeLLM(raw=raw)
    item = Evidence(source_type="docs", url="https://x", snippet="s", relevance=0.7)
    subs = score_evidence_item(item, llm)
    assert subs == SubScores(source_count_weight=0.2, agreement=0.8, recency=0.6, specificity=0.9)


def test_score_evidence_item_calls_llm_exactly_once():
    raw = json.dumps(
        {"source_count_weight": 0.1, "agreement": 0.8, "recency": 0.7, "specificity": 0.6}
    )
    llm = FakeLLM(raw=raw)
    item = Evidence(source_type="github", url="https://github.com/x", snippet="issue", relevance=0.7)
    score_evidence_item(item, llm)
    score_evidence_item(item, llm)
    assert llm.calls == 2


# --- score: batch interface, drop-and-continue on malformed items ---


def test_score_batch_aggregates_all_valid():
    raw = json.dumps(
        {"source_count_weight": 0.2, "agreement": 0.8, "recency": 0.6, "specificity": 0.9}
    )
    llm = FakeLLM(raw=raw)
    items = [Evidence(source_type="docs", url="https://x/1", snippet="s", relevance=0.7)] * 2
    breakdown = score(items, llm)
    assert breakdown.aggregate == pytest.approx(0.1 * 0.2 + 0.4 * 0.8 + 0.2 * 0.6 + 0.3 * 0.9)
    assert breakdown.agreement == pytest.approx(0.8)


def test_score_drops_malformed_item_and_continues():
    good = json.dumps(
        {"source_count_weight": 1.0, "agreement": 1.0, "recency": 1.0, "specificity": 1.0}
    )
    items = [
        Evidence(source_type="docs", url="https://x/1", snippet="s", relevance=0.7),
        Evidence(source_type="docs", url="https://x/2", snippet="s", relevance=0.7),
    ]

    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "not json {{{"
            return good

    breakdown = score(items, FlakyLLM())
    assert breakdown.aggregate == 1.0


def test_score_all_malformed_yields_zero():
    llm = FakeLLM(raw="not json {{{")
    items = [Evidence(source_type="docs", url="https://x/1", snippet="s", relevance=0.7)]
    breakdown = score(items, llm)
    assert breakdown.aggregate == 0.0
    assert breakdown.agreement == 0.0


def test_score_empty_items_yields_zero_breakdown():
    breakdown = score([], FakeLLM())
    assert breakdown.aggregate == 0.0
    assert breakdown.specificity == 0.0
