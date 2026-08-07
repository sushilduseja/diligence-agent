import json

import pytest

from dd_agent.nodes.query_planner import PlanningError, plan_queries


class FakeLLM:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def invoke(self, *args, **kwargs):
        self.calls += 1
        return self.raw


def test_valid_structured_output():
    raw = json.dumps(
        {
            "docs": "stripe connect docs",
            "github": "stripe connect issues",
            "community": "stripe connect stackoverflow",
        }
    )
    result = plan_queries("Should we integrate Stripe Connect?", FakeLLM(raw))
    assert set(result.keys()) == {"docs", "github", "community"}
    assert all(isinstance(v, str) and v for v in result.values())


def test_missing_key_raises_planning_error():
    raw = json.dumps({"docs": "x", "github": "y"})
    with pytest.raises(PlanningError):
        plan_queries("question", FakeLLM(raw))


def test_empty_query_key_dropped():
    raw = json.dumps({"docs": "", "github": "gh query", "community": "so query"})
    result = plan_queries("question", FakeLLM(raw))
    assert "docs" not in result
    assert set(result.keys()) == {"github", "community"}
