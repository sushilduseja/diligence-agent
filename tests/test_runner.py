import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from dd_agent.graph import build_graph
from dd_agent.runner import run

PLAN = json.dumps(
    {
        "docs": "stripe connect docs",
        "github": "stripe connect github issues",
        "community": "stripe connect stackoverflow",
    }
)


class FakeLLM:
    def __init__(self, score):
        self.score = score

    def invoke(self, prompt):
        if prompt.startswith("Decompose"):
            return PLAN
        if prompt.startswith("Score the following"):
            return json.dumps(
                {
                    "source_count_weight": self.score,
                    "agreement": self.score,
                    "recency": self.score,
                    "specificity": self.score,
                }
            )
        raise AssertionError(prompt)


class FakeClient:
    def get(self, url, **kwargs):
        return type("R", (), {"status_code": 200, "text": '{"items": []}'})()


INDEX = [{"url": "https://docs.example.com/a", "chunk": "stripe connect docs overview"}]


def build_low_conf_graph():
    return build_graph(
        llm=FakeLLM(0.2),
        http_client=FakeClient(),
        docs_index=INDEX,
        checkpointer=InMemorySaver(),
    )


def test_low_confidence_run_pending_approval():
    runner = run(build_low_conf_graph(), "Should we integrate Stripe Connect?")
    assert runner.pending_approval is True
    assert runner.question == "Should we integrate Stripe Connect?"
    assert runner.confidence < 0.8


def test_resume_approved_continues_to_final_answer():
    runner = run(build_low_conf_graph(), "Should we integrate Stripe Connect?")
    result = runner.resume(approved=True)
    assert result.pending_approval is False
    assert result.answer != ""


def test_resume_rejected_explicit_no_recommendation():
    runner = run(build_low_conf_graph(), "Should we integrate Stripe Connect?")
    result = runner.resume(approved=False)
    assert result.pending_approval is False
    assert "no recommendation" in result.answer.lower()
    assert "should not" in result.answer.lower()


def test_resume_without_pending_raises():
    runner = run(build_low_conf_graph(), "Should we integrate Stripe Connect?")
    runner.resume(approved=True)
    with pytest.raises(RuntimeError):
        runner.resume(approved=True)
