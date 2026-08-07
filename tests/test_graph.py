import json
import threading
import time

import pytest
from langgraph.errors import GraphInterrupt
from langgraph.checkpoint.memory import InMemorySaver

from dd_agent.graph import build_graph
from dd_agent.schema import AgentState

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
        self.plan = PLAN
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if prompt.startswith("Decompose"):
            return self.plan
        if prompt.startswith("Score the following"):
            return json.dumps(
                {
                    "source_count_weight": self.score,
                    "agreement": self.score,
                    "recency": self.score,
                    "specificity": self.score,
                }
            )
        raise AssertionError(f"unexpected prompt: {prompt[:60]}")


class FakeResponse:
    def __init__(self, status_code=200, text="{}"):
        self.status_code = status_code
        self._text = text

    @property
    def text(self):
        return self._text


class FakeClient:
    """HTTP client that asserts concurrent access via a barrier."""

    def __init__(self, n_participants=2, empty=False):
        self.barrier = threading.Barrier(n_participants, timeout=10)
        self.started_at = []
        self.empty = empty

    def get(self, url, **kwargs):
        self.started_at.append((url, time.time()))
        self.barrier.wait()
        if self.empty:
            return FakeResponse(200, json.dumps({"items": []}))
        return FakeResponse(
            200,
            json.dumps(
                {
                    "items": [
                        {
                            "html_url": f"{url}/result/1",
                            "title": "title 1",
                            "body": "body 1",
                        }
                    ]
                }
            ),
        )


INDEX = [{"url": "https://docs.example.com/connect", "chunk": "stripe connect docs overview"}]


def collect_nodes(graph, question, thread_id="t-1", expect_interrupt=False):
    nodes = []
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    saw_interrupt = False
    for chunk in graph.stream({"question": question}, config):
        for key in chunk:
            if key == "__interrupt__":
                saw_interrupt = True
            elif isinstance(key, str):
                nodes.append(key)
    if expect_interrupt:
        assert saw_interrupt, "expected a __interrupt__ chunk but the run completed"
    return nodes


def test_high_confidence_path():
    llm = FakeLLM(0.9)
    graph = build_graph(
        llm=llm, http_client=FakeClient(), docs_index=INDEX, checkpointer=InMemorySaver()
    )
    nodes = collect_nodes(graph, "Should we integrate Stripe Connect?")
    final = graph.invoke(
        {"question": "Should we integrate Stripe Connect?"},
        {"configurable": {"thread_id": "t-high"}},
    )
    assert "needs_review" not in nodes
    assert final["answer"] != ""
    assert final["confidence"] >= 0.8
    AgentState.model_validate(final)


def test_low_confidence_path_interrupts():
    llm = FakeLLM(0.2)
    graph = build_graph(
        llm=llm, http_client=FakeClient(), docs_index=INDEX, checkpointer=InMemorySaver()
    )
    nodes = collect_nodes(
        graph,
        "Should we integrate Stripe Connect?",
        thread_id="t-low",
        expect_interrupt=True,
    )
    assert "final_answer" not in nodes


def test_retrieval_nodes_ran_in_parallel():
    llm = FakeLLM(0.9)
    client = FakeClient()
    graph = build_graph(
        llm=llm, http_client=client, docs_index=INDEX, checkpointer=InMemorySaver()
    )
    graph.invoke(
        {"question": "Should we integrate Stripe Connect?"},
        {"configurable": {"thread_id": "t-par"}},
    )
    assert len(client.started_at) == 2
    spread = max(t for _, t in client.started_at) - min(t for _, t in client.started_at)
    assert spread < 1.0


def test_planner_empty_docs_key_graph_still_completes():
    llm = FakeLLM(0.9)
    llm.plan = json.dumps(
        {"docs": "", "github": "stripe connect github issues", "community": "stripe connect stackoverflow"}
    )
    graph = build_graph(
        llm=llm, http_client=FakeClient(), docs_index=INDEX, checkpointer=InMemorySaver()
    )
    final = graph.invoke(
        {"question": "Should we integrate Stripe Connect?"},
        {"configurable": {"thread_id": "t-missing-docs"}},
    )
    assert final["docs_evidence"] == []
    assert len(final["github_evidence"]) == 1
    assert final["answer"] != ""


def test_zero_evidence_yields_zero_confidence():
    llm = FakeLLM(0.9)
    graph = build_graph(
        llm=llm,
        http_client=FakeClient(empty=True),
        docs_index=[],
        checkpointer=InMemorySaver(),
    )
    final = graph.invoke(
        {"question": "Question with no matches anywhere"},
        {"configurable": {"thread_id": "t-zero"}},
    )
    assert final["confidence"] == 0.0
    nodes = collect_nodes(
        graph,
        "Question with no matches anywhere",
        thread_id="t-zero-2",
        expect_interrupt=True,
    )
    assert "final_answer" not in nodes
