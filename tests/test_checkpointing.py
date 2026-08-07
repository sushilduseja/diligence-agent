import json

import pytest
from pydantic import ValidationError

from dd_agent.checkpoint import make_sqlite_checkpointer
from dd_agent.graph import build_graph
from dd_agent.runner import RunResult, run

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
    def __init__(self, tag=""):
        self.tag = tag

    def get(self, url, **kwargs):
        q = kwargs.get("params", {}).get("q", "")
        tag = self.tag or ("A" if "A" in q else "B")
        if "stackexchange.com" in url:
            body = {"items": [{"link": f"https://stackoverflow.com/q/{tag}", "title": f"q {tag}"}]}
        else:
            body = {"items": [{"html_url": f"https://github.com/{tag}/issue/1", "title": f"t {tag}", "body": f"b {tag}"}]}
        return type("R", (), {"status_code": 200, "text": json.dumps(body)})()


INDEX = [{"url": "https://docs.example.com/a", "chunk": "stripe connect docs overview"}]


def test_restart_resume_completes(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    saver = make_sqlite_checkpointer(db)
    graph1 = build_graph(llm=FakeLLM(0.2), http_client=FakeClient(), docs_index=INDEX, checkpointer=saver)
    r1 = run(graph1, "Should we integrate Stripe Connect?", thread_id="t-restart")
    assert r1.pending_approval is True
    saver2 = make_sqlite_checkpointer(db)
    graph2 = build_graph(llm=FakeLLM(0.2), http_client=FakeClient(), docs_index=INDEX, checkpointer=saver2)
    r2 = run(graph2, "Should we integrate Stripe Connect?", thread_id="t-restart")
    assert r2.pending_approval is True
    r3 = r2.resume(approved=True)
    assert r3.pending_approval is False
    assert r3.answer != ""


def test_resume_never_started_thread_raises_clear_error(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    saver = make_sqlite_checkpointer(db)
    graph = build_graph(
        llm=FakeLLM(0.2), http_client=FakeClient(), docs_index=INDEX, checkpointer=saver
    )
    r = RunResult(graph, "t-never", "Question", interrupt_payload={"question": "Question"})
    with pytest.raises(RuntimeError, match="no checkpoint"):
        r.resume(approved=True)


def test_concurrent_threads_do_not_leak(tmp_path):
    db = str(tmp_path / "checkpoints.db")
    saver = make_sqlite_checkpointer(db)
    index_a = [{"url": "https://docs.example.com/A", "chunk": "stripe connect docs overview"}]
    index_b = [{"url": "https://docs.example.com/B", "chunk": "stripe connect docs overview"}]
    graph = build_graph(
        llm=FakeLLM(0.9),
        http_client=FakeClient(tag="A"),
        docs_index=index_a,
        checkpointer=saver,
    )
    run(graph, "Question A?", thread_id="t-A")
    graph2 = build_graph(
        llm=FakeLLM(0.9),
        http_client=FakeClient(tag="B"),
        docs_index=index_b,
        checkpointer=saver,
    )
    run(graph2, "Question B?", thread_id="t-B")
    a_state = graph.get_state({"configurable": {"thread_id": "t-A"}})
    b_state = graph2.get_state({"configurable": {"thread_id": "t-B"}})
    a_evidence = a_state.values.get("evidence", [])
    b_evidence = b_state.values.get("evidence", [])
    a_urls = {e.url for e in a_evidence}
    b_urls = {e.url for e in b_evidence}
    assert all("A" in u for u in a_urls)
    assert all("B" in u for u in b_urls)
    assert not (a_urls & b_urls)
