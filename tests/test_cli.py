import json

import pytest

from dd_agent import cli
from dd_agent.cli import run_cli
from dd_agent.schema import Evidence


def make_chunk(node, update):
    return {node: update}


CHUNKS = [
    make_chunk("query_planner", {"sub_queries": {"docs": "q1", "github": "q2", "community": "q3"}}),
    make_chunk(
        "docs_rag",
        {"docs_evidence": [Evidence(source_type="docs", url="https://docs.example.com/a", snippet="s", relevance=0.5)]},
    ),
    make_chunk(
        "github_search",
        {"github_evidence": [Evidence(source_type="github", url="https://github.com/x/1", snippet="s", relevance=0.5)]},
    ),
    make_chunk("__interrupt__", [object()]),
]


class FakeGraph:
    def __init__(self, chunks, resume_error=False, final_state=None):
        self.chunks = chunks
        self.resume_error = resume_error
        self.final_state = final_state or {"answer": "Final recommendation"}

    def stream(self, input, config):
        yield from self.chunks

    def invoke(self, command, config):
        if self.resume_error:
            raise RuntimeError("no interrupt pending; nothing to resume")
        return self.final_state


def test_cli_streams_chunks_then_approval_prompt():
    captured = []
    prompt_text = []

    def fake_input(prompt=""):
        prompt_text.append(prompt)
        return "y"

    code = run_cli(
        FakeGraph(CHUNKS),
        "Should we integrate Stripe Connect?",
        out=captured.append,
        err=captured.append,
        prompt_input=fake_input,
    )
    assert code == 0
    joined = "\n".join(captured)
    assert joined.index("[query_planner]") < joined.index("[docs_rag]")
    assert joined.index("[docs_rag]") < joined.index("[github_search]")
    assert "https://docs.example.com/a" in joined
    assert "https://github.com/x/1" in joined
    assert prompt_text and "approve this recommendation" in prompt_text[0]
    assert "Final recommendation" in joined


def test_cli_resume_runtime_error_exits_nonzero_no_traceback():
    captured = []
    code = run_cli(
        FakeGraph(CHUNKS, resume_error=True),
        "Question",
        out=captured.append,
        err=captured.append,
        prompt_input=lambda prompt="": "y",
    )
    assert code == 1
    joined = "\n".join(captured)
    assert "error: no interrupt pending" in joined
    assert "Traceback" not in joined


def test_cli_no_interrupt_prints_recommendation_without_prompt():
    captured = []
    graph = FakeGraph(
        [
            make_chunk("query_planner", {"sub_queries": {"docs": "q1"}}),
            make_chunk("final_answer", {"answer": "Recommendation: proceed"}),
        ]
    )
    code = run_cli(
        graph,
        "Question",
        out=captured.append,
        err=captured.append,
        prompt_input=lambda prompt="": pytest.fail("prompt should not be shown"),
    )
    assert code == 0
    joined = "\n".join(captured)
    assert "Recommendation: proceed" in joined


def test_cli_breakdown_rendered():
    captured = []
    graph = FakeGraph(
        [
            make_chunk(
                "confidence_scorer",
                {"confidence": 0.85, "confidence_breakdown": "breakdown-line"},
            )
        ]
    )
    run_cli(graph, "Question", out=captured.append, err=captured.append)
    joined = "\n".join(captured)
    assert "confidence: 0.85" in joined
    assert "breakdown: breakdown-line" in joined


def test_main_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.main(["Should we integrate Stripe Connect?"])


def test_build_default_graph_wiring(tmp_path, monkeypatch):
    from dd_agent import checkpoint, graph, llm

    index_file = tmp_path / "docs_index.json"
    index_file.write_text(json.dumps([{"url": "u", "chunk": "c"}]), encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")
    calls = {}

    class FakeLLM:
        def __init__(self, key):
            calls["llm_key"] = key

    class FakeCheckpointer:
        def __init__(self, path):
            calls["db_path"] = path

    def fake_build(llm, client, index, checkpointer, query_cache):
        calls["index"] = index
        calls["client_headers"] = client.headers
        calls["query_cache"] = query_cache
        return ("graph", llm, checkpointer)

    monkeypatch.setattr(llm, "AnthropicLLM", FakeLLM)
    monkeypatch.setattr(checkpoint, "make_sqlite_checkpointer", FakeCheckpointer)
    monkeypatch.setattr(graph, "build_graph", fake_build)
    result = cli.build_default_graph(index_path=index_file)
    graph_, llm_, checkpointer = result
    assert calls["llm_key"] == "sk-test"
    assert calls["index"] == [{"url": "u", "chunk": "c"}]
    assert calls["client_headers"]["Authorization"] == "Bearer ghp-test"
    assert calls["db_path"] == "checkpoints.db"
    assert calls["query_cache"] is not None


def test_build_default_graph_missing_index(tmp_path, monkeypatch):
    from dd_agent import graph

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    calls = {}

    def fake_build(llm, client, index, checkpointer, query_cache):
        calls["index"] = index
        return "ok"

    monkeypatch.setattr(graph, "build_graph", fake_build)
    result = cli.build_default_graph(index_path=tmp_path / "missing.json")
    assert calls["index"] == []
