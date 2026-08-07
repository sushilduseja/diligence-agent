import json

from langgraph.checkpoint.memory import InMemorySaver

from dd_agent.export import export_answer_md, export_mermaid
from dd_agent.graph import build_graph
from dd_agent.schema import AgentState, Evidence

NODE_NAMES = [
    "query_planner",
    "docs_rag",
    "github_search",
    "community",
    "normalizer",
    "confidence_scorer",
    "needs_review",
    "final_answer",
    "rejected",
]


class FakeLLM:
    def invoke(self, prompt):
        if prompt.startswith("Decompose"):
            return json.dumps({"docs": "x", "github": "y", "community": "z"})
        return json.dumps({"source_count_weight": 0.9, "agreement": 0.9, "recency": 0.9, "specificity": 0.9})


class FakeClient:
    def get(self, url, **kwargs):
        body = {"items": [{"html_url": f"{url}/1", "title": "t", "body": "b"}]}
        return type("R", (), {"status_code": 200, "text": json.dumps(body)})()


INDEX = [{"url": "https://docs.example.com/a", "chunk": "stripe connect docs overview"}]


def test_mermaid_contains_all_node_names():
    graph = build_graph(
        llm=FakeLLM(),
        http_client=FakeClient(),
        docs_index=INDEX,
        checkpointer=InMemorySaver(),
    )
    mermaid = export_mermaid(graph)
    for name in NODE_NAMES:
        assert name in mermaid


def test_export_answer_md_every_url_once():
    state = AgentState(
        question="Q",
        evidence=[
            Evidence(source_type="docs", url="https://a.example.com", snippet="s1", relevance=0.5),
            Evidence(source_type="github", url="https://b.example.com", snippet="s2", relevance=0.5),
        ],
        confidence=0.9,
        answer="Recommendation.",
    )
    md = export_answer_md(state)
    assert "https://a.example.com" in md
    assert "https://b.example.com" in md
    assert md.count("https://a.example.com") == 1
    assert md.count("https://b.example.com") == 1
    assert "[1]" in md
    assert "[2]" in md
