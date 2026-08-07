from dd_agent.nodes.docs_rag import docs_rag_node

INDEX = [
    {"url": "https://docs.example.com/a", "chunk": "The Stripe Connect API supports onboarding."},
    {"url": "https://docs.example.com/b", "chunk": "Payouts are batched nightly for connected accounts."},
    {"url": "https://docs.example.com/c", "chunk": "Webhooks are delivered with signed payloads."},
]


def test_query_present_in_chunk_2_returns_chunk_2_first():
    result = docs_rag_node("payouts batched", INDEX)
    assert len(result) >= 1
    assert result[0].url == "https://docs.example.com/b"


def test_no_match_returns_empty():
    assert docs_rag_node("zzzzz nonexistent term", INDEX) == []


def test_source_type_always_docs():
    for item in docs_rag_node("onboarding", INDEX):
        assert item.source_type == "docs"


def test_empty_query_returns_empty_without_error():
    assert docs_rag_node("", INDEX) == []
