from dd_agent.nodes.normalizer import normalize
from dd_agent.schema import Evidence


def _e(source_type, url, snippet="snippet"):
    return Evidence(source_type=source_type, url=url, snippet=snippet, relevance=0.5)


def test_dedupes_by_url():
    lists = [
        [_e("docs", "https://a"), _e("github", "https://b")],
        [_e("github", "https://a"), _e("community", "https://c")],
    ]
    result = normalize(lists)
    urls = [e.url for e in result]
    assert urls.count("https://a") == 1
    assert len(result) == 3


def test_drops_empty_snippet():
    result = normalize([[_e("docs", "https://a", snippet="")]])
    assert result == []


def test_empty_input_lists():
    assert normalize([]) == []
    assert normalize([[], []]) == []


def test_order_stable_docs_github_community():
    lists = [
        [_e("github", "https://gh"), _e("community", "https://so")],
        [_e("docs", "https://doc")],
    ]
    result = normalize(lists)
    assert [e.source_type for e in result] == ["docs", "github", "community"]
