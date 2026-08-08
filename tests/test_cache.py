import json

from dd_agent.cache import QueryCache
from dd_agent.nodes.community import community_node
from dd_agent.nodes.github_search import github_search_node


class CountingClient:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "text": json.dumps(
                    {"items": [{"html_url": "https://github.com/x/1", "title": "t", "body": "b"}]}
                ),
            },
        )()


def test_second_call_hits_cache_zero_http_calls(tmp_path):
    cache = QueryCache.open(str(tmp_path / "cache.db"))
    client = CountingClient()
    first = github_search_node("stripe connect", client, cache=cache)
    assert client.calls == 1
    second = github_search_node("stripe connect", client, cache=cache)
    assert client.calls == 1
    assert [e.model_dump() for e in first] == [e.model_dump() for e in second]


def test_cache_miss_different_query(tmp_path):
    cache = QueryCache.open(str(tmp_path / "cache.db"))
    client = CountingClient()
    github_search_node("query one", client, cache=cache)
    github_search_node("query two", client, cache=cache)
    assert client.calls == 2


def test_cache_persists_across_instances(tmp_path):
    path = str(tmp_path / "cache.db")
    client = CountingClient()
    github_search_node("stripe connect", client, cache=QueryCache.open(path))
    assert client.calls == 1
    github_search_node("stripe connect", client, cache=QueryCache.open(path))
    assert client.calls == 1


def test_same_query_different_source_does_not_share_cache(tmp_path):
    cache = QueryCache.open(str(tmp_path / "cache.db"))
    client = CountingClient()
    github_search_node("stripe connect", client, cache=cache)
    community_node("stripe connect", client, cache=cache)
    assert client.calls == 2
