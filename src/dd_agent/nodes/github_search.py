"""GitHub REST search. Client is injected; never constructed here."""

from dd_agent.nodes.retrieval import retrieve


def github_search_node(query: str, client, cache=None) -> list:
    return retrieve(query, client, "github", cache)
