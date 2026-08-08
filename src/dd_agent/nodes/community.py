"""Stack Exchange search, capped at MAX_COMMUNITY_RESULTS. Client injected."""

from dd_agent.nodes.retrieval import retrieve


def community_node(query: str, client, cache=None) -> list:
    return retrieve(query, client, "community", cache)
