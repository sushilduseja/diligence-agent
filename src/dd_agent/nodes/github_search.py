import json
import logging

import httpx

from dd_agent.nodes.http_retry import get_with_retry
from dd_agent.schema import Evidence

logger = logging.getLogger("dd_agent")


def github_search_node(query: str, client, cache=None) -> list[Evidence]:
    """GitHub REST search. Client is injected; never constructed here."""
    if not query:
        return []
    if cache is not None:
        cached = cache.get(query)
        if cached is not None:
            return cached
    try:
        resp = get_with_retry(client, "https://api.github.com/search/issues", {"q": query})
        if resp.status_code != 200:
            logger.warning("GitHub search returned status %s", resp.status_code)
            return []
        items = json.loads(resp.text).get("items", [])
    except (json.JSONDecodeError, httpx.HTTPError) as e:
        logger.warning("GitHub search failed: %s", e)
        return []
    result = []
    for it in items:
        snippet = f"{it.get('title', '')} — {it.get('body', '')}"[:500]
        result.append(
            Evidence(
                source_type="github",
                url=it.get("html_url", ""),
                snippet=snippet,
                relevance=1.0,
            )
        )
    if cache is not None:
        cache.set(query, result)
    return result
