import json
import logging

import httpx

from dd_agent.nodes.http_retry import get_with_retry
from dd_agent.schema import Evidence

logger = logging.getLogger("dd_agent")

MAX_RESULTS = 5


def community_node(query: str, client, cache=None) -> list[Evidence]:
    """Stack Exchange search, capped at MAX_RESULTS. Client injected."""
    if not query:
        return []
    if cache is not None:
        cached = cache.get(query)
        if cached is not None:
            return cached
    try:
        resp = get_with_retry(
            client,
            "https://api.stackexchange.com/2.3/search/advanced",
            {
                "site": "stackoverflow",
                "q": query,
                "pagesize": MAX_RESULTS,
                "order": "desc",
                "sort": "relevance",
            },
        )
        if resp.status_code != 200:
            logger.warning("Stack Exchange search returned status %s", resp.status_code)
            return []
        items = json.loads(resp.text).get("items", [])
    except (json.JSONDecodeError, httpx.HTTPError) as e:
        logger.warning("Stack Exchange search failed: %s", e)
        return []
    result = []
    for it in items[:MAX_RESULTS]:
        result.append(
            Evidence(
                source_type="community",
                url=it.get("link", ""),
                snippet=it.get("title", ""),
                relevance=1.0,
            )
        )
    if cache is not None:
        cache.set(query, result)
    return result
