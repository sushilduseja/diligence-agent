"""Shared HTTP retrieval behind thin per-source wrappers."""

import json
import logging

import httpx

from dd_agent.nodes.http_retry import get_with_retry
from dd_agent.schema import Evidence, SourceType

logger = logging.getLogger("dd_agent")

MAX_COMMUNITY_RESULTS = 5


def _community_params(query: str) -> dict:
    return {
        "site": "stackoverflow",
        "q": query,
        "pagesize": MAX_COMMUNITY_RESULTS,
        "order": "desc",
        "sort": "relevance",
    }


SOURCES: dict[SourceType, dict] = {
    "github": {
        "url": "https://api.github.com/search/issues",
        "params": lambda q: {"q": q},
        "url_field": "html_url",
        "snippet": lambda it: f"{it.get('title', '')} - {it.get('body', '')}"[:500],
        "max_results": None,
    },
    "community": {
        "url": "https://api.stackexchange.com/2.3/search/advanced",
        "params": _community_params,
        "url_field": "link",
        "snippet": lambda it: it.get("title", ""),
        "max_results": MAX_COMMUNITY_RESULTS,
    },
}


def retrieve(query: str, client, source: SourceType, cache=None) -> list[Evidence]:
    """Fetch evidence for one source. Client and cache are injected; source selects the endpoint config."""
    if not query:
        return []
    if cache is not None:
        cached = cache.get(source, query)
        if cached is not None:
            return cached
    cfg = SOURCES[source]
    try:
        resp = get_with_retry(client, cfg["url"], cfg["params"](query))
        if resp.status_code != 200:
            logger.warning("%s search returned status %s", source, resp.status_code)
            return []
        items = json.loads(resp.text).get("items", [])
    except (json.JSONDecodeError, httpx.HTTPError) as e:
        logger.warning("%s search failed: %s", source, e)
        return []
    if cfg["max_results"]:
        items = items[: cfg["max_results"]]
    result = [
        Evidence(
            source_type=source,
            url=it.get(cfg["url_field"], ""),
            snippet=cfg["snippet"](it),
            relevance=1.0,
        )
        for it in items
    ]
    if cache is not None:
        cache.set(source, query, result)
    return result
