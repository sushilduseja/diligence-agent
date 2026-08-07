"""Merge parallel node outputs into one deduplicated, ordered list."""

from dd_agent.schema import Evidence

_ORDER = {"docs": 0, "github": 1, "community": 2}


def normalize(evidence_lists: list[list[Evidence]]) -> list[Evidence]:
    """Flatten, dedupe by URL, drop empty snippets, keep source order stable."""
    seen: set[str] = set()
    result: list[Evidence] = []
    for lst in evidence_lists:
        for item in lst:
            if not item.snippet:
                continue
            if item.url in seen:
                continue
            seen.add(item.url)
            result.append(item)
    result.sort(key=lambda e: _ORDER[e.source_type])
    return result
