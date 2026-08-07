from dd_agent.schema import Evidence


def docs_rag_node(query: str, index: list[dict]) -> list[Evidence]:
    """Top-k chunk retrieval over a pre-built local chunk index. No live fetch."""
    if not query:
        return []
    terms = {t for t in query.lower().split() if t}
    if not terms:
        return []
    scored = []
    for entry in index:
        chunk_terms = set(entry["chunk"].lower().split())
        overlap = terms & chunk_terms
        if overlap:
            relevance = len(overlap) / len(terms)
            scored.append(
                Evidence(
                    source_type="docs",
                    url=entry["url"],
                    snippet=entry["chunk"],
                    relevance=relevance,
                )
            )
    scored.sort(key=lambda e: e.relevance, reverse=True)
    return scored
