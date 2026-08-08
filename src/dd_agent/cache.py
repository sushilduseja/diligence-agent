"""SQLite-backed query cache, keyed by hash of source + sub-query."""

import hashlib
import json
import sqlite3

from dd_agent.schema import Evidence


def _key(source: str, query: str) -> str:
    return hashlib.sha256(f"{source}\0{query}".encode("utf-8")).hexdigest()


class QueryCache:
    """Cache of raw retrieval results keyed on the sub-query. Persists in SQLite."""

    def __init__(self, conn):
        self._conn = conn
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS query_cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self._conn.commit()

    @classmethod
    def open(cls, path: str) -> "QueryCache":
        return cls(sqlite3.connect(path, check_same_thread=False))

    def get(self, source: str, query: str) -> list[Evidence] | None:
        row = self._conn.execute(
            "SELECT payload FROM query_cache WHERE key = ?", (_key(source, query),)
        ).fetchone()
        if row is None:
            return None
        return [Evidence.model_validate(item) for item in json.loads(row[0])]

    def set(self, source: str, query: str, evidence: list[Evidence]) -> None:
        payload = json.dumps([e.model_dump() for e in evidence])
        self._conn.execute(
            "INSERT OR REPLACE INTO query_cache (key, payload) VALUES (?, ?)",
            (_key(source, query), payload),
        )
        self._conn.commit()
