"""SQLite checkpointer construction with msgpack module registration."""

import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

SCHEMA_TYPES = ["Evidence", "ConfidenceBreakdown", "ReviewRequest"]


def make_sqlite_checkpointer(path: str) -> SqliteSaver:
    """Build a long-lived SqliteSaver, registering our schema types for msgpack."""
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(
        conn,
        serde=JsonPlusSerializer(allowed_msgpack_modules=[("dd_agent.schema", t) for t in SCHEMA_TYPES]),
    )
    saver.setup()
    return saver
