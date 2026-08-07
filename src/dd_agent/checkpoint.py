"""SQLite checkpointer construction with msgpack module registration."""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver


def make_sqlite_checkpointer(path: str) -> SqliteSaver:
    """Build a long-lived SqliteSaver, registering our schema types for msgpack."""
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.allowed_msgpack_modules = [
        ("dd_agent.schema", "Evidence"),
        ("dd_agent.schema", "ConfidenceBreakdown"),
    ]
    saver.setup()
    return saver
