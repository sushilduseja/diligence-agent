"""Composition root: assemble the graph from environment, files, and adapters."""

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from dd_agent.cache import QueryCache
from dd_agent.checkpoint import make_sqlite_checkpointer
from dd_agent.graph import build_graph
from dd_agent.llm import GroqLLM

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


def build_default_graph(index_path=None):
    """Assemble the graph with real dependencies from the environment."""
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set")
    if index_path is None:
        index_path = DATA_DIR / "docs_index.json"
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    client = httpx.Client(timeout=30, headers=headers, follow_redirects=True)
    checkpointer = make_sqlite_checkpointer(str(REPO_ROOT / "checkpoints.db"))
    return build_graph(
        GroqLLM(api_key), client, index, checkpointer, QueryCache.open(str(REPO_ROOT / "query_cache.db"))
    )
