"""Terminal UX: live evidence arrival, approval prompt, final recommendation."""

import argparse
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

from dd_agent.schema import Evidence


def _evidence_urls(value):
    urls = []
    for item in value:
        if isinstance(item, dict):
            urls.append(item.get("url", ""))
        else:
            urls.append(getattr(item, "url", ""))
    return urls


def run_cli(graph, question, *, out=print, err=print, prompt_input=input) -> int:
    """Stream one run to the terminal, pausing for approval when interrupted."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    interrupted = False
    for chunk in graph.stream({"question": question}, config):
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupted = True
                continue
            out(f"[{node}]")
            if not isinstance(update, dict):
                continue
            for key, value in update.items():
                if isinstance(value, list) and value:
                    for url in _evidence_urls(value):
                        out(f"  evidence: {url}")
                elif key == "confidence_breakdown" and value is not None:
                    parts = value.model_dump() if hasattr(value, "model_dump") else value
                    if isinstance(parts, dict):
                        rendered = "  ".join(f"{k}={v:.2f}" for k, v in parts.items())
                    else:
                        rendered = str(parts)
                    out(f"  breakdown: {rendered}")
                elif key == "confidence":
                    out(f"  confidence: {value:.2f}")
                elif key == "answer" and value:
                    out(f"Recommendation: {value}")
    if interrupted:
        decision = prompt_input("Human review required: approve this recommendation? [y/N] ")
        approved = decision.strip().lower() in ("y", "yes")
        try:
            final = graph.invoke(Command(resume=approved), config)
        except RuntimeError as e:
            err(f"error: {e}")
            return 1
        out(f"Recommendation: {final.get('answer', '')}")
    return 0


def build_default_graph(index_path=None):
    """Assemble the graph with real dependencies from the environment."""
    import json
    import os

    import httpx

    from dotenv import load_dotenv

    from dd_agent.cache import QueryCache
    from dd_agent.checkpoint import make_sqlite_checkpointer
    from dd_agent.graph import build_graph
    from dd_agent.llm import GroqLLM

    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set")
    if index_path is None:
        index_path = Path(__file__).resolve().parent.parent.parent / "data" / "docs_index.json"
    index_path = Path(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    client = httpx.Client(timeout=30, headers=headers, follow_redirects=True)
    checkpointer = make_sqlite_checkpointer("checkpoints.db")
    return build_graph(GroqLLM(api_key), client, index, checkpointer, QueryCache.open("query_cache.db"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dd_agent", description="AI due diligence agent")
    parser.add_argument("question", help="the integration or technology question to research")
    args = parser.parse_args(argv)
    graph = build_default_graph()
    return run_cli(graph, args.question)


if __name__ == "__main__":
    sys.exit(main())
