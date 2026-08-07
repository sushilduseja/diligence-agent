"""Build the local docs chunk index from a fixed list of doc URLs.

Run once (not part of the graph): python -m scripts.build_docs_index
Writes data/docs_index.json consumed by the docs RAG node.
"""

import json
import re
from pathlib import Path

import httpx

DOC_URLS = [
    "https://docs.stripe.com/connect",
    "https://docs.stripe.com/payouts",
    "https://docs.stripe.com/webhooks",
]

OUT = Path(__file__).resolve().parent.parent / "data" / "docs_index.json"


def fetch_and_chunk(url: str, client: httpx.Client) -> list[dict]:
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = re.sub(r"\s+", " ", text).strip()
    step = 200
    chunks = [text[i : i + step] for i in range(0, len(text), step) if text[i : i + step]]
    return [{"url": url, "chunk": c} for c in chunks]


def main() -> None:
    with httpx.Client(follow_redirects=True) as client:
        entries = []
        for url in DOC_URLS:
            try:
                entries.extend(fetch_and_chunk(url, client))
            except httpx.HTTPError as e:
                print(f"warning: could not fetch {url}: {e}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(entries), encoding="utf-8")
    print(f"wrote {len(entries)} chunks to {OUT}")


if __name__ == "__main__":
    main()
