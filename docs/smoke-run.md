# Smoke Run Log

Real (non-mocked) end-to-end run against live APIs. Manual, not part of the
automated suite.

## How to run

1. Set credentials: `GROQ_API_KEY` (required), `GITHUB_TOKEN` (optional, raises rate limits).
2. Build the docs index: `python -m scripts.build_docs_index`
3. Run: `python -m dd_agent.cli "Should we integrate Stripe Connect?"`

## Procedure

- [ ] High-confidence question (expected: no approval prompt, final recommendation printed)
- [ ] Low-confidence question (expected: approval prompt; approve → recommendation; reject → "No recommendation")
- [ ] Re-run the same question (expected: second retrieval served from `query_cache.db`)

## Run log

| Date | Question | Confidence | Evidence count | Path (auto / approved / rejected) |
| ---- | -------- | ---------- | -------------- | --------------------------------- |
|      |          |            |                |                                   |

Note: no smoke run has been recorded yet — this environment has no
`GROQ_API_KEY`, so the end-to-end run against real APIs is still pending.
