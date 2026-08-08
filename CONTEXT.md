# CONTEXT - dd-agent domain model

## Goal

Answer integration due-diligence questions (e.g. "Should we integrate Stripe
Connect?") by retrieving evidence in parallel from three sources, scoring
confidence with a rubric, and pausing for human approval when confidence is low.

## Domain terms

- **question** - the integration/technology question under diligence.
- **sub-query** - a per-source query string produced by the planner from the
  question. A source with an absent/empty sub-query returns no evidence.
- **evidence** - a single retrieval result (`Evidence`): source_type, url,
  snippet, relevance. **relevance** is currently always 1.0 and unconsumed.
- **retrieval source** - one of `docs` (local index), `github`, `community`
  (both HTTP). The two HTTP sources share one retrieval module and a
  source-scoped query cache.
- **confidence** - a 0-1 rubric score; `aggregate` is the weighted sum over
  the four axes (source_count_weight, agreement, recency, specificity).
- **breakdown** - the per-axis mean sub-scores plus the aggregate.
- **review request** - the typed payload (`ReviewRequest`) sent at the approval
  interrupt: question, confidence, evidence summary. Emitted by the graph,
  read by the runner.
- **approval** - the human decision at the interrupt; `approved` routes to
  final answer, else an explicit "no recommendation".

## Flow

`question` → planner → parallel retrieval (docs, github, community) →
normalizer (dedupe/order) → scorer (sub-scores → breakdown) → gate:
confidence ≥ 0.8 → final answer; else review request interrupt → approval →
final answer or rejected.

## Module map

- `compose.py` - composition root: env, paths (repo-root anchored), adapters.
- `graph.py` - LangGraph wiring: fan-out, gate, interrupt routing.
- `confidence.py` - deep scoring module: `score(items, llm) -> ConfidenceBreakdown`.
- `nodes/retrieval.py` - shared HTTP retrieval behind thin per-source wrappers.
- `runner.py` - run/resume wrapper around the graph.
- `checkpoint.py`, `cache.py` - SQLite persistence.
- `cli.py` - streaming terminal renderer only.
