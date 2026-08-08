# AI Due Diligence Agent - TDD Build Plan

Target: LangGraph prototype, parallel retrieval, rubric-based confidence gate, human-in-the-loop interrupt, SQLite checkpointing. Under 400 lines excluding tests.

Rule for every phase: write the failing test first, implement the minimum code to pass, refactor, commit. No phase starts until the prior phase's tests are green.

---

## Phase 0 - Project Scaffold

**Goal:** Repo skeleton, dependency lock, CI-runnable test command.

**Tasks**
- `pyproject.toml` with `langgraph`, `langchain-core`, `pytest`, `pytest-mock`, `httpx`, `pydantic`
- `src/dd_agent/` package, `tests/` mirror structure
- `.env.example` with `GITHUB_TOKEN`, `GROQ_API_KEY`
- `pytest.ini` with `testpaths = tests`

**Tests**
- `tests/test_scaffold.py`: import `dd_agent` package succeeds, package has `__version__`

**Exit criteria**
- `pytest` runs, 1 test passes, 0 collection errors

---

## Phase 1 - State and Evidence Schema

**Goal:** Typed state model with validation, no graph logic yet.

**Tasks**
- `src/dd_agent/schema.py`: `Evidence` (TypedDict or pydantic model), `AgentState`, `ConfidenceBreakdown`
- Fields: `Evidence.source_type: Literal["docs","github","community"]`, `url`, `snippet`, `relevance: float`
- `AgentState`: `question`, `sub_queries: dict[str,str]`, `evidence: list[Evidence]`, `confidence: float`, `confidence_breakdown: ConfidenceBreakdown`, `answer: str`

**Tests** (`tests/test_schema.py`)
- Constructing `Evidence` with valid fields succeeds
- `relevance` outside `[0,1]` raises `ValidationError`
- `source_type` outside the three literals raises
- `AgentState` with empty `evidence` list is valid (initial state)
- Serialization round-trip: `model_dump()` → `model_validate()` produces equal object

**Exit criteria**
- 100% branch coverage on `schema.py`, all tests green

---

## Phase 2 - Confidence Scorer (rubric, deterministic core)

**Goal:** Separate the deterministic weighted-sum math from the LLM sub-score assignment. Ship and test the math first, mock the LLM call second.

**Tasks**
- `src/dd_agent/confidence.py`
- `score_evidence_item(item: Evidence, llm) -> SubScores` - calls LLM once, returns structured output: `source_count_weight`, `agreement`, `recency`, `specificity`, each `0-1`
- `aggregate_confidence(sub_scores: list[SubScores], weights: dict[str,float]) -> float` - pure function, no LLM, weighted sum clipped to `[0,1]`
- Default weights: agreement 0.4, specificity 0.3, recency 0.2, source_count 0.1

**Tests** (`tests/test_confidence.py`)
- `aggregate_confidence` pure function:
  - all sub-scores 1.0 → aggregate 1.0
  - all sub-scores 0.0 → aggregate 0.0
  - known mixed input → known expected output (hand-computed, hardcoded assertion)
  - empty `sub_scores` list → returns 0.0, does not raise
  - weights that don't sum to 1.0 → raises `ValueError` at call time
- `score_evidence_item` with mocked LLM:
  - LLM returns valid structured JSON → parsed into `SubScores` correctly
  - LLM returns malformed JSON → raises a typed `ScoringError`, not a bare exception
  - LLM call is invoked exactly once per evidence item (mock assertion)

**Exit criteria**
- Aggregate function has zero LLM dependency in its test path
- Malformed-LLM-output case is explicitly covered

---

## Phase 3 - Retrieval Nodes (isolated, mocked transport)

**Goal:** Each retrieval node is a pure function of `(query, http_client) -> list[Evidence]`. No live network calls in tests.

### 3a. Docs RAG node
**Tasks**
- `src/dd_agent/nodes/docs_rag.py`
- Pre-built local chunk index (small JSON or SQLite FTS) built from a fixed list of doc URLs, built once via a separate `scripts/build_docs_index.py` (not part of the graph)
- `docs_rag_node(query: str, index) -> list[Evidence]`: top-k chunk retrieval, no live fetch during graph run

**Tests**
- Given a fixture index with 3 known chunks, querying a term present in chunk 2 returns chunk 2 first
- Query matching nothing returns empty list, not an error
- Returned `Evidence.source_type == "docs"` always

### 3b. GitHub node
**Tasks**
- `src/dd_agent/nodes/github_search.py`
- `github_search_node(query: str, client: httpx.Client) -> list[Evidence]` using GitHub REST search API
- Client is injected, never constructed inside the node

**Tests** (`pytest-mock` / `respx` to fake HTTP)
- Mocked 200 response with 2 issues → 2 `Evidence` items, `source_type == "github"`
- Mocked 403 rate-limit response → node returns empty list and logs a warning, does not raise
- Mocked timeout → node returns empty list, does not raise
- Malformed JSON body → node returns empty list, does not raise

### 3c. Community node (optional - build only if 3a/3b land with time to spare)
**Tasks**
- Same shape as GitHub node, Stack Exchange API, capped at N=5 results

**Tests**
- Same four cases as GitHub node

**Exit criteria for Phase 3**
- Every node tested for: happy path, empty result, transport failure, malformed response
- No test in this phase makes a real network call

---

## Phase 4 - Query Planner

**Goal:** Decompose one user question into per-source sub-queries.

**Tasks**
- `src/dd_agent/nodes/query_planner.py`
- `plan_queries(question: str, llm) -> dict[str, str]` returns keys `"docs"`, `"github"`, `"community"`

**Tests**
- Mocked LLM returns valid structured output → dict has exactly the three expected keys, all non-empty strings
- Mocked LLM returns a dict missing a key → raises typed `PlanningError`
- Mocked LLM returns empty string for a query → that key is dropped, remaining keys still returned (downstream nodes must handle a missing sub-query key)

**Exit criteria**
- Downstream nodes (Phase 3) already handle an absent/empty query without crashing - add a regression test confirming this integration point

---

## Phase 5 - Evidence Normalizer

**Goal:** Merge parallel node outputs into one deduplicated list.

**Tasks**
- `src/dd_agent/nodes/normalizer.py`
- `normalize(evidence_lists: list[list[Evidence]]) -> list[Evidence]`: flatten, dedupe by URL, drop items with empty snippet

**Tests**
- Two lists with an overlapping URL → deduped output has one entry for that URL
- Item with empty `snippet` is dropped
- Empty input lists → empty output, no error
- Order is stable: docs items precede github items precede community items (assert exact order given fixed input)

---

## Phase 6 - Graph Wiring (no interrupt yet)

**Goal:** Assemble Phases 1-5 into a LangGraph `StateGraph` with parallel fan-out and conditional routing, confidence gate wired to a stub "final answer" node on both branches.

**Tasks**
- `src/dd_agent/graph.py`
- Nodes: `query_planner`, `docs_rag`, `github_search`, `community` (optional), `normalizer`, `confidence_scorer`, `final_answer`
- Conditional edge on `state["confidence"] >= 0.8` → `final_answer` directly; else → placeholder `needs_review` node (stub, replaced in Phase 7)
- Compile without a checkpointer first

**Tests** (`tests/test_graph.py`, using a fully mocked LLM and mocked HTTP for every node)
- End-to-end run with mocked high-confidence path → final state has non-empty `answer`, `confidence >= 0.8`, graph never enters `needs_review`
- End-to-end run with mocked low-confidence path → graph enters `needs_review`, does not reach `final_answer` in this test
- Assert the three retrieval nodes ran in parallel, not sequentially: use call-order timestamps on mocks, or assert LangGraph's execution trace shows them in the same superstep
- Assert `AgentState` at the end validates against the Phase 1 schema

**Exit criteria**
- Full graph runs start-to-finish under mocks with zero network calls
- Both branches of the confidence gate are covered

---

## Phase 7 - Human-in-the-Loop Interrupt

**Goal:** Replace the `needs_review` stub with a real `interrupt()` call and `Command(resume=...)` flow.

**Tasks**
- Replace stub with `interrupt({"question": ..., "confidence": ..., "evidence_summary": ...})`
- Graph invocation wrapper in `src/dd_agent/runner.py`: `run(question) -> RunResult`, catches the interrupt, exposes `.pending_approval` and `.resume(approved: bool)`

**Tests**
- Low-confidence run: invoking the graph raises/returns a `GraphInterrupt` (per LangGraph's actual interrupt signal), captured by the runner as `pending_approval == True`
- `resume(approved=True)` continues to `final_answer` and returns a populated `answer`
- `resume(approved=False)` continues to a distinct terminal state (`answer` explicitly states no recommendation, does not fabricate one)
- Calling `resume()` when no interrupt is pending raises a clear `RuntimeError`, not a silent no-op

---

## Phase 8 - Checkpointing

**Goal:** Persist state across process restarts using `SqliteSaver`.

**Tasks**
- Wire `SqliteSaver.from_conn_string("checkpoints.db")` into graph compilation
- `runner.py`: every invocation carries a `thread_id`

**Tests**
- Start a run, hit the interrupt, simulate a process restart by re-instantiating the graph object against the same SQLite file, resume with the same `thread_id` → run completes correctly
- Resuming with a `thread_id` that was never started raises a clear error
- Two concurrent `thread_id`s do not leak state into each other (run both to completion, assert each final state only contains its own evidence)

---

## Phase 9 - CLI and Streaming Output

**Goal:** Terminal UX matching the demo transcript: live evidence arrival, approval prompt, final recommendation.

**Tasks**
- `src/dd_agent/cli.py` using the graph's `.stream()` API, print each node's output as it arrives
- Render `confidence_breakdown` alongside the aggregate score, not just the single number

**Tests**
- Given a mocked graph that yields 3 stream chunks then an interrupt, CLI output contains all 3 chunk markers in order, then the approval prompt string
- CLI exits non-zero on `resume()` `RuntimeError` (Phase 7 case), does not print a traceback to stdout

**Exit criteria for Phases 0-9**
- `pytest --cov=src/dd_agent` reports coverage on every module above; no module under 80%
- Manual run: `python -m dd_agent.cli "Should we integrate Stripe Connect?"` completes end-to-end against real APIs at least once, logged as a smoke test, not part of the automated suite

---

## Phase 10 - Stretch Goals (only after Phase 9 is fully green)

Pick in this order, stop when time runs out:

1. **Mermaid export**: `graph.get_graph().draw_mermaid()` written to `docs/graph.mmd`. Test: output contains all 7 node names as strings.
2. **Markdown citation export**: `export_answer_md(state) -> str`, each evidence URL rendered as a footnote. Test: every `Evidence.url` in state appears exactly once in output.
3. **Retry with backoff**: wrap the GitHub/community HTTP calls with 3 retries, exponential backoff. Test: mocked client fails twice then succeeds → node still returns correct evidence, exactly 3 calls made.
4. **Query cache**: SQLite table keyed on `hash(sub_query)`, checked before any live retrieval call. Test: second call with an identical sub-query makes zero HTTP calls, returns identical evidence to the first.

---

## Definition of Done

- All phases 0-9 complete, `pytest` green, coverage ≥ 80% overall
- One real (non-mocked) end-to-end smoke run recorded in `docs/smoke-run.md` with actual confidence score and evidence
- `docs/graph.mmd` mermaid diagram present
- README documents: setup, env vars, how to run, how to run tests, architecture diagram
