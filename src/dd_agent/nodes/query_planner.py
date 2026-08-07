"""Query planner: decompose one question into per-source sub-queries."""

import json

from pydantic import BaseModel, ValidationError


class PlanningError(RuntimeError):
    """Raised when the planner output is missing required keys."""


class Plan(BaseModel):
    docs: str
    github: str
    community: str


def plan_queries(question: str, llm) -> dict[str, str]:
    """Return a dict keyed by source name; empty sub-queries are dropped."""
    prompt = (
        "Decompose the following question into three targeted search queries, one per source. "
        'Respond with JSON: {"docs": "...", "github": "...", "community": "..."}.\n'
        f"question: {question}\n"
    )
    raw = llm.invoke(prompt)
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        plan = Plan.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise PlanningError(f"could not parse plan: {e}") from e
    return {k: v for k, v in plan.model_dump().items() if v}
