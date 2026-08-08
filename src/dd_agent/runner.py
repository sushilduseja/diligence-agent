"""Graph invocation wrapper: catches the human-approval interrupt, exposes resume."""

from uuid import uuid4

from langgraph.types import Command
from pydantic import ValidationError

from dd_agent.schema import AgentState


class RunResult:
    def __init__(self, graph, thread_id, question, final_state=None, interrupt_payload=None):
        self._graph = graph
        self._thread_id = thread_id
        self._final_state: dict | None = final_state
        self._interrupt_payload = interrupt_payload
        self.question = question

    @property
    def pending_approval(self) -> bool:
        return self._interrupt_payload is not None

    @property
    def confidence(self) -> float | None:
        if self._interrupt_payload is not None:
            return self._interrupt_payload.confidence
        if self._final_state is None:
            return None
        return self._final_state.get("confidence", 0.0)

    @property
    def answer(self) -> str:
        if self._final_state is None:
            return ""
        return self._final_state.get("answer", "")

    @property
    def evidence_summary(self) -> list:
        if self._interrupt_payload is not None:
            return self._interrupt_payload.evidence_summary
        return []

    def resume(self, approved: bool) -> "RunResult":
        if not self.pending_approval:
            raise RuntimeError("no interrupt pending; nothing to resume")
        config = {"configurable": {"thread_id": self._thread_id}}
        try:
            state = self._graph.invoke(Command(resume=approved), config)
        except ValidationError as e:
            raise RuntimeError(
                f"no checkpoint exists for thread {self._thread_id!r}; start a new run first"
            ) from e
        self._interrupt_payload = None
        self._final_state = state
        return self


def run(graph, question: str, thread_id: str | None = None) -> RunResult:
    tid = thread_id or str(uuid4())
    config = {"configurable": {"thread_id": tid}}
    state = graph.invoke({"question": question}, config)
    interrupt = state.get("__interrupt__")
    if interrupt:
        payload = interrupt[0].value
        return RunResult(graph, tid, question, interrupt_payload=payload)
    return RunResult(graph, tid, question, final_state=state)
