"""Answer rendering: mermaid diagram export and markdown citation export."""

from dd_agent.schema import AgentState


def export_mermaid(graph) -> str:
    """Return the graph's mermaid representation."""
    return graph.get_graph().draw_mermaid()


def export_answer_md(state: AgentState) -> str:
    """Render the answer with numbered footnote citations; every URL appears once."""
    lines = [state.answer, ""]
    for i, item in enumerate(state.evidence, start=1):
        lines.append(f"[{i}]: {item.url}")
    return "\n".join(lines)
