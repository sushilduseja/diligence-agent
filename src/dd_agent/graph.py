"""Assemble Phases 1-5 into a LangGraph StateGraph: parallel fan-out, confidence gate, human approval."""

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from dd_agent.confidence import score
from dd_agent.nodes.community import community_node
from dd_agent.nodes.docs_rag import docs_rag_node
from dd_agent.nodes.github_search import github_search_node
from dd_agent.nodes.normalizer import normalize
from dd_agent.nodes.query_planner import plan_queries
from dd_agent.schema import AgentState, ReviewRequest

CONFIDENCE_THRESHOLD = 0.8


def build_graph(llm, http_client, docs_index, checkpointer=None, query_cache=None):
    """Compile the graph. llm, http_client, docs_index are injected, never constructed here."""

    def query_planner(state):
        return {"sub_queries": plan_queries(state.question, llm)}

    def docs_rag(state):
        return {
            "docs_evidence": docs_rag_node(state.sub_queries.get("docs", ""), docs_index)
        }

    def github_search(state):
        return {
            "github_evidence": github_search_node(
                state.sub_queries.get("github", ""), http_client, cache=query_cache
            )
        }

    def community(state):
        return {
            "community_evidence": community_node(
                state.sub_queries.get("community", ""), http_client, cache=query_cache
            )
        }

    def normalizer(state):
        return {
            "evidence": normalize(
                [
                    state.docs_evidence,
                    state.github_evidence,
                    state.community_evidence,
                ]
            )
        }

    def confidence_scorer(state):
        breakdown = score(state.evidence, llm)
        return {"confidence": breakdown.aggregate, "confidence_breakdown": breakdown}

    def final_answer(state):
        urls = ", ".join(e.url for e in state.evidence)
        return {
            "answer": (
                f"Recommendation based on {len(state.evidence)} sources with "
                f"confidence {state.confidence:.2f}. Evidence: {urls}"
            )
        }

    def needs_review(state):
        decision = interrupt(
            ReviewRequest(
                question=state.question,
                confidence=state.confidence,
                evidence_summary=state.evidence,
            )
        )
        return {"approved": decision}

    def rejected(state):
        return {
            "answer": (
                "No recommendation: the evidence was rejected or insufficient. "
                "Should not proceed with the integration without further review."
            )
        }

    def route_after_score(state):
        if state.confidence >= CONFIDENCE_THRESHOLD:
            return "final_answer"
        return "needs_review"

    def route_after_approval(state):
        if state.approved:
            return "final_answer"
        return "rejected"

    g = StateGraph(AgentState)
    g.add_node("query_planner", query_planner)
    g.add_node("docs_rag", docs_rag)
    g.add_node("github_search", github_search)
    g.add_node("community", community)
    g.add_node("normalizer", normalizer)
    g.add_node("confidence_scorer", confidence_scorer)
    g.add_node("final_answer", final_answer)
    g.add_node("needs_review", needs_review)
    g.add_node("rejected", rejected)

    g.add_edge(START, "query_planner")
    g.add_edge("query_planner", "docs_rag")
    g.add_edge("query_planner", "github_search")
    g.add_edge("query_planner", "community")
    g.add_edge("docs_rag", "normalizer")
    g.add_edge("github_search", "normalizer")
    g.add_edge("community", "normalizer")
    g.add_edge("normalizer", "confidence_scorer")
    g.add_conditional_edges(
        "confidence_scorer",
        route_after_score,
        {"final_answer": "final_answer", "needs_review": "needs_review"},
    )
    g.add_conditional_edges(
        "needs_review",
        route_after_approval,
        {"final_answer": "final_answer", "rejected": "rejected"},
    )
    g.add_edge("final_answer", END)
    g.add_edge("rejected", END)

    return g.compile(checkpointer=checkpointer)
