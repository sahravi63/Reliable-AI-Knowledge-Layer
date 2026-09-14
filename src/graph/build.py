"""
Assembles the LangGraph state machine described in §6.1 / §6.7.

Each ablation variant enables a prefix of the full node set, so the marginal
contribution of each component can be isolated (§6.7, §8):

  A — Plain RAG:            retrieve -> generate
  B — + relevance grading:  retrieve -> grade -> generate (discard irrelevant, no retry)
  C — + correction loop:    retrieve -> grade -> [rewrite/web-search loop] -> generate
  D — + conflict resolution: ... -> conflict resolution -> generate
  E — + hallucination check (full pipeline): ... -> generate -> hallucination grade -> evidence status

variant is a plain string in state; the graph is a single fixed structure and
each node checks `state["variant"]` at its own conditional-edge decision points
rather than building five separate graphs, so intervention behavior stays
identical to the full pipeline it's ablating from.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.config import settings
from src.graph.nodes.conflict import conflict_resolution_node
from src.graph.nodes.evidence_status import evidence_status_node
from src.graph.nodes.generator import generator_node
from src.graph.nodes.grader import grade_chunks_node
from src.graph.nodes.hallucination import hallucination_grader_node
from src.graph.nodes.retrieve import retrieve_node
from src.graph.nodes.rewrite import rewrite_query_node
from src.graph.nodes.router import router_node
from src.graph.nodes.web_search import web_search_node
from src.graph.state import PipelineState

VARIANT_ORDER = ["A", "B", "C", "D", "E"]


def _variant_at_least(state: PipelineState, minimum: str) -> bool:
    variant = state.get("variant", "E")
    return VARIANT_ORDER.index(variant) >= VARIANT_ORDER.index(minimum)


# --- conditional edge functions -------------------------------------------------

def route_after_router(state: PipelineState) -> str:
    return "retrieve" if state.get("needs_retrieval", True) else "generate"


def route_after_grading(state: PipelineState) -> str:
    """Variant A/B still run grading, but its result never gates routing.
    From C onward: sufficiency met -> proceed; not met -> correction loop."""
    if state.get("sufficiency_met", False):
        return "proceed"

    if not _variant_at_least(state, "C"):
        # Variant B has no correction loop: proceed with whatever was graded relevant.
        return "proceed"

    relevant_count = sum(1 for c in state.get("chunks", []) if c.relevant)
    rewrite_count = state.get("rewrite_count", 0)

    if relevant_count == 0:
        return "web_search"
    if rewrite_count < settings.max_rewrite_attempts:
        return "rewrite"
    return "web_search"


def route_after_rewrite(_state: PipelineState) -> str:
    return "retrieve"


def route_after_web_search(_state: PipelineState) -> str:
    return "proceed"


def route_after_proceed(state: PipelineState) -> str:
    if _variant_at_least(state, "D"):
        return "conflict_resolution"
    return "generate"


def route_after_conflict(_state: PipelineState) -> str:
    return "generate"


def route_after_generate(state: PipelineState) -> str:
    if _variant_at_least(state, "E"):
        return "hallucination_check"
    return "evidence_status"


def route_after_hallucination(state: PipelineState) -> str:
    faithfulness_ratio = state.get("faithfulness_ratio", 1.0)
    regeneration_count = state.get("regeneration_count", 0)

    if faithfulness_ratio >= settings.faithfulness_threshold:
        return "evidence_status"

    if regeneration_count < settings.max_regenerations:
        return "regenerate"

    return "evidence_status"  # cap hit -> evidence_status derives UNVERIFIED/PARTIALLY VERIFIED


def bump_regeneration_count(state: PipelineState) -> PipelineState:
    return {**state, "regeneration_count": state.get("regeneration_count", 0) + 1}


# --- graph assembly -------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_chunks_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("conflict_resolution", conflict_resolution_node)
    graph.add_node("generate", generator_node)
    graph.add_node("hallucination_check", hallucination_grader_node)
    graph.add_node("bump_regeneration", bump_regeneration_count)
    graph.add_node("evidence_status", evidence_status_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", route_after_router, {"retrieve": "retrieve", "generate": "generate"})

    # Variant A retains grading for consistent chunk metadata, but A/B always proceed.
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grading,
        {"proceed": "proceed_marker", "rewrite": "rewrite", "web_search": "web_search"},
    )

    # tiny passthrough node so route_after_proceed has a clean place to branch from
    graph.add_node("proceed_marker", lambda s: s)
    graph.add_conditional_edges(
        "proceed_marker",
        route_after_proceed,
        {"conflict_resolution": "conflict_resolution", "generate": "generate"},
    )

    graph.add_conditional_edges("rewrite", route_after_rewrite, {"retrieve": "retrieve"})
    graph.add_conditional_edges("web_search", route_after_web_search, {"proceed": "proceed_marker"})
    graph.add_conditional_edges("conflict_resolution", route_after_conflict, {"generate": "generate"})

    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {"hallucination_check": "hallucination_check", "evidence_status": "evidence_status"},
    )
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {"regenerate": "bump_regeneration", "evidence_status": "evidence_status"},
    )
    graph.add_edge("bump_regeneration", "generate")
    graph.add_edge("evidence_status", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(query: str, variant: str = "E") -> PipelineState:
    graph = get_graph()
    initial_state: PipelineState = {
        "query": query,
        "original_query": query,
        "variant": variant,
        "rewrite_count": 0,
        "generation_count": 0,
        "regeneration_count": 0,
        "used_web_search": False,
        "trace": [f"[Pipeline] variant={variant} query={query!r}"],
    }
    final_state = graph.invoke(initial_state)
    return final_state
