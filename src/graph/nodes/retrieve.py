from __future__ import annotations

from src.graph.state import PipelineState
from src.retriever import Retriever

_retriever = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve_node(state: PipelineState) -> PipelineState:
    retriever = _get_retriever()
    chunks = retriever.search(state["query"])

    trace = state.get("trace", [])
    trace.append(f"[Retriever] fetched {len(chunks)} chunks for query: {state['query']!r}")

    return {**state, "chunks": chunks, "trace": trace}
