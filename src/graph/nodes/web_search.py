"""Web search fallback (Tavily) when internal retrieval + rewriting still
can't find sufficient relevant evidence. Purely additive: results are tagged
as external so the conflict/authority resolution and citation logic can treat
them differently from internal, versioned policy documents.
"""

from __future__ import annotations

from src.config import settings
from src.graph.state import PipelineState
from src.retriever import Chunk


def web_search_node(state: PipelineState) -> PipelineState:
    trace = state.get("trace", [])

    if not settings.tavily_api_key:
        trace.append("[Web Search] skipped — no TAVILY_API_KEY configured")
        return {**state, "used_web_search": False, "web_search_results": [], "trace": trace}

    try:
        from tavily import TavilyClient
    except ImportError:
        trace.append("[Web Search] skipped — tavily-python not installed")
        return {**state, "used_web_search": False, "web_search_results": [], "trace": trace}

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=state["query"], max_results=3)
    results = response.get("results", [])

    web_chunks = [
        Chunk(
            chunk_id=f"web::{i}",
            text=r.get("content", ""),
            metadata={
                "doc_id": f"web-{i}",
                "title": r.get("title", "Web result"),
                "version": "",
                "effective_date": "",
                "scope": "external",
                "department": "",
                "source": "web",
                "url": r.get("url", ""),
            },
            relevant=True,  # web results are pre-filtered by Tavily relevance
        )
        for i, r in enumerate(results)
    ]

    existing_chunks = state.get("chunks", [])
    trace.append(f"[Web Search] fallback triggered — found {len(web_chunks)} external results")

    return {
        **state,
        "used_web_search": True,
        "web_search_results": results,
        "chunks": existing_chunks + web_chunks,
        "sufficiency_met": len(web_chunks) > 0,
        "trace": trace,
    }
