"""§6 correction loop — query rewriting when retrieval is insufficient.

Three-way retrieval-outcome routing (see build.py for the conditional edges):
  - zero relevant chunks       -> go straight to web search
  - partial (some but not enough) -> rewrite query, re-retrieve (max N attempts)
    -> after N attempts, fall through to web search too
"""

from __future__ import annotations

from src.config import settings
from src.graph.state import PipelineState
from src.providers.base import get_provider

SYSTEM_PROMPT = """You rewrite user questions to improve retrieval recall against
an internal policy document corpus. The original query didn't retrieve enough
relevant evidence. Produce a reformulated query that surfaces different phrasing,
synonyms, or a more specific/general framing of the same underlying question.
Do not change what is being asked — only how it is phrased for search."""

SCHEMA_HINT = """Respond with JSON matching exactly:
{"rewritten_query": "...", "reasoning": "one sentence"}"""


def rewrite_query_node(state: PipelineState) -> PipelineState:
    provider = get_provider(settings.provider_for("router"))
    result = provider.complete_json(
        system=SYSTEM_PROMPT,
        user=(
            f"Original question: {state.get('original_query', state['query'])}\n"
            f"Most recent query tried: {state['query']}\n"
            f"Attempt number: {state.get('rewrite_count', 0) + 1} of {settings.max_rewrite_attempts}"
        ),
        schema_hint=SCHEMA_HINT,
    )

    rewritten = str(result.get("rewritten_query", state["query"]))
    rewrite_count = state.get("rewrite_count", 0) + 1

    trace = state.get("trace", [])
    trace.append(f"[Query Rewriter] attempt {rewrite_count}: {rewritten!r}")

    return {
        **state,
        "query": rewritten,
        "rewrite_count": rewrite_count,
        "trace": trace,
    }
