"""§6.2 Router — decides whether retrieval is needed at all.

Rule: default to retrieval when confidence is low. A false "no retrieval
needed" is worse than an unnecessary retrieval, since the relevance grader
downstream can still discard bad context but skipping retrieval removes
that safety net entirely.
"""

from __future__ import annotations

from src.config import settings
from src.graph.state import PipelineState
from src.providers.base import get_provider

SYSTEM_PROMPT = """You are the routing component of an enterprise knowledge assistant.
Given a user question, decide whether it requires retrieving internal documents
(HR / procurement / travel / compliance policies) to answer accurately, or whether
it can be answered directly (e.g. it's a greeting, a meta question about the assistant,
or pure small talk with no factual content).

Default to needing retrieval whenever there is ANY chance the question touches
company policy, process, or documented facts. Only skip retrieval for clearly
conversational, non-factual input."""

SCHEMA_HINT = """Respond with JSON matching exactly:
{"needs_retrieval": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


def router_node(state: PipelineState) -> PipelineState:
    provider = get_provider(settings.provider_for("router"))
    result = provider.complete_json(
        system=SYSTEM_PROMPT,
        user=f"User question: {state['query']}",
        schema_hint=SCHEMA_HINT,
    )

    needs_retrieval = bool(result.get("needs_retrieval", True))
    confidence = float(result.get("confidence", 0.5))
    reasoning = str(result.get("reasoning", ""))

    # Safety rule: low-confidence "no retrieval" gets overridden to "retrieve anyway".
    if not needs_retrieval and confidence < 0.7:
        needs_retrieval = True
        reasoning += " [overridden: low-confidence no-retrieval decision defaulted to retrieval]"

    trace = state.get("trace", [])
    trace.append(
        f"[Router] needs_retrieval={needs_retrieval} confidence={confidence:.2f} — {reasoning}"
    )

    return {
        **state,
        "needs_retrieval": needs_retrieval,
        "router_confidence": confidence,
        "router_reasoning": reasoning,
        "trace": trace,
    }
