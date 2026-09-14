"""§6.5 Hallucination Grader — claim-level, not answer-level.

Decompose the generated answer into atomic claims, check each against context.
  - >= threshold: return as-is.
  - below threshold, context was well-graded: generation failure -> regenerate
    with a stricter prompt.
  - still fails after 1-2 regenerations: likely a retrieval failure the grader
    missed -> escalate back to query-rewrite (handled by the graph's edges).
  - cap regenerations -> return a flagged "low confidence, unverified" answer
    rather than silently shipping an ungrounded one.
"""

from __future__ import annotations

from src.config import settings
from src.graph.state import Claim, PipelineState
from src.providers.base import get_provider

SYSTEM_PROMPT = """You are a strict faithfulness grader. You will be given a
generated answer and the context it was supposed to be grounded in. Decompose
the answer into atomic, independently-checkable claims (each a single fact or
assertion). For each claim, decide whether it is directly supported by the
context — "supported" means the claim can be verified from the context text
without inference beyond what's explicitly stated. Be strict: vague hedges
("you should check with your manager") count as supported if the context
itself recommends that; specific numbers or rules not present in the context
are NOT supported."""

SCHEMA_HINT = """Respond with JSON matching exactly:
{"claims": [{"text": "...", "supported": true/false, "reasoning": "..."}]}"""


def hallucination_grader_node(state: PipelineState) -> PipelineState:
    provider = get_provider(settings.provider_for("hallucination"))
    context_chunks = state.get("resolved_context") or state.get("chunks", [])
    context_blob = "\n\n".join(c.text for c in context_chunks)

    result = provider.complete_json(
        system=SYSTEM_PROMPT,
        user=f"Context:\n{context_blob}\n\nGenerated answer:\n{state.get('answer', '')}",
        schema_hint=SCHEMA_HINT,
    )

    raw_claims = result.get("claims", [])
    claims: list[Claim] = [
        Claim(
            text=c.get("text", ""),
            supported=bool(c.get("supported", False)),
            reasoning=c.get("reasoning", ""),
        )
        for c in raw_claims
    ]

    answer = str(state.get("answer", "")).strip()
    insufficient_answer = (
        len(answer) <= 160
        and any(
            phrase in answer.lower()
            for phrase in ("insufficient evidence", "cannot determine", "not enough information")
        )
    )

    if claims:
        faithfulness_ratio = sum(1 for c in claims if c["supported"]) / len(claims)
    elif insufficient_answer:
        faithfulness_ratio = 1.0
    else:
        faithfulness_ratio = 0.0

    trace = state.get("trace", [])
    trace.append(
        f"[Hallucination Grader] {sum(1 for c in claims if c['supported'])}/{len(claims)} "
        f"claims supported -> faithfulness_ratio={faithfulness_ratio:.2f}"
    )

    return {
        **state,
        "claims": claims,
        "claims_checked": True,
        "faithfulness_ratio": faithfulness_ratio,
        "trace": trace,
    }
