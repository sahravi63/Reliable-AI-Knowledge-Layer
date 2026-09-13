from __future__ import annotations

from src.config import settings
from src.graph.state import PipelineState
from src.providers.base import get_provider

NORMAL_SYSTEM_PROMPT = """You are an enterprise policy assistant. Answer the
user's question using ONLY the provided context chunks. Cite which document
each part of your answer comes from by title. Do not use outside knowledge.
If the context is insufficient to fully answer, say so explicitly rather than
filling gaps with assumptions."""

CONFLICT_SYSTEM_PROMPT = """You are an enterprise policy assistant. The context
below contains a CONFLICT between two or more documents that could not be
automatically resolved. Do NOT pick a side or give a single confident number.
Instead: (1) briefly state that applicable policies disagree, (2) show what
each document says, with its title/version, (3) explain what the user should
check or confirm (e.g. their department, the purchase category) to determine
which applies, (4) recommend they confirm with the relevant approver if still
unclear."""

STRICT_REGENERATE_SUFFIX = """

IMPORTANT: A previous answer attempt included claims not clearly supported by
the context. Be more conservative this time — only state what is explicitly
present in the provided chunks. Do not infer, extrapolate, or generalize
beyond the text."""


def _format_context(chunks) -> str:
    parts = []
    for c in chunks:
        m = c.metadata
        label = f"{m.get('title')} ({m.get('version') or 'undated'})"
        parts.append(f"[{label}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generator_node(state: PipelineState) -> PipelineState:
    provider = get_provider(settings.provider_for("generator"))
    context_chunks = state.get("resolved_context") or state.get("chunks", [])
    context_blob = _format_context(context_chunks)

    is_conflict = state.get("conflict_detected", False) and not state.get("conflict_resolved", True)
    system_prompt = CONFLICT_SYSTEM_PROMPT if is_conflict else NORMAL_SYSTEM_PROMPT

    if state.get("regeneration_count", 0) > 0:
        system_prompt += STRICT_REGENERATE_SUFFIX

    result = provider.complete_text(
        system=system_prompt,
        user=f"Question: {state.get('original_query', state['query'])}\n\nContext:\n{context_blob}",
    )

    generation_count = state.get("generation_count", 0) + 1
    trace = state.get("trace", [])
    trace.append(
        f"[Generator] produced answer (attempt {generation_count}, conflict_mode={is_conflict})"
    )

    return {**state, "answer": result.text, "generation_count": generation_count, "trace": trace}
