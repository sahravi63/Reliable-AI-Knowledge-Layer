"""§6.3 Relevance Grader — grades each retrieved chunk independently.

Binary relevant/irrelevant is used instead of a continuous score: it is more
reproducible across runs and across graders (the approach used in the
Corrective RAG / CRAG paper). Sufficiency is an explicit, reported threshold
(settings.relevance_sufficiency_threshold), not an implicit judgment call.
"""

from __future__ import annotations

from src.config import settings
from src.graph.state import PipelineState
from src.providers.base import get_provider

SYSTEM_PROMPT = """You are a strict relevance grader for a retrieval-augmented
question-answering system. Given a user question and a single retrieved
document chunk, decide whether the chunk contains information that is directly
relevant to answering the question. Be strict: a chunk that only shares
keywords but doesn't address the question's actual subject is NOT relevant."""

SCHEMA_HINT = """Respond with JSON matching exactly:
{"relevant": true/false, "reasoning": "one sentence"}"""


def grade_chunks_node(state: PipelineState) -> PipelineState:
    provider = get_provider(settings.provider_for("grader"))
    chunks = state.get("chunks", [])

    graded = []
    for chunk in chunks:
        result = provider.complete_json(
            system=SYSTEM_PROMPT,
            user=(
                f"Question: {state['query']}\n\n"
                f"Chunk (from '{chunk.metadata.get('title', chunk.metadata.get('doc_id'))}'):\n"
                f"{chunk.text}"
            ),
            schema_hint=SCHEMA_HINT,
        )
        chunk.relevant = bool(result.get("relevant", False))
        chunk.grader_reasoning = str(result.get("reasoning", ""))
        graded.append(chunk)

    relevant_count = sum(1 for c in graded if c.relevant)
    sufficiency_met = relevant_count >= settings.relevance_sufficiency_threshold

    trace = state.get("trace", [])
    trace.append(
        f"[Relevance Grader] {relevant_count}/{len(graded)} chunks relevant "
        f"(threshold={settings.relevance_sufficiency_threshold}) -> sufficiency_met={sufficiency_met}"
    )

    return {**state, "chunks": graded, "sufficiency_met": sufficiency_met, "trace": trace}
