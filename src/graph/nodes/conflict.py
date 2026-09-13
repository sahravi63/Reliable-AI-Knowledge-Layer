"""§6.4 Conflict & Authority Resolution — the node the procurement example
exposed, and the core differentiator of the "reliability layer" pitch.

Binary relevance isn't enough: two chunks can both be relevant and still
disagree (a 2024 vs. 2025 policy). This node:
  1. Groups relevant chunks by topic/claim overlap.
  2. Checks for contradiction using available metadata (effective date,
     version number, department/scope tags).
  3. If metadata resolves the conflict (e.g. 2025 supersedes 2024): use the
     authoritative document, and note the resolution in the citations.
  4. If metadata CANNOT resolve it: don't silently pick one — surface the
     conflict explicitly to the user, with both cited.

This is deliberately NOT allowed to silently guess. Failure mode to test for
explicitly (§11): "silent conflict-picking" — the whole pitch collapses if
this node guesses instead of resolving-or-surfacing.
"""

from __future__ import annotations

from src.config import settings
from src.graph.state import Conflict, PipelineState
from src.providers.base import get_provider
from src.retriever import Chunk

SYSTEM_PROMPT = """You are the conflict-and-authority resolution component of an
enterprise knowledge assistant. You will be given a user question and a list of
relevant document chunks, each with metadata (document title, version,
effective_date, scope, department, supersedes/superseded_by fields).

Your job:
1. Determine whether any of the chunks disagree with each other on a
   fact relevant to the question (e.g. different numeric thresholds, different
   procedures, contradictory rules).
2. For each disagreement found, decide whether the METADATA resolves it:
   - A later effective_date / higher version number that explicitly
     supersedes another document resolves a general-vs-general conflict.
   - A department-specific document does NOT automatically override a
     general company-wide document unless the department document's text
     explicitly states it overrides the general policy for its scope.
   - If two documents could both plausibly apply (e.g. ambiguous department
     scope, or a department doc that doesn't explicitly claim override
     authority against a newer general policy), the conflict is NOT resolved
     by metadata alone.
3. Never resolve a conflict by picking the answer that "seems more reasonable"
   if the metadata does not clearly support it. When in doubt, mark unresolved.

Be conservative: it is much worse to silently pick a side than to correctly
report that human confirmation is needed."""

SCHEMA_HINT = """Respond with JSON matching exactly:
{
  "conflict_detected": true/false,
  "conflicts": [
    {
      "doc_ids": ["doc_id_1", "doc_id_2"],
      "description": "what they disagree about",
      "resolved": true/false,
      "resolution": "which doc_id is authoritative and why, OR empty string if unresolved"
    }
  ],
  "authoritative_doc_ids": ["doc_id of chunks that should be used for generation"]
}"""


def _format_chunk(chunk: Chunk) -> str:
    m = chunk.metadata
    return (
        f"[doc_id={m.get('doc_id')} | title={m.get('title')} | version={m.get('version')} | "
        f"effective_date={m.get('effective_date')} | scope={m.get('scope')} | "
        f"department={m.get('department') or 'none'} | supersedes={m.get('supersedes') or 'none'} | "
        f"superseded_by={m.get('superseded_by') or 'none'}]\n{chunk.text}"
    )


def conflict_resolution_node(state: PipelineState) -> PipelineState:
    relevant_chunks = [c for c in state.get("chunks", []) if c.relevant]

    if len(relevant_chunks) < 2:
        # Nothing to conflict with.
        trace = state.get("trace", [])
        trace.append("[Conflict Resolution] fewer than 2 relevant chunks — skipping conflict check")
        return {
            **state,
            "conflict_detected": False,
            "conflicts": [],
            "conflict_resolved": True,
            "resolved_context": relevant_chunks,
            "trace": trace,
        }

    provider = get_provider(settings.provider_for("conflict"))
    chunk_blob = "\n\n---\n\n".join(_format_chunk(c) for c in relevant_chunks)

    result = provider.complete_json(
        system=SYSTEM_PROMPT,
        user=f"Question: {state['query']}\n\nRelevant chunks:\n\n{chunk_blob}",
        schema_hint=SCHEMA_HINT,
    )

    conflict_detected = bool(result.get("conflict_detected", False))
    raw_conflicts = result.get("conflicts", [])
    conflicts: list[Conflict] = [
        Conflict(
            doc_ids=c.get("doc_ids", []),
            description=c.get("description", ""),
            resolved=bool(c.get("resolved", False)),
            resolution=c.get("resolution", ""),
        )
        for c in raw_conflicts
    ]

    all_resolved = all(c["resolved"] for c in conflicts) if conflicts else True
    authoritative_ids = set(result.get("authoritative_doc_ids", []))

    if not conflict_detected:
        resolved_context = relevant_chunks
    elif all_resolved and authoritative_ids:
        resolved_context = [c for c in relevant_chunks if c.metadata.get("doc_id") in authoritative_ids]
        if not resolved_context:  # safety net if the model's ids didn't match
            resolved_context = relevant_chunks
    else:
        # Unresolved conflict: withhold a confident generation. The generator
        # node will see conflict_resolved=False and produce a conflict-surfacing
        # response instead of a normal answer.
        resolved_context = relevant_chunks

    trace = state.get("trace", [])
    if conflict_detected:
        status = "resolved" if all_resolved else "UNRESOLVED — surfacing to user"
        trace.append(f"[Conflict Resolution] conflict detected ({len(conflicts)} found) — {status}")
        for c in conflicts:
            trace.append(f"    - {c['doc_ids']}: {c['description']} (resolved={c['resolved']})")
    else:
        trace.append("[Conflict Resolution] no conflict detected among relevant chunks")

    return {
        **state,
        "conflict_detected": conflict_detected,
        "conflicts": conflicts,
        "conflict_resolved": all_resolved,
        "resolved_context": resolved_context,
        "trace": trace,
    }
