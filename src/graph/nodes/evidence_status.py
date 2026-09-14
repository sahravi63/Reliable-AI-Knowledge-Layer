"""§6.6 Evidence Status — the actual product surface.

The output is NOT "answer + citations." It's an evidence status, derived
DETERMINISTICALLY from signals already computed elsewhere in the pipeline
(sufficiency §6.3, conflict resolution §6.4, claim support §6.5) — never a
new subjective judgment, and never a self-reported LLM confidence score.

| Sufficiency | Conflict resolution         | Claim support                  | -> Status             |
|-------------|------------------------------|---------------------------------|------------------------|
| Met         | No conflict                  | All/most claims supported       | VERIFIED               |
| Met         | No conflict                  | Some unsupported after cap      | PARTIALLY VERIFIED     |
| Met         | Conflict, unresolved          | Generation withheld/conditional | CONFLICTING            |
| Not met     | (after rewrite + web search)  | —                               | INSUFFICIENT EVIDENCE  |
| Met         | Resolved                     | Still below threshold after cap | UNVERIFIED             |
"""

from __future__ import annotations

from src.config import settings
from src.graph.state import PipelineState


def evidence_status_node(state: PipelineState) -> PipelineState:
    sufficiency_met = state.get("sufficiency_met", False)
    conflict_detected = state.get("conflict_detected", False)
    conflict_resolved = state.get("conflict_resolved", True)
    # Default to 1.0 ("no penalty"), not 0.0: ablation variants below E never
    # run the hallucination-check node at all, so an absent ratio means "not
    # graded," not "failed grading." Only a node that actually ran and scored
    # below threshold should push status away from VERIFIED.
    faithfulness_ratio = state.get("faithfulness_ratio", 1.0)
    regeneration_count = state.get("regeneration_count", 0)
    threshold = settings.faithfulness_threshold
    answer = str(state.get("answer", "")).lower()
    explicit_gap = any(
        phrase in answer
        for phrase in (
            "cannot be determined from context",
            "insufficient context",
            "not enough information",
            "unable to determine",
        )
    )

    if not sufficiency_met:
        status = "INSUFFICIENT EVIDENCE"
    elif conflict_detected and not conflict_resolved:
        status = "CONFLICTING"
    elif explicit_gap:
        status = "PARTIALLY VERIFIED"
    elif faithfulness_ratio >= threshold:
        status = "VERIFIED"
    elif regeneration_count >= settings.max_regenerations:
        # Distinguish "some claims unsupported, sufficiency/conflict were fine"
        # (PARTIALLY VERIFIED) from "conflict was resolved but generation still
        # can't stay faithful" (UNVERIFIED) using whether a conflict existed.
        status = "UNVERIFIED" if conflict_detected else "PARTIALLY VERIFIED"
    else:
        # Shouldn't normally reach here (graph should have regenerated), but
        # default conservatively.
        status = "PARTIALLY VERIFIED"

    citations = [
        {
            "doc_id": c.metadata.get("doc_id"),
            "title": c.metadata.get("title"),
            "version": c.metadata.get("version"),
            "effective_date": c.metadata.get("effective_date"),
        }
        for c in (state.get("resolved_context") or state.get("chunks", []))
        if getattr(c, "relevant", False)
    ]
    # de-duplicate citations by doc_id
    seen = set()
    deduped = []
    for c in citations:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            deduped.append(c)

    trace = state.get("trace", [])
    trace.append(f"[Evidence Status] -> {status}")

    return {**state, "evidence_status": status, "citations": deduped, "trace": trace}
