from __future__ import annotations

from typing import Optional, TypedDict

from src.retriever import Chunk


class Claim(TypedDict):
    text: str
    supported: bool
    reasoning: str


class Conflict(TypedDict):
    doc_ids: list[str]
    description: str
    resolved: bool
    resolution: str          # how it was resolved, or "" if unresolved


class PipelineState(TypedDict, total=False):
    # --- input ---
    query: str
    original_query: str

    # --- variant switches (for the ablation ladder, §6.7 / §8) ---
    variant: str              # "A" | "B" | "C" | "D" | "E"

    # --- router ---
    needs_retrieval: bool
    router_confidence: float
    router_reasoning: str

    # --- retrieval / correction loop ---
    chunks: list[Chunk]
    rewrite_count: int
    used_web_search: bool
    web_search_results: list[dict]
    sufficiency_met: bool

    # --- conflict & authority resolution ---
    conflicts: list[Conflict]
    conflict_detected: bool
    conflict_resolved: bool
    resolved_context: list[Chunk]     # chunks to actually generate from

    # --- generation ---
    answer: Optional[str]
    generation_count: int

    # --- hallucination / groundedness ---
    claims: list[Claim]
    claims_checked: bool
    faithfulness_ratio: float
    regeneration_count: int

    # --- final output (§6.6) ---
    evidence_status: str       # VERIFIED | PARTIALLY VERIFIED | CONFLICTING | INSUFFICIENT EVIDENCE | UNVERIFIED
    citations: list[dict]
    trace: list[str]           # human-readable reasoning trace for the UI
