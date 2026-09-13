"""
Validates the LangGraph routing logic (conditional edges, ablation variants,
regeneration/rewrite caps) using a fully scripted fake LLM + fake retriever —
no network, no real embeddings, no API keys. This is NOT a replacement for
running the real pipeline; it only proves the graph wiring in build.py behaves
as designed for each branch (conflict/no-conflict, sufficient/insufficient,
faithful/unfaithful) before spending API credits on the real thing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import Chunk


class FakeProvider:
    """Returns canned JSON/text based on which node is calling it, inferred
    from a marker in the system prompt."""

    def __init__(self, script: dict):
        self.script = script  # {"router": {...}, "grader": [...], ...}
        self.grader_call_index = 0

    def complete_json(self, system: str, user: str, schema_hint: str = ""):
        if "routing component" in system:
            return self.script["router"]
        if "relevance grader" in system:
            resp = self.script["grader"][self.grader_call_index % len(self.script["grader"])]
            self.grader_call_index += 1
            return resp
        if "conflict-and-authority" in system:
            return self.script["conflict"]
        if "faithfulness grader" in system:
            return self.script["hallucination"]
        if "rewrite" in system.lower():
            return self.script.get("rewrite", {"rewritten_query": "rewritten", "reasoning": "test"})
        raise ValueError(f"FakeProvider: no script entry matches system prompt: {system[:80]}")

    def complete_text(self, system: str, user: str):
        from src.providers.base import CompletionResult
        return CompletionResult(text=self.script["generate"], model="fake", provider="fake")


def make_fake_chunks(n=4, relevant_flags=None):
    chunks = []
    for i in range(n):
        chunks.append(
            Chunk(
                chunk_id=f"c{i}",
                text=f"chunk text {i}",
                metadata={"doc_id": f"doc{i}", "title": f"Doc {i}", "version": "2025", "effective_date": "2025-01-01"},
            )
        )
    return chunks


def run_case(name: str, script: dict, variant: str, fake_chunks, expected_status: str):
    print(f"\n=== {name} (variant {variant}) ===")
    fake_provider = FakeProvider(script)

    with patch("src.graph.nodes.router.get_provider", return_value=fake_provider), \
         patch("src.graph.nodes.grader.get_provider", return_value=fake_provider), \
         patch("src.graph.nodes.conflict.get_provider", return_value=fake_provider), \
         patch("src.graph.nodes.generator.get_provider", return_value=fake_provider), \
         patch("src.graph.nodes.hallucination.get_provider", return_value=fake_provider), \
         patch("src.graph.nodes.rewrite.get_provider", return_value=fake_provider), \
         patch("src.graph.nodes.retrieve._get_retriever") as mock_get_retriever:

        mock_retriever = mock_get_retriever.return_value
        mock_retriever.search.return_value = fake_chunks

        from src.graph.build import build_graph
        graph = build_graph()

        initial_state = {
            "query": "test question",
            "original_query": "test question",
            "variant": variant,
            "rewrite_count": 0,
            "generation_count": 0,
            "regeneration_count": 0,
            "used_web_search": False,
            "trace": [],
        }
        final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    for line in final_state.get("trace", []):
        print(" ", line)
    status = final_state.get("evidence_status")
    result = "PASS" if status == expected_status else "FAIL"
    print(f"  -> evidence_status = {status}  [expected {expected_status}]  {result}")
    assert status == expected_status, f"{name}: expected {expected_status}, got {status}"
    return final_state


def main():
    all_relevant = [{"relevant": True, "reasoning": "matches"} for _ in range(4)]
    none_relevant = [{"relevant": False, "reasoning": "no match"} for _ in range(4)]

    # --- Case 1: clean VERIFIED path, full pipeline ---
    run_case(
        "Clean verified answer",
        {
            "router": {"needs_retrieval": True, "confidence": 0.9, "reasoning": "policy question"},
            "grader": all_relevant,
            "conflict": {"conflict_detected": False, "conflicts": [], "authoritative_doc_ids": []},
            "generate": "Per Doc 0, the answer is X.",
            "hallucination": {"claims": [{"text": "The answer is X", "supported": True, "reasoning": "matches doc0"}]},
        },
        variant="E",
        fake_chunks=make_fake_chunks(),
        expected_status="VERIFIED",
    )

    # --- Case 2: unresolved conflict -> CONFLICTING ---
    run_case(
        "Unresolved conflict",
        {
            "router": {"needs_retrieval": True, "confidence": 0.9, "reasoning": "policy question"},
            "grader": all_relevant,
            "conflict": {
                "conflict_detected": True,
                "conflicts": [
                    {"doc_ids": ["doc0", "doc1"], "description": "disagree on threshold", "resolved": False, "resolution": ""}
                ],
                "authoritative_doc_ids": [],
            },
            "generate": "Two policies disagree; please confirm which applies.",
            "hallucination": {"claims": [{"text": "policies disagree", "supported": True, "reasoning": "stated in context"}]},
        },
        variant="E",
        fake_chunks=make_fake_chunks(),
        expected_status="CONFLICTING",
    )

    # --- Case 3: no relevant chunks, no web search configured -> INSUFFICIENT EVIDENCE ---
    run_case(
        "No relevant evidence found (rewrite loop exhausted, no web search)",
        {
            "router": {"needs_retrieval": True, "confidence": 0.9, "reasoning": "policy question"},
            "grader": none_relevant,
            "rewrite": {"rewritten_query": "rephrased question", "reasoning": "try again"},
            "conflict": {"conflict_detected": False, "conflicts": [], "authoritative_doc_ids": []},
            "generate": "I could not find sufficient evidence to answer this question.",
            "hallucination": {"claims": []},
        },
        variant="E",
        fake_chunks=make_fake_chunks(),
        expected_status="INSUFFICIENT EVIDENCE",
    )

    # --- Case 4: hallucination fails repeatedly, no conflict -> PARTIALLY VERIFIED ---
    run_case(
        "Low faithfulness after regeneration cap, no conflict",
        {
            "router": {"needs_retrieval": True, "confidence": 0.9, "reasoning": "policy question"},
            "grader": all_relevant,
            "conflict": {"conflict_detected": False, "conflicts": [], "authoritative_doc_ids": []},
            "generate": "The answer is X, Y, and also Z (Z is made up).",
            "hallucination": {
                "claims": [
                    {"text": "The answer is X", "supported": True, "reasoning": "ok"},
                    {"text": "Z is true", "supported": False, "reasoning": "not in context"},
                ]
            },
        },
        variant="E",
        fake_chunks=make_fake_chunks(),
        expected_status="PARTIALLY VERIFIED",
    )

    # --- Case 5: Variant A (plain RAG) skips conflict resolution + hallucination check ---
    run_case(
        "Variant A — plain RAG, no safety nets",
        {
            "router": {"needs_retrieval": True, "confidence": 0.9, "reasoning": "policy question"},
            "grader": all_relevant,
            "generate": "Plain RAG answer with no verification.",
        },
        variant="A",
        fake_chunks=make_fake_chunks(),
        expected_status="VERIFIED",  # sufficiency_met defaults true from grading; no conflict/hallucination checks run
    )

    print("\nAll graph-logic tests passed.")


if __name__ == "__main__":
    main()
