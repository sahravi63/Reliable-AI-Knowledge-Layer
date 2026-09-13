"""
§6.7 / §10 Weeks 11-12 — run all five variants (A-E) through RAGAS on the
identical eval set, broken out by question category, with intervention rates
and paired comparisons rather than a single blended average.

Run: python -m eval.run_ablation --variants A B C D E
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR
from src.graph.build import run_pipeline

RESULTS_PATH = Path(__file__).resolve().parent / "ablation_results.json"


def load_eval_questions() -> list[dict]:
    with open(DATA_DIR / "eval_questions.json") as f:
        return json.load(f)


def run_variant(variant: str, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        start = time.time()
        try:
            final_state = run_pipeline(q["question"], variant=variant)
            latency = time.time() - start
            rows.append(
                {
                    "question_id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "gold_answer": q["gold_answer"],
                    "gold_doc_ids": q["gold_doc_ids"],
                    "variant": variant,
                    "answer": final_state.get("answer"),
                    "evidence_status": final_state.get("evidence_status"),
                    "citations": final_state.get("citations"),
                    "conflict_detected": final_state.get("conflict_detected", False),
                    "conflict_resolved": final_state.get("conflict_resolved", True),
                    "faithfulness_ratio": final_state.get("faithfulness_ratio"),
                    "sufficiency_met": final_state.get("sufficiency_met"),
                    "rewrite_count": final_state.get("rewrite_count", 0),
                    "used_web_search": final_state.get("used_web_search", False),
                    "regeneration_count": final_state.get("regeneration_count", 0),
                    "latency_seconds": round(latency, 2),
                    "trace": final_state.get("trace", []),
                    "error": None,
                }
            )
        except Exception as e:  # noqa: BLE001 — log and continue; one bad question shouldn't kill the run
            rows.append(
                {
                    "question_id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "variant": variant,
                    "error": str(e),
                }
            )
        print(f"  [{variant}] {q['id']} ({q['category']}) done")
    return rows


def summarize(rows: list[dict]) -> dict:
    """Intervention rates + outcome breakdown by category and variant (§6.7)."""
    summary: dict = {}
    variants = sorted(set(r["variant"] for r in rows))
    for variant in variants:
        v_rows = [r for r in rows if r["variant"] == variant and not r.get("error")]
        by_category = {}
        for category in sorted(set(r["category"] for r in v_rows)):
            c_rows = [r for r in v_rows if r["category"] == category]
            n = len(c_rows)
            if n == 0:
                continue
            by_category[category] = {
                "n": n,
                "avg_faithfulness": round(
                    sum(r.get("faithfulness_ratio") or 0 for r in c_rows) / n, 3
                ),
                "conflict_detected_rate": round(
                    sum(1 for r in c_rows if r.get("conflict_detected")) / n, 3
                ),
                "conflict_resolved_rate": round(
                    sum(1 for r in c_rows if r.get("conflict_detected") and r.get("conflict_resolved")) / n, 3
                ),
                "insufficient_evidence_rate": round(
                    sum(1 for r in c_rows if r.get("evidence_status") == "INSUFFICIENT EVIDENCE") / n, 3
                ),
                "verified_rate": round(
                    sum(1 for r in c_rows if r.get("evidence_status") == "VERIFIED") / n, 3
                ),
                "avg_latency_seconds": round(sum(r.get("latency_seconds", 0) for r in c_rows) / n, 2),
                "rewrite_intervention_rate": round(
                    sum(1 for r in c_rows if r.get("rewrite_count", 0) > 0) / n, 3
                ),
                "web_search_intervention_rate": round(
                    sum(1 for r in c_rows if r.get("used_web_search")) / n, 3
                ),
            }
        summary[variant] = by_category
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["A", "B", "C", "D", "E"])
    args = parser.parse_args()

    questions = load_eval_questions()
    all_rows = []
    for variant in args.variants:
        print(f"Running variant {variant} ({len(questions)} questions)...")
        all_rows.extend(run_variant(variant, questions))

    summary = summarize(all_rows)

    with open(RESULTS_PATH, "w") as f:
        json.dump({"rows": all_rows, "summary": summary}, f, indent=2)

    print(f"\nResults written to {RESULTS_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
