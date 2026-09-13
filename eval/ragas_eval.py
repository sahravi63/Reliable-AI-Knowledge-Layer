"""
Optional deeper RAGAS scoring pass over eval/ablation_results.json.
Kept separate from run_ablation.py so the fast custom-metric pass (faithfulness
ratio, conflict/sufficiency rates, latency) can run without a RAGAS/judge-model
dependency, and this heavier pass runs on top when you want the standard
faithfulness / answer-relevancy / context-precision / context-recall numbers
for the report (§7, §10 Weeks 11-12).

Run: python -m eval.ragas_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_PATH = Path(__file__).resolve().parent / "ablation_results.json"
RAGAS_OUTPUT_PATH = Path(__file__).resolve().parent / "ragas_scores.json"


def build_ragas_dataset(rows: list[dict]):
    """Convert our pipeline output rows into the RAGAS EvaluationDataset shape."""
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

    samples = []
    for r in rows:
        if r.get("error"):
            continue
        contexts = [c.get("title", "") for c in (r.get("citations") or [])]
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r.get("answer") or "",
                retrieved_contexts=contexts or ["(no context retrieved)"],
                reference=r.get("gold_answer", ""),
            )
        )
    return EvaluationDataset(samples=samples)


def main():
    if not RESULTS_PATH.exists():
        raise SystemExit("Run `python -m eval.run_ablation` first to produce ablation_results.json")

    with open(RESULTS_PATH) as f:
        data = json.load(f)
    rows = data["rows"]

    try:
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
    except ImportError:
        raise SystemExit("pip install ragas to run this script")

    results_by_variant = {}
    for variant in sorted(set(r["variant"] for r in rows)):
        v_rows = [r for r in rows if r["variant"] == variant]
        dataset = build_ragas_dataset(v_rows)
        scores = evaluate(
            dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                LLMContextPrecisionWithoutReference(),
                LLMContextRecall(),
            ],
        )
        results_by_variant[variant] = scores.to_pandas().mean(numeric_only=True).to_dict()
        print(f"Variant {variant}: {results_by_variant[variant]}")

    with open(RAGAS_OUTPUT_PATH, "w") as f:
        json.dump(results_by_variant, f, indent=2)
    print(f"\nWritten to {RAGAS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
