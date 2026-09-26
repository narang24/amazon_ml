"""
evaluate.py — Standalone Evaluation & Reporting Script
=======================================================
Computes and displays all key metrics from existing output files.
Can be run after pipeline.py or independently.

Usage:
    python evaluate.py --output-dir output --gt-path data/train/train_ground_truth.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_predictions(results_path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(results_path, sep="\t", dtype=str, keep_default_na=False)
    pred: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        pred[s1_id] = [m.strip() for m in matched.split(",") if m.strip()]
    return pred


def full_report(
    pred: dict[str, list[str]],
    gt: dict[str, list[str]],
    candidates: pd.DataFrame | None = None,
) -> dict:
    from model import macro_f05
    from blocking import candidate_recall

    metrics = macro_f05(pred, gt)

    # Candidate recall (if candidates provided)
    if candidates is not None:
        metrics["candidate_recall"] = round(candidate_recall(candidates, gt), 6)

    # Per-country breakdown
    # (requires s1 preprocessed to be available — skip gracefully if not)
    metrics["n_s1_entities"]      = len(gt)
    metrics["n_predicted_matches"] = sum(len(v) for v in pred.values())
    metrics["n_true_matches"]      = sum(len(v) for v in gt.values())
    metrics["singleton_frac"]      = round(
        sum(1 for v in gt.values() if not v) / max(len(gt), 1), 4)

    return metrics


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("  ENTITY RESOLUTION EVALUATION REPORT")
    print("=" * 60)
    key_order = ["f05", "precision", "recall", "candidate_recall",
                 "threshold", "n_s1_entities", "n_predicted_matches",
                 "n_true_matches", "singleton_frac", "runtime_s",
                 "total_runtime_s", "block_time_s"]
    for k in key_order:
        if k in metrics:
            print(f"  {k:<30} {metrics[k]}")
    if "top_features" in metrics:
        print("\n  Top-15 features by importance:")
        for fname, imp in list(metrics["top_features"].items())[:15]:
            print(f"    {fname:<35} {imp:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",   default="output")
    ap.add_argument("--gt-path",      default="data/train/train_ground_truth.tsv")
    ap.add_argument("--metrics-json", default=None,
                    help="If set, also print saved metrics from JSON")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)

    if args.metrics_json:
        with open(args.metrics_json) as f:
            metrics = json.load(f)
        print_report(metrics)
    else:
        from preprocess import load_ground_truth
        gt = load_ground_truth(args.gt_path)

        results_path = out_dir / "matching_results.tsv"
        if not results_path.exists():
            print(f"No matching_results.tsv found at {results_path}")
            sys.exit(1)

        pred = load_predictions(results_path)

        cands_path = out_dir / "candidate_pairs.tsv"
        candidates = None
        if cands_path.exists():
            candidates = pd.read_csv(cands_path, sep="\t", dtype=str,
                                     keep_default_na=False)

        metrics = full_report(pred, gt, candidates)
        print_report(metrics)

        # Save to output
        with open(out_dir / "eval_report.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved eval report to {out_dir}/eval_report.json")
