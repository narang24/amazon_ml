"""
pipeline.py — End-to-End Orchestrator
======================================
Orchestrates the full Business Entity Resolution pipeline:

  Step 1: Load & preprocess raw TSV sources (via preprocess.py)
  Step 2: Generate candidate pairs (via blocking.py)
  Step 3: Extract pairwise features (via features.py)
  Step 4: Train / load model, sweep threshold (via model.py)
  Step 5: Run inference on test set, write matching_results.tsv
  Step 6: Compute & log metrics (MLflow + JSON file)
  Step 7: Upload outputs to S3 (optional)

Run locally:
    python pipeline.py --config config/pipeline.yaml

Run on EC2 (invoked by GitHub Actions):
    python pipeline.py --config config/pipeline.yaml --s3-sync
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_split, load_ground_truth, gt_to_pairs
from blocking import generate_candidates, candidate_recall
from features import build_feature_matrix
from model import (
    assign_labels, split_features, run_training_pipeline,
    predict, write_matching_results, macro_f05,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_dir":          "data",
    "processed_dir":     "processed",
    "output_dir":        "output",
    "mlflow_uri":        "mlruns",
    "experiment_name":   "entity_resolution",
    "lsh_enabled":       True,
    "lsh_threshold":     0.25,
    "lgb_params":        {},
    "n_estimators":      500,
    "early_stopping":    50,
    "val_frac":          0.2,
    "seed":              42,
    "s3_bucket":         "",
    "s3_prefix":         "entity-resolution",
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and Path(path).exists():
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    # Environment variable overrides (useful for GitHub Actions secrets)
    for key in ("s3_bucket", "mlflow_uri"):
        env_val = os.environ.get(f"ER_{key.upper()}")
        if env_val:
            cfg[key] = env_val
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# S3 sync utilities
# ─────────────────────────────────────────────────────────────────────────────

def s3_upload_dir(local_dir: Path, bucket: str, prefix: str) -> None:
    """Upload a directory to S3 using boto3."""
    try:
        import boto3
        s3 = boto3.client("s3")
        for fp in local_dir.rglob("*"):
            if fp.is_file():
                key = f"{prefix}/{fp.relative_to(local_dir).as_posix()}"
                logger.info("S3 upload: %s → s3://%s/%s", fp, bucket, key)
                s3.upload_file(str(fp), bucket, key)
    except Exception as e:
        logger.error("S3 upload failed: %s", e)


def s3_download_dir(bucket: str, prefix: str, local_dir: Path) -> None:
    """Download a S3 prefix to a local directory."""
    try:
        import boto3
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix) + 1:]
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info("S3 download: s3://%s/%s → %s", bucket, key, dest)
                s3.download_file(bucket, key, str(dest))
    except Exception as e:
        logger.error("S3 download failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet / TSV loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_processed(proc_dir: Path, name: str) -> pd.DataFrame:
    p = proc_dir / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    t = proc_dir / f"{name}.tsv"
    return pd.read_csv(t, sep="\t", dtype=str, keep_default_na=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict, s3_sync: bool = False) -> dict:
    t_total = time.time()
    data_dir  = Path(cfg["data_dir"])
    proc_dir  = Path(cfg["processed_dir"])
    out_dir   = Path(cfg["output_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional: pull data from S3 ────────────────────────────────────────
    if s3_sync and cfg.get("s3_bucket"):
        logger.info("Downloading data from S3 …")
        s3_download_dir(cfg["s3_bucket"],
                        f"{cfg['s3_prefix']}/data", data_dir)

    # ── MLflow tracking ────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_params({k: v for k, v in cfg.items()
                           if isinstance(v, (str, int, float, bool))})

        # ─── Step 1: Preprocess ──────────────────────────────────────────
        logger.info("=== Step 1: Preprocessing ===")
        from preprocess import load_split, load_ground_truth
        try:
            s1_tr, s2_tr, s3_tr = load_split(data_dir, "train")
        except FileNotFoundError as e:
            logger.error("Train data not found: %s", e)
            raise

        # Cache processed data
        for name, df in [("train_source1", s1_tr), ("train_source2", s2_tr),
                          ("train_source3", s3_tr)]:
            _save_processed(df, proc_dir, name)

        # ─── Step 2: Candidate Generation ────────────────────────────────
        logger.info("=== Step 2: Blocking / Candidate Generation ===")
        t_block = time.time()
        candidates_train = generate_candidates(
            s1_tr, s2_tr, s3_tr,
            lsh_enabled=cfg["lsh_enabled"],
            lsh_threshold=cfg["lsh_threshold"],
        )
        block_time = time.time() - t_block
        logger.info("Blocking: %d pairs in %.1fs", len(candidates_train), block_time)

        # Load ground truth & compute candidate recall
        gt_path = data_dir / "train" / "train_ground_truth.tsv"
        gt = load_ground_truth(gt_path)
        cand_recall = candidate_recall(candidates_train, gt)
        logger.info("Candidate recall: %.4f", cand_recall)
        mlflow.log_metric("candidate_recall", cand_recall)

        # Save candidate pairs
        candidates_train.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)

        # ─── Step 3: Feature Extraction ──────────────────────────────────
        logger.info("=== Step 3: Feature Extraction ===")
        t_feat = time.time()
        feat_df = build_feature_matrix(candidates_train, s1_tr, s2_tr, s3_tr)
        logger.info("Features: %d rows, %d cols in %.1fs",
                    len(feat_df), feat_df.shape[1], time.time() - t_feat)

        # Label + train/val split
        feat_df = assign_labels(feat_df, gt)
        pos_rate = feat_df["label"].mean()
        logger.info("Positive rate: %.4f", pos_rate)
        mlflow.log_metric("positive_rate", pos_rate)

        split_train_path = proc_dir / "split_train_ids.tsv"
        split_val_path   = proc_dir / "split_val_ids.tsv"
        if split_train_path.exists() and split_val_path.exists():
            train_ids = pd.read_csv(split_train_path)["s1_id"].tolist()
            val_ids   = pd.read_csv(split_val_path)["s1_id"].tolist()
        else:
            from preprocess import split_s1_ids
            s1_idx = s1_tr.set_index("entity_id")
            strat = (s1_idx["country_norm"] + "|" +
                     pd.Series({k: str(min(len(v), 2)) for k, v in gt.items()}))
            train_ids, val_ids = split_s1_ids(
                list(gt), cfg["val_frac"], cfg["seed"], strat)

        feat_train, feat_val = split_features(feat_df, train_ids, val_ids)
        logger.info("Split: %d train / %d val pairs", len(feat_train), len(feat_val))

        # ─── Step 4: Train & Threshold Sweep ─────────────────────────────
        logger.info("=== Step 4: Model Training ===")
        result = run_training_pipeline(
            feat_train, feat_val, gt, out_dir,
            params=cfg.get("lgb_params"),
        )
        model     = result["model"]
        threshold = result["threshold"]
        metrics   = result["metrics"]
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        mlflow.log_metric("block_time_s", block_time)
        mlflow.log_metric("candidate_recall", cand_recall)
        logger.info("Val metrics: %s", metrics)

        # ─── Step 5: Test set prediction ─────────────────────────────────
        logger.info("=== Step 5: Test Set Prediction ===")
        try:
            s1_te, s2_te, s3_te = load_split(data_dir, "test")
            cands_test = generate_candidates(
                s1_te, s2_te, s3_te,
                lsh_enabled=cfg["lsh_enabled"],
                lsh_threshold=cfg["lsh_threshold"],
            )
            feat_test = build_feature_matrix(cands_test, s1_te, s2_te, s3_te)
            pred_test = predict(model, feat_test, threshold=threshold)
            all_s1_test = s1_te["entity_id"].tolist()
            write_matching_results(
                pred_test, all_s1_test, out_dir / "matching_results.tsv")
        except FileNotFoundError:
            logger.warning("Test data not found — skipping test prediction.")

        # ─── Step 6: Save metrics ─────────────────────────────────────────
        metrics["candidate_recall"] = round(cand_recall, 6)
        metrics["block_time_s"]     = round(block_time, 2)
        metrics["total_runtime_s"]  = round(time.time() - t_total, 2)
        metrics["run_id"]           = run.info.run_id
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(out_dir / "metrics.json"))
        logger.info("=== Pipeline complete in %.1fs ===", metrics["total_runtime_s"])

        # ─── Step 7: S3 upload ────────────────────────────────────────────
        if s3_sync and cfg.get("s3_bucket"):
            logger.info("Uploading outputs to S3 …")
            s3_upload_dir(out_dir, cfg["s3_bucket"],
                          f"{cfg['s3_prefix']}/output")
            s3_upload_dir(Path(cfg["mlflow_uri"]), cfg["s3_bucket"],
                          f"{cfg['s3_prefix']}/mlruns")

    return metrics


def _save_processed(df: pd.DataFrame, proc_dir: Path, name: str) -> None:
    try:
        df.to_parquet(proc_dir / f"{name}.parquet", index=False)
    except Exception:
        df.to_csv(proc_dir / f"{name}.tsv", sep="\t", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    ap.add_argument("--config",   default="config/pipeline.yaml",
                    help="Path to YAML config file")
    ap.add_argument("--s3-sync",  action="store_true",
                    help="Download data from S3 before run; upload outputs after")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metrics = run_pipeline(cfg, s3_sync=args.s3_sync)
    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        if k != "top_features":
            print(f"  {k}: {v}")
