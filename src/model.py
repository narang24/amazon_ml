"""
model.py — Training, Prediction, and Evaluation
================================================
Trains a LightGBM binary classifier on pairwise features to predict whether
a (S1, S2/S3) candidate pair is a true match.

Threshold selection
-------------------
After training, we sweep the decision threshold on the validation set to
maximise F0.5. This is important because the competition metric heavily
rewards precision over recall.

Output
------
  matching_results.tsv  : source1_entity_id  matched_entity_ids (comma-joined)
  metrics.json          : precision, recall, F0.5, candidate_recall, runtime
  model.lgb             : saved LightGBM model
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger(__name__)

FEATURE_EXCLUDE = {"s1_id", "candidate_id", "label"}


# ─────────────────────────────────────────────────────────────────────────────
# Label assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_labels(
    feat_df: pd.DataFrame,
    gt: dict[str, list[str]],
) -> pd.DataFrame:
    """Add a binary `label` column (1 = true match)."""
    gt_set: set[tuple[str, str]] = set()
    for s1_id, matches in gt.items():
        for m in matches:
            gt_set.add((s1_id, m))
    feat_df = feat_df.copy()
    feat_df["label"] = feat_df.apply(
        lambda r: int((r["s1_id"], r["candidate_id"]) in gt_set), axis=1)
    return feat_df


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split helpers
# ─────────────────────────────────────────────────────────────────────────────

def split_features(
    feat_df: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_set = set(train_ids)
    val_set   = set(val_ids)
    return (
        feat_df[feat_df["s1_id"].isin(train_set)].reset_index(drop=True),
        feat_df[feat_df["s1_id"].isin(val_set)].reset_index(drop=True),
    )


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM training
# ─────────────────────────────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples":20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             42,
}


def train_model(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    p = {**LGB_PARAMS, **(params or {})}
    feat_cols = _feature_cols(train_df)
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["label"].values

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
    callbacks = [lgb.log_evaluation(period=50)]
    valid_sets = [dtrain]
    valid_names = ["train"]

    if val_df is not None and len(val_df) > 0:
        X_val = val_df[feat_cols].values.astype(np.float32)
        y_val = val_df["label"].values
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=True))

    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep (maximise F0.5 on validation set)
# ─────────────────────────────────────────────────────────────────────────────

def f05_score(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 1.25 * p * r / (0.25 * p + r)


def sweep_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Return (best_threshold, best_f05)."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        if preds.sum() == 0:
            continue
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f = f05_score(p, r)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ─────────────────────────────────────────────────────────────────────────────
# Macro-averaged F0.5  (mirrors official metric)
# ─────────────────────────────────────────────────────────────────────────────

def macro_f05(
    pred: dict[str, list[str]],
    gt: dict[str, list[str]],
) -> dict[str, float]:
    """Returns dict with f05, precision, recall (all macro-averaged)."""
    f05s, precs, recs = [], [], []
    for s1_id, truth in gt.items():
        p_set = set(pred.get(s1_id, []))
        t_set = set(truth)
        if not t_set:
            f05s.append(1.0 if not p_set else 0.0)
            precs.append(1.0 if not p_set else 0.0)
            recs.append(1.0)
            continue
        if not p_set:
            f05s.append(0.0); precs.append(0.0); recs.append(0.0)
            continue
        tp = len(p_set & t_set)
        pr = tp / len(p_set)
        rc = tp / len(t_set)
        precs.append(pr); recs.append(rc)
        f05s.append(f05_score(pr, rc) if tp > 0 else 0.0)
    return {
        "f05":       round(np.mean(f05s), 6),
        "precision": round(np.mean(precs), 6),
        "recall":    round(np.mean(recs), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predict and format output
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model: lgb.Booster,
    feat_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, list[str]]:
    """Returns {s1_id: [matched_ids]} for scores >= threshold."""
    feat_cols = _feature_cols(feat_df)
    X = feat_df[feat_cols].values.astype(np.float32)
    scores = model.predict(X)
    feat_df = feat_df.copy()
    feat_df["score"] = scores
    feat_df["match"] = scores >= threshold
    matched = feat_df[feat_df["match"]]
    result: dict[str, list[str]] = {}
    for row in matched.itertuples(index=False):
        result.setdefault(row.s1_id, []).append(row.candidate_id)
    return result


def write_matching_results(
    pred: dict[str, list[str]],
    all_s1_ids: list[str],
    out_path: Path,
) -> None:
    """Write matching_results.tsv with ALL S1 entities (singletons get empty)."""
    rows = []
    for s1_id in all_s1_ids:
        matches = pred.get(s1_id, [])
        rows.append({"source1_entity_id": s1_id,
                     "matched_entity_ids": ",".join(sorted(matches))})
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote %d rows to %s", len(rows), out_path)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline entry (called from pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(
    feat_train: pd.DataFrame,
    feat_val: pd.DataFrame,
    gt: dict[str, list[str]],
    out_dir: Path,
    params: Optional[dict] = None,
) -> dict:
    """Train, threshold-sweep, evaluate on val, save model + results."""
    t0 = time.time()

    logger.info("Training LightGBM …")
    model = train_model(feat_train, feat_val, params=params)

    # Threshold sweep on val
    feat_cols = _feature_cols(feat_val)
    X_val = feat_val[feat_cols].values.astype(np.float32)
    val_scores = model.predict(X_val)
    best_t, best_f = sweep_threshold(val_scores, feat_val["label"].values)
    logger.info("Best val threshold=%.3f  F0.5=%.4f", best_t, best_f)

    # Macro F0.5 on val (official metric style)
    val_s1_ids = sorted(feat_val["s1_id"].unique())
    val_gt = {k: v for k, v in gt.items() if k in set(val_s1_ids)}
    pred_val = predict(model, feat_val, threshold=best_t)
    metrics = macro_f05(pred_val, val_gt)
    metrics["threshold"] = round(best_t, 4)
    metrics["runtime_s"] = round(time.time() - t0, 2)

    # Feature importance
    feat_cols_list = _feature_cols(feat_train)
    imp = dict(zip(feat_cols_list, model.feature_importance("gain").tolist()))
    metrics["top_features"] = dict(sorted(imp.items(), key=lambda x: -x[1])[:15])

    # Save model
    model_path = out_dir / "model.lgb"
    model.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)

    return {"model": model, "threshold": best_t, "metrics": metrics}
