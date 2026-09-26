"""
blocking.py — Candidate Pair Generation
========================================
Generates (S1_id, S2_or_S3_id) candidate pairs using multiple blocking strategies
in union to maximise recall while keeping pair volume tractable.

Strategies used:
  1. Exact-key blocking on bk_name_prefix (first 4 chars of name_nospace)
  2. Exact-key blocking on bk_country_postcode (country|postcode, skips empty)
  3. Exact-key blocking on bk_country_city
  4. Exact-key blocking on bk_name_translit
  5. Name-sorted blocking (word-order invariant)
  6. Trigram / MinHash LSH on name_nospace for fuzzy near-duplicate blocking

Outputs
-------
  candidate_pairs.tsv : s1_id, candidate_id, source  (deduplicated union)
"""

from __future__ import annotations

import hashlib
import logging
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trigrams(s: str) -> set[str]:
    s = f"  {s}  "
    return {s[i:i+3] for i in range(len(s) - 2)}


def _minhash(s: str, n_hashes: int = 64, seed: int = 0) -> np.ndarray:
    """Lightweight MinHash without external deps."""
    tgs = _trigrams(s) if s else {""}
    sig = np.full(n_hashes, np.iinfo(np.uint32).max, dtype=np.uint32)
    for tg in tgs:
        h = int(hashlib.md5((tg + str(seed)).encode()).hexdigest(), 16)
        for i in range(n_hashes):
            val = (h ^ (i * 2654435761)) & 0xFFFFFFFF
            if val < sig[i]:
                sig[i] = val
    return sig


def _jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    return float((sig1 == sig2).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Exact-key blocking
# ─────────────────────────────────────────────────────────────────────────────

def exact_block(
    s1: pd.DataFrame,
    others: list[pd.DataFrame],
    key: str,
    min_key_len: int = 1,
) -> set[tuple[str, str]]:
    """Return all (s1_id, other_id) pairs sharing the same blocking key."""
    pairs: set[tuple[str, str]] = set()
    idx: dict[str, list[str]] = {}
    for df in others:
        for eid, k in zip(df["entity_id"], df[key]):
            if k and len(k) >= min_key_len:
                idx.setdefault(k, []).append(eid)

    for eid1, k in zip(s1["entity_id"], s1[key]):
        if k and len(k) >= min_key_len and k in idx:
            for eid2 in idx[k]:
                pairs.add((eid1, eid2))
    logger.info("  exact_block key=%s  -> %d pairs", key, len(pairs))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# LSH / MinHash fuzzy blocking
# ─────────────────────────────────────────────────────────────────────────────

def lsh_block(
    s1: pd.DataFrame,
    others: list[pd.DataFrame],
    text_col: str = "name_nospace",
    n_hashes: int = 128,
    n_bands: int = 16,
    jaccard_threshold: float = 0.25,
) -> set[tuple[str, str]]:
    """MinHash-LSH blocking on a text column."""
    rows_per_band = n_hashes // n_bands
    assert rows_per_band * n_bands == n_hashes, "n_hashes must be divisible by n_bands"

    logger.info("  lsh_block col=%s  computing signatures …", text_col)
    all_dfs = [s1] + others
    sigs: dict[str, np.ndarray] = {}
    for df in all_dfs:
        for eid, txt in zip(df["entity_id"], df[text_col]):
            sigs[eid] = _minhash(str(txt), n_hashes)

    # Build band buckets from the other-side entities
    buckets: dict[tuple, list[str]] = {}
    for df in others:
        for eid in df["entity_id"]:
            sig = sigs[eid]
            for b in range(n_bands):
                band_key = (b, tuple(sig[b * rows_per_band: (b + 1) * rows_per_band].tolist()))
                buckets.setdefault(band_key, []).append(eid)

    pairs: set[tuple[str, str]] = set()
    s1_ids = list(s1["entity_id"])
    for eid1 in tqdm(s1_ids, desc="LSH blocking", leave=False):
        sig1 = sigs[eid1]
        candidates: set[str] = set()
        for b in range(n_bands):
            band_key = (b, tuple(sig1[b * rows_per_band: (b + 1) * rows_per_band].tolist()))
            for eid2 in buckets.get(band_key, []):
                candidates.add(eid2)
        for eid2 in candidates:
            if _jaccard_estimate(sig1, sigs[eid2]) >= jaccard_threshold:
                pairs.add((eid1, eid2))
    logger.info("  lsh_block -> %d pairs", len(pairs))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    lsh_enabled: bool = True,
    lsh_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Run all blocking strategies in union.
    Returns DataFrame with columns [s1_id, candidate_id, candidate_source].
    """
    others = [s2, s3]
    all_pairs: set[tuple[str, str]] = set()

    # --- Exact key blocking ---
    for key in [
        "bk_name_prefix",
        "bk_country_postcode",
        "bk_country_city",
        "bk_name_translit",
        "name_sorted",      # word-order invariant
    ]:
        if key in s1.columns:
            min_len = 4 if key == "bk_name_prefix" else 3
            all_pairs |= exact_block(s1, others, key, min_key_len=min_len)

    # --- LSH fuzzy blocking ---
    if lsh_enabled:
        all_pairs |= lsh_block(s1, others, "name_nospace",
                               jaccard_threshold=lsh_threshold)

    # --- Assemble DataFrame ---
    logger.info("Total candidate pairs (union): %d", len(all_pairs))
    if not all_pairs:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source"])

    pairs_df = pd.DataFrame(list(all_pairs), columns=["s1_id", "candidate_id"])
    pairs_df["candidate_source"] = pairs_df["candidate_id"].str.split("-").str[0]
    pairs_df = pairs_df.drop_duplicates().reset_index(drop=True)
    return pairs_df


def candidate_recall(
    candidates: pd.DataFrame,
    gt: dict[str, list[str]],
) -> float:
    """Fraction of ground-truth positives covered by the candidate set."""
    found = total = 0
    cand_set = set(zip(candidates["s1_id"], candidates["candidate_id"]))
    for s1_id, match_ids in gt.items():
        for m in match_ids:
            total += 1
            if (s1_id, m) in cand_set:
                found += 1
    return found / total if total else 1.0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(name):
        p = data_dir / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.read_csv(
            data_dir / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False)

    s1 = _load("train_source1")
    s2 = _load("train_source2")
    s3 = _load("train_source3")
    cands = generate_candidates(s1, s2, s3)
    cands.to_csv(out_dir / "candidate_pairs.tsv", sep="\t", index=False)
    print(f"Wrote {len(cands):,} candidate pairs to {out_dir}/candidate_pairs.tsv")
