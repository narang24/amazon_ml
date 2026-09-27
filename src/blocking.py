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


# Precomputed per-hash multipliers (avoids recomputing inside the hot loop)
_MINHASH_MULTIPLIERS: np.ndarray | None = None


def _get_multipliers(n_hashes: int) -> np.ndarray:
    global _MINHASH_MULTIPLIERS
    if _MINHASH_MULTIPLIERS is None or len(_MINHASH_MULTIPLIERS) != n_hashes:
        _MINHASH_MULTIPLIERS = (
            np.arange(n_hashes, dtype=np.uint64) * np.uint64(2654435761)
        ) & np.uint64(0xFFFFFFFF)
    return _MINHASH_MULTIPLIERS


def _minhash(s: str, n_hashes: int = 128, seed: int = 0) -> np.ndarray:
    """Vectorised MinHash — ~20-50x faster than the scalar version."""
    tgs = _trigrams(s) if s else {""}
    multipliers = _get_multipliers(n_hashes)
    sig = np.full(n_hashes, np.iinfo(np.uint32).max, dtype=np.uint32)
    for tg in tgs:
        h = np.uint64(int(hashlib.md5(tg.encode()).hexdigest()[:8], 16) ^ seed)
        # Vectorised: one XOR across all hash slots at once
        vals = (h ^ multipliers).astype(np.uint32)
        np.minimum(sig, vals, out=sig)
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
    s1_batch_size: int = 5_000,
    max_candidates_per_entity: int = 500,
) -> set[tuple[str, str]]:
    """MinHash-LSH blocking on a text column.

    Memory-efficient: only the other-side signatures are held in RAM
    permanently; S1 signatures are computed and discarded in batches.

    Args:
        s1_batch_size: Number of S1 rows to process per batch (tune down
            if still OOM-killed; tune up for speed on large RAM hosts).
        max_candidates_per_entity: Hard cap on LSH candidates per S1 entity
            before Jaccard re-scoring — prevents quadratic blowup on
            extremely common name prefixes.
    """
    rows_per_band = n_hashes // n_bands
    assert rows_per_band * n_bands == n_hashes, "n_hashes must be divisible by n_bands"

    # ── Step 1: Build band-bucket index from the other-side entities only ──
    logger.info("  lsh_block col=%s  indexing other-side signatures …", text_col)
    other_sigs: dict[str, np.ndarray] = {}
    for df in others:
        for eid, txt in zip(df["entity_id"], df[text_col]):
            other_sigs[eid] = _minhash(str(txt), n_hashes)

    buckets: dict[tuple, list[str]] = {}
    for eid, sig in other_sigs.items():
        for b in range(n_bands):
            band_key = (b, sig[b * rows_per_band: (b + 1) * rows_per_band].tobytes())
            buckets.setdefault(band_key, []).append(eid)
    logger.info("  lsh_block  bucket index built (%d buckets)", len(buckets))

    # ── Step 2: Stream S1 in batches, look up candidates, re-score ──
    pairs: set[tuple[str, str]] = set()
    s1_ids   = s1["entity_id"].tolist()
    s1_texts = s1[text_col].tolist()
    n = len(s1_ids)

    for start in tqdm(range(0, n, s1_batch_size), desc="LSH blocking", leave=False):
        batch_ids   = s1_ids[start: start + s1_batch_size]
        batch_texts = s1_texts[start: start + s1_batch_size]

        # Compute S1 sigs for this batch only
        batch_sigs: dict[str, np.ndarray] = {
            eid: _minhash(str(txt), n_hashes)
            for eid, txt in zip(batch_ids, batch_texts)
        }

        for eid1, sig1 in batch_sigs.items():
            candidates: set[str] = set()
            for b in range(n_bands):
                band_key = (b, sig1[b * rows_per_band: (b + 1) * rows_per_band].tobytes())
                for eid2 in buckets.get(band_key, []):
                    candidates.add(eid2)
                    if len(candidates) >= max_candidates_per_entity:
                        break
                if len(candidates) >= max_candidates_per_entity:
                    break
            for eid2 in candidates:
                if _jaccard_estimate(sig1, other_sigs[eid2]) >= jaccard_threshold:
                    pairs.add((eid1, eid2))

        # Explicitly release batch sigs after each batch
        del batch_sigs

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

    # Use a generator to avoid double-materialising the full set as a list
    s1_ids_out, cand_ids_out = zip(*all_pairs)
    pairs_df = pd.DataFrame({"s1_id": s1_ids_out, "candidate_id": cand_ids_out})
    del s1_ids_out, cand_ids_out, all_pairs  # free immediately
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
