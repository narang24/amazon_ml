"""
features.py — Pairwise Feature Extraction
==========================================
Given a DataFrame of (s1_id, candidate_id) pairs and the preprocessed source
DataFrames, computes a rich vector of similarity features used by the
classifier.

Feature groups
--------------
  NAME  : edit-distance, token-set ratio, Jaro-Winkler, exact-match flags,
           sorted-token match, DBA/trade name matches, acronym match
  ADDR  : token-set ratio, number overlap, postcode exact, city match,
           translit ratio, structural Jaccard
  META  : same country, same legal form, name missing, addr missing,
           token count diff, numeric count diff
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance as rfdist

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(func, a, b, default=0.0):
    try:
        return func(a, b)
    except Exception:
        return default


def _tok_set(a: str, b: str) -> float:
    return fuzz.token_set_ratio(a, b) / 100.0


def _tok_sort(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(a, b) / 100.0


def _partial(a: str, b: str) -> float:
    return fuzz.partial_ratio(a, b) / 100.0


def _jaro_winkler(a: str, b: str) -> float:
    return rfdist.JaroWinkler.normalized_similarity(a, b)


def _levenshtein_norm(a: str, b: str) -> float:
    return rfdist.Levenshtein.normalized_similarity(a, b)


def _exact(a: str, b: str) -> float:
    return 1.0 if a == b and a != "" else 0.0


def _set_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _number_overlap(a: str, b: str) -> float:
    """Fraction of numeric tokens in `a` that appear in `b`."""
    na = set(t for t in a.split() if t.isdigit())
    nb = set(t for t in b.split() if t.isdigit())
    if not na:
        return 1.0  # no numbers to mismatch
    return len(na & nb) / len(na)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_pair_features(r1: dict, r2: dict) -> dict:
    """
    r1 / r2 are row dicts from the preprocessed source DataFrames.
    Returns a flat feature dict (all float / int values).
    """
    f: dict = {}

    # ── NAME features ──────────────────────────────────────────────────────
    n1c, n2c   = r1.get("name_clean", ""),   r2.get("name_clean", "")
    n1k, n2k   = r1.get("name_core", ""),    r2.get("name_core", "")
    n1s, n2s   = r1.get("name_sorted", ""),  r2.get("name_sorted", "")
    n1ns, n2ns = r1.get("name_nospace", ""), r2.get("name_nospace", "")
    n1tr, n2tr = r1.get("name_translit", ""),r2.get("name_translit", "")
    n1db, n2db = r1.get("name_dba", ""),     r2.get("name_dba", "")
    n1ac, n2ac = r1.get("name_acronym", ""), r2.get("name_acronym", "")

    f["name_tok_set"]       = _tok_set(n1c, n2c)
    f["name_tok_sort"]      = _tok_sort(n1c, n2c)
    f["name_partial"]       = _partial(n1c, n2c)
    f["name_jaro"]          = _jaro_winkler(n1c, n2c)
    f["name_lev"]           = _levenshtein_norm(n1c, n2c)
    f["name_core_tok_set"]  = _tok_set(n1k, n2k)
    f["name_core_jaro"]     = _jaro_winkler(n1k, n2k)
    f["name_sorted_exact"]  = _exact(n1s, n2s)
    f["name_sorted_lev"]    = _levenshtein_norm(n1s, n2s)
    f["name_nospace_exact"] = _exact(n1ns, n2ns)
    f["name_nospace_lev"]   = _levenshtein_norm(n1ns, n2ns)
    f["name_translit_exact"]= _exact(n1tr, n2tr)
    f["name_translit_jaro"] = _jaro_winkler(n1tr, n2tr)
    f["name_jaccard"]       = _set_jaccard(n1k, n2k)

    # DBA / trade name cross-matches
    f["dba_vs_core_1"]      = _tok_set(n1db, n2k) if n1db else 0.0
    f["dba_vs_core_2"]      = _tok_set(n2db, n1k) if n2db else 0.0
    f["dba_vs_dba"]         = _tok_set(n1db, n2db) if (n1db and n2db) else 0.0

    # Acronym match
    f["acronym_exact"]      = _exact(n1ac, n2ac)

    # Legal form agreement
    lf1, lf2 = r1.get("name_legal_form", ""), r2.get("name_legal_form", "")
    f["legal_form_exact"]   = _exact(lf1, lf2)
    f["legal_form_both_empty"] = int(lf1 == "" and lf2 == "")

    # Token count difference (normalised)
    nt1 = int(r1.get("name_n_tokens", 1) or 1)
    nt2 = int(r2.get("name_n_tokens", 1) or 1)
    f["name_ntok_diff"] = abs(nt1 - nt2) / max(nt1, nt2)

    # ── ADDRESS features ────────────────────────────────────────────────────
    a1c, a2c   = r1.get("addr_clean", ""),     r2.get("addr_clean", "")
    a1s, a2s   = r1.get("addr_sorted", ""),    r2.get("addr_sorted", "")
    a1w, a2w   = r1.get("addr_words", ""),     r2.get("addr_words", "")
    a1n, a2n   = r1.get("addr_numbers", ""),   r2.get("addr_numbers", "")
    a1tr, a2tr = r1.get("addr_translit", ""),  r2.get("addr_translit", "")
    pc1, pc2   = r1.get("addr_postcode", ""),  r2.get("addr_postcode", "")
    cy1, cy2   = r1.get("addr_city_guess", ""),r2.get("addr_city_guess", "")
    lm1, lm2   = r1.get("addr_landmark", ""), r2.get("addr_landmark", "")

    f["addr_tok_set"]        = _tok_set(a1c, a2c)
    f["addr_tok_sort"]       = _tok_sort(a1c, a2c)
    f["addr_jaro"]           = _jaro_winkler(a1c, a2c)
    f["addr_jaccard"]        = _set_jaccard(a1w, a2w)
    f["addr_translit_jaro"]  = _jaro_winkler(a1tr, a2tr)
    f["addr_sorted_exact"]   = _exact(a1s, a2s)
    f["addr_num_overlap"]    = _number_overlap(a1n, a2n)
    f["addr_postcode_exact"] = _exact(pc1, pc2)
    f["addr_postcode_both_empty"] = int(pc1 == "" and pc2 == "")
    f["addr_city_exact"]     = _exact(cy1, cy2)
    f["addr_city_jaro"]      = _jaro_winkler(cy1, cy2)
    f["addr_landmark_tok"]   = _tok_set(lm1, lm2) if (lm1 and lm2) else 0.0

    at1 = int(r1.get("addr_n_tokens", 0) or 0)
    at2 = int(r2.get("addr_n_tokens", 0) or 0)
    f["addr_ntok_diff"] = abs(at1 - at2) / max(at1, at2, 1)

    # ── META features ───────────────────────────────────────────────────────
    c1, c2 = r1.get("country_norm", ""), r2.get("country_norm", "")
    f["same_country"]    = _exact(c1, c2)
    f["name_missing_1"]  = int(r1.get("name_missing", 0))
    f["name_missing_2"]  = int(r2.get("name_missing", 0))
    f["addr_missing_1"]  = int(r1.get("addr_missing", 0))
    f["addr_missing_2"]  = int(r2.get("addr_missing", 0))

    # Combined scores (useful derived features)
    f["name_addr_mean"]  = (f["name_tok_set"] + f["addr_tok_set"]) / 2.0
    f["name_addr_min"]   = min(f["name_tok_set"], f["addr_tok_set"])
    f["name_addr_max"]   = max(f["name_tok_set"], f["addr_tok_set"])

    return f


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction over a candidate DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    candidates : DataFrame with columns [s1_id, candidate_id]
    s1/s2/s3   : preprocessed source DataFrames
    Returns     : feature DataFrame (one row per candidate pair)
    """
    idx1 = s1.set_index("entity_id").to_dict("index")
    idx2 = s2.set_index("entity_id").to_dict("index")
    idx3 = s3.set_index("entity_id").to_dict("index")
    idx_other = {**idx2, **idx3}

    rows = []
    iterator = candidates.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(candidates), desc="Extracting features")

    for row in iterator:
        r1 = idx1.get(row.s1_id, {})
        r2 = idx_other.get(row.candidate_id, {})
        if not r1 or not r2:
            continue
        feat = extract_pair_features(r1, r2)
        feat["s1_id"] = row.s1_id
        feat["candidate_id"] = row.candidate_id
        rows.append(feat)

    df = pd.DataFrame(rows)
    # Move ID columns to front
    id_cols = ["s1_id", "candidate_id"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    return df[id_cols + feat_cols].reset_index(drop=True)


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


if __name__ == "__main__":
    import sys, logging
    logging.basicConfig(level=logging.INFO)
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed")
    cand_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/candidate_pairs.tsv")
    out_path  = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output/features.parquet")
    from pathlib import Path
    import pandas as pd

    def _load(name):
        p = data_dir / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.read_csv(
            data_dir / f"{name}.tsv", sep="\t", dtype=str, keep_default_na=False)

    s1, s2, s3 = _load("train_source1"), _load("train_source2"), _load("train_source3")
    cands = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
    feat_df = build_feature_matrix(cands, s1, s2, s3)
    feat_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(feat_df):,} feature rows → {out_path}")
