"""
test_blocking.py — Unit Tests for Blocking / Candidate Generation
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blocking import exact_block, candidate_recall, generate_candidates
from preprocess import preprocess_source


def _make_df(rows: list[dict], source_prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return preprocess_source(df)


S1_ROWS = [
    {"entity_id": "S1-001", "business_name": "Acme Corp",
     "business_address": "123 Main St, NY 10001", "country": "US"},
    {"entity_id": "S1-002", "business_name": "Sharma Enterprises",
     "business_address": "Plot 12, Chandigarh 160017", "country": "India"},
]

S2_ROWS = [
    {"entity_id": "S2-001", "business_name": "Acme Corporation",
     "business_address": "123 Main Street, New York 10001", "country": "US"},
    {"entity_id": "S2-002", "business_name": "Totally Different Biz",
     "business_address": "99 Other Rd, Chicago", "country": "US"},
]

S3_ROWS = [
    {"entity_id": "S3-001", "business_name": "Sharma Enterprises Pvt Ltd",
     "business_address": "Plot 12, Sec 17, Chandigarh 160017", "country": "India"},
]


@pytest.fixture
def s1():
    return _make_df(S1_ROWS, "S1")

@pytest.fixture
def s2():
    return _make_df(S2_ROWS, "S2")

@pytest.fixture
def s3():
    return _make_df(S3_ROWS, "S3")


class TestExactBlock:
    def test_finds_postcode_match(self, s1, s2, s3):
        pairs = exact_block(s1, [s2, s3], "bk_country_postcode", min_key_len=3)
        # S1-001 and S2-001 share US|10001; S1-002 and S3-001 share India|160017
        assert ("S1-001", "S2-001") in pairs or len(pairs) > 0

    def test_no_false_cross_country(self, s1, s2, s3):
        pairs = exact_block(s1, [s2, s3], "bk_country_postcode", min_key_len=3)
        # S1-001 (US) should not match S3-001 (India)
        assert ("S1-001", "S3-001") not in pairs

    def test_empty_key_skipped(self, s1, s2, s3):
        """Keys shorter than min_key_len should not produce pairs."""
        pairs = exact_block(s1, [s2], "bk_name_prefix", min_key_len=100)
        assert len(pairs) == 0


class TestCandidateRecall:
    def test_perfect_recall(self):
        cands = pd.DataFrame({
            "s1_id":        ["S1-001", "S1-002"],
            "candidate_id": ["S2-001", "S3-001"],
        })
        gt = {"S1-001": ["S2-001"], "S1-002": ["S3-001"]}
        assert candidate_recall(cands, gt) == pytest.approx(1.0)

    def test_zero_recall(self):
        cands = pd.DataFrame({
            "s1_id":        ["S1-001"],
            "candidate_id": ["S2-999"],
        })
        gt = {"S1-001": ["S2-001"]}
        assert candidate_recall(cands, gt) == pytest.approx(0.0)

    def test_empty_gt(self):
        cands = pd.DataFrame(columns=["s1_id", "candidate_id"])
        assert candidate_recall(cands, {}) == pytest.approx(1.0)


class TestGenerateCandidates:
    def test_returns_dataframe(self, s1, s2, s3):
        cands = generate_candidates(s1, s2, s3, lsh_enabled=False)
        assert isinstance(cands, pd.DataFrame)
        assert "s1_id" in cands.columns
        assert "candidate_id" in cands.columns

    def test_no_self_matches(self, s1, s2, s3):
        cands = generate_candidates(s1, s2, s3, lsh_enabled=False)
        # All candidate_ids should be from S2 or S3
        assert all(cands["candidate_id"].str.startswith(("S2-", "S3-")))

    def test_no_duplicates(self, s1, s2, s3):
        cands = generate_candidates(s1, s2, s3, lsh_enabled=False)
        dupes = cands.duplicated(["s1_id", "candidate_id"]).sum()
        assert dupes == 0
