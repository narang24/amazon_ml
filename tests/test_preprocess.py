"""
test_preprocess.py — Unit Tests for the Preprocessing Module
=============================================================
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import (
    base_normalize,
    normalize_name,
    normalize_address,
    normalize_country,
    split_dba,
    extract_legal,
    parse_id_list,
    f05_macro,
    preprocess_source,
)


# ─────────────────────────────────────────────────────────────────────────────
# base_normalize
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseNormalize:
    def test_ampersand_to_and(self):
        assert "a and b" in base_normalize("A & B")

    def test_accents(self):
        assert base_normalize("café") == "cafe"

    def test_apostrophe(self):
        assert "mcdonalds" in base_normalize("McDonald's")

    def test_dotted_acronym(self):
        result = base_normalize("s.c.o.")
        assert "." not in result

    def test_empty_string(self):
        assert base_normalize("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# normalize_name
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_basic_name(self):
        r = normalize_name("Acme Corporation")
        assert "acme" in r["name_core"]
        assert r["name_legal_form"] in ("corp", "")

    def test_dba_split(self):
        r = normalize_name("Acme Holdings LLC dba Acme Cafe")
        assert "acme" in r["name_dba"] or "cafe" in r["name_dba"]

    def test_pvt_ltd(self):
        r = normalize_name("Sharma Enterprises Pvt. Ltd.")
        assert "pvt" in r["name_legal_form"] or "ltd" in r["name_legal_form"]

    def test_ms_prefix_stripped(self):
        r = normalize_name("M/s Gupta Traders")
        assert "gupta" in r["name_core"]

    def test_sorted_invariant(self):
        r1 = normalize_name("Global Tech Solutions")
        r2 = normalize_name("Solutions Tech Global")
        assert r1["name_sorted"] == r2["name_sorted"]

    def test_nospace(self):
        r = normalize_name("McDonald Corporation")
        assert " " not in r["name_nospace"]

    def test_abbreviation_expansion(self):
        r = normalize_name("ABC Corp")
        assert "corporation" in r["name_clean"]


# ─────────────────────────────────────────────────────────────────────────────
# normalize_address
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeAddress:
    def test_us_postcode(self):
        r = normalize_address("123 Main St, San Francisco, CA 94103", "us")
        assert r["addr_postcode"] == "94103"

    def test_india_pin(self):
        r = normalize_address("H.No. 12, Sector 17, Chandigarh 160017", "india")
        assert r["addr_postcode"] == "160017"

    def test_landmark_extraction(self):
        r = normalize_address("Shop 5, Near SBI ATM, MG Road", "india")
        assert r["addr_landmark"] != ""

    def test_city_aliases(self):
        r = normalize_address("MG Road, Bangalore, Karnataka", "india")
        assert "bengaluru" in r["addr_clean"] or "bengaluru" in r["addr_city_guess"]

    def test_empty_address(self):
        r = normalize_address("", "us")
        assert r["addr_missing"] == 1

    def test_unit_prefix_removed(self):
        r = normalize_address("Plot No. 45, Industrial Area", "india")
        assert "45" in r["addr_numbers"]

    def test_state_expansion_us(self):
        r = normalize_address("100 Main St, Austin TX", "us")
        assert "texas" in r["addr_clean"]


# ─────────────────────────────────────────────────────────────────────────────
# normalize_country
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeCountry:
    @pytest.mark.parametrize("raw,expected", [
        ("US",            "us"),
        ("USA",           "us"),
        ("United States", "us"),
        ("India",         "india"),
        ("IN",            "india"),
        ("France",        "france"),
        ("FR",            "france"),
        ("Unknown",       "unknown"),   # open-set passthrough
        ("",              ""),
    ])
    def test_aliases(self, raw, expected):
        assert normalize_country(raw) == expected


# ─────────────────────────────────────────────────────────────────────────────
# split_dba
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitDba:
    def test_dba_keyword(self):
        legal, dba = split_dba("Acme Holdings LLC dba Acme Cafe")
        assert "acme" in legal.lower()
        assert "cafe" in dba.lower()

    def test_no_dba(self):
        legal, dba = split_dba("Acme Holdings LLC")
        assert dba == ""

    def test_parenthetical(self):
        _, dba = split_dba("Foo Ltd (Bar Foods)")
        assert "bar" in dba.lower() or "foo" in dba.lower()


# ─────────────────────────────────────────────────────────────────────────────
# parse_id_list
# ─────────────────────────────────────────────────────────────────────────────

class TestParseIdList:
    def test_basic(self):
        assert parse_id_list("S2-001,S3-002") == ["S2-001", "S3-002"]

    def test_dedup(self):
        assert len(parse_id_list("S2-001,S2-001")) == 1

    def test_empty(self):
        assert parse_id_list("") == []

    def test_spaces(self):
        result = parse_id_list(" S2-001 , S3-002 ")
        assert result == ["S2-001", "S3-002"]


# ─────────────────────────────────────────────────────────────────────────────
# f05_macro
# ─────────────────────────────────────────────────────────────────────────────

class TestF05Macro:
    def test_perfect_match(self):
        gt   = {"S1-1": ["S2-1"], "S1-2": ["S3-1"]}
        pred = {"S1-1": ["S2-1"], "S1-2": ["S3-1"]}
        assert f05_macro(pred, gt) == pytest.approx(1.0)

    def test_all_wrong(self):
        gt   = {"S1-1": ["S2-1"]}
        pred = {"S1-1": ["S2-99"]}
        assert f05_macro(pred, gt) == pytest.approx(0.0)

    def test_singleton_no_pred(self):
        gt   = {"S1-1": []}
        pred = {}
        assert f05_macro(pred, gt) == pytest.approx(1.0)

    def test_singleton_wrong_pred(self):
        gt   = {"S1-1": []}
        pred = {"S1-1": ["S2-99"]}
        assert f05_macro(pred, gt) == pytest.approx(0.0)

    def test_partial_match(self):
        gt   = {"S1-1": ["S2-1", "S3-1"]}
        pred = {"S1-1": ["S2-1"]}
        score = f05_macro(pred, gt)
        assert 0.0 < score < 1.0

    def test_precision_weighted(self):
        """F0.5 should be higher when precision is high even if recall is low."""
        gt = {"S1-1": ["S2-1", "S2-2", "S2-3"]}
        # High precision (1/1 = 1.0), low recall (1/3)
        pred_hp = {"S1-1": ["S2-1"]}
        # Low precision (1/3), high recall (1/1)
        pred_hr = {"S1-1": ["S2-1", "S2-99", "S2-98"]}
        assert f05_macro(pred_hp, gt) > f05_macro(pred_hr, gt)


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_source (integration)
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessSource:
    SAMPLE_ROWS = [
        {"entity_id": "S1-00001", "business_name": "Acme Corp", 
         "business_address": "123 Main St, New York, NY 10001", "country": "US"},
        {"entity_id": "S1-00002", "business_name": "M/s Sharma Pvt Ltd",
         "business_address": "Plot 12, Sector 17, Chandigarh 160017", "country": "India"},
        {"entity_id": "S1-00003", "business_name": "Boulangerie Martin SARL",
         "business_address": "12 Rue de la Paix, 75001 Paris", "country": "France"},
    ]

    def _df(self):
        return pd.DataFrame(self.SAMPLE_ROWS)

    def test_output_columns(self):
        result = preprocess_source(self._df())
        for col in ["name_core", "name_legal_form", "addr_clean",
                    "addr_postcode", "country_norm",
                    "bk_name_prefix", "bk_country_city"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_country_normalised(self):
        result = preprocess_source(self._df())
        assert set(result["country_norm"]) == {"us", "india", "france"}

    def test_blocking_keys_nonempty(self):
        result = preprocess_source(self._df())
        # At least the name prefix key should be populated for all rows
        assert result["bk_name_prefix"].str.len().min() >= 1

    def test_us_postcode_extracted(self):
        result = preprocess_source(self._df())
        us_row = result[result["country_norm"] == "us"].iloc[0]
        assert us_row["addr_postcode"] == "10001"

    def test_india_postcode_extracted(self):
        result = preprocess_source(self._df())
        in_row = result[result["country_norm"] == "india"].iloc[0]
        assert in_row["addr_postcode"] == "160017"
