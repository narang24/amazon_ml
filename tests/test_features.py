"""
test_features.py — Unit Tests for the Feature Extraction Module
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import extract_pair_features, build_feature_matrix, _tok_set, _exact


class TestLowLevelSimilarity:
    def test_tok_set_identical(self):
        assert _tok_set("acme corp", "acme corp") == pytest.approx(1.0)

    def test_tok_set_reordered(self):
        assert _tok_set("acme corp", "corp acme") == pytest.approx(1.0)

    def test_tok_set_different(self):
        assert _tok_set("acme corp", "xyz holdings") < 0.5

    def test_exact_match(self):
        assert _exact("acme", "acme") == 1.0

    def test_exact_mismatch(self):
        assert _exact("acme", "xyz") == 0.0

    def test_exact_empty(self):
        assert _exact("", "") == 0.0   # empty == empty is NOT considered a match


class TestExtractPairFeatures:
    BASE_R1 = {
        "name_clean": "acme corporation",
        "name_core":  "acme",
        "name_sorted": "acme",
        "name_nospace": "acme",
        "name_translit": "acm",
        "name_dba": "",
        "name_acronym": "a",
        "name_legal_form": "corp",
        "name_n_tokens": 1,
        "addr_clean": "123 main street new york",
        "addr_sorted": "123 main new street york",
        "addr_words":  "main street new york",
        "addr_numbers": "123",
        "addr_translit": "main nw streit york",
        "addr_postcode": "10001",
        "addr_city_guess": "new york",
        "addr_landmark": "",
        "addr_n_tokens": 5,
        "addr_missing": 0,
        "country_norm": "us",
        "name_missing": 0,
    }

    def _pair(self, overrides=None):
        r2 = dict(self.BASE_R1)
        if overrides:
            r2.update(overrides)
        return extract_pair_features(self.BASE_R1, r2)

    def test_self_similarity_high(self):
        f = self._pair()
        assert f["name_tok_set"] == pytest.approx(1.0)
        assert f["addr_tok_set"] == pytest.approx(1.0)
        assert f["same_country"] == pytest.approx(1.0)

    def test_different_names_low_score(self):
        f = self._pair({"name_clean": "xyz holdings", "name_core": "xyz",
                        "name_sorted": "xyz", "name_nospace": "xyz",
                        "name_translit": "xiz"})
        assert f["name_tok_set"] < 0.5

    def test_postcode_match_flag(self):
        f = self._pair({"addr_postcode": "10001"})
        assert f["addr_postcode_exact"] == 1.0

    def test_postcode_mismatch_flag(self):
        f = self._pair({"addr_postcode": "99999"})
        assert f["addr_postcode_exact"] == 0.0

    def test_all_features_present(self):
        f = self._pair()
        required = [
            "name_tok_set", "name_jaro", "name_sorted_exact",
            "addr_tok_set", "addr_postcode_exact", "addr_city_exact",
            "same_country", "legal_form_exact", "name_addr_mean",
        ]
        for key in required:
            assert key in f, f"Missing feature: {key}"

    def test_feature_values_in_range(self):
        f = self._pair()
        for k, v in f.items():
            assert 0.0 <= v <= 1.0 or isinstance(v, int), \
                f"Feature {k}={v} out of [0,1]"


class TestBuildFeatureMatrix:
    def test_output_shape(self):
        from preprocess import preprocess_source
        rows = [
            {"entity_id": "S1-001", "business_name": "Acme Corp",
             "business_address": "123 Main St NY 10001", "country": "US"},
        ]
        rows2 = [
            {"entity_id": "S2-001", "business_name": "Acme Corporation",
             "business_address": "123 Main Street NY 10001", "country": "US"},
        ]
        s1 = preprocess_source(pd.DataFrame(rows))
        s2 = preprocess_source(pd.DataFrame(rows2))
        s3 = preprocess_source(pd.DataFrame(rows2))
        cands = pd.DataFrame({"s1_id": ["S1-001"], "candidate_id": ["S2-001"]})
        feat_df = build_feature_matrix(cands, s1, s2, s3, show_progress=False)
        assert len(feat_df) == 1
        assert "name_tok_set" in feat_df.columns
        assert "s1_id" in feat_df.columns
