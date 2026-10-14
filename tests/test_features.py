"""Feature engineering smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.data import basic_clean, load_raw
from credit_default.features import engineer_features, split_xy


def test_load_raw_shape():
    raw = load_raw()
    assert raw.shape[0] == 30000
    assert "default payment next month" in raw.columns or "default" in basic_clean(raw).columns


def test_engineer_features_no_nulls_and_expected_columns():
    raw = basic_clean(load_raw()).head(200)
    X, y = split_xy(raw)
    assert len(X) == len(y) == 200
    assert not X.isna().any().any()
    for col in ["PAY_MEAN", "UTIL_MEAN", "PAY_RATIO_MEAN", "STRESS_SCORE", "FEMALE"]:
        assert col in X.columns
    assert "ID" not in X.columns
    assert "default" not in X.columns
    assert set(y.unique()).issubset({0, 1})


def test_education_and_marriage_harmonized():
    raw = basic_clean(load_raw()).head(500).copy()
    raw.loc[raw.index[0], "EDUCATION"] = 6
    raw.loc[raw.index[1], "MARRIAGE"] = 0
    feats = engineer_features(raw)
    assert set(feats["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(feats["MARRIAGE"].unique()).issubset({1, 2, 3})
    assert np.isfinite(feats["UTIL_MEAN"]).all()
