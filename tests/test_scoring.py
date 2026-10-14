"""Tests for the shared scoring helpers used by CLI and Streamlit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.scoring import (
    explain_client,
    get_client_by_id,
    load_bundle,
    load_client_table,
    resolve_threshold,
    score_client_frame,
)


@pytest.fixture(scope="module")
def bundle_and_clients():
    model_path = ROOT / "artifacts" / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("best_model.joblib not present")
    return load_bundle(model_path), load_client_table()


def test_resolve_threshold_policies(bundle_and_clients):
    bundle, _ = bundle_and_clients
    cost_thr = resolve_threshold(bundle, "cost")
    f1_thr = resolve_threshold(bundle, "f1")
    assert 0.0 <= cost_thr <= 1.0
    assert 0.0 <= f1_thr <= 1.0


def test_score_and_explain_client(bundle_and_clients):
    bundle, clients = bundle_and_clients
    row = get_client_by_id(clients, int(clients["ID"].iloc[0]))
    scored = score_client_frame(bundle, row, policy="cost")
    assert len(scored) == 1
    assert 0.0 <= float(scored["default_probability"].iloc[0]) <= 1.0
    assert int(scored["predicted_default"].iloc[0]) in {0, 1}

    local = explain_client(bundle, row, top_k=5)
    assert len(local) == 5
    assert {"feature", "shap_value", "abs_shap"} <= set(local.columns)


def test_get_client_by_id_missing(bundle_and_clients):
    _, clients = bundle_and_clients
    with pytest.raises(KeyError):
        get_client_by_id(clients, client_id=-999999)
