"""Threshold tuning and metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.evaluate import (
    classification_metrics,
    expected_cost,
    tune_threshold,
)


def test_tune_threshold_f1_between_0_and_1():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    proba = np.clip(y * 0.7 + rng.normal(0, 0.15, size=100), 0, 1)
    thr = tune_threshold(y, proba, optimize="f1")
    assert 0.0 <= thr <= 1.0


def test_cost_threshold_prefers_recall_when_fn_expensive():
    rng = np.random.default_rng(1)
    y = np.array([0] * 70 + [1] * 30)
    # Higher scores for positives, with overlap.
    proba = np.clip(
        np.where(y == 1, rng.normal(0.65, 0.15, size=100), rng.normal(0.35, 0.15, size=100)),
        0,
        1,
    )
    thr_balanced = tune_threshold(y, proba, optimize="cost", fp_cost=1, fn_cost=1)
    thr_fn_heavy = tune_threshold(y, proba, optimize="cost", fp_cost=1, fn_cost=20)
    # With expensive false negatives, the optimal cutoff should not be higher
    # (typically lower → more positives flagged).
    assert thr_fn_heavy <= thr_balanced + 1e-9


def test_expected_cost_matches_confusion_counts():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.8, 0.2, 0.9])
    # threshold 0.5 => pred [0,1,0,1] => FP=1, FN=1
    assert expected_cost(y, proba, 0.5, fp_cost=2, fn_cost=3) == 2 * 1 + 3 * 1
    metrics = classification_metrics(y, proba, threshold=0.5, fp_cost=2, fn_cost=3)
    assert metrics["expected_cost"] == 5
    assert "brier" in metrics
    assert 0 <= metrics["roc_auc"] <= 1
