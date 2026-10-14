"""Probability calibration utilities for default risk scores."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .evaluate import predict_proba_positive
from .models import fit_pipeline


def make_calibrated_pipeline(
    base_pipeline: Pipeline,
    *,
    method: str = "isotonic",
    cv_folds: int = 3,
    random_state: int = 1738,
) -> Pipeline:
    """Return a new pipeline that calibrates the terminal classifier.

    Structure: preprocess -> CalibratedClassifierCV(clone(model)).
    The preprocessor is fitted jointly when the returned pipeline is fit.
    """
    preprocess = clone(base_pipeline.named_steps["preprocess"])
    model = clone(base_pipeline.named_steps["model"])
    calibrator = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", calibrator)])


def fit_calibrated_pipeline(
    base_pipeline: Pipeline,
    X,
    y,
    *,
    method: str = "isotonic",
    cv_folds: int = 3,
    random_state: int = 1738,
) -> Pipeline:
    """Fit a calibrated clone of ``base_pipeline`` on ``X``, ``y``."""
    calibrated = make_calibrated_pipeline(
        base_pipeline,
        method=method,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    return fit_pipeline(calibrated, X, y)


def calibration_report(
    y_true,
    prob_before,
    prob_after,
) -> dict[str, Any]:
    """Compare Brier score before/after calibration (lower is better)."""
    y_true = np.asarray(y_true).astype(int)
    before = np.clip(np.asarray(prob_before, dtype=float), 1e-7, 1 - 1e-7)
    after = np.clip(np.asarray(prob_after, dtype=float), 1e-7, 1 - 1e-7)
    brier_before = float(brier_score_loss(y_true, before))
    brier_after = float(brier_score_loss(y_true, after))
    return {
        "method": "isotonic",
        "brier_before": brier_before,
        "brier_after": brier_after,
        "brier_improvement": brier_before - brier_after,
    }


def score_with_optional_bundle(bundle: dict[str, Any], X) -> np.ndarray:
    """Predict default probabilities from a saved artifact bundle."""
    return predict_proba_positive(bundle["pipeline"], X)
