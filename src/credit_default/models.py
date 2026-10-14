"""Model zoo and training utilities for credit default prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


@dataclass
class FittedModel:
    name: str
    pipeline: Pipeline
    best_threshold: float
    metrics: dict[str, float]
    threshold_f1: float | None = None
    threshold_cost: float | None = None
    calibration: dict[str, float] | None = None


def _pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y)
    n_pos = max((y == 1).sum(), 1)
    n_neg = max((y == 0).sum(), 1)
    return float(n_neg / n_pos)


def get_model_candidates(y_train: np.ndarray | None = None) -> dict[str, Any]:
    """Return a dictionary of ready-to-fit estimators with sensible defaults."""
    spw = _pos_weight(y_train) if y_train is not None else 1.0

    # Keep estimator counts moderate and n_jobs conservative so nested CV
    # does not oversubscribe CPU/RAM on laptop-sized machines.
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=1738,
            n_jobs=2,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=1738,
            n_jobs=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_depth=6,
            max_iter=200,
            random_state=1738,
            class_weight="balanced",
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=spw,
            random_state=1738,
            n_jobs=2,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=1738,
            n_jobs=2,
            verbose=-1,
        )

    return models


def make_pipeline(estimator: Any, X_sample) -> Pipeline:
    """Wrap an estimator with the project's standard preprocessor."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_sample)),
            ("model", estimator),
        ]
    )


def fit_pipeline(pipeline: Pipeline, X, y) -> Pipeline:
    """Fit the full preprocessing + model pipeline."""
    pipeline.fit(X, y)
    return pipeline
