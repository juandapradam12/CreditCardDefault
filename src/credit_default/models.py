"""Model zoo and training utilities for credit default prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
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

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


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


def _tuned_xgboost(spw: float) -> Any:
    """XGBoost defaults from Optuna (maximize CV ROC-AUC on this dataset)."""
    return XGBClassifier(
        n_estimators=573,
        learning_rate=0.01948459411873672,
        max_depth=3,
        min_child_weight=6.030921671484922,
        subsample=0.8539760134155758,
        colsample_bytree=0.9334914716395416,
        reg_lambda=0.05825931002811421,
        reg_alpha=1.1319021382524528,
        gamma=3.5447383293760906,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        random_state=1738,
        n_jobs=2,
    )


def get_model_candidates(y_train: np.ndarray | None = None) -> dict[str, Any]:
    """Return a dictionary of ready-to-fit estimators with strong defaults."""
    spw = _pos_weight(y_train) if y_train is not None else 1.0

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=1738,
            n_jobs=2,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=1738,
            n_jobs=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=5,
            max_iter=350,
            l2_regularization=0.1,
            random_state=1738,
            class_weight="balanced",
        ),
    }

    boosters: list[tuple[str, Any]] = []

    if XGBClassifier is not None:
        xgb = _tuned_xgboost(spw)
        models["xgboost"] = xgb
        boosters.append(("xgb", xgb))

    if LGBMClassifier is not None:
        lgb = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=24,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=1738,
            n_jobs=2,
            verbose=-1,
        )
        models["lightgbm"] = lgb
        boosters.append(("lgb", lgb))

    if CatBoostClassifier is not None:
        cat = CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=1738,
            verbose=False,
            thread_count=2,
        )
        models["catboost"] = cat
        boosters.append(("cat", cat))

    if len(boosters) >= 2:
        # Soft vote and stacking usually edge out any single booster on this benchmark.
        models["boosting_soft_vote"] = VotingClassifier(
            estimators=[(name, clf) for name, clf in boosters],
            voting="soft",
            weights=[2 if name == "xgb" else 1 for name, _ in boosters],
        )
        models["boosting_stack"] = StackingClassifier(
            estimators=[(name, clf) for name, clf in boosters],
            final_estimator=LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
            stack_method="predict_proba",
            passthrough=False,
            cv=3,
            n_jobs=1,
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
