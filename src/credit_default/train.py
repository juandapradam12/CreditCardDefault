"""End-to-end training, model comparison, and artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import basic_clean, load_raw
from .evaluate import (
    classification_metrics,
    metrics_table,
    predict_proba_positive,
    tune_threshold,
)
from .features import split_xy
from .models import FittedModel, fit_pipeline, get_model_candidates, make_pipeline

RANDOM_STATE = 1738


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    cv_folds: int = 3,
) -> tuple[pd.DataFrame, dict[str, FittedModel], dict[str, Any]]:
    """Train several classifiers and return leaderboard + fitted winners."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = get_model_candidates(y_train.to_numpy())
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    cv_rows: list[dict[str, Any]] = []
    fitted: dict[str, FittedModel] = {}
    test_metrics: dict[str, dict[str, float]] = {}

    for name, estimator in candidates.items():
        print(f"Training {name}...", flush=True)
        pipe = make_pipeline(estimator, X_train)

        # Out-of-fold probabilities for threshold selection without peeking at test.
        # n_jobs=1 avoids nested parallelism with tree models that already use threads.
        oof_prob = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold = tune_threshold(y_train, oof_prob, optimize="f1")
        oof_metrics = classification_metrics(y_train, oof_prob, threshold=threshold)

        # Refit on full training fold and evaluate hold-out test.
        pipe = make_pipeline(get_model_candidates(y_train.to_numpy())[name], X_train)
        pipe = fit_pipeline(pipe, X_train, y_train)
        test_prob = predict_proba_positive(pipe, X_test)
        holdout = classification_metrics(y_test, test_prob, threshold=threshold)

        fitted[name] = FittedModel(
            name=name,
            pipeline=pipe,
            best_threshold=threshold,
            metrics=holdout,
        )
        test_metrics[name] = holdout
        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc": oof_metrics["roc_auc"],
                "cv_pr_auc": oof_metrics["pr_auc"],
                "cv_f1": oof_metrics["f1"],
                "test_roc_auc": holdout["roc_auc"],
                "test_pr_auc": holdout["pr_auc"],
                "test_f1": holdout["f1"],
                "test_recall": holdout["recall"],
                "test_precision": holdout["precision"],
                "threshold": threshold,
            }
        )
        print(
            f"  done — test ROC-AUC={holdout['roc_auc']:.4f}, "
            f"PR-AUC={holdout['pr_auc']:.4f}, F1={holdout['f1']:.4f}",
            flush=True,
        )

    leaderboard = pd.DataFrame(cv_rows).sort_values(
        by=["test_roc_auc", "test_pr_auc", "test_f1"],
        ascending=False,
    ).reset_index(drop=True)

    summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
        "best_model": leaderboard.iloc[0]["model"],
        "features": list(X.columns),
    }
    return leaderboard, fitted, summary


def run_training(
    data_path: str | Path | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Load data, train models, and write leaderboard + best model artifacts."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = basic_clean(load_raw(data_path))
    X, y = split_xy(raw)

    leaderboard, fitted, summary = compare_models(X, y)
    best_name = summary["best_model"]
    best = fitted[best_name]

    leaderboard_path = artifacts_dir / "model_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    model_path = artifacts_dir / "best_model.joblib"
    joblib.dump(
        {
            "name": best.name,
            "pipeline": best.pipeline,
            "threshold": best.best_threshold,
            "metrics": best.metrics,
            "feature_names": list(X.columns),
        },
        model_path,
    )

    metrics_path = artifacts_dir / "best_model_metrics.json"
    payload = {
        "summary": summary,
        "best_model": best.name,
        "threshold": best.best_threshold,
        "test_metrics": best.metrics,
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    # Also dump a compact markdown-friendly table for docs regeneration.
    pretty = metrics_table({k: v.metrics for k, v in fitted.items()})
    pretty.to_csv(artifacts_dir / "test_metrics.csv")

    return {
        "leaderboard": leaderboard,
        "summary": summary,
        "best_model": best.name,
        "artifacts": {
            "leaderboard": str(leaderboard_path),
            "model": str(model_path),
            "metrics": str(metrics_path),
        },
    }


if __name__ == "__main__":
    result = run_training()
    print(result["leaderboard"].to_string(index=False))
    print(f"\nBest model: {result['best_model']}")
    print(f"Artifacts: {result['artifacts']}")
