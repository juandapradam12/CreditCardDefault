"""End-to-end training, model comparison, and artifact export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .calibration import calibration_report, fit_calibrated_pipeline
from .data import basic_clean, load_raw
from .evaluate import (
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    classification_metrics,
    metrics_table,
    predict_proba_positive,
    tune_threshold,
)
from .explain import save_shap_artifacts
from .features import split_xy
from .models import FittedModel, fit_pipeline, get_model_candidates, make_pipeline

RANDOM_STATE = 1738


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    cv_folds: int = 3,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> tuple[pd.DataFrame, dict[str, FittedModel], dict[str, Any], tuple]:
    """Train several classifiers and return leaderboard + fitted winners.

    Also returns ``(X_train, X_test, y_train, y_test)`` for post-processing
    (calibration / SHAP) without reshuffling.
    """
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

    for name, estimator in candidates.items():
        print(f"Training {name}...", flush=True)
        pipe = make_pipeline(estimator, X_train)

        # Out-of-fold probabilities for threshold selection without peeking at test.
        oof_prob = cross_val_predict(
            pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        threshold_f1 = tune_threshold(y_train, oof_prob, optimize="f1")
        threshold_cost = tune_threshold(
            y_train,
            oof_prob,
            optimize="cost",
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        oof_metrics = classification_metrics(
            y_train,
            oof_prob,
            threshold=threshold_cost,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )

        # Refit on full training fold and evaluate hold-out test.
        pipe = make_pipeline(get_model_candidates(y_train.to_numpy())[name], X_train)
        pipe = fit_pipeline(pipe, X_train, y_train)
        test_prob = predict_proba_positive(pipe, X_test)
        holdout = classification_metrics(
            y_test,
            test_prob,
            threshold=threshold_cost,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )
        holdout_f1 = classification_metrics(
            y_test,
            test_prob,
            threshold=threshold_f1,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        )

        fitted[name] = FittedModel(
            name=name,
            pipeline=pipe,
            best_threshold=threshold_cost,
            threshold_f1=threshold_f1,
            threshold_cost=threshold_cost,
            metrics=holdout,
        )
        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc": oof_metrics["roc_auc"],
                "cv_pr_auc": oof_metrics["pr_auc"],
                "cv_f1": oof_metrics["f1"],
                "test_roc_auc": holdout["roc_auc"],
                "test_pr_auc": holdout["pr_auc"],
                "test_ks": holdout["ks"],
                "test_f1": holdout_f1["f1"],
                "test_recall_f1": holdout_f1["recall"],
                "test_precision_f1": holdout_f1["precision"],
                "test_recall_cost": holdout["recall"],
                "test_precision_cost": holdout["precision"],
                "test_expected_cost": holdout["expected_cost"],
                "test_brier": holdout["brier"],
                "threshold_f1": threshold_f1,
                "threshold_cost": threshold_cost,
            }
        )
        print(
            f"  done — ROC-AUC={holdout['roc_auc']:.4f}, "
            f"PR-AUC={holdout['pr_auc']:.4f}, KS={holdout['ks']:.4f}, "
            f"F1={holdout_f1['f1']:.4f}",
            flush=True,
        )

    leaderboard = pd.DataFrame(cv_rows).sort_values(
        by=["test_roc_auc", "test_pr_auc", "test_expected_cost"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "default_rate_train": float(y_train.mean()),
        "default_rate_test": float(y_test.mean()),
        "best_model": leaderboard.iloc[0]["model"],
        "features": list(X.columns),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
    }
    splits = (X_train, X_test, y_train, y_test)
    return leaderboard, fitted, summary, splits


def finalize_best_model(
    fitted: dict[str, FittedModel],
    summary: dict[str, Any],
    splits: tuple,
    *,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    calibrate: bool = True,
) -> FittedModel:
    """Calibrate the winning model and retune F1 / cost thresholds."""
    X_train, X_test, y_train, y_test = splits
    best_name = summary["best_model"]
    base = fitted[best_name]

    raw_test_prob = predict_proba_positive(base.pipeline, X_test)

    # Hold out a threshold-tuning slice so cutoffs are not fit on the final test set
    # or on in-sample training scores.
    X_fit, X_thresh, y_fit, y_thresh = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    if calibrate:
        print(f"Calibrating best model ({best_name}) with isotonic regression...", flush=True)
        pipeline_for_thresh = fit_calibrated_pipeline(
            base.pipeline, X_fit, y_fit, method="isotonic"
        )
        # Refit calibrator on the full training set for the exported artifact.
        pipeline = fit_calibrated_pipeline(base.pipeline, X_train, y_train, method="isotonic")
        cal_test_prob = predict_proba_positive(pipeline, X_test)
        cal_info = calibration_report(y_test, raw_test_prob, cal_test_prob)
        test_prob = cal_test_prob
        print(
            f"  Brier {cal_info['brier_before']:.4f} → {cal_info['brier_after']:.4f}",
            flush=True,
        )
    else:
        pipeline_for_thresh = fit_pipeline(
            make_pipeline(
                get_model_candidates(y_fit.to_numpy())[best_name],
                X_fit,
            ),
            X_fit,
            y_fit,
        )
        pipeline = base.pipeline
        test_prob = raw_test_prob
        cal_info = {
            "method": "none",
            "brier_before": float(classification_metrics(y_test, test_prob)["brier"]),
            "brier_after": float(classification_metrics(y_test, test_prob)["brier"]),
            "brier_improvement": 0.0,
        }

    thresh_prob = predict_proba_positive(pipeline_for_thresh, X_thresh)
    threshold_f1 = tune_threshold(y_thresh, thresh_prob, optimize="f1")
    threshold_cost = tune_threshold(
        y_thresh,
        thresh_prob,
        optimize="cost",
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )

    metrics_cost = classification_metrics(
        y_test,
        test_prob,
        threshold=threshold_cost,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    metrics_f1 = classification_metrics(
        y_test,
        test_prob,
        threshold=threshold_f1,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )

    return FittedModel(
        name=best_name,
        pipeline=pipeline,
        best_threshold=threshold_cost,
        threshold_f1=threshold_f1,
        threshold_cost=threshold_cost,
        metrics={
            **metrics_cost,
            "f1_at_f1_threshold": metrics_f1["f1"],
            "recall_at_f1_threshold": metrics_f1["recall"],
            "precision_at_f1_threshold": metrics_f1["precision"],
            "expected_cost_at_f1_threshold": metrics_f1["expected_cost"],
        },
        calibration=cal_info,
    )


def run_training(
    data_path: str | Path | None = None,
    artifacts_dir: str | Path = "artifacts",
    *,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    calibrate: bool = True,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Load data, train models, calibrate winner, explain, and write artifacts."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw = basic_clean(load_raw(data_path))
    X, y = split_xy(raw)

    leaderboard, fitted, summary, splits = compare_models(
        X,
        y,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
    best = finalize_best_model(
        fitted,
        summary,
        splits,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        calibrate=calibrate,
    )
    summary["best_model"] = best.name
    summary["calibrated"] = calibrate

    leaderboard_path = artifacts_dir / "model_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    model_path = artifacts_dir / "best_model.joblib"
    joblib.dump(
        {
            "name": best.name,
            "pipeline": best.pipeline,
            "threshold": best.best_threshold,
            "threshold_f1": best.threshold_f1,
            "threshold_cost": best.threshold_cost,
            "cost_params": {"fp_cost": fp_cost, "fn_cost": fn_cost},
            "metrics": best.metrics,
            "calibration": best.calibration,
            "feature_names": list(X.columns),
        },
        model_path,
    )

    shap_paths: dict[str, str] = {}
    if compute_shap:
        print("Computing SHAP feature importances...", flush=True)
        # Explain on a training sample using the fitted pipeline.
        X_train = splits[0]
        try:
            shap_paths = save_shap_artifacts(best.pipeline, X_train, artifacts_dir)
            print(f"  wrote {shap_paths}", flush=True)
        except Exception as exc:  # pragma: no cover - environment-specific
            print(f"  SHAP skipped: {exc}", flush=True)

    metrics_path = artifacts_dir / "best_model_metrics.json"
    payload = {
        "summary": summary,
        "best_model": best.name,
        "threshold": best.best_threshold,
        "threshold_f1": best.threshold_f1,
        "threshold_cost": best.threshold_cost,
        "cost_params": {"fp_cost": fp_cost, "fn_cost": fn_cost},
        "calibration": best.calibration,
        "test_metrics": best.metrics,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "shap_artifacts": shap_paths,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    pretty = metrics_table({k: v.metrics for k, v in fitted.items()})
    pretty.to_csv(artifacts_dir / "test_metrics.csv")

    return {
        "leaderboard": leaderboard,
        "summary": summary,
        "best_model": best.name,
        "best": best,
        "artifacts": {
            "leaderboard": str(leaderboard_path),
            "model": str(model_path),
            "metrics": str(metrics_path),
            **shap_paths,
        },
    }


if __name__ == "__main__":
    result = run_training()
    print(result["leaderboard"].to_string(index=False))
    print(f"\nBest model: {result['best_model']}")
    print(f"Artifacts: {result['artifacts']}")
