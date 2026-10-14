"""Evaluation helpers: metrics, threshold tuning, and reporting."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Default business costs: missing a default is more expensive than a false alarm.
DEFAULT_FP_COST = 1.0
DEFAULT_FN_COST = 5.0


def predict_proba_positive(model, X) -> np.ndarray:
    """Return P(default=1) from a fitted classifier or pipeline."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Map decision scores to (0,1) with a logistic squash for threshold search.
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError("Model does not support probabilistic predictions.")


def expected_cost(
    y_true,
    y_prob,
    threshold: float,
    *,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> float:
    """Total decision cost = fp_cost * FP + fn_cost * FN."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp_cost * fp + fn_cost * fn)


def classification_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
    *,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> dict[str, float]:
    """Compute ranking, operating-point, calibration, and cost metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    cost = float(fp_cost * fp + fn_cost * fn)

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "expected_cost": cost,
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def tune_threshold(
    y_true,
    y_prob,
    *,
    min_precision: float | None = None,
    optimize: str = "f1",
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> float:
    """Pick an operating threshold on a validation probability vector.

    Modes:
      - ``f1``: maximize F1 on the default class
      - ``recall``: maximize recall
      - ``cost``: minimize fp_cost*FP + fn_cost*FN
      - optional ``min_precision`` floor with maximum recall among eligible cutoffs
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if optimize == "cost":
        # Quantile grid keeps the search fast while covering the score distribution.
        thresholds = np.unique(
            np.concatenate(
                (
                    [0.0, 1.0],
                    np.quantile(y_prob, np.linspace(0.0, 1.0, 201)),
                )
            )
        )
        costs = np.array(
            [expected_cost(y_true, y_prob, t, fp_cost=fp_cost, fn_cost=fn_cost) for t in thresholds]
        )
        return float(thresholds[int(np.argmin(costs))])

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    if min_precision is not None:
        eligible = precision >= min_precision
        if eligible.any():
            idx = np.where(eligible)[0][np.argmax(recall[eligible])]
            return float(thresholds[idx])

    if optimize == "f1":
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        return float(thresholds[int(np.argmax(f1))])

    if optimize == "recall":
        return float(thresholds[int(np.argmax(recall))])

    raise ValueError(f"Unknown optimize mode: {optimize}")


def metrics_table(
    results: dict[str, dict[str, float]],
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pretty DataFrame of model metrics sorted by ROC-AUC then PR-AUC."""
    columns = list(
        columns
        or [
            "roc_auc",
            "pr_auc",
            "f1",
            "recall",
            "precision",
            "brier",
            "expected_cost",
            "accuracy",
            "threshold",
        ]
    )
    frame = pd.DataFrame(results).T
    keep = [c for c in columns if c in frame.columns]
    frame = frame[keep].sort_values(by=["roc_auc", "pr_auc"], ascending=False)
    return frame.round(4)
