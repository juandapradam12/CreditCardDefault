"""SHAP-based explainability for credit default models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None


def _transformed_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor and return a named DataFrame."""
    preprocess = pipeline.named_steps["preprocess"]
    Xt = preprocess.transform(X)
    try:
        names = list(preprocess.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    # Strip ColumnTransformer prefixes for readability.
    clean = [n.split("__", 1)[-1] for n in names]
    return pd.DataFrame(Xt, columns=clean, index=getattr(X, "index", None))


def _unwrap_estimator(pipeline):
    """Return the underlying predictive estimator (unwrap calibration if present)."""
    model = pipeline.named_steps["model"]
    # CalibratedClassifierCV stores fitted calibrators per fold.
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        # Use the first fold's base estimator for TreeExplainer compatibility.
        first = model.calibrated_classifiers_[0]
        if hasattr(first, "estimator"):
            return first.estimator
        if hasattr(first, "base_estimator"):
            return first.base_estimator
    return model


def explain_model(
    pipeline,
    X: pd.DataFrame,
    *,
    max_samples: int = 500,
    random_state: int = 1738,
) -> dict[str, Any]:
    """Compute SHAP values for a fitted preprocess+model pipeline.

    Returns mean |SHAP| importances and the raw Explanation object when available.
    """
    if shap is None:
        raise ImportError("shap is required for explainability. Install with: pip install shap")

    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=random_state)
    else:
        X_sample = X

    X_trans = _transformed_frame(pipeline, X_sample)
    estimator = _unwrap_estimator(pipeline)

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans)
        # Binary classifiers may return a list [class0, class1].
        if isinstance(shap_values, list):
            shap_matrix = np.asarray(shap_values[1])
        else:
            shap_matrix = np.asarray(shap_values)
            if shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, :, 1]
    except Exception:
        # Model-agnostic fallback (slower, but works for calibrated wrappers).
        predict = lambda data: pipeline.predict_proba(  # noqa: E731
            pd.DataFrame(data, columns=X_sample.columns)
        )[:, 1]
        background = shap.sample(X_sample, min(100, len(X_sample)), random_state=random_state)
        explainer = shap.KernelExplainer(predict, background)
        shap_matrix = np.asarray(explainer.shap_values(X_sample, nsamples=100))
        X_trans = X_sample

    importances = (
        pd.DataFrame(
            {
                "feature": list(X_trans.columns),
                "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "importances": importances,
        "shap_values": shap_matrix,
        "feature_frame": X_trans,
        "sample_index": list(X_sample.index),
    }


def save_shap_artifacts(
    pipeline,
    X: pd.DataFrame,
    artifacts_dir: str | Path,
    *,
    max_samples: int = 500,
) -> dict[str, str]:
    """Write SHAP importance CSV and summary plot under ``artifacts_dir``."""
    import matplotlib.pyplot as plt

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    result = explain_model(pipeline, X, max_samples=max_samples)
    importances: pd.DataFrame = result["importances"]
    csv_path = artifacts_dir / "shap_feature_importance.csv"
    importances.to_csv(csv_path, index=False)

    plot_path = artifacts_dir / "shap_summary.png"
    top = importances.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#1B4965")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Top features driving default risk")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    return {"importance_csv": str(csv_path), "summary_plot": str(plot_path)}


def local_explanation(
    pipeline,
    X_row: pd.DataFrame,
    *,
    top_k: int = 8,
) -> pd.DataFrame:
    """Return the top local SHAP contributions for a single client row."""
    result = explain_model(pipeline, X_row, max_samples=1)
    shap_row = np.asarray(result["shap_values"][0])
    features = list(result["feature_frame"].columns)
    frame = pd.DataFrame({"feature": features, "shap_value": shap_row})
    frame["abs_shap"] = frame["shap_value"].abs()
    return frame.sort_values("abs_shap", ascending=False).head(top_k).reset_index(drop=True)
