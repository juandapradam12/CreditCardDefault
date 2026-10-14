"""End-to-end pipeline smoke tests on a data subset."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.calibration import fit_calibrated_pipeline
from credit_default.data import basic_clean, load_raw
from credit_default.evaluate import classification_metrics, predict_proba_positive, tune_threshold
from credit_default.explain import explain_model
from credit_default.features import split_xy
from credit_default.models import fit_pipeline, get_model_candidates, make_pipeline
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def sample_xy():
    raw = basic_clean(load_raw()).sample(n=2500, random_state=1738)
    return split_xy(raw)


def test_train_calibrate_predict_roundtrip(sample_xy, tmp_path):
    X, y = sample_xy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1738, stratify=y
    )
    model = get_model_candidates(y_train.to_numpy())["hist_gradient_boosting"]
    base = fit_pipeline(make_pipeline(model, X_train), X_train, y_train)
    calibrated = fit_calibrated_pipeline(base, X_train, y_train, method="isotonic", cv_folds=2)

    proba = predict_proba_positive(calibrated, X_test)
    assert proba.shape == (len(X_test),)
    assert np.all((proba >= 0) & (proba <= 1))

    thr = tune_threshold(y_test, proba, optimize="cost", fp_cost=1, fn_cost=5)
    metrics = classification_metrics(y_test, proba, threshold=thr, fp_cost=1, fn_cost=5)
    assert metrics["roc_auc"] >= 0.70
    assert metrics["pr_auc"] >= 0.35

    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {
            "name": "hist_gradient_boosting",
            "pipeline": calibrated,
            "threshold": thr,
            "threshold_cost": thr,
            "feature_names": list(X.columns),
        },
        artifact,
    )
    bundle = joblib.load(artifact)
    reloaded = predict_proba_positive(bundle["pipeline"], X_test.head(5))
    assert reloaded.shape == (5,)


def test_shap_importance_nonempty(sample_xy):
    X, y = sample_xy
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=1738, stratify=y
    )
    model = get_model_candidates(y_train.to_numpy())["hist_gradient_boosting"]
    pipe = fit_pipeline(make_pipeline(model, X_train), X_train, y_train)
    result = explain_model(pipe, X_train, max_samples=120)
    assert isinstance(result["importances"], pd.DataFrame)
    assert len(result["importances"]) > 0
    assert result["importances"]["mean_abs_shap"].iloc[0] >= 0
