"""Helpers for loading the saved model and scoring individual clients."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd

from .data import basic_clean, load_raw
from .explain import local_explanation
from .features import engineer_features

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"


def load_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load the trained artifact bundle written by ``run_training``."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model artifact at {path}. Run `python scripts/train_models.py` first."
        )
    return joblib.load(path)


def resolve_threshold(
    bundle: dict[str, Any],
    policy: Literal["cost", "f1"] = "cost",
) -> float:
    """Return the operating threshold for the requested policy."""
    if policy == "f1" and bundle.get("threshold_f1") is not None:
        return float(bundle["threshold_f1"])
    if bundle.get("threshold_cost") is not None:
        return float(bundle["threshold_cost"])
    return float(bundle["threshold"])


def load_client_table() -> pd.DataFrame:
    """Return the cleaned raw client table used by demos."""
    return basic_clean(load_raw())


def score_client_frame(
    bundle: dict[str, Any],
    raw_rows: pd.DataFrame,
    *,
    policy: Literal["cost", "f1"] = "cost",
) -> pd.DataFrame:
    """Score one or more raw client rows and return probabilities + decisions."""
    pipe = bundle["pipeline"]
    threshold = resolve_threshold(bundle, policy=policy)
    X = engineer_features(raw_rows)
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = raw_rows.copy()
    if "ID" not in out.columns:
        out.insert(0, "ID", range(1, len(out) + 1))
    out["default_probability"] = proba
    out["predicted_default"] = pred
    out["threshold"] = threshold
    out["threshold_policy"] = policy
    return out


def explain_client(
    bundle: dict[str, Any],
    raw_row: pd.DataFrame,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Local SHAP drivers for a single-row raw client frame."""
    if len(raw_row) != 1:
        raise ValueError("explain_client expects exactly one raw client row")
    X = engineer_features(raw_row)
    return local_explanation(bundle["pipeline"], X, top_k=top_k)


def get_client_by_id(raw: pd.DataFrame, client_id: int) -> pd.DataFrame:
    """Select a single client by ID."""
    if "ID" not in raw.columns:
        raise KeyError("Raw table has no ID column")
    row = raw.loc[raw["ID"] == client_id]
    if row.empty:
        raise KeyError(f"No client with ID={client_id}")
    return row.copy()
