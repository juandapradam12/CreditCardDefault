#!/usr/bin/env python3
"""Score sample clients with the saved best model artifact (+ optional SHAP)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.data import basic_clean, load_raw  # noqa: E402
from credit_default.explain import local_explanation  # noqa: E402
from credit_default.features import engineer_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="Number of sample clients")
    parser.add_argument(
        "--explain-id",
        type=int,
        default=None,
        help="Optional client ID for a local SHAP breakdown",
    )
    parser.add_argument(
        "--threshold",
        choices=["cost", "f1"],
        default="cost",
        help="Which saved operating point to use",
    )
    args = parser.parse_args()

    model_path = ROOT / "artifacts" / "best_model.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"Missing {model_path}. Run `python scripts/train_models.py` first."
        )

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    if args.threshold == "f1" and bundle.get("threshold_f1") is not None:
        threshold = float(bundle["threshold_f1"])
    else:
        threshold = float(bundle.get("threshold_cost", bundle["threshold"]))

    raw = basic_clean(load_raw())
    sample = raw.head(args.n)
    ids = sample["ID"] if "ID" in sample.columns else pd.Series(range(1, len(sample) + 1))
    X = engineer_features(sample)
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "ID": ids.values,
            "default_probability": proba.round(4),
            "predicted_default": pred,
            "threshold": threshold,
            "threshold_policy": args.threshold,
        }
    )
    print(f"Model: {bundle['name']}")
    if bundle.get("calibration"):
        print(f"Calibration: {bundle['calibration']}")
    if bundle.get("cost_params"):
        print(f"Cost params: {bundle['cost_params']}")
    print(out.to_string(index=False))

    explain_id = args.explain_id
    if explain_id is not None:
        row_raw = raw.loc[raw["ID"] == explain_id]
        if row_raw.empty:
            raise SystemExit(f"No client with ID={explain_id}")
        row_x = engineer_features(row_raw)
        local = local_explanation(pipe, row_x)
        print(f"\nLocal SHAP drivers for ID={explain_id}:")
        print(local.to_string(index=False))


if __name__ == "__main__":
    main()
