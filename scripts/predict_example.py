#!/usr/bin/env python3
"""Score sample clients with the saved best model artifact."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.data import basic_clean, load_raw  # noqa: E402
from credit_default.features import engineer_features  # noqa: E402


def main() -> None:
    model_path = ROOT / "artifacts" / "best_model.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"Missing {model_path}. Run `python scripts/train_models.py` first."
        )

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    threshold = float(bundle["threshold"])

    raw = basic_clean(load_raw()).head(10)
    ids = raw["ID"] if "ID" in raw.columns else pd.Series(range(1, len(raw) + 1))
    X = engineer_features(raw)
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame(
        {
            "ID": ids.values,
            "default_probability": proba.round(4),
            "predicted_default": pred,
            "threshold": threshold,
        }
    )
    print(f"Model: {bundle['name']}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
