#!/usr/bin/env python3
"""Rebuild global SHAP artifacts for the saved best model."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.data import basic_clean, load_raw  # noqa: E402
from credit_default.explain import save_shap_artifacts  # noqa: E402
from credit_default.features import split_xy  # noqa: E402


def main() -> None:
    model_path = ROOT / "artifacts" / "best_model.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"Missing {model_path}. Run `python scripts/train_models.py` first."
        )
    bundle = joblib.load(model_path)
    X, _ = split_xy(basic_clean(load_raw()))
    paths = save_shap_artifacts(bundle["pipeline"], X, ROOT / "artifacts")
    print(f"Model: {bundle['name']}")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
