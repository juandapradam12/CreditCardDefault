#!/usr/bin/env python3
"""CLI entrypoint: train, calibrate, explain, and compare credit-default models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_default.evaluate import DEFAULT_FN_COST, DEFAULT_FP_COST  # noqa: E402
from credit_default.train import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=None,
        help="Optional path to CSV/XLS. Defaults to data/credit_card_defaults.csv",
    )
    parser.add_argument(
        "--artifacts",
        default=str(ROOT / "artifacts"),
        help="Directory for model artifacts and metrics",
    )
    parser.add_argument("--fp-cost", type=float, default=DEFAULT_FP_COST)
    parser.add_argument("--fn-cost", type=float, default=DEFAULT_FN_COST)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    result = run_training(
        data_path=args.data,
        artifacts_dir=args.artifacts,
        fp_cost=args.fp_cost,
        fn_cost=args.fn_cost,
        calibrate=not args.no_calibrate,
        compute_shap=not args.no_shap,
    )
    print(result["leaderboard"].to_string(index=False))
    print(f"\nBest model: {result['best_model']}")
    best = result["best"]
    print(
        f"Thresholds — cost: {best.threshold_cost:.4f} | f1: {best.threshold_f1:.4f}"
    )
    if best.calibration:
        print(
            "Calibration Brier: "
            f"{best.calibration.get('brier_before', float('nan')):.4f} → "
            f"{best.calibration.get('brier_after', float('nan')):.4f}"
        )
    for key, path in result["artifacts"].items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
