#!/usr/bin/env python3
"""CLI entrypoint: train and compare credit-default models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    args = parser.parse_args()

    result = run_training(data_path=args.data, artifacts_dir=args.artifacts)
    print(result["leaderboard"].to_string(index=False))
    print(f"\nBest model: {result['best_model']}")
    for key, path in result["artifacts"].items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
