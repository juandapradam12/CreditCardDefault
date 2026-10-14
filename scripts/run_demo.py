#!/usr/bin/env python3
"""Launch the Streamlit credit-default scoring demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def main() -> None:
    if not (ROOT / "artifacts" / "best_model.joblib").exists():
        raise SystemExit(
            "Missing artifacts/best_model.joblib. Run `python scripts/train_models.py` first."
        )
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.headless=true",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
