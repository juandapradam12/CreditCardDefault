"""Data loading helpers for the UCI credit-card default dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COL = "default payment next month"
ID_COL = "ID"

# Canonical relative path from the repository root.
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "credit_card_defaults.csv"
DEFAULT_XLS = Path(__file__).resolve().parents[2] / "data" / "default_of_credit_card_clients.xls"


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw UCI Taiwan credit-card default dataset.

    Prefers CSV for speed/reliability; falls back to the original Excel file.
    """
    if path is not None:
        path = Path(path)
        if path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(path, header=1)
        return pd.read_csv(path)

    if DEFAULT_CSV.exists():
        return pd.read_csv(DEFAULT_CSV)

    if DEFAULT_XLS.exists():
        return pd.read_excel(DEFAULT_XLS, header=1)

    # Legacy root locations from the original repository layout.
    root = Path(__file__).resolve().parents[2]
    legacy_xls = root / "default of credit card clients.xls"
    if legacy_xls.exists():
        return pd.read_excel(legacy_xls, header=1)

    raise FileNotFoundError(
        "Could not find credit card default data. Expected "
        f"{DEFAULT_CSV} or {DEFAULT_XLS}."
    )


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and categorical codes used downstream."""
    out = df.copy()
    if "PAY_0" in out.columns:
        out = out.rename(columns={"PAY_0": "PAY_1"})
    if TARGET_COL in out.columns:
        out = out.rename(columns={TARGET_COL: "default"})
    return out
