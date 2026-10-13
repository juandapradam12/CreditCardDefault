"""Feature engineering and preprocessing for default risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def _safe_div(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray) -> np.ndarray:
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(numer, denom)
    out[~np.isfinite(out)] = 0.0
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features that capture repayment stress and credit usage.

    Designed to be leakage-safe: uses only fields available at prediction time
    (limit, demographics, and the trailing six-month bill/payment history).
    """
    out = df.copy()

    # Harmonize education / marriage rare codes into "unknown/other".
    if "EDUCATION" in out.columns:
        out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in out.columns:
        out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})

    # Sex as binary female indicator (original: 1=male, 2=female).
    if "SEX" in out.columns:
        out["FEMALE"] = (out["SEX"] == 2).astype(int)
        out = out.drop(columns=["SEX"])

    # Aggregate repayment delay signals.
    pay = out[PAY_COLS]
    out["PAY_MEAN"] = pay.mean(axis=1)
    out["PAY_MAX"] = pay.max(axis=1)
    out["PAY_STD"] = pay.std(axis=1).fillna(0.0)
    out["N_DELAYED"] = (pay > 0).sum(axis=1)
    out["N_ON_TIME"] = (pay <= 0).sum(axis=1)
    out["WORST_RECENT"] = pay.iloc[:, :3].max(axis=1)

    # Utilization and bill dynamics relative to credit limit.
    limit = out["LIMIT_BAL"].replace(0, np.nan)
    for i, col in enumerate(BILL_COLS, start=1):
        out[f"UTIL_{i}"] = _safe_div(out[col], limit)
    util_cols = [f"UTIL_{i}" for i in range(1, 7)]
    out["UTIL_MEAN"] = out[util_cols].mean(axis=1)
    out["UTIL_MAX"] = out[util_cols].max(axis=1)
    out["UTIL_TREND"] = out["UTIL_1"] - out["UTIL_6"]

    bill = out[BILL_COLS]
    out["BILL_MEAN"] = bill.mean(axis=1)
    out["BILL_STD"] = bill.std(axis=1).fillna(0.0)
    out["BILL_TREND"] = out["BILL_AMT1"] - out["BILL_AMT6"]

    # Payment intensity vs. prior bill (classic repayment-capacity proxies).
    for i in range(1, 6):
        out[f"PAY_RATIO_{i}"] = _safe_div(out[f"PAY_AMT{i}"], out[f"BILL_AMT{i + 1}"].abs() + 1.0)
    pay_ratio_cols = [f"PAY_RATIO_{i}" for i in range(1, 6)]
    out["PAY_RATIO_MEAN"] = out[pay_ratio_cols].mean(axis=1)
    out["PAY_AMT_MEAN"] = out[PAY_AMT_COLS].mean(axis=1)
    out["PAY_AMT_STD"] = out[PAY_AMT_COLS].std(axis=1).fillna(0.0)
    out["ZERO_PAY_MONTHS"] = (out[PAY_AMT_COLS] == 0).sum(axis=1)

    # Soft signals of stress: high utilization + recent delays.
    out["STRESS_SCORE"] = out["UTIL_MEAN"] * (1.0 + out["PAY_MEAN"].clip(lower=0))

    drop_cols = [c for c in ["ID", "default"] if c in out.columns]
    return out.drop(columns=drop_cols)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return engineered features and binary default target."""
    if "default" not in df.columns and "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    y = df["default"].astype(int)
    X = engineer_features(df)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Column-wise preprocessing for mixed numeric/categorical features."""
    categorical = [c for c in ["EDUCATION", "MARRIAGE"] if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
