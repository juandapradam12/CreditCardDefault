# Documentation — Credit Card Default Risk Modeling

## 1. Problem statement

Banks and consumer lenders need an early signal that a revolving credit account is likely to miss its next payment. False negatives (missed defaults) create credit loss; false positives (flagging healthy accounts) create unnecessary collections cost and customer friction.

This project frames the problem as **binary classification**:

> Given demographics, credit limit, and the last six months of repayment / bill / payment history, will the client default next month?

## 2. Dataset

Source: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) (Yeh & Lien, 2009), Taiwanese credit-card portfolio.

| Property | Value |
|---|---|
| Rows | 30,000 clients |
| Raw columns | 23 predictors + ID + target |
| Target | `default payment next month` ∈ {0, 1} |
| Positive rate | ~22.1% |
| Missing values | 0 |

Key raw fields:

- `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- `PAY_1` … `PAY_6` — repayment status history (delay codes)
- `BILL_AMT1` … `BILL_AMT6` — monthly bill statements
- `PAY_AMT1` … `PAY_AMT6` — monthly amounts paid

Notes from exploratory analysis:

- Codebook education values are 1–4, but the file also contains 0, 5, 6. These are mapped to a single “other/unknown” bucket.
- Marriage value `0` is treated as “other”.
- `PAY_*` codes include `-2` and `0`, which are not fully described in every codebook version; they are retained as ordinal repayment signals.

Local copies live in `data/credit_card_defaults.csv` (preferred) and `data/default_of_credit_card_clients.xls` (original).

## 3. Project layout

```text
CreditCardDefault/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── credit_card_defaults.csv
│   └── default_of_credit_card_clients.xls
├── src/credit_default/
│   ├── data.py          # loading / light cleaning
│   ├── features.py      # feature engineering + preprocessing
│   ├── models.py        # model zoo
│   ├── evaluate.py      # metrics + threshold tuning
│   └── train.py         # comparison / artifact export
├── scripts/
│   ├── train_models.py
│   └── predict_example.py
├── notebooks/
│   ├── credit_default_risk_modeling.ipynb   # enhanced analysis
│   └── original_default_of_credit_card_clients.ipynb
├── artifacts/           # leaderboard + fitted best model
└── docs/DOCUMENTATION.md
```

## 4. Methodology

### 4.1 Validation design

1. Stratified train / test split (75% / 25%, `random_state=1738`).
2. Stratified 3-fold CV on the training set to obtain out-of-fold probabilities.
3. Threshold selected on OOF probabilities by maximizing F1 for the default class.
4. Final model refit on the full training set and scored once on the hold-out test set.

This avoids tuning the operating point directly on the test set.

### 4.2 Feature engineering

Leakage-safe engineered features include:

- delay aggregates: `PAY_MEAN`, `PAY_MAX`, `PAY_STD`, `N_DELAYED`, `WORST_RECENT`
- utilization: bill amount / credit limit per month, plus mean / max / trend
- repayment capacity: payment / previous bill ratios, zero-payment month counts
- composite `STRESS_SCORE = UTIL_MEAN * (1 + max(PAY_MEAN, 0))`

Categorical fields (`EDUCATION`, `MARRIAGE`) are one-hot encoded; numeric fields are median-imputed and standardized inside a scikit-learn `ColumnTransformer`.

### 4.3 Algorithms

| Model | Role |
|---|---|
| Logistic Regression (balanced) | Strong linear baseline / interpretable coefficients after scaling |
| Random Forest / Extra Trees | Bagged trees with class weighting (original project family) |
| HistGradientBoosting | Fast sklearn boosting baseline |
| XGBoost / LightGBM | Modern gradient boosting with imbalance handling |

Improvements versus the original notebook:

- stratified splits instead of an unstratified 70/30 split
- ranking metrics (ROC-AUC, PR-AUC), not only classification reports
- class-weight / `scale_pos_weight` for the minority default class
- engineered repayment-stress features
- F1-oriented threshold tuning instead of default 0.5 / tiny forests that inflate recall
- reproducible package + CLI artifacts

### 4.4 Metrics

Because defaults are uncommon (~22%), accuracy alone is misleading.

| Metric | Why it matters |
|---|---|
| ROC-AUC | Threshold-free ranking quality |
| PR-AUC | Ranking quality under class imbalance |
| Precision / Recall / F1 | Operating-point quality after threshold tuning |
| Specificity | Impact on non-default customers |

## 5. Benchmark results

Hold-out test results from `python scripts/train_models.py` (seed 1738):

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.777** | **0.555** | **0.536** | 0.541 | 0.531 |
| HistGradientBoosting | 0.776 | 0.550 | 0.534 | 0.530 | 0.539 |
| LightGBM | 0.773 | 0.552 | 0.525 | 0.588 | 0.475 |
| Extra Trees | 0.773 | 0.546 | 0.533 | 0.529 | 0.538 |
| Random Forest | 0.772 | 0.548 | 0.533 | 0.550 | 0.518 |
| Logistic Regression | 0.752 | 0.499 | 0.526 | 0.563 | 0.494 |

Original-style untuned Random Forest on raw/dummified columns (for reference): ROC-AUC ≈ 0.757 with default-class recall ≈ 0.36 at the 0.5 threshold.

The enhanced XGBoost pipeline improves ranking quality and, with a tuned threshold (~0.59), recovers a much stronger precision/recall balance on defaults.

Artifacts:

- `artifacts/model_leaderboard.csv`
- `artifacts/best_model_metrics.json`
- `artifacts/best_model.joblib`

## 6. How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full model comparison + artifact export
python scripts/train_models.py

# Score a few rows with the saved best model
python scripts/predict_example.py

# Interactive exploration
jupyter notebook notebooks/credit_default_risk_modeling.ipynb
```

## 7. Using the saved model

```python
import joblib
import pandas as pd
from credit_default.data import basic_clean, load_raw
from credit_default.features import engineer_features

bundle = joblib.load("artifacts/best_model.joblib")
pipe = bundle["pipeline"]
threshold = bundle["threshold"]

raw = basic_clean(load_raw()).head(5)
X = engineer_features(raw)
proba = pipe.predict_proba(X)[:, 1]
pred = (proba >= threshold).astype(int)
```

## 8. Limitations and next steps

- The UCI file is a static public snapshot; production portfolios drift and need monitoring.
- PAY codes are treated as ordinal; a pure categorical encoding could be compared.
- Cost-sensitive thresholds (expected loss / collections cost) would beat F1 for business use.
- Calibration (Platt / isotonic) would improve probability quality for pricing.
- Model explainability (SHAP) would help credit officers trust individual scores.
- A temporal split would be preferable if true statement dates were available.

## 9. References

- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
- UCI Machine Learning Repository: Default of Credit Card Clients Dataset.
