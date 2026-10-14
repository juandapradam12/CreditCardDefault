# Credit Card Default Risk Scoring

### Turn six months of repayment history into an actionable default probability

This project is a complete, reproducible machine-learning pipeline that predicts whether a revolving credit client will **default next month** on the UCI Taiwan credit-card dataset. It delivers a portfolio-ready scoring workflow: engineered risk features, imbalance-aware models, **isotonic probability calibration**, **cost-sensitive thresholds**, **SHAP explainability**, and exportable artifacts.

<p align="center">
  <img alt="default risk concept" src="https://img.shields.io/badge/task-binary%20classification-0B6E4F?style=for-the-badge" />
  <img alt="best model" src="https://img.shields.io/badge/best%20model-XGBoost-1B4965?style=for-the-badge" />
  <img alt="roc auc" src="https://img.shields.io/badge/hold--out%20ROC--AUC-0.777-5FA8D3?style=for-the-badge" />
</p>

---

## Why this matters

Missed defaults are expensive. Over-flagging good customers is also expensive. A useful credit-risk model must do more than maximize accuracy on an imbalanced dataset — it must **rank risky accounts well** and expose a **tunable operating point** for collections, underwriting overlays, or early-warning workflows.

This repository shows how to do that end-to-end on a public benchmark of **30,000 credit-card clients** (~22% default rate), with feature engineering, calibrated boosting models, cost-sensitive thresholds, SHAP explanations, a Streamlit demo, and CI-tested tooling.

## Headline results

Hold-out test metrics (`random_state=1738`):

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| **XGBoost (best)** | **0.777** | **0.555** | **0.536** | 0.541 | 0.531 |
| HistGradientBoosting | 0.776 | 0.550 | 0.534 | 0.530 | 0.539 |
| LightGBM | 0.773 | 0.552 | 0.525 | 0.588 | 0.475 |
| Extra Trees | 0.773 | 0.546 | 0.533 | 0.529 | 0.538 |
| Random Forest | 0.772 | 0.548 | 0.533 | 0.550 | 0.518 |
| Logistic Regression | 0.752 | 0.499 | 0.526 | 0.563 | 0.494 |

On the winning XGBoost model:

- **Calibration**: hold-out Brier **0.175 → 0.135** after isotonic calibration
- **Cost policy** (FN cost = 5× FP): threshold ≈ **0.17**, test recall ≈ **0.75** at expected cost **4222**
- **Top SHAP drivers**: `PAY_MAX`, `WORST_RECENT`, `PAY_1`, `PAY_AMT1`, `LIMIT_BAL`

## Quickstart

```bash
git clone https://github.com/juandapradam12/CreditCardDefault.git
cd CreditCardDefault

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train, calibrate, explain, export artifacts/
python scripts/train_models.py

# Score sample clients (cost-sensitive threshold by default)
python scripts/predict_example.py
python scripts/predict_example.py --explain-id 1

# Interactive Streamlit demo (probability + decision + SHAP)
python scripts/run_demo.py

# Rebuild global SHAP artifacts
python scripts/explain_model.py

# Run automated tests
pytest -q
```

Custom loss weights:

```bash
python scripts/train_models.py --fp-cost 1 --fn-cost 10
```

Explore interactively:

```bash
jupyter notebook notebooks/credit_default_risk_modeling.ipynb
```

## Project structure

```text
data/                  # CSV + Excel dataset
src/credit_default/    # load → features → models → calibrate → explain → score
app/streamlit_app.py   # interactive scoring demo
scripts/               # CLI entrypoints
tests/                 # pytest smoke suite
.github/workflows/ci.yml
notebooks/             # analysis notebooks
artifacts/             # leaderboard, metrics, SHAP plots, best_model.joblib
docs/DOCUMENTATION.md  # full methodology write-up
```

## How the scoring pipeline works

1. **Load** the UCI credit-card default dataset
2. **Engineer** repayment-stress features (utilization, delay summaries, payment ratios)
3. **Train** imbalance-aware classifiers with stratified validation
4. **Select** the best ranker, then **calibrate** probabilities (isotonic)
5. **Tune** both F1 and expected-cost thresholds on a held-out train slice
6. **Explain** with SHAP and **export** artifacts / leaderboard

```python
from credit_default.train import run_training

result = run_training()
print(result["best_model"])
print(result["leaderboard"].head())
```

## Use cases

- Early-warning lists for revolving accounts likely to miss the next payment
- Champion/challenger experiments for credit risk scorecards
- Cost-aware default scoring with explainable drivers for risk and collections workflows

## Documentation

Methodology details: **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)**

## Dataset credit

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications.

Data hosted by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

## Author

**Juan David Prada Malagon**
