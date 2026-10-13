# Credit Card Default Risk Scoring

### Turn six months of repayment history into an actionable default probability

This project is a complete, reproducible machine-learning pipeline that predicts whether a revolving credit client will **default next month**. It upgrades a classic tree-based student analysis of the UCI Taiwan credit-card dataset into a portfolio-ready scoring workflow: engineered risk features, imbalance-aware models, threshold-tuned decisions, and exportable artifacts.

<p align="center">
  <img alt="default risk concept" src="https://img.shields.io/badge/task-binary%20classification-0B6E4F?style=for-the-badge" />
  <img alt="best model" src="https://img.shields.io/badge/best%20model-XGBoost-1B4965?style=for-the-badge" />
  <img alt="roc auc" src="https://img.shields.io/badge/hold--out%20ROC--AUC-0.777-5FA8D3?style=for-the-badge" />
</p>

---

## Why this matters

Missed defaults are expensive. Over-flagging good customers is also expensive. A useful credit-risk model must do more than maximize accuracy on an imbalanced dataset — it must **rank risky accounts well** and expose a **tunable operating point** for collections, underwriting overlays, or early-warning workflows.

This repository shows how to do that end-to-end on a public benchmark of **30,000 credit-card clients** (~22% default rate).

## What you get

- **Business-ready framing** — predict next-month default from limit, demographics, and trailing repayment / bill / payment history
- **Domain feature engineering** — utilization, payment-to-bill ratios, delay aggregates, and a composite stress score
- **Modern model zoo** — Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM
- **Honest evaluation** — stratified splits, ROC-AUC / PR-AUC, and F1-optimized thresholds chosen without peeking at the test set
- **Reproducible package** — `src/credit_default` + CLI training script + saved leaderboard / model artifacts
- **Notebooks** — enhanced analysis plus the preserved original exploratory notebook

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

Versus the original untuned Random Forest baseline on raw features (**ROC-AUC ≈ 0.757**, default recall ≈ **0.36** at threshold 0.5), the enhanced pipeline improves ranking quality and recovers a far more usable precision/recall balance after threshold tuning.

## Quickstart

```bash
git clone https://github.com/juandapradam12/CreditCardDefault.git
cd CreditCardDefault

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train, compare models, export artifacts/
python scripts/train_models.py

# Score a sample of clients with the saved best model
python scripts/predict_example.py
```

Explore interactively:

```bash
jupyter notebook notebooks/credit_default_risk_modeling.ipynb
```

## Project structure

```text
data/                  # CSV + original Excel dataset
src/credit_default/    # load → features → models → evaluate → train
scripts/               # CLI entrypoints
notebooks/             # enhanced + original notebooks
artifacts/             # leaderboard, metrics JSON, best_model.joblib
docs/DOCUMENTATION.md  # full methodology write-up
```

## How the scoring pipeline works

1. **Load** the UCI credit-card default dataset
2. **Engineer** repayment-stress features (utilization, delay summaries, payment ratios)
3. **Train** imbalance-aware classifiers with stratified validation
4. **Tune** the decision threshold on out-of-fold probabilities
5. **Export** the winning model and a transparent leaderboard

```python
from credit_default.train import run_training

result = run_training()
print(result["best_model"])
print(result["leaderboard"].head())
```

## Ideal use cases

- Portfolio **early-warning** lists for accounts likely to roll delinquent
- Offline **champion/challenger** experiments for risk scorecards
- Teaching / portfolio piece demonstrating **imbalanced classification done properly**
- Starting point for cost-sensitive or explainable credit models (SHAP, expected-loss thresholds)

## Documentation

Deep dive on data quirks, feature design, validation protocol, limitations, and extension ideas:

**[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)**

## Dataset credit

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications.

Data hosted by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

## Author

**Juan David Prada Malagon** — data scientist. Original exploratory analysis modernized into a reproducible risk-scoring package.

---

If this project is useful for your hiring review, teaching material, or risk-modeling sandbox, star the repo and adapt the threshold to your own false-positive / false-negative cost ratio.
