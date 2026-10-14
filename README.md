# Credit Card Default Risk Scoring

### Who is about to miss next month’s payment — and what should you do about it?

Every month, a lender stares at the same tradeoff: chase too few accounts and credit losses pile up; chase too many and you burn cash harassing customers who would have paid. Accuracy on a dataset where only ~22% default is a trap — a naive “nobody defaults” rule is already ~78% accurate and completely useless.

This repository turns six months of repayment, bill, and payment history from the UCI Taiwan credit-card portfolio (**30,000 clients**) into a **next-month default probability**, then makes that score usable: calibrated for decisioning, tunable for business cost, and explainable enough to show *why* an account was flagged.

<p align="center">
  <img alt="default risk concept" src="https://img.shields.io/badge/task-binary%20classification-0B6E4F?style=for-the-badge" />
  <img alt="best model" src="https://img.shields.io/badge/best%20model-soft%20vote%20ensemble-1B4965?style=for-the-badge" />
  <img alt="roc auc" src="https://img.shields.io/badge/hold--out%20ROC--AUC-0.780-5FA8D3?style=for-the-badge" />
</p>

---

## Why ranking beats raw accuracy

A useful credit-risk model must **rank risky accounts well** and expose a **tunable operating point** for collections or early-warning workflows. Missing a default and bothering a good customer are not equally expensive — so the cutoff is a policy choice, not a fixed 0.5.

## Headline results

Ranking quality is the primary score here; precision/recall change with the threshold you pick.

Hold-out ranking metrics (`random_state=1738`):

| Model | ROC-AUC | PR-AUC | KS |
|---|---:|---:|---:|
| **Soft-vote ensemble (best)** | **0.780** | **0.556** | **0.420** |
| Stacking ensemble | 0.779 | 0.555 | 0.424 |
| CatBoost | 0.779 | 0.553 | 0.422 |
| XGBoost (Optuna-tuned) | 0.779 | 0.554 | 0.418 |
| LightGBM | 0.777 | 0.556 | 0.410 |
| HistGradientBoosting | 0.777 | 0.549 | 0.418 |
| Random Forest | 0.772 | 0.550 | 0.416 |
| Logistic Regression | 0.759 | 0.511 | 0.413 |

Calibrated soft-vote operating points on the hold-out set:

| Policy | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| F1-optimized | 0.30 | 0.545 | 0.531 | 0.538 |
| Cost-sensitive (FN=5×FP) | 0.20 | 0.419 | 0.672 | 0.516 |

- **Calibration**: Brier **0.175 → 0.135** after isotonic calibration
- **3-fold CV ROC-AUC** for the soft-vote ensemble: **0.790**
- Cost-sensitive mode trades precision for higher recall on purpose (catch more defaults)

### What “good” looks like on this dataset

There is no proven mathematical ceiling, but **modern published results on this same UCI Taiwan file repeatedly land around ROC-AUC 0.77–0.79**. That range is an empirical benchmark, not a hard limit: with only six months of PAY / BILL / PAY_AMT history plus basic demographics, most of the signal sits in recent delinquency, so extra model complexity usually yields small hold-out gains.

Examples from public work on this dataset:

| Source | Reported ROC-AUC |
|---|---:|
| XGBoost credit-risk study (Morgan State / related write-up) | ~0.775 |
| Calibration + rejection study on UCI clients (JACS, 2025) | 0.778 |
| Comparative data-mining study (XGBoost / LightGBM) | 0.779 / 0.790 |
| This repo — soft-vote ensemble (hold-out / 3-fold CV) | **0.780 / 0.790** |

So 0.78 hold-out here is competitive with strong published baselines on the same features — not a weak result dressed up with metrics. Jumping to 0.85+ without new information would usually mean leakage, an odd split, or a different evaluation setup.

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
