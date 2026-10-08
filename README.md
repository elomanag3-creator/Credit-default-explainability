# Credit Default Prediction: Explainable, Calibrated, Audited

A lender wants to predict which credit-card customers will **miss their next payment**, so it can
intervene early (adjust limits, offer payment plans) **without unfairly penalising any group**.

The point of this project is not "XGBoost got 82% accuracy". It is to show a model that can be
**justified, calibrated and audited**, which is what regulated industries (banking, insurance) care about.

| Question a regulator / risk committee asks | Where it is answered |
|---|---|
| Is the model better than a simple, transparent baseline? | `02_modeling`, logistic regression vs LightGBM, with and without feature engineering |
| Do the probabilities mean what they say? | Calibration curves, Brier/ECE, Platt vs isotonic |
| Why this decision threshold? | Cost-based threshold (missed defaulter = 5x false alarm) plus a sensitivity table |
| Why was *this* customer flagged? | SHAP waterfall plots |
| Does it treat groups differently? | Fairness audit on sex, age, education and marital status, with bootstrap CIs |
| Where does it fail? | High-confidence errors, error rates by segment, missed vs caught defaulters |

## Results

> Run the pipeline and paste your numbers here. `python -m src.train` writes CSVs to `reports/tables/`
> and figures to `reports/figures/`. I deliberately don't ship numbers I haven't produced from your run.

| Model (test set) | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|
| Logistic regression | | | | |
| LightGBM (raw scores) | | | | |
| LightGBM (calibrated) | | | | |

Headline findings to fill in: top 3 SHAP drivers · whether SMOTE helped · calibration before/after ·
chosen threshold and cost per customer vs naive policies · largest fairness gaps (with CIs) · main failure mode.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. get the data (see data/README.md), then:
python -m src.train --data data/default_of_credit_card_clients.xls --n-trials 40   # headless pipeline

# 2. or walk through it interactively
jupyter lab notebooks/

# 3. tests (synthetic data, no download needed)
python -m pytest tests
```

## Repository layout

```
├── README.md
├── requirements.txt
├── data/                 gitignored; download instructions in data/README.md
├── notebooks/
│   ├── 01_eda.ipynb                       class balance, undocumented codes, outliers, signal
│   ├── 02_modeling.ipynb                  baseline, LightGBM, imbalance, Optuna, calibration, cost threshold
│   └── 03_explainability_fairness.ipynb   SHAP, fairness audit, failure analysis
├── src/
│   ├── features.py       load, data-quality report, cleaning, feature engineering
│   ├── train.py          models, CV comparison, Optuna, calibrators, end-to-end pipeline (CLI)
│   ├── evaluate.py       metrics, calibration, cost curves, failure analysis
│   ├── fairness.py       group metrics, bootstrap CIs, proxy test, mitigation trade-off experiment
│   └── explain.py        SHAP global and local plots
├── tests/                smoke tests on synthetic data
├── artifacts/            gitignored; trained model and predictions
└── reports/figures/      plots written by the notebooks / pipeline
```

## Methodology (decisions worth defending)

1. **Hold-out discipline.** A stratified 20% test set is untouched until the end. Model choice, tuning,
   calibration and the threshold all use CV on the other 80%.
2. **Data quality.** EDUCATION 0/5/6 and MARRIAGE 0 are undocumented, so they are merged into "Other". PAY_x
   values -2 and 0 are undocumented but behave differently from -1, so they are kept as distinct ordinal values.
   Duplicates are dropped.
3. **Features.** Utilisation, repayment-to-bill ratios, delinquency counts and trends, built from the six-month history.
   The notebook measures their lift separately from the lift of the stronger model.
4. **Imbalance.** Plain vs class weights vs SMOTE with identical settings. SMOTE runs inside an `imblearn`
   pipeline so synthetic rows never leak across folds. Plain is preferred unless an alternative wins PR-AUC by a margin.
5. **Metrics.** PR-AUC (tuning objective) and ROC-AUC for ranking; Brier, log-loss and ECE for probability quality.
   Accuracy is not reported as a headline.
6. **Calibration.** Platt and isotonic are fit on **out-of-fold** predictions and compared with cross-fitting.
7. **Cost-based threshold.** `flag if p > c_fp / (c_fp + c_fn)` for calibrated probabilities (1/6 at a 5x ratio).
   The empirical optimum is checked against that theory, chosen on OOF scores, and stress-tested across cost ratios 2x to 20x.
8. **Explainability.** SHAP on the raw log-odds (calibration is monotone, so direction and ranking are unchanged).
   Waterfalls for a confident TP, FN, FP and TN.
9. **Fairness.** SEX and MARRIAGE are **not** model inputs; all four attributes are audited on flag rate, TPR, FPR, PPV and
   calibration by group, with bootstrap CIs. A proxy test checks whether the other features still encode sex, and a
   with/without-SEX comparison measures what unawareness buys. A group-threshold experiment shows the cost of forcing
   equal TPR, as analysis and not as a recommendation.
10. **Failure analysis.** Most confident errors, miss/false-alarm rates by segment, and standardised differences
    between missed and caught defaulters.

## Interview talking points

- *"Why not just report accuracy?"* A "nobody defaults" model scores about 78%. PR-AUC and expected cost reflect the actual decision.
- *"Did SMOTE help?"* Report your measured answer. It usually leaves ranking unchanged and degrades calibration.
- *"Why calibrate?"* Lenders use the probability itself (expected loss, limits). Reweighting distorts it; calibration fixes it, and Brier/ECE prove it.
- *"Why is the threshold 0.17 and not 0.5?"* The costs are asymmetric, and the theoretical and empirical optima agree.
  The 5x ratio is a business assumption, hence the sensitivity table.
- *"You removed SEX, so it's fair?"* No. Proxy AUC and group metrics show what remains. Fairness criteria also conflict when base rates differ.
- *"Why not equalise error rates with group thresholds?"* It is disparate treatment on a protected attribute (generally
  unlawful for sex in EU credit/insurance pricing since the CJEU's Test-Achats ruling, and under US ECOA). It is shown
  here only to quantify the trade-off. This is not legal advice.

## Limitations

- **Old, single-market data** (Taiwan, 2005). Patterns will not transfer to other markets or today's behaviour.
- **No time dimension**, so no out-of-time validation. Random CV likely overstates real-world performance.
- **Selection bias.** Only customers who were granted credit are observed, and limits already reflect the lender's earlier decisions.
- **No income, employment or bureau data.** Missed defaulters are probably shocks the features cannot see.
- **Binary sex coding and coarse groups.** Intersectional effects (e.g. sex x age) are not audited.
- **Cost ratio is assumed**, not estimated from real intervention economics.
- **SHAP explains the model, not causality.** Correlated features share credit.

## Possible extensions

Home Credit Default Risk (multi-table, messier) · monotonic constraints on PAY_x for regulatory plausibility ·
conformal prediction intervals · adversarial debiasing or reweighing vs thresholding comparison · a drift/monitoring
sketch (PSI on score and key features) · a one-page model card.
