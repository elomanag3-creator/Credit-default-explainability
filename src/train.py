"""Modeling: baseline, LightGBM, imbalance handling, Optuna tuning, calibration, pipeline.

Run the whole thing end to end:

    python -m src.train --data data/default_of_credit_card_clients.xls --n-trials 40

Methodology notes
-----------------
* The 20% test set is held out up front and used only for final reporting.
* Everything else (model choice, tuning, calibration, threshold) uses cross-validation
  on the 80% training set. Calibrators and the cost threshold are fit on
  out-of-fold predictions, so no test information leaks into them.
* Tuning optimises PR-AUC (average precision), which is more informative than
  ROC-AUC when the positive class is ~22%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from . import features as ft

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def skf(n_splits: int = 5, seed: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def make_logreg(C: float = 1.0):
    """Baseline: median-impute -> standardise -> logistic regression."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=5000)),
    ])


def make_lgbm(**overrides):
    """LightGBM with sensible defaults for ~30k rows of tabular data."""
    from lightgbm import LGBMClassifier

    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    params.update(overrides)
    return LGBMClassifier(**params)


def extract_gbm(model):
    """Return the underlying tree model even if wrapped in a (SMOTE) pipeline."""
    return model.named_steps["clf"] if hasattr(model, "named_steps") else model


# --------------------------------------------------------------------------- #
# Cross-validated comparison
# --------------------------------------------------------------------------- #
def cv_compare(models: dict, X, y, n_splits: int = 5) -> pd.DataFrame:
    """Mean +/- std of ROC-AUC, PR-AUC and Brier across stratified folds.

    `models` maps name -> estimator, or name -> (estimator, X_override) when a model
    should see different features (e.g. raw vs engineered).
    """
    rows = []
    for name, spec in models.items():
        est, Xm = spec if isinstance(spec, tuple) else (spec, X)
        res = cross_validate(est, Xm, y, cv=skf(n_splits), n_jobs=1,
                             scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"})
        rows.append({
            "model": name,
            "roc_auc": res["test_roc_auc"].mean(), "roc_auc_sd": res["test_roc_auc"].std(),
            "pr_auc": res["test_pr_auc"].mean(), "pr_auc_sd": res["test_pr_auc"].std(),
            "brier": -res["test_brier"].mean(), "brier_sd": res["test_brier"].std(),
        })
    return pd.DataFrame(rows).set_index("model")


def imbalance_strategies(params: dict | None = None) -> dict:
    """Plain vs class weights vs SMOTE, all with the same LightGBM settings.

    SMOTE lives inside an imblearn Pipeline so synthetic rows are created only from the
    training part of each fold. Oversampling before cross-validation leaks and inflates scores.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    params = params or {}
    return {
        "plain": make_lgbm(**params),
        "class_weight": make_lgbm(class_weight="balanced", **params),
        "smote": ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", make_lgbm(**params))]),
    }


def choose_strategy(table: pd.DataFrame, margin: float = 0.002) -> str:
    """Pick the best PR-AUC strategy, but prefer 'plain' unless something wins by `margin`."""
    best = table["pr_auc"].idxmax()
    if best != "plain" and table.loc[best, "pr_auc"] - table.loc["plain", "pr_auc"] < margin:
        return "plain"
    return best


# --------------------------------------------------------------------------- #
# Hyper-parameter tuning
# --------------------------------------------------------------------------- #
def tune_lgbm(X, y, build=make_lgbm, n_trials: int = 40, n_splits: int = 5, seed: int = RANDOM_STATE):
    """Optuna (TPE) search maximising cross-validated PR-AUC.

    `build(**params)` returns an estimator, so the same search works for plain,
    class-weighted or SMOTE variants.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        }
        return cross_val_score(build(**params), X, y, cv=skf(n_splits, seed), scoring="average_precision").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study


# --------------------------------------------------------------------------- #
# Calibration (fit on out-of-fold predictions)
# --------------------------------------------------------------------------- #
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class IdentityCalibrator:
    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, float)


class PlattScaler:
    """Sigmoid calibration: logistic regression on the logit of the raw score."""

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone step-function calibration. Flexible, but needs a few thousand rows."""

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        return self.iso_.predict(np.asarray(p, float))


CALIBRATORS = {"none": IdentityCalibrator, "platt": PlattScaler, "isotonic": IsotonicCalibrator}


def oof_proba(estimator, X, y, n_splits: int = 5, seed: int = RANDOM_STATE) -> np.ndarray:
    """Out-of-fold P(default) for every training row."""
    return cross_val_predict(clone(estimator), X, y, cv=skf(n_splits, seed), method="predict_proba")[:, 1]


def cv_calibrate(p, y, method: str, n_splits: int = 5, seed: int = RANDOM_STATE) -> np.ndarray:
    """Cross-fitted calibration: each row is calibrated by a calibrator that never saw it."""
    p, y = np.asarray(p), np.asarray(y)
    out = np.zeros_like(p, dtype=float)
    for tr, va in skf(n_splits, seed).split(p.reshape(-1, 1), y):
        out[va] = CALIBRATORS[method]().fit(p[tr], y[tr]).predict(p[va])
    return out


def compare_calibrators(p_oof, y) -> tuple[pd.DataFrame, str]:
    """Brier / log-loss / ECE for raw vs Platt vs isotonic. Returns (table, best method by Brier)."""
    rows = [ev.score_summary(y, cv_calibrate(p_oof, y, m), name=m) for m in CALIBRATORS]
    table = pd.DataFrame(rows).set_index("model")
    return table, table["brier"].idxmin()


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(data_path, n_trials: int = 40, c_fn: float = 5.0, c_fp: float = 1.0,
                 out_dir: Path | str = ROOT / "artifacts", tables_dir: Path | str = ROOT / "reports" / "tables",
                 make_estimator=None) -> dict:
    """Train, tune, calibrate, pick a cost-based threshold, and save everything.

    `make_estimator` lets tests inject a stand-in model; leave it None for LightGBM.
    """
    out_dir, tables_dir = Path(out_dir), Path(tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    build = make_estimator or make_lgbm

    # 1. Data
    df = ft.add_group_labels(ft.clean(ft.load_raw(data_path)))
    tr_idx, te_idx = ft.make_split(df, seed=RANDOM_STATE)
    X, y = ft.get_xy(df, engineered=True)
    X_raw, _ = ft.get_xy(df, engineered=False)
    X_tr, X_te, y_tr, y_te = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]

    # 2. Baselines and feature-engineering lift
    base = cv_compare({
        "LogReg (raw features)": (make_logreg(), X_raw.iloc[tr_idx]),
        "LogReg (engineered)": (make_logreg(), X_tr),
        "LightGBM default (engineered)": (build(), X_tr),
    }, X_tr, y_tr)
    base.to_csv(tables_dir / "cv_baselines.csv")

    # 3. Imbalance handling, same settings for every strategy
    if make_estimator is None:
        strat = imbalance_strategies()
    else:  # stand-in model: class_weight only (no imblearn dependency)
        strat = {"plain": build(), "class_weight": build(class_weight="balanced")}
    imb = cv_compare(strat, X_tr, y_tr)
    imb.to_csv(tables_dir / "cv_imbalance.csv")
    chosen = choose_strategy(imb)

    # 4. Tune the chosen strategy
    def build_chosen(**params):
        if make_estimator is not None:
            return build(**params, **({"class_weight": "balanced"} if chosen == "class_weight" else {}))
        if chosen == "class_weight":
            return make_lgbm(class_weight="balanced", **params)
        if chosen == "smote":
            return imbalance_strategies(params)["smote"]
        return make_lgbm(**params)

    best_params, study = tune_lgbm(X_tr, y_tr, build=build_chosen, n_trials=n_trials)
    model = build_chosen(**best_params)

    # 5. Calibration on out-of-fold predictions
    p_oof = oof_proba(model, X_tr, y_tr)
    cal_table, best_cal = compare_calibrators(p_oof, y_tr)
    cal_table.to_csv(tables_dir / "calibration_methods.csv")
    calibrator = CALIBRATORS[best_cal]().fit(p_oof, y_tr)
    p_oof_cal = cv_calibrate(p_oof, y_tr, best_cal)

    # 6. Cost-based threshold, chosen on out-of-fold calibrated scores
    thr = ev.pick_threshold(y_tr, p_oof_cal, c_fn, c_fp)

    # 7. Fit on all training data, score the held-out test set once
    model.fit(X_tr, y_tr)
    p_raw = model.predict_proba(X_te)[:, 1]
    p_cal = calibrator.predict(p_raw)
    logreg = make_logreg().fit(X_tr, y_tr)
    p_lr = logreg.predict_proba(X_te)[:, 1]

    test_scores = pd.DataFrame([
        ev.score_summary(y_te, p_lr, "LogReg baseline"),
        ev.score_summary(y_te, p_raw, f"LightGBM ({chosen}), raw"),
        ev.score_summary(y_te, p_cal, f"LightGBM ({chosen}), {best_cal}-calibrated"),
    ]).set_index("model")
    test_scores.to_csv(tables_dir / "test_scores.csv")
    ev.baseline_costs(y_te, p_cal, thr, c_fn, c_fp).to_csv(tables_dir / "test_policy_costs.csv", index=False)

    # 8. Persist artifacts for the notebooks
    pred = df.iloc[te_idx].copy()
    pred["row_id"] = te_idx
    pred["p_logreg"], pred["p_raw"], pred["p_cal"] = p_lr, p_raw, p_cal
    pred.to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame({"y": y_tr.values, "p_oof": p_oof, "p_oof_cal": p_oof_cal}).to_csv(out_dir / "oof_predictions.csv", index=False)
    joblib.dump({"model": model, "calibrator": calibrator, "threshold": thr, "features": list(X.columns),
                 "strategy": chosen, "calibration": best_cal, "params": best_params,
                 "test_idx": te_idx, "train_idx": tr_idx, "c_fn": c_fn, "c_fp": c_fp}, out_dir / "model.joblib")
    summary = {"strategy": chosen, "calibration": best_cal, "threshold": thr, "best_params": best_params,
               "cv_best_pr_auc": study.best_value, "test": test_scores.round(4).reset_index().to_dict("records")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to the UCI .xls/.xlsx/.csv")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--c-fn", type=float, default=5.0, help="Cost of a missed defaulter")
    ap.add_argument("--c-fp", type=float, default=1.0, help="Cost of a false alarm")
    a = ap.parse_args()
    print(json.dumps(run_pipeline(a.data, a.n_trials, a.c_fn, a.c_fp), indent=2, default=float))
