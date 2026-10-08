"""SHAP explanations for the tree model.

SHAP values are computed on the RAW model output (log-odds). The isotonic/Platt
calibrator is a monotone transform applied afterwards, so it changes the probability
scale but not the ranking of customers or the direction of any feature's effect.
When quoting a customer's explanation, report the calibrated probability alongside
the SHAP waterfall and say which scale each number is on.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import FIG_DIR
from .train import extract_gbm


def compute_shap(model, X: pd.DataFrame):
    """Return a shap.Explanation (n_samples, n_features) in log-odds of the positive class."""
    import shap

    explainer = shap.TreeExplainer(extract_gbm(model))
    sv = explainer(X)
    if sv.values.ndim == 3:  # some versions return one slice per class
        sv = sv[:, :, 1]
    return sv


def _save(name):
    if name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")


def plot_global(sv, max_display: int = 15, save_prefix: str | None = "shap"):
    """Beeswarm (direction + magnitude) and bar (mean |SHAP|) summaries."""
    import shap

    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.title("Which features drive predicted default risk?")
    _save(f"{save_prefix}_beeswarm.png" if save_prefix else None)
    plt.show()

    shap.plots.bar(sv, max_display=max_display, show=False)
    _save(f"{save_prefix}_bar.png" if save_prefix else None)
    plt.show()


def pick_cases(y, p, thr: float) -> dict[str, int]:
    """Positions of four illustrative customers: a confident TP, FN, FP and TN."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))

    def pick(mask, largest):
        cand = idx[mask]
        return int(cand[np.argmax(p[cand])] if largest else cand[np.argmin(p[cand])])

    return {
        "true_positive (caught defaulter)": pick((y == 1) & (p >= thr), True),
        "false_negative (missed defaulter)": pick((y == 1) & (p < thr), False),
        "false_positive (false alarm)": pick((y == 0) & (p >= thr), True),
        "true_negative (safe customer)": pick((y == 0) & (p < thr), False),
    }


def plot_waterfalls(sv, cases: dict[str, int], p_cal, y, max_display: int = 12, save_prefix: str | None = "shap_waterfall"):
    """One waterfall per case, titled with outcome, actual label and calibrated probability."""
    import shap

    for label, i in cases.items():
        shap.plots.waterfall(sv[i], max_display=max_display, show=False)
        plt.title(f"{label}: P(default)={p_cal[i]:.2f}, actual={int(np.asarray(y)[i])}", fontsize=10)
        _save(f"{save_prefix}_{label.split(' ')[0]}.png" if save_prefix else None)
        plt.show()


def top_features(sv, n: int = 15) -> pd.DataFrame:
    """Mean |SHAP| ranking as a table (handy for the README)."""
    imp = pd.Series(np.abs(sv.values).mean(axis=0), index=sv.feature_names, name="mean_abs_shap")
    return imp.sort_values(ascending=False).head(n).to_frame()
