"""Metrics, calibration, cost-based thresholding, plots and failure analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def _save(fig, name: str | None):
    if name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Ranking and probability-quality metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """Weighted mean |observed rate - mean predicted| over equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def score_summary(y, p, name: str = "model") -> dict:
    """ROC-AUC, PR-AUC (the headline for a 22% positive class) and probability-quality metrics."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(np.mean(y)),
    }


def plot_roc_pr(y, preds: dict[str, np.ndarray], save: str | None = None):
    """ROC and precision-recall curves for several models side by side."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax[1].plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax[1].axhline(np.mean(y), color="k", ls=":", lw=1, label=f"Prevalence {np.mean(y):.2f}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for a in ax:
        a.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, save)
    return fig


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def plot_calibration(y, preds: dict[str, np.ndarray], n_bins: int = 10, save: str | None = None):
    """Reliability diagram (quantile bins) plus histogram of predicted probabilities."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].plot([0, 1], [0, 1], "k:", label="Perfect")
    for name, p in preds.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax[0].plot(mean_pred, frac, "o-", ms=4, label=f"{name} (ECE {expected_calibration_error(y, p):.3f})")
        ax[1].hist(p, bins=40, alpha=0.5, label=name)
    ax[0].set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Reliability diagram")
    ax[1].set(xlabel="Predicted probability", ylabel="Customers", title="Score distribution")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save)
    return fig


# --------------------------------------------------------------------------- #
# Cost-based threshold
# --------------------------------------------------------------------------- #
def cost_curve(y, p, c_fn: float = 5.0, c_fp: float = 1.0, thresholds=None) -> pd.DataFrame:
    """Total and per-customer cost at every threshold.

    c_fn: cost of missing a defaulter (no early intervention).
    c_fp: cost of a false alarm (needless limit cut / outreach).
    """
    y, p = np.asarray(y).astype(bool), np.asarray(p)
    thresholds = np.linspace(0.01, 0.99, 197) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        flag = p >= t
        fn = int((y & ~flag).sum())
        fp = int((~y & flag).sum())
        tp = int((y & flag).sum())
        rows.append(
            {
                "threshold": t,
                "fn": fn,
                "fp": fp,
                "tp": tp,
                "cost": c_fn * fn + c_fp * fp,
                "cost_per_customer": (c_fn * fn + c_fp * fp) / len(y),
                "flag_rate": flag.mean(),
                "recall": tp / max(y.sum(), 1),
                "precision": tp / max(flag.sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def theoretical_threshold(c_fn: float, c_fp: float) -> float:
    """For perfectly calibrated probabilities, flag when p * c_fn > (1-p) * c_fp."""
    return c_fp / (c_fp + c_fn)


def pick_threshold(y, p, c_fn: float = 5.0, c_fp: float = 1.0) -> float:
    """Empirical cost-minimising threshold. Use out-of-fold scores, not the test set."""
    cc = cost_curve(y, p, c_fn, c_fp)
    return float(cc.loc[cc["cost"].idxmin(), "threshold"])


def baseline_costs(y, p, thr: float, c_fn: float = 5.0, c_fp: float = 1.0) -> pd.DataFrame:
    """Compare the chosen policy against naive ones, in cost per customer."""
    y = np.asarray(y).astype(bool)
    n = len(y)

    def cost(flag):
        return (c_fn * (y & ~flag).sum() + c_fp * (~y & flag).sum()) / n

    policies = {
        "Flag nobody": np.zeros(n, bool),
        "Flag everybody": np.ones(n, bool),
        "Model @ 0.50": p >= 0.5,
        f"Model @ {thr:.2f} (cost-optimal)": p >= thr,
    }
    out = pd.DataFrame({"policy": list(policies), "cost_per_customer": [cost(f) for f in policies.values()]})
    out["flag_rate"] = [f.mean() for f in policies.values()]
    out["recall"] = [(f & y).sum() / y.sum() for f in policies.values()]
    out["precision"] = [(f & y).sum() / max(f.sum(), 1) for f in policies.values()]
    return out


def plot_cost_curve(y, p, c_fn=5.0, c_fp=1.0, chosen: float | None = None, save: str | None = None):
    cc = cost_curve(y, p, c_fn, c_fp)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(cc["threshold"], cc["cost_per_customer"], lw=2)
    ax.axvline(0.5, color="gray", ls=":", label="Default 0.50")
    ax.axvline(theoretical_threshold(c_fn, c_fp), color="tab:green", ls="--",
               label=f"Theory c_fp/(c_fp+c_fn) = {theoretical_threshold(c_fn, c_fp):.2f}")
    if chosen is not None:
        ax.axvline(chosen, color="tab:red", label=f"Chosen {chosen:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Expected cost per customer",
           title=f"Cost vs threshold (missed defaulter = {c_fn:g}x false alarm)")
    ax.legend()
    fig.tight_layout()
    _save(fig, save)
    return fig


def cost_ratio_sensitivity(y, p_oof, p_test, y_test, ratios=(2, 3, 5, 10, 20)) -> pd.DataFrame:
    """How the optimal threshold and flag rate move if the cost assumption is wrong.

    The threshold is picked on out-of-fold scores (y, p_oof) and evaluated on the test set.
    """
    rows = []
    for r in ratios:
        thr = pick_threshold(y, p_oof, c_fn=r, c_fp=1.0)
        flag = np.asarray(p_test) >= thr
        yt = np.asarray(y_test).astype(bool)
        cost = (r * (yt & ~flag).sum() + (~yt & flag).sum()) / len(yt)
        rows.append({"cost_ratio": r, "threshold": thr, "theory": theoretical_threshold(r, 1.0),
                     "flag_rate": flag.mean(), "recall": (flag & yt).sum() / yt.sum(),
                     "test_cost_per_customer": cost})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Failure analysis
# --------------------------------------------------------------------------- #
def confident_errors(X: pd.DataFrame, y, p, thr: float, n: int = 15) -> dict[str, pd.DataFrame]:
    """Most confident false negatives (defaulters scored low) and false positives."""
    d = X.copy()
    d["y"], d["p"] = np.asarray(y), np.asarray(p)
    fn = d[(d.y == 1) & (d.p < thr)].sort_values("p").head(n)
    fp = d[(d.y == 0) & (d.p >= thr)].sort_values("p", ascending=False).head(n)
    return {"false_negatives": fn, "false_positives": fp}


def error_by_segment(df: pd.DataFrame, y, p, thr: float, by: str, bins=None) -> pd.DataFrame:
    """Miss rate (1 - recall) and false-alarm rate within a segment of customers.

    `by` is a column of `df`; pass `bins` to bucket a numeric column first.
    """
    d = pd.DataFrame({"seg": pd.cut(df[by], bins) if bins is not None else df[by].values,
                      "y": np.asarray(y).astype(int), "flag": (np.asarray(p) >= thr).astype(int)})
    g = d.groupby("seg", observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "default_rate": g["y"].mean(),
        "flag_rate": g["flag"].mean(),
        "miss_rate": g.apply(lambda s: ((s.y == 1) & (s.flag == 0)).sum() / max((s.y == 1).sum(), 1), include_groups=False),
        "false_alarm_rate": g.apply(lambda s: ((s.y == 0) & (s.flag == 1)).sum() / max((s.y == 0).sum(), 1), include_groups=False),
    })
    return out


def compare_groups(X: pd.DataFrame, mask_a, mask_b, top: int = 12) -> pd.DataFrame:
    """Standardised mean difference of every feature between two sets of customers.

    Example: mask_a = missed defaulters (FN), mask_b = caught defaulters (TP).
    Large |SMD| shows what the model's blind spot looks like.
    """
    a, b = X[np.asarray(mask_a)], X[np.asarray(mask_b)]
    pooled = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
    out = pd.DataFrame({"mean_a": a.mean(), "mean_b": b.mean(), "smd": (a.mean() - b.mean()) / pooled})
    return out.reindex(out["smd"].abs().sort_values(ascending=False).index).head(top)
