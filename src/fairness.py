"""Group-fairness audit for a binary "flag for early intervention" decision.

Terminology used here:
  flag rate    = P(flagged | group)                      (demographic parity / selection rate)
  TPR          = P(flagged | defaults, group)            (equal opportunity)
  FPR          = P(flagged | does not default, group)    (with TPR: equalised odds)
  PPV          = P(defaults | flagged, group)            (predictive parity)
  calibration  = mean predicted probability vs observed default rate in the group

Important: these criteria cannot all hold at once when base rates differ between
groups (Chouldechova 2017; Kleinberg et al. 2016). The audit exists to quantify
the gaps and make the trade-off explicit, not to find a metric that turns green.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_metrics(y, flag, score, groups, min_n: int = 30) -> pd.DataFrame:
    """One row per group with the standard fairness quantities."""
    d = pd.DataFrame({"y": np.asarray(y).astype(int), "flag": np.asarray(flag).astype(int),
                      "score": np.asarray(score, dtype=float), "g": np.asarray(groups)})
    rows = []
    for g, s in d.groupby("g", observed=True):
        pos, neg = s[s.y == 1], s[s.y == 0]
        rows.append({
            "group": g,
            "n": len(s),
            "base_rate": s.y.mean(),
            "mean_score": s.score.mean(),
            "calibration_gap": s.score.mean() - s.y.mean(),
            "flag_rate": s.flag.mean(),
            "tpr": pos.flag.mean() if len(pos) else np.nan,
            "fpr": neg.flag.mean() if len(neg) else np.nan,
            "ppv": s[s.flag == 1].y.mean() if (s.flag == 1).any() else np.nan,
            "auc": roc_auc_score(s.y, s.score) if s.y.nunique() == 2 and len(s) >= min_n else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def disparity_summary(gm: pd.DataFrame) -> pd.Series:
    """Collapse a group table into headline gaps (worst-case across groups)."""
    return pd.Series({
        "flag_rate_ratio (min/max)": gm.flag_rate.min() / gm.flag_rate.max(),
        "tpr_gap (max-min)": gm.tpr.max() - gm.tpr.min(),
        "fpr_gap (max-min)": gm.fpr.max() - gm.fpr.min(),
        "ppv_gap (max-min)": gm.ppv.max() - gm.ppv.min(),
        "base_rate_gap (max-min)": gm.base_rate.max() - gm.base_rate.min(),
        "max_abs_calibration_gap": gm.calibration_gap.abs().max(),
    })


def bootstrap_group_ci(y, flag, score, groups, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Percentile bootstrap CIs for flag rate, TPR and FPR per group.

    Gaps that look alarming but have overlapping CIs are often just noise in small
    groups (for example customers aged 60+).
    """
    rng = np.random.default_rng(seed)
    y, flag, groups = np.asarray(y).astype(int), np.asarray(flag).astype(int), np.asarray(groups)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        gm = group_metrics(y[i], flag[i], np.asarray(score)[i], groups[i], min_n=10**9)
        draws.append(gm[["flag_rate", "tpr", "fpr"]].stack())
    boot = pd.concat(draws, axis=1)
    lo = boot.quantile(alpha / 2, axis=1).unstack()
    hi = boot.quantile(1 - alpha / 2, axis=1).unstack()
    point = group_metrics(y, flag, score, groups)[["flag_rate", "tpr", "fpr"]]
    out = point.join(lo.add_suffix("_lo")).join(hi.add_suffix("_hi"))
    return out[[c for m in ["flag_rate", "tpr", "fpr"] for c in (m, f"{m}_lo", f"{m}_hi")]]


def proxy_auc(X: pd.DataFrame, sensitive, cv: int = 5, seed: int = 0) -> float:
    """How well can the model's own input features predict a sensitive attribute?

    If SEX is removed from the model but this AUC is well above 0.5, other features
    carry information about sex ("proxies"), and unawareness does not mean the model
    cannot treat the groups differently.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    s = pd.Series(np.asarray(sensitive))
    s = (s == s.value_counts().index[0]).astype(int)  # binarise against the largest group
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
    return float(cross_val_score(clf, X, s, cv=cv, scoring="roc_auc").mean())


def group_threshold_experiment(y, score, groups, global_thr: float, c_fn: float = 5.0, c_fp: float = 1.0) -> pd.DataFrame:
    """EXPLORATORY: what would it cost to equalise TPR across groups with group-specific thresholds?

    For each group we pick the threshold that reproduces the overall TPR obtained at
    `global_thr`. This quantifies the accuracy/cost price of "equal opportunity".

    Caveat: using a protected attribute to set decision thresholds is disparate
    treatment. For sex this is generally unlawful in EU credit/insurance pricing
    (CJEU Test-Achats, 2011) and under US ECOA. This function is for analysis and
    for explaining the trade-off. It is not a recommended deployment.
    """
    y, score, groups = np.asarray(y).astype(int), np.asarray(score, float), np.asarray(groups)
    target_tpr = ((score >= global_thr) & (y == 1)).sum() / (y == 1).sum()

    thr_by_group = {}
    for g in np.unique(groups):
        m = (groups == g) & (y == 1)
        thr_by_group[g] = float(np.quantile(score[m], 1 - target_tpr)) if m.any() else global_thr

    def evaluate(flag, label):
        gm = group_metrics(y, flag, score, groups)
        cost = (c_fn * ((y == 1) & (flag == 0)).sum() + c_fp * ((y == 0) & (flag == 1)).sum()) / len(y)
        return {"policy": label, "cost_per_customer": cost, "flag_rate": flag.mean(),
                "tpr_gap": gm.tpr.max() - gm.tpr.min(), "fpr_gap": gm.fpr.max() - gm.fpr.min(),
                "flag_rate_ratio": gm.flag_rate.min() / gm.flag_rate.max()}

    flag_global = (score >= global_thr).astype(int)
    flag_group = np.array([score[i] >= thr_by_group[groups[i]] for i in range(len(y))]).astype(int)
    out = pd.DataFrame([evaluate(flag_global, "Single global threshold"),
                        evaluate(flag_group, "Group-specific thresholds (equal TPR)")])
    out.attrs["thresholds"] = thr_by_group
    return out


def plot_group_ci(ci: pd.DataFrame, title: str = "", save: str | None = None):
    """Flag rate / TPR / FPR per group with bootstrap CIs (output of `bootstrap_group_ci`)."""
    import matplotlib.pyplot as plt

    from .evaluate import _save

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, (m, label) in zip(axes, [("flag_rate", "Flag rate"), ("tpr", "True positive rate"), ("fpr", "False positive rate")]):
        x = np.arange(len(ci))
        err = np.vstack([ci[m] - ci[f"{m}_lo"], ci[f"{m}_hi"] - ci[m]]).clip(min=0)
        ax.bar(x, ci[m], yerr=err, capsize=4, color="tab:blue", alpha=0.8)
        ax.set_xticks(x, ci.index.astype(str), rotation=0)
        ax.set(title=label, ylim=(0, min(1.0, float(ci[f"{m}_hi"].max()) * 1.15)))
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, save)
    return fig
