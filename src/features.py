"""Data loading, data-quality audit, cleaning and feature engineering.

Month indexing in this dataset is reversed: suffix 1 is the MOST RECENT month
(September 2005) and suffix 6 is the oldest (April 2005). The target is whether
the customer misses the payment in October 2005.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "DEFAULT"
MONTHS = range(1, 7)
PAY_COLS = [f"PAY_{i}" for i in MONTHS]
BILL_COLS = [f"BILL_AMT{i}" for i in MONTHS]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in MONTHS]

# Attributes we audit for fairness. SEX and MARRIAGE are never given to the model
# (see `get_xy`); AGE and EDUCATION are kept as features but still audited.
AUDITED = ["SEX", "AGE_GROUP", "EDUCATION", "MARRIAGE"]
PROTECTED_DROPPED = ["SEX", "MARRIAGE"]

SEX_LABELS = {1: "Male", 2: "Female"}
EDU_LABELS = {1: "Grad school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the UCI file (.xls/.xlsx/.csv) and normalise column names."""
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=1)  # row 0 holds X1..X23 placeholders
    else:
        df = pd.read_csv(path)
        if "LIMIT_BAL" not in df.columns:  # csv exported with the X1..X23 row
            df = pd.read_csv(path, header=1)

    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.rename(
        columns={
            "PAY_0": "PAY_1",
            "default payment next month": TARGET,
            "default.payment.next.month": TARGET,
        }
    )
    if "ID" in df.columns:
        df = df.drop(columns="ID")
    missing = {TARGET, "LIMIT_BAL", "SEX", "AGE", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Flag the known oddities of this dataset. Returns one row per issue."""
    n = len(df)
    rows = []

    def add(issue, count, note):
        rows.append({"issue": issue, "rows": int(count), "pct": round(100 * count / n, 2), "note": note})

    add("Duplicate rows", df.duplicated().sum(), "Exact duplicates (the dataset ships with a few)")
    add("EDUCATION in {0,5,6}", df["EDUCATION"].isin([0, 5, 6]).sum(), "Undocumented codes; merged into 'Other' (4)")
    add("MARRIAGE == 0", (df["MARRIAGE"] == 0).sum(), "Undocumented code; merged into 'Other' (3)")
    pay = df[PAY_COLS]
    add("PAY_x == -2", (pay == -2).any(axis=1).sum(), "Undocumented. Most plausibly 'no consumption'")
    add("PAY_x == 0", (pay == 0).any(axis=1).sum(), "Undocumented. Most plausibly 'revolving credit, minimum paid'")
    add("Negative BILL_AMT", (df[BILL_COLS] < 0).any(axis=1).sum(), "Credit balance / refunds")
    add("BILL_AMT > LIMIT_BAL", (df[BILL_COLS].gt(df["LIMIT_BAL"], axis=0)).any(axis=1).sum(), "Over-limit statements")
    add("All six PAY_AMT == 0", (df[PAYAMT_COLS] == 0).all(axis=1).sum(), "Customers who never paid in the window")
    add("AGE > 70", (df["AGE"] > 70).sum(), "Sparse tail; fairness estimates for these ages are noisy")
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Resolve undocumented category codes. PAY_x codes are left as-is (ordinal)."""
    out = df.copy()
    if drop_duplicates:
        out = out.drop_duplicates().reset_index(drop=True)
    out["EDUCATION"] = out["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: 3})
    return out


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural features built from the six-month history."""
    out = df.copy()
    limit = out["LIMIT_BAL"].replace(0, np.nan)
    new = {}

    # Utilisation: how much of the limit is used each month
    for i in MONTHS:
        new[f"UTIL_{i}"] = out[f"BILL_AMT{i}"] / limit
    util = pd.DataFrame({k: v for k, v in new.items()})
    new["UTIL_MEAN"] = util.mean(axis=1)
    new["UTIL_MAX"] = util.max(axis=1)

    # Repayment ratio: payment in month i vs the bill issued the month before (i+1).
    # If nothing was owed, count it as fully paid.
    for i in range(1, 6):
        prev_bill = out[f"BILL_AMT{i + 1}"]
        ratio = np.where(prev_bill > 0, out[f"PAY_AMT{i}"] / prev_bill.where(prev_bill > 0, 1), 1.0)
        new[f"PAY_RATIO_{i}"] = np.clip(ratio, 0, 2)
    ratio_cols = [new[f"PAY_RATIO_{i}"] for i in range(1, 6)]
    new["PAY_RATIO_MEAN"] = np.mean(ratio_cols, axis=0)

    # Delinquency summary
    pay = out[PAY_COLS]
    new["PAY_MAX"] = pay.max(axis=1)
    new["PAY_MEAN"] = pay.mean(axis=1)
    new["N_DELAYED"] = (pay >= 1).sum(axis=1)
    new["N_REVOLVING"] = (pay == 0).sum(axis=1)
    new["N_PAID_FULL"] = (pay == -1).sum(axis=1)
    new["N_NO_CONSUMPTION"] = (pay == -2).sum(axis=1)
    new["PAY_TREND"] = out["PAY_1"] - out["PAY_6"]  # >0: getting worse

    # Amounts and trends
    new["BILL_MEAN"] = out[BILL_COLS].mean(axis=1)
    new["BILL_TREND"] = (out["BILL_AMT1"] - out["BILL_AMT6"]) / limit
    new["PAYAMT_MEAN"] = out[PAYAMT_COLS].mean(axis=1)
    new["N_ZERO_PAYMENTS"] = (out[PAYAMT_COLS] == 0).sum(axis=1)
    total_bills = out[BILL_COLS[1:]].clip(lower=0).sum(axis=1)
    total_paid = out[PAYAMT_COLS[:-1]].sum(axis=1)
    new["PAID_TO_BILLED"] = np.where(total_bills > 0, total_paid / total_bills.where(total_bills > 0, 1), 1.0).clip(0, 3)

    out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    return out.replace([np.inf, -np.inf], np.nan)


def add_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable sensitive-attribute columns for auditing (never model inputs)."""
    out = df.copy()
    out["SEX_LABEL"] = out["SEX"].map(SEX_LABELS)
    out["EDU_LABEL"] = out["EDUCATION"].map(EDU_LABELS)
    out["MARRIAGE_LABEL"] = out["MARRIAGE"].map(MARRIAGE_LABELS)
    out["AGE_GROUP"] = pd.cut(
        out["AGE"], bins=[0, 29, 39, 49, 59, 120], labels=["<30", "30-39", "40-49", "50-59", "60+"]
    ).astype(str)
    return out


# --------------------------------------------------------------------------- #
# Model matrices
# --------------------------------------------------------------------------- #
RAW_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]


def get_xy(
    df: pd.DataFrame,
    engineered: bool = True,
    drop_protected: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y).

    drop_protected=True removes SEX and MARRIAGE ("fairness through unawareness").
    This does NOT guarantee fairness because other features can act as proxies,
    which is why `fairness.proxy_auc` and the group audit exist.
    """
    work = engineer(df) if engineered else df
    drop = {TARGET, "SEX_LABEL", "EDU_LABEL", "MARRIAGE_LABEL", "AGE_GROUP"}
    if drop_protected:
        drop |= set(PROTECTED_DROPPED)
    X = work[[c for c in work.columns if c not in drop]].copy()
    return X, df[TARGET].astype(int)


def make_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test row positions. The test set is touched only for final reporting."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    return train_test_split(idx, test_size=test_size, stratify=df[TARGET], random_state=seed)
