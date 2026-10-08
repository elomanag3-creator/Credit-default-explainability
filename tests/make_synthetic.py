"""Generate a synthetic dataset with the same schema as the UCI file (for smoke tests only).

NOT a substitute for the real data: it just lets CI exercise the code without a download.
"""
import numpy as np
import pandas as pd


def make_synthetic(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = rng.choice([10_000, 30_000, 50_000, 80_000, 120_000, 200_000, 400_000], n, p=[.05, .2, .2, .2, .15, .15, .05])
    sex = rng.choice([1, 2], n, p=[.4, .6])
    edu = rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[.002, .35, .47, .16, .01, .005, .003])
    mar = rng.choice([0, 1, 2, 3], n, p=[.002, .46, .53, .008])
    age = np.clip((21 + rng.gamma(2.5, 5.5, n)).astype(int), 21, 79)

    risk = rng.normal(0, 1, n) + 0.3 * (sex == 1) - 0.01 * (age - 35)
    pay = np.zeros((n, 6), int)
    state = np.clip(np.round(risk * 0.9), -2, 4).astype(int)
    for m in range(5, -1, -1):  # oldest -> newest
        state = np.clip(state + rng.choice([-1, 0, 0, 1], n), -2, 8)
        pay[:, m] = np.where(state == 1, rng.choice([0, 1], n), state)
    util = np.clip(rng.beta(2, 3, n) + 0.1 * np.maximum(risk, 0), 0, 1.3)
    bills = np.outer(limit * util, np.ones(6)) * rng.uniform(0.7, 1.1, (n, 6))
    pay_amt = bills[:, 1:].tolist()
    pay_amt = np.column_stack([bills[:, [min(m + 1, 5)]] * rng.uniform(0, 0.6, (n, 1)) * (pay[:, [m]] <= 0) for m in range(6)])
    pay_amt = np.where(rng.random((n, 6)) < 0.2, 0, pay_amt)

    logit = -1.6 + 0.55 * pay[:, 0] + 0.2 * pay[:, 1] + 0.8 * (util > 1) - 0.5 * np.log1p(limit / 50_000) + 0.4 * risk
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({"LIMIT_BAL": limit, "SEX": sex, "EDUCATION": edu, "MARRIAGE": mar, "AGE": age})
    for i in range(6):
        df[f"PAY_{i + 1}"] = pay[:, i]
    for i in range(6):
        df[f"BILL_AMT{i + 1}"] = bills[:, i].round().astype(int)
    for i in range(6):
        df[f"PAY_AMT{i + 1}"] = pay_amt[:, i].round().astype(int)
    df["default payment next month"] = y
    df.insert(0, "ID", np.arange(1, n + 1))
    return df.rename(columns={"PAY_1": "PAY_0"})  # mimic the original naming quirk
