"""Smoke tests on synthetic data. They check that the code runs and that key invariants hold.
They do not test model quality."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from src import evaluate as ev
from src import fairness as fr
from src import features as ft
from src import train as tr
from tests.make_synthetic import make_synthetic


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "UCI_Credit_Card.csv"
    make_synthetic().to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def data(csv_path):
    df = ft.add_group_labels(ft.clean(ft.load_raw(csv_path)))
    tr_idx, te_idx = ft.make_split(df)
    X, y = ft.get_xy(df)
    return df, X, y, tr_idx, te_idx


def hgb(**kw):
    kw.pop("n_estimators", None)
    return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, random_state=0, **kw)


def test_loading_and_cleaning(csv_path):
    raw = ft.load_raw(csv_path)
    assert ft.TARGET in raw and "PAY_1" in raw and "ID" not in raw
    rep = ft.data_quality_report(raw)
    assert rep.loc[rep.issue == "EDUCATION in {0,5,6}", "rows"].iloc[0] > 0
    df = ft.clean(raw)
    assert set(df.EDUCATION) <= {1, 2, 3, 4} and set(df.MARRIAGE) <= {1, 2, 3}


def test_features_exclude_protected_and_have_no_inf(data):
    df, X, y, *_ = data
    assert "SEX" not in X and "MARRIAGE" not in X and ft.TARGET not in X
    assert not np.isinf(X.to_numpy(float)).any()
    assert len(X) == len(y) == len(df)
    Xs, _ = ft.get_xy(df, drop_protected=False)
    assert "SEX" in Xs


def test_split_is_stratified(data):
    df, X, y, tr_idx, te_idx = data
    assert abs(y.iloc[tr_idx].mean() - y.iloc[te_idx].mean()) < 0.01
    assert not set(tr_idx) & set(te_idx)


def test_calibration_and_cost(data):
    df, X, y, tr_idx, te_idx = data
    Xt, yt = X.iloc[tr_idx], y.iloc[tr_idx]
    p = tr.oof_proba(hgb(class_weight="balanced"), Xt, yt)  # deliberately miscalibrated
    table, best = tr.compare_calibrators(p, yt)
    assert table.loc["none", "ece"] > table.loc[best, "ece"] or best == "none"
    pc = tr.cv_calibrate(p, yt, best)
    assert ((pc >= 0) & (pc <= 1)).all()
    cc = ev.cost_curve(yt, pc, 5, 1)
    assert cc.cost.min() <= cc.cost.iloc[0] and cc.cost.min() <= cc.cost.iloc[-1]
    assert ev.theoretical_threshold(5, 1) == pytest.approx(1 / 6)
    thr = ev.pick_threshold(yt, pc, 5, 1)
    assert 0.03 < thr < 0.5  # cost-aware threshold should sit well below 0.5
    pol = ev.baseline_costs(yt, pc, thr, 5, 1).set_index("policy")
    assert pol.cost_per_customer.min() == pol.cost_per_customer.iloc[-1]


def test_fairness_and_failure_analysis(data):
    df, X, y, tr_idx, te_idx = data
    m = hgb().fit(X.iloc[tr_idx], y.iloc[tr_idx])
    p = m.predict_proba(X.iloc[te_idx])[:, 1]
    yt, d = y.iloc[te_idx].values, df.iloc[te_idx]
    flag = (p >= 0.25).astype(int)
    gm = fr.group_metrics(yt, flag, p, d["SEX_LABEL"])
    assert set(gm.index) == {"Male", "Female"} and gm.n.sum() == len(yt)
    assert fr.disparity_summary(gm)["flag_rate_ratio (min/max)"] <= 1
    ci = fr.bootstrap_group_ci(yt, flag, p, d["AGE_GROUP"], n_boot=100)
    assert (ci.tpr_lo <= ci.tpr).all() and (ci.tpr <= ci.tpr_hi).all()
    assert 0.3 < fr.proxy_auc(X.iloc[te_idx], d["SEX"], cv=3) < 1.0
    exp = fr.group_threshold_experiment(yt, p, d["SEX_LABEL"], 0.25)
    assert exp.tpr_gap.iloc[1] <= exp.tpr_gap.iloc[0] + 1e-9
    seg = ev.error_by_segment(d.reset_index(drop=True), yt, p, 0.25, "LIMIT_BAL", bins=[0, 50_000, 150_000, 1e7])
    assert seg.n.sum() == len(yt)
    errs = ev.confident_errors(X.iloc[te_idx].reset_index(drop=True), yt, p, 0.25)
    assert (errs["false_negatives"].y == 1).all() and (errs["false_positives"].y == 0).all()
    xs = X.iloc[te_idx].reset_index(drop=True)
    assert len(ev.compare_groups(xs, (yt == 1) & (p < .25), (yt == 1) & (p >= .25))) > 0


def test_pipeline_end_to_end(csv_path, tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "tune_lgbm", lambda X, y, build, n_trials: ({}, SimpleNamespace(best_value=0.5)))
    s = tr.run_pipeline(csv_path, n_trials=1, out_dir=tmp_path / "a", tables_dir=tmp_path / "t", make_estimator=hgb)
    assert (tmp_path / "a" / "model.joblib").exists() and (tmp_path / "t" / "test_scores.csv").exists()
    assert 0 < s["threshold"] < 1
    pred = pd.read_csv(tmp_path / "a" / "test_predictions.csv")
    assert {"p_logreg", "p_raw", "p_cal", "SEX_LABEL", "AGE_GROUP"} <= set(pred.columns)
