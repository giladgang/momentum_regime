import numpy as np
import pandas as pd
import pytest

from paper.src import metrics as M

IDX = pd.date_range('2020-01-31', periods=24, freq='ME')


def test_basic_metrics_hand_computed():
    r = pd.Series([0.01] * 24, index=IDX)
    assert M.ann_ret(r) == pytest.approx((1.01 ** 12) - 1, rel=1e-9)
    assert M.ann_vol(r) == pytest.approx(0.0, abs=1e-12)
    b = pd.Series([0.005] * 24, index=IDX)
    a = M.active(r, b)
    assert a.iloc[0] == pytest.approx(0.005)
    assert np.isnan(M.ir(r, b))  # zero-TE edge case


def test_ir_te_and_mdd():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0.008, 0.04, 24), index=IDX)
    r = b + pd.Series(rng.normal(0.002, 0.01, 24), index=IDX)
    a = r - b
    assert M.te(r, b) == pytest.approx(a.std() * np.sqrt(12), rel=1e-9)
    assert M.ir(r, b) == pytest.approx(a.mean() / a.std() * np.sqrt(12), rel=1e-9)
    dd = M.max_dd(pd.Series([0.10, -0.50, 0.10], index=IDX[:3]))
    assert dd == pytest.approx(-0.50, rel=1e-9)


def test_capture_and_rolling_beta():
    b = pd.Series([0.02, -0.02] * 12, index=IDX)
    r = 0.5 * b
    up, down = M.capture(r, b)   # arithmetic-mean capture
    assert up == pytest.approx(0.5, rel=1e-9)
    assert down == pytest.approx(0.5, rel=1e-9)
    beta = M.rolling_beta(r, b, window=12)
    assert beta.dropna().iloc[-1] == pytest.approx(0.5, rel=1e-6)


def test_one_way_turnover():
    h = pd.DataFrame({
        'date': ['2020-01-31'] * 2 + ['2020-02-29'] * 2,
        'permno': [1, 2, 2, 3],
        'weight': [0.5, 0.5, 0.5, 0.5]})
    h['date'] = pd.to_datetime(h['date'])
    to = M.one_way_turnover(h)
    # sell all of 1 (0.5) + buy all of 3 (0.5) -> one-way = 0.5*(0.5+0.5)=0.5
    assert to.loc[pd.Timestamp('2020-02-29')] == pytest.approx(0.5)
