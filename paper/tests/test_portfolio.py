import pandas as pd
import pytest

from paper.src import portfolio as P


def _toy():
    rows = []
    for p in range(1, 11):
        rows.append({'date': pd.Timestamp('2020-01-31'), 'permno': p,
                     'me': 100.0 if p == 10 else 10.0,
                     'score': float(p), 'ret_fwd': 0.01 * p})
    return pd.DataFrame(rows)


def test_long_only_top_decile_vw():
    df = _toy()
    r, h = P.long_only_top(df, 'score', frac=0.10)
    # top decile of 10 names = 1 name: permno 10 -> ret 0.10, weight 1.0
    assert r.loc[pd.Timestamp('2020-01-31')] == pytest.approx(0.10)
    assert h['weight'].sum() == pytest.approx(1.0)
    assert set(h['permno']) == {10}


def test_vw_benchmark():
    df = _toy()
    b = P.vw_benchmark(df)
    expected = (100 * 0.10 + sum(10 * 0.01 * p for p in range(1, 10))) / 190
    assert b.iloc[0] == pytest.approx(expected)


def test_two_names_weighting():
    df = _toy()
    r, h = P.long_only_top(df, 'score', frac=0.20)   # permnos 9 & 10
    w = h.set_index('permno')['weight']
    assert w.loc[10] == pytest.approx(100 / 110)
    assert r.iloc[0] == pytest.approx((100 * 0.10 + 10 * 0.09) / 110)
