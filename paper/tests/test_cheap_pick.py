import numpy as np
import pandas as pd

from paper import execution as X


def _fixture(hs_of):
    # 3 months x 20 stocks, k (top decile) = 2. Scores fixed: permno 100 best,
    # 119 worst. hs_lag set per stock via hs_of(permno).
    rows = []
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    for d in dates:
        for i in range(20):
            rows.append({'date': d, 'permno': 100 + i, 'me': 1.0, 'pi': 0.0,
                         'score': 20 - i, 'mom_12': 0.0, 'ret_fwd': 0.0,
                         'state3': 0, 'hs_lag': float(hs_of(100 + i))})
    return pd.DataFrame(rows)


def test_cheap_pick_reduces_to_nmv_when_pool_is_decile():
    x = _fixture(lambda p: 10.0)
    a, _ = X.simulate(x, ('cheap_pick', (20, 10)), score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)


def test_cheap_pick_substitutes_cheap_for_expensive():
    # pool = top 20% (4 names: 100,101,102,103). 100 & 101 are EXPENSIVE,
    # 102 & 103 cheap. k=2 -> cheap_pick must hold {102,103}, not {100,101}.
    x = _fixture(lambda p: 9.0 if p in (100, 101) else 1.0)
    _, led = X.simulate(x, ('cheap_pick', (20, 20)), score_col='score')
    d0 = x['date'].unique()[0]
    entered = set(led[(led['date'] == d0) & (led['cause'] == 'entry')]['permno'])
    assert entered == {102, 103}
