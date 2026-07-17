import numpy as np
import pandas as pd

from paper import execution as X


def _fixture(ebands):
    # 3 months x 10 stocks; stock 100 sits at rank 5/10 (50th pct) in months
    # 2-3 (score 5.5 vs 9,8,7,6 above). eband set per month via `ebands`.
    rows = []
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    for mi, d in enumerate(dates):
        for i in range(10):
            score = 10 - i if mi == 0 else (10 - i if i != 0 else 5.5)
            rows.append({'date': d, 'permno': 100 + i, 'me': 1.0, 'pi': 0.0,
                         'score': score, 'mom_12': 0.0, 'ret_fwd': 0.0,
                         'state3': 0, 'eband': ebands[mi]})
    return pd.DataFrame(rows)


def test_var_band_constant_reduces_to_nmv():
    x = _fixture([20, 20, 20])
    a, _ = X.simulate(x, ('var_band', None), score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)


def test_var_band_switches_on_eband():
    # month 2 wide band (70) -> stock 100 (50th pct) KEPT; month 3 tight (10)
    # -> stock 100 expelled.
    x = _fixture([10, 70, 10])
    _, led = X.simulate(x, ('var_band', None), score_col='score')
    d2, d3 = x['date'].unique()[1], x['date'].unique()[2]
    assert 100 not in set(led[(led['date'] == d2)
                           & (led['cause'] == 'exit_rank')]['permno'])
    assert 100 in set(led[(led['date'] == d3)
                          & (led['cause'] == 'exit_rank')]['permno'])
