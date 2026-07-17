import numpy as np
import pandas as pd

from paper import execution as X


def _fixture(eband_of):
    # 3 months x 20 stocks (permno 100..119), top-decile k=2. Month 0: stock
    # 100,101 are the top-2 (held). Months 1-2: 100,101 drift to ~50th pct
    # (score ~10.5 while others hold 18..1), so they are held-but-drifted names
    # whose retention is decided by their PER-STOCK band. eband_of: permno->E.
    rows = []
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    for mi, d in enumerate(dates):
        for i in range(20):
            permno = 100 + i
            if mi == 0:
                score = 20 - i                      # 100 highest, 101 next
            elif permno in (100, 101):
                score = 10.5 - (permno - 100) * 0.1  # drift to mid rank (~50th)
            else:
                score = 20 - i
            rows.append({'date': d, 'permno': permno, 'me': 1.0, 'pi': 0.0,
                         'score': score, 'mom_12': 0.0, 'ret_fwd': 0.0,
                         'state3': 0, 'eband': float(eband_of(permno))})
    return pd.DataFrame(rows)


def test_stock_var_band_constant_reduces_to_nmv():
    x = _fixture(lambda p: 20)
    a, _ = X.simulate(x, ('stock_var_band', None), score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)


def test_stock_var_band_per_stock_discrimination():
    # same rank (~50th pct), different per-stock band: stock 100 wide (kept),
    # stock 101 tight (expelled), in the SAME month.
    x = _fixture(lambda p: 70 if p == 100 else 10)
    _, led = X.simulate(x, ('stock_var_band', None), score_col='score')
    d1 = x['date'].unique()[1]
    exited = set(led[(led['date'] == d1)
                     & (led['cause'] == 'exit_rank')]['permno'])
    assert 100 not in exited      # wide band -> held despite drifting to 50th pct
    assert 101 in exited          # tight band -> expelled at the same rank
