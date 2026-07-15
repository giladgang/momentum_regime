import numpy as np
import pandas as pd

from paper import execution as X


def _xsec_fixture():
    # 3 months x 10 stocks; scores fixed so ranks are stable; k = 1 (decile)
    # month 1 calm(0), month 2 crash(1), month 3 recovery(2)
    rows = []
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    st3 = [0, 1, 2]
    for mi, d in enumerate(dates):
        for i in range(10):
            score = 10 - i if mi == 0 else (10 - i if i != 0 else 5.5)
            # month>=2: stock 0 (prev top) slips to rank 5 of 10 (50th pct)
            rows.append({'date': d, 'permno': 100 + i, 'me': 1.0,
                         'pi': 0.0 if mi == 0 else 0.9, 'score': score,
                         'mom_12': 0.0, 'ret_fwd': 0.0, 'state3': st3[mi]})
    return pd.DataFrame(rows)


def test_regime3_band_switches_on_state():
    x = _xsec_fixture()
    # month >= 2: stock 100 (score 5.5) sits at rank 5 of 10 = 50th pct.
    # E_crash = 70%: 0.50 <= 0.70 -> KEPT in the crash month;
    # E_recovery = 10%: 0.50 > 0.10 -> DROPPED in the recovery month.
    r, ledger = X.simulate(x, ('regime3_band', (10, 70, 10)),
                           score_col='score')
    # crash month: no forced exit of stock 100
    assert 100 not in set(
        ledger[(ledger['date'] == x['date'].unique()[1])
               & (ledger['cause'] == 'exit_rank')]['permno'])
    # recovery month: stock 100 (rank 6 > E=10%) is expelled
    assert 100 in set(
        ledger[(ledger['date'] == x['date'].unique()[2])
               & (ledger['cause'] == 'exit_rank')]['permno'])


def test_regime3_all_equal_reduces_to_nmv_band():
    x = _xsec_fixture()
    a, _ = X.simulate(x, ('regime3_band', (20, 20, 20)), score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)
