import numpy as np
import pandas as pd

from paper import execution as X


def _fixture():
    # 3 months x 10 stocks; scores fixed; month clusters [0,1,2]; stock 100
    # sits at rank 5/10 (50th pct) in months 2-3 (score 5.5 vs 9,8,7,6 above).
    rows = []
    dates = pd.date_range('2020-01-31', periods=3, freq='ME')
    for mi, d in enumerate(dates):
        for i in range(10):
            score = 10 - i if mi == 0 else (10 - i if i != 0 else 5.5)
            rows.append({'date': d, 'permno': 100 + i, 'me': 1.0, 'pi': 0.0,
                         'score': score, 'mom_12': 0.0, 'ret_fwd': 0.0,
                         'state3': 0, 'cluster': mi,
                         'stock_cluster': (0 if i == 0 else 1)})
    return pd.DataFrame(rows)


def test_cluster_band_equal_reduces_to_nmv():
    x = _fixture()
    a, _ = X.simulate(x, ('cluster_band', (20, 20, 20, 20)), score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)


def test_stock_cluster_band_equal_reduces_to_nmv():
    x = _fixture()
    a, _ = X.simulate(x, ('stock_cluster_band', {0: 20, 1: 20, 2: 20, 3: 20}),
                      score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)


def test_cluster_band_switches_on_month_cluster():
    x = _fixture()
    # cluster 1 (month 2) gets E=70 -> stock 100 (50th pct) KEPT; cluster 2
    # (month 3) gets E=10 -> stock 100 expelled.
    _, ledger = X.simulate(x, ('cluster_band', (10, 70, 10, 10)),
                           score_col='score')
    d2, d3 = x['date'].unique()[1], x['date'].unique()[2]
    kept_m2 = 100 not in set(ledger[(ledger['date'] == d2)
                             & (ledger['cause'] == 'exit_rank')]['permno'])
    dropped_m3 = 100 in set(ledger[(ledger['date'] == d3)
                            & (ledger['cause'] == 'exit_rank')]['permno'])
    assert kept_m2 and dropped_m3


def test_stock_cluster_band_switches_on_stock_cluster():
    x = _fixture()
    # stock 100 is stock_cluster 0; give cluster 0 a wide band (70) -> kept,
    # cluster 1 tight (10). Stock 100 at 50th pct stays.
    _, ledger = X.simulate(x, ('stock_cluster_band', {0: 70, 1: 10}),
                           score_col='score')
    d2 = x['date'].unique()[1]
    assert 100 not in set(ledger[(ledger['date'] == d2)
                          & (ledger['cause'] == 'exit_rank')]['permno'])
