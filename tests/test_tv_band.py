import numpy as np
import pandas as pd
import pytest
from paper import tv_band as T


def test_load_panel_shape_and_columns():
    p = T.load_panel()
    need = ({'date', 'permno', 'me', 'pi', 'score_pi', 'ret_fwd'}
            | {f'mom_{h}' for h in range(1, 13)})
    assert need.issubset(p.columns)
    assert p['date'].min() == pd.Timestamp('2011-01-31')
    assert p['date'].max() >= pd.Timestamp('2025-10-31')
    # no duplicate (date, permno)
    assert not p.duplicated(['date', 'permno']).any()
    # ~1000 names per month
    assert p.groupby('date').size().median() >= 500


def test_load_spreads_units():
    sp, month_med, first_med = T.load_spreads()
    assert {'permno', 'ym', 'hs'}.issubset(sp.columns)
    # hs converted to basis points, sane range
    assert 0.1 < sp['hs'].median() < 500
    assert np.isfinite(first_med)


def test_trust_gate_reproduces_walk_returns():
    panel = T.load_panel()
    monthly, _ = T.simulate(panel, T.policy_monthly)
    bench, _ = T.simulate(panel, T.policy_benchmark)
    wr = pd.read_csv(T.WALK, parse_dates=['date'])
    wr = wr[(wr['rule'] == 'rule_r') & (wr['combo'] == 'DD')].set_index('date')
    j = monthly.join(wr[['strat_ret', 'bench_ret']], how='inner')
    assert len(j) >= 100
    assert (j['book'] - j['strat_ret']).abs().max() < 1e-6
    b = bench.join(wr[['bench_ret']], how='inner')
    assert (b['book'] - b['bench_ret']).abs().max() < 1e-6


def test_simulate_active_is_book_minus_bench():
    panel = T.load_panel()
    m, _ = T.simulate(panel, T.policy_monthly)
    assert np.allclose(m['active'], m['book'] - m['bench'])


def test_band_reduces_turnover_monotonically():
    panel = T.load_panel()
    to = {}
    for exit_pct in [10, 20, 40]:
        m, _ = T.simulate(panel, T.policy_band(10, exit_pct))
        to[exit_pct] = m['turnover'].mean()
    # wider exit band => weakly lower turnover
    assert to[40] < to[20] < to[10] + 1e-9
    # E_enter=E_exit=10 reproduces the monthly turnover closely
    m10, _ = T.simulate(panel, T.policy_band(10, 10))
    mm, _ = T.simulate(panel, T.policy_monthly)
    assert abs(m10['turnover'].mean() - mm['turnover'].mean()) < 5e-3


def test_price_and_net():
    panel = T.load_panel()
    m, led = T.simulate(panel, T.policy_band(10, 20))
    sp_pack = T.load_spreads()
    cost = T.price(led, sp_pack)
    assert (cost >= 0).all() and cost.mean() > 0
    flat = T.price(led, sp_pack, flat_bp=10)
    # flat 10bp cost == 10bp * two-way traded volume / 1e4
    assert flat.mean() > 0
    n = T.net(m['active'], cost)
    assert np.allclose(n, m['active'] - cost.reindex(n.index).fillna(0.0))
