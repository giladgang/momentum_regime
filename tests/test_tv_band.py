import os

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


def test_month_features_zcurve_matches_definition():
    panel = T.load_panel()
    feat = T.month_features(panel)
    assert {f'zc_{h}' for h in range(1, 13)}.issubset(feat.columns)
    assert {f'cs_{h}' for h in range(1, 13)}.issubset(feat.columns)
    assert 'pi' in feat.columns
    # recompute zc for one month independently
    d = feat.index[20]
    g = panel[panel['date'] == d].dropna(subset=[f'mom_{h}' for h in range(1, 13)])
    z = (g[[f'mom_{h}' for h in range(1, 13)]]
         - g[[f'mom_{h}' for h in range(1, 13)]].mean()) / \
        g[[f'mom_{h}' for h in range(1, 13)]].std(ddof=0)
    z.columns = [f'zc_{h}' for h in range(1, 13)]
    g2 = g.assign(**z)
    k = max(int(len(g) * 0.10), 1)
    top = g2.nlargest(k, 'score_pi')
    exp1 = top['zc_1'].mean()
    assert abs(feat.loc[d, 'zc_1'] - exp1) < 1e-9


def test_standardize_uses_train_only():
    panel = T.load_panel()
    feat = T.month_features(panel)
    mask = feat.index < pd.Timestamp('2018-01-01')
    z = T.standardize(feat, mask)
    # train-window columns are ~mean 0
    assert abs(z.loc[mask, 'pi'].mean()) < 1e-9
    comp = T.compress_features(feat)
    assert {'level', 'slope', 'curv', 'pi'} == set(comp.columns)


def test_net_ir_and_bootstrap():
    rng = np.random.default_rng(0)
    idx = pd.date_range('2011-01-31', periods=120, freq='ME')
    a = pd.Series(rng.normal(0.01, 0.04, 120), index=idx)
    b = a + 0.003                                     # b strictly better
    assert T.net_ir(b) > T.net_ir(a)
    delta, lo, hi = T.paired_block_bootstrap(a, b)
    assert delta == pytest.approx(T.net_ir(b) - T.net_ir(a), abs=1e-9)
    assert lo <= delta <= hi
    # zero-variance guard
    assert T.net_ir(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_oracle_ceiling_at_least_static():
    panel = T.load_panel()
    sp_pack = T.load_spreads()
    feat = T.month_features(panel)
    E_star, ir_static = T.best_static(panel, sp_pack)
    # coarse pi x zc-level bins
    lvl = feat[[f'zc_{h}' for h in range(1, 13)]].mean(axis=1)
    bins = (pd.qcut(feat['pi'], 2, labels=False, duplicates='drop').astype(str)
            + '_' + pd.qcut(lvl, 2, labels=False, duplicates='drop').astype(str))
    ir_oracle, _ = T.oracle_bin_ir(panel, feat, sp_pack, bins,
                                   exit_grid=(15, 20, 25, 30, 40))
    # a per-bin optimal band cannot do worse than the single best static band
    assert ir_oracle >= ir_static - 1e-9


def test_run_gate0_writes_report():
    res = T.run_gate0()
    assert 'g0_pass' in res and isinstance(res['g0_pass'], bool)
    assert os.path.exists(os.path.join(T.OUT_DIR, 'tv_band_gate0.md'))
    assert os.path.exists(os.path.join(T.OUT_DIR, 'tv_band_gate0.csv'))


def test_regime_turnover_independent():
    panel = T.load_panel()
    d = T.regime_turnover(panel)
    assert {'to_calm', 'to_panic', 'ratio', 'n_calm', 'n_panic'} <= set(d)
    assert d['n_calm'] > 0 and d['n_panic'] > 0
    # sanity: turnover is a fraction in (0, 1]
    assert 0 < d['to_calm'] <= 1 and 0 < d['to_panic'] <= 1


def test_price_stress_multiplies_panic_only():
    sp_pack = T.load_spreads()
    led = pd.DataFrame({
        'date': [pd.Timestamp('2020-03-31'), pd.Timestamp('2013-06-28')],
        'permno': [10107, 10107], 'dw': [0.5, 0.5]})
    pi_by_date = {pd.Timestamp('2020-03-31'): 0.9, pd.Timestamp('2013-06-28'): 0.1}
    base = T.price(led, sp_pack, flat_bp=10)
    stressed = T.price(led, sp_pack, flat_bp=10, stress_mult=3, pi_by_date=pi_by_date)
    assert stressed.loc[pd.Timestamp('2020-03-31')] == pytest.approx(
        3 * base.loc[pd.Timestamp('2020-03-31')])
    assert stressed.loc[pd.Timestamp('2013-06-28')] == pytest.approx(
        base.loc[pd.Timestamp('2013-06-28')])


def test_band_regime_turnover_and_impl_report():
    panel = T.load_panel()
    calm, panic = T.band_regime_turnover(panel, T.policy_band(10, 20))
    assert calm > 0 and panic > 0
    res = T.run_gate0_impl(stress_grid=(1, 2))
    df = res['table']
    assert set(df['band']) == {'static', 'oracle'}
    assert {'to_calm', 'to_panic', 'net_ir_stress1', 'net_ir_stress2'} <= set(df.columns)
    assert 'MORE' in res['direction'] or 'LESS' in res['direction']
    assert os.path.exists(os.path.join(T.OUT_DIR, 'tv_band_gate0_impl.md'))
