"""Tests for paper/tv_band_multi.py (per-strategy flexible-banding study)."""
import os

import numpy as np
import pandas as pd
import pytest

from paper import tv_band as T
from paper import tv_band_multi as M


def test_load_strat_panel_momentum():
    p = M.load_strat_panel('momentum')
    need = ({'date', 'permno', 'me', 'pi', 'score_pi', 'ret_fwd'}
            | {f'mom_{h}' for h in range(1, 13)})
    assert need.issubset(p.columns)
    assert not p.duplicated(['date', 'permno']).any()
    assert p['date'].min().year <= 1993 and p['date'].max().year >= 2024
    assert p.groupby('date').size().median() >= 500


def test_trust_check_monthly_turnover_vs_prior_campaign():
    # cross-engine check: the clean-room monthly momentum book's one-way turnover
    # over the prior campaign's window should match turnover_diagnostic.csv
    prior = pd.read_csv(os.path.join(T.OUT_DIR, 'turnover_diagnostic.csv'))
    prior_to = float(prior.loc[prior['strategy'] == 'momentum', 'to_mo'].iloc[0])
    p = M.load_strat_panel('momentum')
    m, _ = T.simulate(p[p['date'] >= '2011-01-01'], T.policy_monthly)
    assert abs(float(m['turnover'].mean()) - prior_to) < 0.03


def test_objective_and_1se_guard():
    idx = pd.date_range('2001-01-31', periods=60, freq='ME')
    rng = np.random.default_rng(0)
    net = pd.Series(rng.normal(0.005, 0.02, 60), index=idx)
    to = pd.Series(0.3, index=idx)
    o = M.monthly_objective(net, to, lam=1.0)
    assert np.allclose(o, net - 1.0 * to)
    # guard: identical arms -> keep static; clearly-better arm -> adopt
    same = M.passes_1se(o, o)
    assert same is False
    better = M.passes_1se(o + 0.05, o)
    assert better is True


def test_trailing_targets_obj_valid():
    p = M.load_strat_panel('momentum')
    p = p[p['date'] >= '2015-01-01']            # small slice for speed
    feat = T.month_features(p)
    p = p[p['date'].isin(feat.index)]
    sp = T.load_spreads()
    tgt = M.trailing_optimal_targets_obj(p, feat, sp, lam=1.0, window=24)
    assert {'tgt_enter', 'tgt_exit'}.issubset(tgt.columns)
    assert (tgt['tgt_enter'] <= tgt['tgt_exit']).all()
    assert tgt['tgt_enter'].isin(M.ENTER_GRID).all()
    assert tgt['tgt_exit'].isin(M.EXIT_GRID).all()


def test_model_factories_both_configs():
    for cfg in ('conservative', 'flexible'):
        for name in ('gbm', 'ridge', 'mlp'):
            f, cols = M.model_factory(name, cfg)
            mdl = f()
            assert hasattr(mdl, 'fit') and hasattr(mdl, 'predict')
            assert len(cols) == (4 if cfg == 'conservative' else 25)


def test_band_monotonic_on_momentum():
    p = M.load_strat_panel('momentum')
    p = p[p['date'] >= '2018-01-01']            # small slice for speed
    to = {}
    for ex in (10, 20, 40):
        m, _ = T.simulate(p, T.policy_band(10, ex))
        to[ex] = m['turnover'].mean()
    assert to[40] < to[20] < to[10] + 1e-9
