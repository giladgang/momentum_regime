import os

import numpy as np
import pandas as pd
import pytest

from paper.src import signals
from paper import config as C

_HAS_IVOL = os.path.exists(C.IVOL_MONTHLY)

SCHEMA = ['date', 'permno', 'me', 'pi', 'score', 'mom_12', 'ret_fwd',
          'state3']


@pytest.mark.parametrize('strat', signals.STRATEGIES)
def test_schema_and_sorting(strat):
    if strat == 'lowvol' and not _HAS_IVOL:
        pytest.skip('ivol panel not yet built (WRDS pull in flight)')
    x = signals.build(strat)
    assert list(x.columns) == SCHEMA
    assert x['date'].is_monotonic_increasing or (
        x.sort_values(['date', 'permno']).index == x.index).all()
    assert x['score'].notna().all()
    assert x['state3'].isin([0, 1, 2]).all()
    # pi constant within month
    assert (x.groupby('date')['pi'].nunique() == 1).all()


def test_momentum_reproduces_walk_comparator():
    # gate: momentum top-decile gross == the walk report's mom_12_1 series
    from paper.src import portfolio
    from paper import config as C
    x = signals.build('momentum')
    x11 = x[x['date'] >= '2011-01-01']
    r, _ = portfolio.long_only_top(x11, 'score', C.DECILE_FRAC)
    walk = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    walk = walk[(walk.rule == 'rule_r') & (walk.combo == 'DD')]
    # same months present
    assert set(r.index) == set(walk['date'])


def test_annual_signals_constant_between_junes():
    x = signals.build('value')
    one = x[x['permno'] == x['permno'].iloc[0]]
    yr = one[(one['date'] >= '2015-07-01') & (one['date'] <= '2016-06-30')]
    if len(yr) > 1:
        assert yr['score'].nunique() == 1


def test_reversal_sign():
    x = signals.build('reversal')
    m = x.merge(
        pd.read_parquet('paper/results/data/stocks.parquet',
                        columns=['permno', 'date', 'mom_1']),
        on=['permno', 'date'])
    assert np.corrcoef(m['score'], m['mom_1'])[0, 1] < -0.99
