import numpy as np
import pandas as pd

from paper.src.ivol import compute_ivol


def _make_daily(n_days=130, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days)
    ff = pd.DataFrame({'mktrf': rng.normal(0, 0.01, n_days),
                       'smb': rng.normal(0, 0.005, n_days),
                       'hml': rng.normal(0, 0.005, n_days),
                       'rf': 0.0}, index=dates)
    # stock A: beta 1 on mkt + known idio sigma; stock B: half the idio sigma
    sig_a, sig_b = 0.02, 0.01
    ra = ff['mktrf'] + rng.normal(0, sig_a, n_days)
    rb = ff['mktrf'] + rng.normal(0, sig_b, n_days)
    daily = pd.concat([
        pd.DataFrame({'permno': 1, 'dlycaldt': dates, 'ret': ra.values}),
        pd.DataFrame({'permno': 2, 'dlycaldt': dates, 'ret': rb.values}),
    ], ignore_index=True)
    return daily, ff, sig_a, sig_b


def test_ivol_recovers_idio_sigma_ordering_and_scale():
    daily, ff, sig_a, sig_b = _make_daily()
    out = compute_ivol(daily, ff)
    last = out[out['date'] == out['date'].max()].set_index('permno')['ivol']
    assert last.loc[1] > last.loc[2]                     # ordering
    assert abs(last.loc[1] - sig_a) / sig_a < 0.30       # scale ~ sigma
    assert abs(last.loc[2] - sig_b) / sig_b < 0.30


def test_pit_no_future_days():
    daily, ff, *_ = _make_daily()
    out = compute_ivol(daily, ff)
    # first month-end has < 60 trailing days -> no row
    first_me = (pd.Series(pd.bdate_range('2020-01-01', periods=1))
                .iloc[0] + pd.offsets.MonthEnd(0))
    assert (out['date'] > first_me).all()
