"""Strategy signal builders -> common xsec schema for execution.simulate.

Every builder returns (date, permno, me, pi, score, mom_12, ret_fwd, state3)
over the top-UNIVERSE_N universe; HIGHER score = long side (signs
pre-flipped). Monthly cadence: momentum, reversal, lowvol, xgb. Annual
June-formation, held July..June: value, profitability, investment.
XGB sample is 2011-2025 (scores exist only for the walk years).
"""
import numpy as np
import pandas as pd

from paper import config as C
from paper.src import regime_labels, universe
from paper.src.fundamentals import load_fundamentals

STRATEGIES = ['momentum', 'reversal', 'lowvol', 'xgb', 'value',
              'profitability', 'investment']
SCHEMA = ['date', 'permno', 'me', 'pi', 'score', 'mom_12', 'ret_fwd',
          'state3']


def _base_universe():
    s = pd.read_parquet(C.STOCKS_PARQUET,
                        columns=['permno', 'date', 'me', 'mom_1', 'mom_12',
                                 'ret_fwd'])
    s['date'] = pd.to_datetime(s['date'])
    # top_n returns the FILTERED frame (all columns), verified 2026-07-15
    s = universe.top_n(s, C.UNIVERSE_N)
    return s[s['date'] >= C.STUDY_START]


def _attach_regime(x):
    pi = pd.read_parquet(C.PI_MONTHLY)
    pi['date'] = pd.to_datetime(pi['date'])
    panel = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
    panel['date'] = pd.to_datetime(panel['date'])
    m = pi.merge(panel, on='date')
    st = regime_labels.label_states(m['date'], m['pi'].values,
                                    m['vwretd'].values)
    m = m.assign(state3=st.values)[['date', 'pi', 'state3']]
    out = x.merge(m, on='date', how='inner')
    return out


def _finalize(x):
    x = x.dropna(subset=['score', 'ret_fwd', 'me'])
    return (x[SCHEMA].sort_values(['date', 'permno'])
            .reset_index(drop=True))


def _annual_scores(field):
    """June-Y formation from fiscal years ending in calendar Y-1."""
    f = load_fundamentals()
    s = pd.read_parquet(C.STOCKS_PARQUET, columns=['permno', 'date', 'me'])
    s['date'] = pd.to_datetime(s['date'])
    dec = s[s['date'].dt.month == 12][['permno', 'date', 'me']]
    dec = dec.rename(columns={'me': 'me_dec'})
    dec['dec_year'] = dec['date'].dt.year

    f = f[f['datadate'].dt.year >= 1986].copy()
    f['form_year'] = f['datadate'].dt.year + 1        # June of Y
    # PIT guard: must be available by June-30 of form_year
    ok = f['avail_date'] <= (pd.to_datetime(f['form_year'].astype(str)
                                            + '-06-30'))
    f = f[ok].sort_values('datadate').drop_duplicates(
        ['permno', 'form_year'], keep='last')

    if field == 'value':
        f = f.merge(dec[['permno', 'dec_year', 'me_dec']],
                    left_on=['permno', f['form_year'] - 1],
                    right_on=['permno', 'dec_year'])
        f['sig'] = f['be'] / f['me_dec']
    elif field == 'profitability':
        f['sig'] = f['cop']
    else:                                             # investment
        f['sig'] = -f['asset_growth']
    return f[['permno', 'form_year', 'sig']].dropna()


def _annual(field):
    x = _base_universe()
    # formation-year key: July..Dec of Y and Jan..June of Y+1 use June-Y score
    yr = x['date'].dt.year - (x['date'].dt.month <= 6).astype(int)
    x = x.assign(form_year=yr)
    sc = _annual_scores(field)
    x = x.merge(sc, on=['permno', 'form_year'], how='inner')
    x = x.rename(columns={'sig': 'score'})
    return _finalize(_attach_regime(x))


def build(strategy):
    if strategy == 'momentum':
        x = _base_universe().assign(score=lambda d: d['mom_12'])
        return _finalize(_attach_regime(x))
    if strategy == 'reversal':
        x = _base_universe().assign(score=lambda d: -d['mom_1'])
        return _finalize(_attach_regime(x))
    if strategy == 'lowvol':
        iv = pd.read_parquet(C.IVOL_MONTHLY)
        iv['date'] = pd.to_datetime(iv['date'])
        x = _base_universe().merge(iv, on=['permno', 'date'], how='inner')
        x = x.assign(score=lambda d: -d['ivol'])
        return _finalize(_attach_regime(x))
    if strategy == 'xgb':
        from paper import execution as X
        x = X.load_xsec().rename(columns={'score_pi': 'score'})
        panel = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
        panel['date'] = pd.to_datetime(panel['date'])
        m = (x[['date', 'pi']].drop_duplicates()
             .merge(panel, on='date'))
        st = regime_labels.label_states(m['date'], m['pi'].values,
                                        m['vwretd'].values)
        x = x.merge(m.assign(state3=st.values)[['date', 'state3']],
                    on='date')
        return _finalize(x)
    if strategy in ('value', 'profitability', 'investment'):
        return _annual(strategy)
    raise ValueError(strategy)
