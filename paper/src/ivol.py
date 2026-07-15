"""IVOL: trailing FF3-residual daily volatility, monthly snapshots (PIT).

ivol(permno, month-end m) = std of residuals from OLS of (ret - rf) on
(mktrf, smb, hml) over the trailing WINDOW=90 trading days ending at m
(minimum MIN_DAYS=60). Only days <= m are used. Low ivol = long side of the
low-volatility strategy (sign applied in signals.py).
"""
import numpy as np
import pandas as pd

WINDOW, MIN_DAYS = 90, 60


def _resid_std(y, X):
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return float((y - Xd @ beta).std(ddof=4))


def compute_ivol(daily, ff):
    d = daily.dropna(subset=['ret']).copy()
    d['dlycaldt'] = pd.to_datetime(d['dlycaldt'])
    ff = ff.sort_index()
    d = d.merge(ff, left_on='dlycaldt', right_index=True, how='inner')
    d['exret'] = d['ret'] - d['rf']
    d = d.sort_values(['permno', 'dlycaldt'])
    d['me_month'] = d['dlycaldt'] + pd.offsets.MonthEnd(0)

    rows = []
    for permno, g in d.groupby('permno', sort=False):
        g = g.reset_index(drop=True)
        ends = g.groupby('me_month').apply(lambda x: x.index[-1],
                                           include_groups=False)
        for m, i_end in ends.items():
            lo = i_end - WINDOW + 1
            if lo < 0 and i_end + 1 < MIN_DAYS:
                continue
            w = g.iloc[max(lo, 0): i_end + 1]
            if len(w) < MIN_DAYS:
                continue
            rows.append((permno, m,
                         _resid_std(w['exret'].values,
                                    w[['mktrf', 'smb', 'hml']].values)))
    return pd.DataFrame(rows, columns=['permno', 'date', 'ivol'])


def build_ivol_panel():
    """Build ivol_monthly.parquet from all dsf_v2 files + ff_daily."""
    import glob
    from paper import config as C
    ff = pd.read_parquet(C.FF_DAILY)
    frames = []
    for f in sorted(glob.glob('experiments/results/spreads/dsf_v2_*.parquet')):
        cols = pd.read_parquet(f).columns
        use = ['permno', 'dlycaldt', 'dlyprc'] + \
              (['dlyret'] if 'dlyret' in cols else [])
        d = pd.read_parquet(f, columns=use)
        if 'dlyret' in d:
            d['ret'] = d['dlyret']
        else:                       # fall back to close-to-close price return
            d = d.sort_values(['permno', 'dlycaldt'])
            d['ret'] = (d.groupby('permno')['dlyprc']
                        .pct_change(fill_method=None))
        frames.append(d[['permno', 'dlycaldt', 'ret']])
    daily = pd.concat(frames, ignore_index=True)
    out = compute_ivol(daily, ff)
    out.to_parquet(C.IVOL_MONTHLY, index=False)
    print(f'Saved {C.IVOL_MONTHLY}: {len(out):,} rows, '
          f'{out.date.min().date()} -> {out.date.max().date()}')
