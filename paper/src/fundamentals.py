"""Fundamental signal levels from the (gated, already-run) Compustat pull.

be / op / cop / asset_growth per (permno, avail_date), PIT-safe: every row
carries avail_date = datadate + 6 months (stamped by paper/pull_fundamentals).
Definitions: FF (2015) book equity and operating profitability; Ball et al.
(2016) cash-based adjustments (missing deltas treated as 0); asset growth
at/at_lag - 1.
"""
import numpy as np
import pandas as pd

from paper import config as C

_KEEP = ['permno', 'datadate', 'avail_date', 'fyear',
         'be', 'op', 'cop', 'asset_growth']


def compute_signals(f):
    f = f.sort_values(['gvkey', 'fyear']).copy()

    seq = f['seq'].fillna(f['ceq'] + f['pstk'].fillna(0))
    seq = seq.fillna(f['at'] - f['lt'])
    ps = f['pstkrv'].fillna(f['pstkl']).fillna(f['pstk']).fillna(0)
    be = seq + f['txditc'].fillna(0) - ps
    f['be'] = be.where(be > 0)

    op_raw = f['revt'] - f['cogs'] - f['xsga'].fillna(0) - f['xint'].fillna(0)
    op_raw = op_raw.where(f['revt'].notna() & f['cogs'].notna())
    f['op'] = op_raw / f['be']

    g = f.groupby('gvkey')
    adj = (-g['rect'].diff().fillna(0) - g['invt'].diff().fillna(0)
           - g['xpp'].diff().fillna(0) + g['ap'].diff().fillna(0)
           + g['xacc'].diff().fillna(0))
    f['cop'] = (op_raw + adj) / f['be']

    ag = f['at'] / f['at_lag'] - 1
    f['asset_growth'] = ag.where(f['at'].notna() & (f['at_lag'] > 0))
    return f[_KEEP].reset_index(drop=True)


def load_fundamentals():
    return compute_signals(pd.read_parquet(C.FUNDA_PARQUET))
