"""Clean-room term-structure time-varying band — Gate 0 (ceiling + trust gate).
Design: docs/superpowers/specs/2026-07-20-tv-band-design.md
Plan:   docs/superpowers/plans/2026-07-20-tv-band-gate0.md

Imports NOTHING from paper.execution / paper.banding_study by design: this is an
independent re-implementation whose `monthly` policy must reproduce walk_returns.csv.
Usage: .venv/bin/python -m paper.tv_band --stage gate0
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper.src.clusters import MOMS                             # noqa: E402  (col names only)

XSEC = os.path.join(C.RESULTS, 'xsec')
HS_PARQUET = os.path.join(C.RESULTS, 's5', 'half_spreads.parquet')
STOCKS = C.STOCKS_PARQUET
WALK = os.path.join(C.RESULTS, 'walk_returns.csv')
OUT_DIR = os.path.join(C.RESULTS, 'banding_study')
YEARS = list(range(2011, 2026))


def load_panel():
    """Monthly main-model panel: xsec _DD frames joined to the 12 momentum horizons."""
    frames = []
    for y in YEARS:
        fp = os.path.join(XSEC, f'xsec_{y}_DD.parquet')
        frames.append(pd.read_parquet(
            fp, columns=['date', 'permno', 'me', 'pi', 'score_pi', 'ret_fwd']))
    x = pd.concat(frames, ignore_index=True)
    st = pd.read_parquet(STOCKS, columns=['date', 'permno'] + MOMS)
    p = x.merge(st, on=['date', 'permno'], how='left')
    return p.sort_values(['date', 'permno']).reset_index(drop=True)


def load_spreads():
    """(sp[permno,ym,hs_bp], per-month median ffilled, earliest-month median)."""
    sp = pd.read_parquet(HS_PARQUET, columns=['permno', 'ym', 'hs'])
    sp = sp.dropna(subset=['hs']).copy()
    sp['hs'] = sp['hs'] * 1e4                        # decimal -> bp
    month_med = sp.groupby('ym')['hs'].median()
    full = pd.period_range(month_med.index.min(), month_med.index.max(), freq='M')
    month_med = month_med.reindex(full).ffill()
    return sp, month_med, float(month_med.iloc[0])
