"""Materialize the primary applied-study dataset from the verified ext files.

Panel: copy of PANEL_EXT (full 1970..2025-11 macro history with frozen-train
z conventions, already integrated by the verified ext2025 import).
Stocks: CIZ-filtered common shares with momentum features and ret_fwd, built
exactly as the validated walk-forward ext loader
(experiments/2026-07-09-walkforward-ext2025.py::_load_stock_panel_ext), plus
`me` kept for universe ranking and value weights.
"""
import os

import numpy as np
import pandas as pd

from paper import config as C


def build():
    os.makedirs(C.DATA_OUT, exist_ok=True)
    panel = pd.read_parquet(C.PANEL_EXT)
    panel['date'] = pd.to_datetime(panel['date'])
    panel.to_parquet(C.PANEL_PARQUET, index=False)

    stocks = pd.read_parquet(C.STOCK_EXT,
                             columns=['permno', 'date', 'ret_adj', 'prc',
                                      'shrcd', 'exchcd', 'me'])
    stocks['date'] = pd.to_datetime(stocks['date'])

    # ext months (2025-*) lack `me`; fill from v2 mthcap (mthcap/1000 == me
    # units — enforced on the overlap months where both exist)
    cap = pd.read_parquet(os.path.join(C.EXT_DIR, 'msf_v2_raw.parquet'),
                          columns=['permno', 'mthcaldt', 'mthcap'])
    cap['ym'] = pd.to_datetime(cap['mthcaldt']).dt.to_period('M')
    cap['me_v2'] = cap['mthcap'] / 1000.0
    cap = cap[['permno', 'ym', 'me_v2']].drop_duplicates(['permno', 'ym'])
    stocks['ym'] = stocks['date'].dt.to_period('M')
    stocks = stocks.merge(cap, on=['permno', 'ym'], how='left')
    both = stocks.dropna(subset=['me', 'me_v2'])
    ratio = (both['me_v2'] / both['me']).median()
    assert 0.95 < ratio < 1.05, f'me vs mthcap/1000 unit mismatch: {ratio:.4f}'
    stocks['me'] = stocks['me'].fillna(stocks['me_v2'])
    stocks.drop(columns=['me_v2', 'ym'], inplace=True)

    stocks = build_stock_frame(stocks, max_date=panel['date'].max())
    stocks.to_parquet(C.STOCKS_PARQUET, index=False)
    print(f'[data_build] panel {panel.date.min().date()}..{panel.date.max().date()} '
          f'| stocks {len(stocks):,} rows ..{stocks.date.max().date()}')


def build_stock_frame(stocks, max_date):
    """Universe filters + momentum features + ret_fwd (validated ext-loader
    logic). Expects columns permno, date, ret_adj, prc, shrcd, exchcd, me."""
    stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)
    stocks = stocks[stocks['shrcd'].isin([10, 11])]
    stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
    stocks = stocks[stocks['prc'].abs() > C.PRICE_MIN].reset_index(drop=True)
    stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
    stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)
    for lb in range(1, 13):
        roll = (stocks.groupby('permno', sort=False)['_lr_s1']
                .rolling(lb, min_periods=lb).sum()
                .reset_index(level='permno', drop=True).sort_index())
        stocks[f'mom_{lb}'] = np.expm1(roll)
    stocks.drop(columns=['_lr', '_lr_s1'], inplace=True)
    stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(
        lambda x: x.shift(-1))
    moms = [f'mom_{lb}' for lb in range(1, 13)]
    stocks = stocks.dropna(subset=['ret_fwd', 'me'] + moms
                           ).reset_index(drop=True)
    return stocks[stocks['date'] <= max_date]


def load_panel():
    p = pd.read_parquet(C.PANEL_PARQUET)
    p['date'] = pd.to_datetime(p['date'])
    return p


def load_stocks(columns=None):
    s = pd.read_parquet(C.STOCKS_PARQUET, columns=columns)
    s['date'] = pd.to_datetime(s['date'])
    return s
