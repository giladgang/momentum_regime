"""
src/utils.py
============
Shared utility functions for the momentum regime shifts pipeline.

Centralises portfolio construction, performance metrics, and data loading
so that every script uses the same logic.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_artefacts(path=None):
    """Load the cross-sectional artefacts pickle."""
    if path is None:
        path = os.path.join(_PROJECT_ROOT, cfg.ARTEFACTS_PATH)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_panel_with_regimes(path=None):
    """Load panel_with_regimes.parquet."""
    if path is None:
        path = os.path.join(_PROJECT_ROOT, cfg.PANEL_WITH_REGIMES_PATH)
    return pd.read_parquet(path)


def load_ff_factors(path=None):
    """Load Fama-French factors."""
    if path is None:
        path = os.path.join(_PROJECT_ROOT, cfg.FF_FACTORS_PATH)
    return pd.read_parquet(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio construction
# ═══════════════════════════════════════════════════════════════════════════════

def long_short_port(df_test, score_col, fee=None):
    """
    Monthly long-short portfolio with NYSE breakpoints.

    Each month: NYSE P10/P90 breakpoints on *score_col*.
    Long >= P90, Short <= P10, value-weighted by market equity.
    Transaction costs applied to turnover in both legs.

    Returns
    -------
    pd.Series
        Monthly long-short returns indexed by date.
    """
    if fee is None:
        fee = cfg.TRADING_FEE
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']


# ═══════════════════════════════════════════════════════════════════════════════
# Performance metrics
# ═══════════════════════════════════════════════════════════════════════════════

def metrics(r):
    """
    Compute annualised performance metrics from monthly returns.

    Returns
    -------
    tuple
        (ann_ret, ann_vol, sharpe, max_drawdown)
    """
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return 0.0, 0.0, 0.0, 0.0
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


def compute_sharpe(r):
    """Compute annualised Sharpe ratio from monthly returns."""
    r = pd.Series(r).dropna()
    if len(r) == 0 or r.std() == 0:
        return 0.0
    return r.mean() / r.std() * np.sqrt(12)


# ═══════════════════════════════════════════════════════════════════════════════
# Path helpers
# ═══════════════════════════════════════════════════════════════════════════════

def project_path(*parts):
    """Join parts relative to the project root."""
    return os.path.join(_PROJECT_ROOT, *parts)


def tables_path(filename):
    """Path to a file in the tables/ directory."""
    return os.path.join(_PROJECT_ROOT, cfg.TABLES_DIR, filename)


def plots_path(filename):
    """Path to a file in the plots output directory."""
    return os.path.join(_PROJECT_ROOT, cfg.PLOTS_DIR, filename)
