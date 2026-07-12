"""Benchmark-relative monthly-return metrics for the applied study."""
import numpy as np
import pandas as pd


def ann_ret(r):
    r = r.dropna()
    return float((1 + r).prod() ** (12 / len(r)) - 1)


def ann_vol(r):
    return float(r.dropna().std() * np.sqrt(12))


def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan


def max_dd(r):
    w = (1 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1).min())


def active(r, b):
    r, b = r.align(b, join='inner')
    return r - b


def te(r, b):
    return float(active(r, b).std() * np.sqrt(12))


def ir(r, b):
    a = active(r, b)
    return float(a.mean() / a.std() * np.sqrt(12)) if a.std() > 0 else np.nan


def rolling_beta(r, b, window=36):
    r, b = r.align(b, join='inner')
    cov = r.rolling(window).cov(b)
    var = b.rolling(window).var()
    return cov / var


def capture(r, b):
    """Arithmetic-mean up/down capture: mean(r | b>0)/mean(b | b>0), same down."""
    r, b = r.align(b, join='inner')
    up, dn = b > 0, b < 0
    cup = float(r[up].mean() / b[up].mean()) if up.any() else np.nan
    cdn = float(r[dn].mean() / b[dn].mean()) if dn.any() else np.nan
    return cup, cdn


def one_way_turnover(holdings):
    piv = (holdings.pivot_table(index='date', columns='permno',
                                values='weight', aggfunc='sum')
           .fillna(0.0).sort_index())
    return 0.5 * piv.diff().abs().sum(axis=1).iloc[1:].reindex(piv.index)
