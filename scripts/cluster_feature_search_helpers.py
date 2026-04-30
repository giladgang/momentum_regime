"""Feature-search primitives for picked-stock cluster prediction.

Helpers:
  pi_panic_freq(pi_series, date, n_months, threshold=0.5)
  past_strategy_sharpe(returns_series, date, n_months)
  cross_section_skew(monthly_panel, mom_cols)
  picked_stock_fingerprint(picks_df, mom_panel, mom_cols)
"""
import warnings

import numpy as np
import pandas as pd


def pi_panic_freq(pi_series, date, n_months, threshold=0.5):
    """Fraction of past n_months months with pi > threshold (excluding `date`).

    Parameters
    ----------
    pi_series : pd.Series indexed by date
    date : pd.Timestamp or compatible
    n_months : int
    threshold : float (default 0.5)

    Returns
    -------
    float in [0, 1] or NaN if insufficient history.
    """
    if date not in pi_series.index:
        return float('nan')
    pos = pi_series.index.get_loc(date)
    if pos < n_months:
        return float('nan')
    window = pi_series.iloc[pos - n_months:pos]
    return float((window > threshold).mean())


def past_strategy_sharpe(returns_series, date, n_months):
    """Annualised Sharpe over past n_months months (excluding `date`).

    Uses sample std (ddof=1) and sqrt(12) annualisation, matching the
    project's bootstrap_helpers convention.

    Returns
    -------
    float Sharpe ratio or NaN if insufficient history or zero std.
    """
    if date not in returns_series.index:
        return float('nan')
    pos = returns_series.index.get_loc(date)
    if pos < n_months:
        return float('nan')
    window = returns_series.iloc[pos - n_months:pos]
    sd = window.std(ddof=1)
    # atol=1e-8 catches IEEE 754 noise (~1e-18) from ddof=1 on constant
    # series; real monthly strategy vol >= 0.005, so no false-positive risk.
    if np.isclose(sd, 0) or not np.isfinite(sd):
        return float('nan')
    return float((window.mean() / sd) * np.sqrt(12))


def cross_section_skew(monthly_panel, mom_cols):
    """Per-month skewness across stocks at each momentum horizon.

    Parameters
    ----------
    monthly_panel : DataFrame with column 'date' + each of mom_cols
    mom_cols : list of column names

    Returns
    -------
    DataFrame indexed by date, columns = mom_cols, values = skewness
    """
    return monthly_panel.groupby('date')[mom_cols].skew()


SHORT_H = [1, 2, 3, 4]
MID_H   = [5, 6, 7, 8]
LONG_H  = [9, 10, 11, 12]


def picked_stock_fingerprint(picks_df, mom_panel, mom_cols):
    """Per-month 15-d fingerprint of long-leg picks.

    Parameters
    ----------
    picks_df : DataFrame with columns 'date', 'permno' (long-leg picks).
        Duplicate (date, permno) rows are deduplicated before merging.
    mom_panel : DataFrame with columns 'date', 'permno', and each of mom_cols.
    mom_cols : list of length 12 -- momentum horizon columns in order
        (mom_1, mom_2, ..., mom_12).

    Returns
    -------
    DataFrame indexed by date with 15 columns:
      pick_mom_1 .. pick_mom_12  : mean of picks at each horizon
      pick_disp_short            : mean of per-horizon std at horizons 1..4
      pick_disp_mid              : mean of per-horizon std at horizons 5..8
      pick_disp_long             : mean of per-horizon std at horizons 9..12
    """
    if len(mom_cols) != 12:
        raise ValueError(f'expected 12 mom columns, got {len(mom_cols)}')
    deduped = picks_df[['date', 'permno']].drop_duplicates()
    merged = deduped.merge(
        mom_panel[['date', 'permno'] + list(mom_cols)],
        on=['date', 'permno'], how='inner',
    )
    n_dropped = len(deduped) - len(merged)
    if n_dropped > 0:
        warnings.warn(
            f'picked_stock_fingerprint: {n_dropped} pick rows had no panel match '
            f'and were dropped (inner merge). Check date/permno alignment.',
            stacklevel=2,
        )
    grp = merged.groupby('date')[list(mom_cols)]
    means = grp.mean()
    stds = grp.std(ddof=1)

    fp = means.rename(columns={c: f'pick_{c}' for c in mom_cols})

    short_cols = [mom_cols[h - 1] for h in SHORT_H]
    mid_cols   = [mom_cols[h - 1] for h in MID_H]
    long_cols  = [mom_cols[h - 1] for h in LONG_H]
    fp['pick_disp_short'] = stds[short_cols].mean(axis=1)
    fp['pick_disp_mid']   = stds[mid_cols].mean(axis=1)
    fp['pick_disp_long']  = stds[long_cols].mean(axis=1)
    return fp
