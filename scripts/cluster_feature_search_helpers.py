"""Feature-search primitives for picked-stock cluster prediction.

Helpers:
  pi_panic_freq(pi_series, date, n_months, threshold=0.5)
  past_strategy_sharpe(returns_series, date, n_months)
  cross_section_skew(monthly_panel, mom_cols)
  picked_stock_fingerprint(picks_df, mom_panel, mom_cols)
"""
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
