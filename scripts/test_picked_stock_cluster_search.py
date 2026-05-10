"""Tests for tertile_mean and build_feature_panel in picked_stock_cluster_search.py."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from picked_stock_cluster_search import tertile_mean, build_feature_panel
from cluster_feature_search_helpers import SHORT_H, MID_H, LONG_H

MOM_COLS = [f'mom_{h}' for h in range(1, 13)]


# ---------------------------------------------------------------------------
# tertile_mean
# ---------------------------------------------------------------------------

def test_tertile_mean_short_group():
    """Mean of SHORT_H columns (mom_1..mom_4) is correct."""
    df = pd.DataFrame({f'mom_{h}': [float(h)] for h in range(1, 13)})
    result = tertile_mean(df, SHORT_H)  # SHORT_H = [1,2,3,4]
    expected = (1.0 + 2.0 + 3.0 + 4.0) / 4
    assert result.iloc[0] == pytest.approx(expected)


def test_tertile_mean_mid_group():
    """Mean of MID_H columns (mom_5..mom_8) is correct."""
    df = pd.DataFrame({f'mom_{h}': [float(h * 10)] for h in range(1, 13)})
    result = tertile_mean(df, MID_H)  # MID_H = [5,6,7,8]
    expected = (50.0 + 60.0 + 70.0 + 80.0) / 4
    assert result.iloc[0] == pytest.approx(expected)


def test_tertile_mean_long_group():
    """Mean of LONG_H columns (mom_9..mom_12) is correct."""
    df = pd.DataFrame({f'mom_{h}': [1.0] for h in range(1, 13)})
    result = tertile_mean(df, LONG_H)
    assert result.iloc[0] == pytest.approx(1.0)


def test_tertile_mean_multiple_rows():
    """Works correctly for DataFrames with multiple rows."""
    df = pd.DataFrame({f'mom_{h}': [float(h), float(h * 2)] for h in range(1, 13)})
    result = tertile_mean(df, SHORT_H)
    assert result.iloc[0] == pytest.approx((1.0 + 2.0 + 3.0 + 4.0) / 4)
    assert result.iloc[1] == pytest.approx((2.0 + 4.0 + 6.0 + 8.0) / 4)


# ---------------------------------------------------------------------------
# build_feature_panel
# ---------------------------------------------------------------------------

def _make_panel(dates, n_stocks=5, seed=0):
    """Create a minimal mock cross-section panel with pi_filter column."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(dates):
        pi_val = 1.0 if i % 3 == 0 else 0.0
        for s in range(n_stocks):
            row = {'date': d, 'permno': s, 'pi_filter': pi_val}
            for h in range(1, 13):
                row[f'mom_{h}'] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


def _make_m2_returns(dates, seed=1):
    """Create a Series of XGB monthly returns indexed by dates."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.005, 0.04, len(dates)), index=dates)


def test_build_feature_panel_shape():
    """Output has one row per date and the expected 15 columns (date + 14 features)."""
    dates = pd.date_range('2010-01-31', periods=24, freq='ME')
    panel = _make_panel(dates)
    m2_ret = _make_m2_returns(dates)
    feat = build_feature_panel(panel, m2_ret, list(dates))
    # 24 rows, columns: date + 14 features
    assert feat.shape == (24, 15)


def test_build_feature_panel_columns_present():
    """All 14 expected feature columns plus 'date' are present."""
    dates = pd.date_range('2010-01-31', periods=24, freq='ME')
    panel = _make_panel(dates)
    m2_ret = _make_m2_returns(dates)
    feat = build_feature_panel(panel, m2_ret, list(dates))
    expected_cols = {
        'date',
        'pi_panic', 'cs_mom_short', 'cs_mom_mid', 'cs_mom_long', 'mom_overall',
        'cs_disp_short', 'cs_disp_mid', 'cs_disp_long',
        'pi_panic_freq_6mo', 'pi_panic_freq_12mo', 'past_sharpe_12mo',
        'cs_skew_short', 'cs_skew_mid', 'cs_skew_long',
    }
    assert set(feat.columns) == expected_cols


def test_build_feature_panel_nan_pattern():
    """First 12 rows have NaN for pi_panic_freq_12mo and past_sharpe_12mo.
    First 6 rows have NaN for pi_panic_freq_6mo. All others are non-null."""
    dates = pd.date_range('2010-01-31', periods=20, freq='ME')
    panel = _make_panel(dates)
    m2_ret = _make_m2_returns(dates)
    feat = build_feature_panel(panel, m2_ret, list(dates))

    # First 12 rows: NaN on 12mo features
    assert feat['pi_panic_freq_12mo'].iloc[:12].isna().all()
    assert feat['past_sharpe_12mo'].iloc[:12].isna().all()
    # Rows 12+ are non-NaN for 12mo features
    assert feat['pi_panic_freq_12mo'].iloc[12:].notna().all()
    assert feat['past_sharpe_12mo'].iloc[12:].notna().all()

    # First 6 rows: NaN on 6mo feature
    assert feat['pi_panic_freq_6mo'].iloc[:6].isna().all()
    # Rows 6+ are non-NaN for 6mo feature
    assert feat['pi_panic_freq_6mo'].iloc[6:].notna().all()


def test_build_feature_panel_no_nan_in_context_features():
    """Group A (non-time-series) features are fully populated for all rows."""
    dates = pd.date_range('2010-01-31', periods=20, freq='ME')
    panel = _make_panel(dates)
    m2_ret = _make_m2_returns(dates)
    feat = build_feature_panel(panel, m2_ret, list(dates))
    context_cols = [
        'pi_panic', 'cs_mom_short', 'cs_mom_mid', 'cs_mom_long', 'mom_overall',
        'cs_disp_short', 'cs_disp_mid', 'cs_disp_long',
        'cs_skew_short', 'cs_skew_mid', 'cs_skew_long',
    ]
    for col in context_cols:
        assert feat[col].isna().sum() == 0, f'{col} has unexpected NaN'


def test_build_feature_panel_pi_panic_is_binary():
    """pi_panic column is 0 or 1 only."""
    dates = pd.date_range('2010-01-31', periods=15, freq='ME')
    panel = _make_panel(dates)
    m2_ret = _make_m2_returns(dates)
    feat = build_feature_panel(panel, m2_ret, list(dates))
    assert feat['pi_panic'].isin([0, 1]).all()


def test_build_feature_panel_date_column_matches_input():
    """The date column contains exactly the input dates in order."""
    dates = pd.date_range('2010-01-31', periods=15, freq='ME')
    panel = _make_panel(dates)
    m2_ret = _make_m2_returns(dates)
    feat = build_feature_panel(panel, m2_ret, list(dates))
    assert list(feat['date']) == list(dates)
