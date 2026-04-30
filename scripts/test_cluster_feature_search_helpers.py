import pandas as pd
import numpy as np
import pytest

from cluster_feature_search_helpers import (
    pi_panic_freq,
    past_strategy_sharpe,
    cross_section_skew,
    picked_stock_fingerprint,
)


def test_pi_panic_freq_all_panic():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    pi = pd.Series([1.0] * 12 + [0.0], index=dates)
    # Past 12 months all panic -> 1.0
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert result == pytest.approx(1.0)


def test_pi_panic_freq_half():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # 6 panic + 6 calm in past 12 months
    pi = pd.Series([1.0]*6 + [0.0]*6 + [0.5], index=dates)
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert result == pytest.approx(0.5)


def test_pi_panic_freq_excludes_current():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # Past 12 = 0, current = 1 — should return 0.0
    pi = pd.Series([0.0]*12 + [1.0], index=dates)
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert result == pytest.approx(0.0)


def test_pi_panic_freq_insufficient_history():
    dates = pd.date_range('2020-01-31', periods=5, freq='ME')
    pi = pd.Series([0.5] * 5, index=dates)
    result = pi_panic_freq(pi, dates[-1], n_months=12)
    assert pd.isna(result)


def test_pi_panic_freq_custom_threshold():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # All values 0.6 — at threshold 0.5 -> all panic; at threshold 0.7 -> none
    pi = pd.Series([0.6] * 13, index=dates)
    assert pi_panic_freq(pi, dates[-1], n_months=12, threshold=0.5) == pytest.approx(1.0)
    assert pi_panic_freq(pi, dates[-1], n_months=12, threshold=0.7) == pytest.approx(0.0)


def test_past_strategy_sharpe_zero_std_returns_nan():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    rets = pd.Series([0.01] * 13, index=dates)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert pd.isna(result)


def test_past_strategy_sharpe_known_value():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.01, 0.05, 13), index=dates)
    window = rets.iloc[:12]
    expected = (window.mean() / window.std(ddof=1)) * np.sqrt(12)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert result == pytest.approx(expected)


def test_past_strategy_sharpe_excludes_current():
    dates = pd.date_range('2020-01-31', periods=13, freq='ME')
    # 12 months of small variance + huge spike at current
    rng = np.random.default_rng(1)
    rets = pd.Series(np.r_[rng.normal(0.01, 0.02, 12), 10.0], index=dates)
    window = rets.iloc[:12]
    expected = (window.mean() / window.std(ddof=1)) * np.sqrt(12)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert result == pytest.approx(expected)


def test_past_strategy_sharpe_insufficient_history():
    dates = pd.date_range('2020-01-31', periods=5, freq='ME')
    rets = pd.Series(np.random.default_rng(2).standard_normal(5), index=dates)
    result = past_strategy_sharpe(rets, dates[-1], n_months=12)
    assert pd.isna(result)


def test_cross_section_skew_returns_dataframe():
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range('2020-01-31', periods=3, freq='ME'):
        for s in range(5):
            row = {'date': d, 'permno': s}
            for h in range(1, 5):
                row[f'mom_{h}'] = rng.standard_normal()
            rows.append(row)
    panel = pd.DataFrame(rows)
    skew = cross_section_skew(panel, [f'mom_{h}' for h in range(1, 5)])
    assert isinstance(skew, pd.DataFrame)
    assert skew.shape == (3, 4)
    assert list(skew.columns) == ['mom_1', 'mom_2', 'mom_3', 'mom_4']


def test_cross_section_skew_zero_for_symmetric():
    rows = []
    for d in pd.date_range('2020-01-31', periods=2, freq='ME'):
        for v in [-2, -1, 0, 1, 2]:
            rows.append({'date': d, 'permno': v + 100, 'mom_1': v})
    panel = pd.DataFrame(rows)
    skew = cross_section_skew(panel, ['mom_1'])
    assert all(abs(s) < 1e-9 for s in skew['mom_1'])


def test_cross_section_skew_positive_for_right_tail():
    rows = []
    for d in pd.date_range('2020-01-31', periods=1, freq='ME'):
        for i, v in enumerate([0, 0, 0, 0, 10]):
            rows.append({'date': d, 'permno': i, 'mom_1': v})
    panel = pd.DataFrame(rows)
    skew = cross_section_skew(panel, ['mom_1'])
    assert skew['mom_1'].iloc[0] > 0


def test_picked_stock_fingerprint_columns_and_index():
    # 2 months, full panel of 5 stocks; pick 3 of 5 each month.
    dates = pd.date_range('2020-01-31', periods=2, freq='ME')
    rng = np.random.default_rng(0)
    panel_rows = []
    for d in dates:
        for permno in range(5):
            row = {'date': d, 'permno': permno}
            for h in range(1, 13):
                row[f'mom_{h}'] = rng.standard_normal()
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    picks = pd.DataFrame([
        {'date': d, 'permno': permno}
        for d in dates for permno in [0, 1, 2]
    ])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]
    fp = picked_stock_fingerprint(picks, panel, mom_cols)
    assert list(fp.index) == list(dates)
    expected_cols = (
        [f'pick_mom_{h}' for h in range(1, 13)]
        + ['pick_disp_short', 'pick_disp_mid', 'pick_disp_long']
    )
    assert list(fp.columns) == expected_cols


def test_picked_stock_fingerprint_mean_correct():
    # Single month, 3 picks, known mom values.
    d = pd.Timestamp('2020-01-31')
    panel = pd.DataFrame([
        {'date': d, 'permno': 0, **{f'mom_{h}': 1.0 for h in range(1, 13)}},
        {'date': d, 'permno': 1, **{f'mom_{h}': 2.0 for h in range(1, 13)}},
        {'date': d, 'permno': 2, **{f'mom_{h}': 3.0 for h in range(1, 13)}},
        {'date': d, 'permno': 3, **{f'mom_{h}': 99.0 for h in range(1, 13)}},  # not picked
    ])
    picks = pd.DataFrame([{'date': d, 'permno': p} for p in [0, 1, 2]])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]
    fp = picked_stock_fingerprint(picks, panel, mom_cols)
    # Mean of picks at every horizon = 2.0
    for h in range(1, 13):
        assert fp[f'pick_mom_{h}'].iloc[0] == pytest.approx(2.0)


def test_picked_stock_fingerprint_dispersion_correct():
    # Picks have known std at short horizon; mid and long different.
    d = pd.Timestamp('2020-01-31')
    panel_rows = []
    for permno, vals in [
        (0, [1, 1, 1, 1,  10, 10, 10, 10,  0, 0, 0, 0]),
        (1, [3, 3, 3, 3,  20, 20, 20, 20,  1, 1, 1, 1]),
    ]:
        row = {'date': d, 'permno': permno}
        for h, v in zip(range(1, 13), vals):
            row[f'mom_{h}'] = float(v)
        panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    picks = pd.DataFrame([{'date': d, 'permno': p} for p in [0, 1]])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]
    fp = picked_stock_fingerprint(picks, panel, mom_cols)
    # short tertile (h=1..4): values are [(1,3),(1,3),(1,3),(1,3)] -> std per horizon = sqrt(2),
    #   mean of those 4 stds = sqrt(2)
    assert fp['pick_disp_short'].iloc[0] == pytest.approx(np.sqrt(2.0))
    # mid tertile (h=5..8): values [(10,20)] -> std per horizon = sqrt(50),
    #   mean = sqrt(50)
    assert fp['pick_disp_mid'].iloc[0] == pytest.approx(np.sqrt(50.0))
    # long tertile (h=9..12): values [(0,1)] -> std per horizon = sqrt(0.5),
    #   mean = sqrt(0.5)
    assert fp['pick_disp_long'].iloc[0] == pytest.approx(np.sqrt(0.5))


def test_picked_stock_fingerprint_dedups_picks():
    # If picks_df has a duplicate (date, permno) row, the helper must NOT
    # double-count that stock's momentum in the mean.
    d = pd.Timestamp('2020-01-31')
    panel = pd.DataFrame([
        {'date': d, 'permno': 0, **{f'mom_{h}': 1.0 for h in range(1, 13)}},
        {'date': d, 'permno': 1, **{f'mom_{h}': 5.0 for h in range(1, 13)}},
    ])
    # Pick permno 0 twice (a duplicate) and permno 1 once.
    picks = pd.DataFrame([
        {'date': d, 'permno': 0},
        {'date': d, 'permno': 0},  # duplicate
        {'date': d, 'permno': 1},
    ])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]
    fp = picked_stock_fingerprint(picks, panel, mom_cols)
    # If dedup works, mean across {0, 1} = 3.0. If duplicates are kept, it would be 7/3 = 2.333.
    for h in range(1, 13):
        assert fp[f'pick_mom_{h}'].iloc[0] == pytest.approx(3.0)
