import pandas as pd
import numpy as np
import pytest

from cluster_feature_search_helpers import pi_panic_freq


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


from cluster_feature_search_helpers import past_strategy_sharpe


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


from cluster_feature_search_helpers import cross_section_skew


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
        for v in [0, 0, 0, 0, 10]:
            rows.append({'date': d, 'permno': v, 'mom_1': v})
    panel = pd.DataFrame(rows)
    skew = cross_section_skew(panel, ['mom_1'])
    assert skew['mom_1'].iloc[0] > 0
