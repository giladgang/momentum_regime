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
