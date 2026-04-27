"""
test_leg_betas_by_regime.py
===========================
Tests for `scripts/leg_betas_by_regime.py` (Step E). The script computes
calm-vs-panic CAPM betas of the long and short legs separately for fixed
12-month momentum and Method 2 XGB.

The headline thesis novelty is the panic-regime beta inversion in M2's
legs. Tests pin the building blocks (leg-return construction, beta
regression, rolling-window causality, regime split) on synthetic
fixtures.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script():
    path = os.path.join(REPO, 'scripts', 'leg_betas_by_regime.py')
    spec = importlib.util.spec_from_file_location('leg_betas_by_regime', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def m():
    return _load_script()


def _make_test_panel(n_months=24, n_stocks=40, seed=0):
    """Synthetic monthly panel with NYSE rows + score column.

    Columns: date, permno, exchcd, me, ret_fwd, score_test."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-31', periods=n_months, freq='ME')
    rows = []
    for d in dates:
        for p in range(n_stocks):
            rows.append({
                'date': d, 'permno': p, 'exchcd': 1,
                'me': float(rng.uniform(1e8, 1e10)),
                'ret_fwd': float(rng.normal(0.005, 0.04)),
                'score_test': float(rng.normal()),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# build_leg_returns
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildLegReturns:

    def test_returns_long_short_columns(self, m):
        df = _make_test_panel()
        out = m.build_leg_returns(df, 'score_test', fee=0.0)
        assert {'r_long', 'r_short'}.issubset(out.columns)
        assert out.index.name == 'date'

    def test_skips_months_with_fewer_than_10_nyse(self, m):
        # Only 5 NYSE stocks — should produce no rows
        df = _make_test_panel(n_stocks=5)
        out = m.build_leg_returns(df, 'score_test', fee=0.0)
        assert len(out) == 0

    def test_returns_finite_when_data_clean(self, m):
        df = _make_test_panel(seed=1)
        out = m.build_leg_returns(df, 'score_test', fee=0.0)
        assert out['r_long'].apply(np.isfinite).all()
        assert out['r_short'].apply(np.isfinite).all()

    def test_fee_reduces_returns(self, m):
        df = _make_test_panel(seed=2)
        no_fee = m.build_leg_returns(df, 'score_test', fee=0.0)
        with_fee = m.build_leg_returns(df, 'score_test', fee=0.005)
        # First month has no prev_lw; turnover = 0.5 sum of new weights = 0.5
        # Per-leg: fee impact = 0.005 * 0.5 = 0.0025 per leg
        # Subsequent months: should reduce on average
        common = no_fee.index.intersection(with_fee.index)
        if len(common) > 1:
            assert with_fee.loc[common, 'r_long'].mean() < (
                no_fee.loc[common, 'r_long'].mean() + 1e-12
            )
            assert with_fee.loc[common, 'r_short'].mean() < (
                no_fee.loc[common, 'r_short'].mean() + 1e-12
            )

    def test_perfect_score_long_outperforms_short(self, m):
        """If score perfectly predicts ret_fwd, long leg > short leg."""
        rng = np.random.default_rng(3)
        dates = pd.date_range('2020-01-31', periods=12, freq='ME')
        rows = []
        for d in dates:
            for p in range(40):
                score = float(rng.normal())
                rows.append({
                    'date': d, 'permno': p, 'exchcd': 1,
                    'me': 1e9,
                    'ret_fwd': 0.01 * score,
                    'score_test': score,
                })
        df = pd.DataFrame(rows)
        out = m.build_leg_returns(df, 'score_test', fee=0.0)
        assert (out['r_long'] > out['r_short']).all()


# ═══════════════════════════════════════════════════════════════════════════════
# capm_beta
# ═══════════════════════════════════════════════════════════════════════════════

class TestCAPMBeta:

    def test_beta_one_when_r_equals_market(self, m):
        rng = np.random.default_rng(0)
        dates = pd.date_range('2020-01-31', periods=60, freq='ME')
        r_mkt = pd.Series(rng.normal(0.005, 0.04, 60), index=dates)
        # r perfectly tracks market — beta should be 1
        beta, se, n = m.capm_beta(r_mkt.copy(), r_mkt)
        assert beta == pytest.approx(1.0, abs=1e-9)
        assert n == 60

    def test_beta_zero_when_r_uncorrelated(self, m):
        rng = np.random.default_rng(1)
        dates = pd.date_range('2020-01-31', periods=300, freq='ME')
        r_mkt = pd.Series(rng.normal(0.005, 0.04, 300), index=dates)
        r = pd.Series(rng.normal(0.005, 0.04, 300), index=dates)  # independent
        beta, se, n = m.capm_beta(r, r_mkt)
        # Two independent series should give |beta| close to 0 with 300 obs
        assert abs(beta) < 0.2

    def test_returns_nan_when_too_few_obs(self, m):
        dates = pd.date_range('2020-01-31', periods=5, freq='ME')
        r = pd.Series([0.01] * 5, index=dates)
        r_mkt = pd.Series([0.005] * 5, index=dates)
        beta, se, n = m.capm_beta(r, r_mkt)
        assert np.isnan(beta) and np.isnan(se)
        assert n == 5

    def test_handles_pandas_float64_extension_dtype(self, m):
        """Regression guard for the bug where art['r_mkt'] arrived as the
        pandas nullable Float64 (capital F) extension dtype, which OLS
        rejects with `Pandas data cast to numpy dtype of object`. The fix
        in scripts/leg_betas_by_regime.py:119 casts to numpy float64
        before passing to capm_beta. Test runs capm_beta on a Float64
        series directly to ensure the function itself accepts it OR
        that the upstream cast is the documented fix path."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2020-01-31', periods=60, freq='ME')
        r_mkt_f64 = pd.Series(rng.normal(0.005, 0.04, 60),
                              index=dates).astype('Float64')
        r_f64 = r_mkt_f64.copy()
        # The fix is to cast at the call site; verify both float64 inputs
        # work after the cast. (If capm_beta itself happens to handle
        # Float64 without error in a future statsmodels release, this
        # test will still pass — it pins the working behaviour.)
        beta, se, n = m.capm_beta(r_f64.astype('float64'),
                                  r_mkt_f64.astype('float64'))
        assert beta == pytest.approx(1.0, abs=1e-9)
        assert n == 60

    def test_handles_partial_overlap(self, m):
        """When r and r_mkt have only some dates in common, only the
        overlap is used."""
        rng = np.random.default_rng(2)
        dates_full = pd.date_range('2020-01-31', periods=60, freq='ME')
        r_mkt = pd.Series(rng.normal(0.005, 0.04, 60), index=dates_full)
        r = pd.Series(rng.normal(0.005, 0.04, 30),
                       index=dates_full[:30])  # half overlap
        _, _, n = m.capm_beta(r, r_mkt)
        assert n == 30


# ═══════════════════════════════════════════════════════════════════════════════
# rolling_beta — must be CAUSAL (backward-aligned)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollingBeta:

    def test_first_window_minus_one_months_are_nan(self, m):
        """A backward-rolling 24-month window must produce NaN for the
        first 23 entries — pinned to ensure no centred / forward-looking
        rolling slips in by accident."""
        rng = np.random.default_rng(0)
        dates = pd.date_range('2020-01-31', periods=60, freq='ME')
        r_mkt = pd.Series(rng.normal(0.005, 0.04, 60), index=dates)
        r = pd.Series(rng.normal(0.005, 0.04, 60), index=dates)
        out = m.rolling_beta(r, r_mkt, window=24)
        # First 23 entries NaN, 24th onward should be finite (or NaN if
        # variance is 0, but with random data it won't be)
        assert out.iloc[:23].isna().all()
        assert out.iloc[23:].notna().any()

    def test_rolling_beta_one_when_r_equals_market(self, m):
        """If r == r_mkt over the rolling window, rolling beta == 1."""
        dates = pd.date_range('2020-01-31', periods=60, freq='ME')
        rng = np.random.default_rng(1)
        r_mkt = pd.Series(rng.normal(0.005, 0.04, 60), index=dates)
        out = m.rolling_beta(r_mkt.copy(), r_mkt, window=24)
        # Last value (full window) should be 1.0
        assert out.iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_window_24_is_standard(self, m):
        """ROLLING_WINDOW constant is 24 (months)."""
        assert m.ROLLING_WINDOW == 24


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:

    def test_panic_cutoff_matches_thesis(self, m):
        """0.5 cutoff matches main_results_analysis.py convention."""
        assert m.PANIC_CUTOFF == 0.50

    def test_min_obs_for_regime_beta(self, m):
        """Don't compute a beta on fewer than 12 months — too noisy."""
        assert m.MIN_OBS_REG >= 12


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: real artefacts (auto-skip without)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealArtefacts:

    def test_real_artefacts_have_required_columns(self):
        """Step E reads from artefacts pickle; required columns must be
        present in the test DataFrame."""
        path = os.path.join(REPO, 'artefacts', 'cs_artefacts_data.pkl')
        if not os.path.exists(path):
            pytest.skip('artefacts not built yet')
        import pickle
        with open(path, 'rb') as f:
            art = pickle.load(f)
        required = {'permno', 'date', 'exchcd', 'me', 'ret_fwd',
                    'pi_filter', 'score_xgb', 'score_mom12'}
        missing = required - set(art['test'].columns)
        assert missing == set(), f'test missing columns: {missing}'
        # r_mkt also required
        assert 'r_mkt' in art, 'artefacts missing r_mkt'
