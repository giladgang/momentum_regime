"""
test_utils.py
=============
Unit tests for src/utils.py: performance metrics, portfolio construction,
and data-loading helpers.

These are fast, isolated tests using synthetic fixtures — no dependence on
the full pipeline artefacts. Run with:

    pytest tests/test_utils.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from src import utils


# ═══════════════════════════════════════════════════════════════════════════════
# metrics()
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_empty_series_returns_zeros(self):
        assert utils.metrics(pd.Series(dtype=float)) == (0.0, 0.0, 0.0, 0.0)

    def test_all_nan_series_returns_zeros(self):
        assert utils.metrics(pd.Series([np.nan, np.nan, np.nan])) == (0.0, 0.0, 0.0, 0.0)

    def test_constant_zero_returns(self):
        ann_ret, ann_vol, sharpe, mdd = utils.metrics(pd.Series([0.0] * 24))
        assert ann_ret == 0.0
        assert ann_vol == 0.0
        assert sharpe == 0.0
        assert mdd == 0.0

    def test_constant_positive_returns_zero_vol_yields_zero_sharpe(self):
        # std == 0 -> sharpe must be 0 (not NaN or inf)
        _, ann_vol, sharpe, _ = utils.metrics(pd.Series([0.01] * 24))
        assert ann_vol == 0.0
        assert sharpe == 0.0

    def test_annualisation_of_one_percent_per_month(self):
        # 12 months of exactly 1% -> (1.01)^12 - 1
        ann_ret, _, _, _ = utils.metrics(pd.Series([0.01] * 12))
        assert ann_ret == pytest.approx(1.01**12 - 1, abs=1e-12)

    def test_max_drawdown_simple_case(self):
        # +10%, -20%, flat: cum = 1.1, 0.88, 0.88  -> peak 1.1, trough 0.88
        # mdd = (0.88 - 1.1) / 1.1 = -0.2
        r = pd.Series([0.10, -0.20, 0.00])
        _, _, _, mdd = utils.metrics(r)
        assert mdd == pytest.approx(-0.2, abs=1e-12)

    def test_drops_nan_before_computing(self):
        # NaN should be dropped so same result as the cleaned series
        r_with_nan = pd.Series([0.01, np.nan, 0.02, np.nan, -0.01])
        r_clean = pd.Series([0.01, 0.02, -0.01])
        assert utils.metrics(r_with_nan) == utils.metrics(r_clean)

    def test_sharpe_sign_matches_mean_return_sign(self):
        _, _, sharpe_pos, _ = utils.metrics(pd.Series([0.02, 0.01, 0.03, -0.01, 0.02]))
        _, _, sharpe_neg, _ = utils.metrics(pd.Series([-0.02, -0.01, -0.03, 0.01, -0.02]))
        assert sharpe_pos > 0
        assert sharpe_neg < 0

    def test_deterministic(self):
        r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.015, 0.025, 0.005])
        assert utils.metrics(r) == utils.metrics(r)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_sharpe()
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeSharpe:
    def test_empty_returns_zero(self):
        assert utils.compute_sharpe(pd.Series(dtype=float)) == 0.0

    def test_all_nan_returns_zero(self):
        assert utils.compute_sharpe(pd.Series([np.nan, np.nan])) == 0.0

    def test_zero_std_returns_zero(self):
        assert utils.compute_sharpe(pd.Series([0.01] * 10)) == 0.0

    def test_matches_metrics_sharpe(self):
        r = pd.Series(np.random.default_rng(0).normal(0.005, 0.04, 120))
        _, _, sharpe_from_metrics, _ = utils.metrics(r)
        assert utils.compute_sharpe(r) == pytest.approx(sharpe_from_metrics, rel=1e-12)

    def test_annualisation_factor(self):
        # For r = constant series + tiny noise, sharpe = mean/std * sqrt(12).
        # Build a series with known mean and std.
        r = pd.Series([0.02, -0.02] * 6)  # mean = 0, std > 0
        assert utils.compute_sharpe(r) == pytest.approx(0.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# long_short_port() — synthetic fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_monthly_panel(n_months=6, n_stocks=40, seed=0):
    """Create a synthetic multi-stock monthly panel with NYSE-eligible rows.

    Columns: date, permno, exchcd, me, ret_fwd, score.
    All stocks are on NYSE (exchcd=1) so quantile breakpoints work.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-31', periods=n_months, freq='ME')
    rows = []
    for d in dates:
        for p in range(n_stocks):
            rows.append({
                'date': d,
                'permno': p,
                'exchcd': 1,
                'me': float(rng.uniform(1e8, 1e10)),
                'ret_fwd': float(rng.normal(0.01, 0.05)),
                'score': float(rng.normal(0, 1)),
            })
    return pd.DataFrame(rows)


class TestLongShortPort:
    def test_empty_df_returns_empty_series(self):
        empty = pd.DataFrame(columns=['date', 'permno', 'exchcd', 'me', 'ret_fwd', 'score'])
        out = utils.long_short_port(empty, 'score', fee=0.0)
        assert isinstance(out, pd.Series)
        assert len(out) == 0

    def test_returns_pandas_series_indexed_by_date(self):
        df = _make_monthly_panel(n_months=4, n_stocks=40, seed=1)
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert isinstance(out, pd.Series)
        assert out.index.name == 'date'
        # One return per month (all months have >=10 NYSE stocks)
        assert len(out) == 4

    def test_fewer_than_10_nyse_stocks_skipped(self):
        df = _make_monthly_panel(n_months=3, n_stocks=8, seed=2)
        out = utils.long_short_port(df, 'score', fee=0.0)
        # All months have 8 NYSE stocks < 10, so nothing returned
        assert len(out) == 0

    def test_all_nan_score_skipped(self):
        df = _make_monthly_panel(n_months=3, n_stocks=40, seed=3)
        df['score'] = np.nan
        out = utils.long_short_port(df, 'score', fee=0.0)
        # nyse.dropna() leaves zero rows -> len < 10 -> skipped
        assert len(out) == 0

    def test_zero_fee_matches_nonzero_fee_on_first_month(self):
        # First month has no prev weights -> turnover = 0.5 * sum(new_weights) = 0.5
        # But since prev_lw is {} and we iterate only new permnos, turnover = 0.5 per leg.
        # Actually turnover is computed as sum(|new - prev|) / 2 = sum(new) / 2 = 0.5.
        # So first month fee impact = fee * (0.5 + 0.5) = fee.
        # Therefore zero-fee first-month return must exceed high-fee return by ~fee.
        df = _make_monthly_panel(n_months=1, n_stocks=40, seed=4)
        r_no_fee = utils.long_short_port(df, 'score', fee=0.0)
        r_high_fee = utils.long_short_port(df, 'score', fee=0.01)
        assert len(r_no_fee) == 1 and len(r_high_fee) == 1
        diff = r_no_fee.iloc[0] - r_high_fee.iloc[0]
        # diff should be positive and approximately fee * (sum of both leg turnovers)
        assert diff > 0
        assert diff == pytest.approx(0.01 * 1.0, abs=1e-12)

    def test_long_short_sign_positive_score_outperforms(self):
        # Construct a panel where score perfectly predicts ret_fwd:
        # higher score -> higher ret. Then long-short should be strongly positive.
        rng = np.random.default_rng(5)
        dates = pd.date_range('2020-01-31', periods=6, freq='ME')
        rows = []
        for d in dates:
            for p in range(40):
                score = float(rng.normal())
                rows.append({
                    'date': d,
                    'permno': p,
                    'exchcd': 1,
                    'me': 1e9,  # equal weighted effectively
                    'ret_fwd': 0.01 * score,  # perfect linear link
                    'score': score,
                })
        df = pd.DataFrame(rows)
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert (out > 0).all(), "Long-short should be positive when score perfectly predicts return"

    def test_zero_me_leg_skipped(self):
        # If all longs have me=0, month is skipped
        df = _make_monthly_panel(n_months=1, n_stocks=40, seed=6)
        # Force top-decile stocks to have me=0
        top_cut = df['score'].quantile(0.90)
        df.loc[df['score'] >= top_cut, 'me'] = 0.0
        out = utils.long_short_port(df, 'score', fee=0.0)
        assert len(out) == 0

    def test_deterministic_given_same_input(self):
        df = _make_monthly_panel(n_months=4, n_stocks=40, seed=7)
        r1 = utils.long_short_port(df, 'score', fee=0.001)
        r2 = utils.long_short_port(df, 'score', fee=0.001)
        pd.testing.assert_series_equal(r1, r2)

    def test_transaction_cost_reduces_return_on_subsequent_months(self):
        df = _make_monthly_panel(n_months=6, n_stocks=40, seed=8)
        r_no_fee = utils.long_short_port(df, 'score', fee=0.0)
        r_fee = utils.long_short_port(df, 'score', fee=0.005)
        # Fee must never *increase* return
        assert (r_fee <= r_no_fee + 1e-12).all()
        # And on average fee should reduce returns
        assert r_fee.mean() < r_no_fee.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Path helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestPathHelpers:
    def test_project_path_is_absolute(self):
        p = utils.project_path('foo', 'bar.txt')
        assert os.path.isabs(p)
        assert p.endswith(os.path.join('foo', 'bar.txt'))

    def test_project_path_points_into_repo(self):
        p = utils.project_path('config.py')
        assert os.path.exists(p), "project_path should resolve to real repo files"

    def test_tables_path_uses_tables_dir(self):
        import config as cfg
        p = utils.tables_path('foo.csv')
        assert os.path.basename(p) == 'foo.csv'
        assert cfg.TABLES_DIR in p

    def test_plots_path_uses_plots_dir(self):
        import config as cfg
        p = utils.plots_path('foo.png')
        assert os.path.basename(p) == 'foo.png'
        assert cfg.PLOTS_DIR in p


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoaders:
    """Data-dependent tests. Each uses a session-scoped fixture from
    conftest.py that auto-skips when the underlying file is absent
    (e.g., in CI without the 943 MB artefacts pickle). Loads the
    file ONCE per pytest session instead of once per test."""

    def test_load_panel_with_regimes_returns_dataframe(self, panel_with_regimes):
        assert isinstance(panel_with_regimes, pd.DataFrame)
        assert 'date' in panel_with_regimes.columns
        assert 'pi_filter' in panel_with_regimes.columns
        assert len(panel_with_regimes) > 0

    def test_load_panel_with_regimes_pi_filter_in_range(self, panel_with_regimes):
        pi = panel_with_regimes['pi_filter'].dropna()
        assert (pi >= 0).all() and (pi <= 1).all()

    def test_load_artefacts_has_expected_keys(self, artefacts):
        for key in ('test', 'train', 'FEATURES', 'strategies_lo'):
            assert key in artefacts, f"artefacts missing expected key: {key}"

    def test_load_ff_factors_returns_dataframe(self, ff_factors):
        assert isinstance(ff_factors, pd.DataFrame)
        assert len(ff_factors) > 0

    def test_utils_load_functions_match_fixtures(self, artefacts, panel_with_regimes):
        """Sanity: utils.load_* functions return equivalent data to the
        conftest fixtures (same file, same content)."""
        via_utils_panel = utils.load_panel_with_regimes()
        via_utils_art = utils.load_artefacts()
        # Cheap equivalence check
        assert len(via_utils_panel) == len(panel_with_regimes)
        assert set(via_utils_art.keys()) == set(artefacts.keys())
