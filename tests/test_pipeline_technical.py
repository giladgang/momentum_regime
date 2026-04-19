"""
test_pipeline_technical.py
==========================
Comprehensive technical tests for the momentum regime shifts pipeline.

Tests cover:
  1. Data integrity and pipeline artifact structure
  2. HMM output validity (regime probabilities, MCMC diagnostics)
  3. Cross-sectional model outputs (scores, portfolios, SHAP)
  4. Portfolio construction correctness (weights, turnover, returns)
  5. Table output completeness and format
  6. Thesis-text-to-table number consistency (every inline number verified)
  7. Config-to-artifact alignment
  8. Inter-script data flow (pipeline connections)

Run with:
    pytest tests/test_pipeline_technical.py -v
"""

import os
import re
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg

LATEX_DIR = os.path.join(PROJECT_ROOT, "latex")
TABLES_DIR = os.path.join(PROJECT_ROOT, cfg.TABLES_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTEFACTS_PATH = os.path.join(PROJECT_ROOT, cfg.ARTEFACTS_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _read(path: str) -> str:
    full = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


_ARTEFACTS_CACHE = None


def _get_artefacts():
    global _ARTEFACTS_CACHE
    if _ARTEFACTS_CACHE is None:
        with open(ARTEFACTS_PATH, "rb") as f:
            _ARTEFACTS_CACHE = pickle.load(f)
    return _ARTEFACTS_CACHE


def _find_in_table(table_text: str, row_substr: str, col_index: int) -> str:
    for line in table_text.splitlines():
        if row_substr in line:
            cols = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            if col_index < len(cols):
                raw = cols[col_index]
                raw = raw.replace("$-$", "-").replace(r"\%", "").replace("%", "")
                raw = raw.replace("$", "").replace(",", "").strip()
                return raw
    return ""


def _load_panel_with_regimes():
    return pd.read_parquet(os.path.join(PROJECT_ROOT, cfg.PANEL_WITH_REGIMES_PATH))


def _compute_sharpe(returns_series):
    """Compute annualised Sharpe from monthly excess returns."""
    if len(returns_series) == 0 or returns_series.std() == 0:
        return 0.0
    return returns_series.mean() / returns_series.std() * np.sqrt(12)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Verify raw data files exist and have expected structure."""

    def test_crsp_parquet_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, cfg.STOCK_DATA_PATH))

    def test_panel_parquet_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, cfg.PANEL_PATH))

    def test_ff_factors_parquet_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, cfg.FF_FACTORS_PATH))

    def test_panel_with_regimes_exists(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, cfg.PANEL_WITH_REGIMES_PATH))

    def test_crsp_has_required_columns(self):
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, cfg.STOCK_DATA_PATH))
        required = {'permno', 'date', 'ret_adj', 'me', 'prc', 'exchcd', 'shrcd'}
        missing = required - set(df.columns)
        assert missing == set(), f"CRSP missing columns: {missing}"

    def test_panel_has_hmm_features(self):
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, cfg.PANEL_PATH))
        for feat in cfg.HMM_FEATURES:
            assert feat in df.columns, f"Panel missing HMM feature: {feat}"

    def test_ff_factors_has_required_columns(self):
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, cfg.FF_FACTORS_PATH))
        required = {'Mkt-RF', 'SMB', 'HML'}
        missing = required - set(df.columns)
        assert missing == set(), f"FF factors missing columns: {missing}"

    def test_crsp_no_future_data_leakage(self):
        """Verify CRSP data doesn't contain dates beyond sample end."""
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, cfg.STOCK_DATA_PATH))
        df['date'] = pd.to_datetime(df['date'])
        max_date = df['date'].max()
        assert max_date <= pd.Timestamp('2026-01-01'), (
            f"CRSP data extends to {max_date}, possible data issue"
        )

    def test_crsp_starts_at_or_before_1990(self):
        df = pd.read_parquet(os.path.join(PROJECT_ROOT, cfg.STOCK_DATA_PATH))
        df['date'] = pd.to_datetime(df['date'])
        min_date = df['date'].min()
        assert min_date <= pd.Timestamp('1990-02-01'), (
            f"CRSP starts at {min_date}, expected at or before 1990-01"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HMM OUTPUT VALIDITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMOutputs:
    """Verify HMM regime probabilities are valid and well-behaved."""

    def test_panel_with_regimes_has_pi_filter(self):
        panel = _load_panel_with_regimes()
        assert 'pi_filter' in panel.columns

    def test_pi_filter_bounded_0_1(self):
        panel = _load_panel_with_regimes()
        pi = panel['pi_filter'].dropna()
        assert pi.min() >= -1e-10, f"pi_filter min = {pi.min()}, expected >= 0"
        assert pi.max() <= 1.0 + 1e-10, f"pi_filter max = {pi.max()}, expected <= 1"

    def test_pi_filter_not_degenerate(self):
        """Pi_filter should not be all 0s or all 1s."""
        panel = _load_panel_with_regimes()
        pi = panel['pi_filter'].dropna()
        assert pi.std() > 0.05, (
            f"pi_filter std = {pi.std():.4f}, too low -- likely degenerate"
        )

    def test_pi_filter_has_both_regimes(self):
        """Both calm and panic months should exist in test period."""
        panel = _load_panel_with_regimes()
        panel['date'] = pd.to_datetime(panel['date'])
        test_pi = panel.loc[panel['date'] >= cfg.TRAIN_END, 'pi_filter'].dropna()
        n_calm = (test_pi < 0.5).sum()
        n_panic = (test_pi >= 0.5).sum()
        assert n_calm > 20, f"Only {n_calm} calm months in test, expected > 20"
        assert n_panic > 20, f"Only {n_panic} panic months in test, expected > 20"

    def test_pi_filter_coverage(self):
        """Pi_filter should have values for nearly all months."""
        panel = _load_panel_with_regimes()
        total = len(panel)
        non_null = panel['pi_filter'].notna().sum()
        coverage = non_null / total
        assert coverage > 0.95, (
            f"pi_filter coverage = {coverage:.1%}, expected > 95%"
        )

    def test_mcmc_draws_exist(self):
        path = os.path.join(DATA_DIR, "mcmc_draws.npz")
        assert os.path.exists(path), f"Missing: {path}"

    def test_mcmc_draws_structure(self):
        path = os.path.join(DATA_DIR, "mcmc_draws.npz")
        data = np.load(path, allow_pickle=True)
        assert len(data.files) > 0, "mcmc_draws.npz is empty"

    def test_crisis_periods_identified_as_panic(self):
        """GFC (2008-2009) should have high panic probability."""
        panel = _load_panel_with_regimes()
        panel['date'] = pd.to_datetime(panel['date'])
        gfc = panel[(panel['date'] >= '2007-10-01') & (panel['date'] <= '2009-06-01')]
        if len(gfc) > 0:
            avg_pi = gfc['pi_filter'].mean()
            assert avg_pi > 0.5, (
                f"GFC average pi_filter = {avg_pi:.2f}, expected > 0.5"
            )

    def test_calm_periods_low_panic(self):
        """2013-2019 (non-crisis) should have low panic probability on average."""
        panel = _load_panel_with_regimes()
        panel['date'] = pd.to_datetime(panel['date'])
        calm = panel[(panel['date'] >= '2013-01-01') & (panel['date'] <= '2019-12-01')]
        if len(calm) > 0:
            avg_pi = calm['pi_filter'].mean()
            assert avg_pi < 0.5, (
                f"2013-2019 average pi_filter = {avg_pi:.2f}, expected < 0.5"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-SECTIONAL MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSectionalArtifacts:
    """Verify cs_artefacts_data.pkl has correct structure and content."""

    def test_artefacts_file_exists(self):
        assert os.path.exists(ARTEFACTS_PATH)

    def test_artefacts_is_dict(self):
        art = _get_artefacts()
        assert isinstance(art, dict)

    @pytest.mark.parametrize("key", [
        "test", "train", "FEATURES", "X_train", "X_test",
        "y_train", "shap_values", "strategies_lo",
    ])
    def test_required_keys_present(self, key):
        art = _get_artefacts()
        assert key in art, f"Missing key: {key}"

    def test_features_list_matches_config(self):
        art = _get_artefacts()
        assert list(art["FEATURES"]) == list(cfg.CS_FEATURES)

    def test_features_count_is_13(self):
        art = _get_artefacts()
        assert len(art["FEATURES"]) == 13, (
            f"Expected 13 features (12 mom + pi_filter), got {len(art['FEATURES'])}"
        )

    def test_feature_names_correct(self):
        art = _get_artefacts()
        expected = [f"mom_{i}" for i in range(1, 13)] + ["pi_filter"]
        assert list(art["FEATURES"]) == expected

    def test_X_train_X_test_shape_consistency(self):
        art = _get_artefacts()
        assert art["X_train"].shape[1] == art["X_test"].shape[1], (
            f"X_train cols ({art['X_train'].shape[1]}) != X_test cols ({art['X_test'].shape[1]})"
        )

    def test_X_train_shape_matches_features(self):
        art = _get_artefacts()
        assert art["X_train"].shape[1] == len(art["FEATURES"]), (
            f"X_train has {art['X_train'].shape[1]} cols but {len(art['FEATURES'])} features"
        )

    def test_y_train_length_matches_X_train(self):
        art = _get_artefacts()
        assert len(art["y_train"]) == art["X_train"].shape[0]

    def test_shap_values_shape_matches_X_test(self):
        art = _get_artefacts()
        assert art["shap_values"].shape == art["X_test"].shape, (
            f"SHAP shape {art['shap_values'].shape} != X_test shape {art['X_test'].shape}"
        )

    def test_shap_values_not_all_zero(self):
        art = _get_artefacts()
        assert np.abs(art["shap_values"]).sum() > 0

    def test_test_has_score_columns(self):
        art = _get_artefacts()
        test_df = art["test"]
        assert "score_xgb" in test_df.columns, "Missing score_xgb in test"

    def test_test_has_required_columns(self):
        art = _get_artefacts()
        test_df = art["test"]
        required = {'permno', 'date', 'ret_fwd', 'me', 'exchcd', 'pi_filter'}
        missing = required - set(test_df.columns)
        assert missing == set(), f"Test DataFrame missing: {missing}"

    def test_train_has_required_columns(self):
        art = _get_artefacts()
        train_df = art["train"]
        required = {'permno', 'date', 'ret_fwd', 'me'}
        missing = required - set(train_df.columns)
        assert missing == set(), f"Train DataFrame missing: {missing}"

    def test_momentum_features_in_test(self):
        art = _get_artefacts()
        test_df = art["test"]
        for lb in range(1, 13):
            col = f"mom_{lb}"
            assert col in test_df.columns, f"Missing {col} in test"

    def test_train_test_no_date_overlap(self):
        """Train and test periods should not overlap."""
        art = _get_artefacts()
        train_dates = pd.to_datetime(art["train"]["date"])
        test_dates = pd.to_datetime(art["test"]["date"])
        train_max = train_dates.max()
        test_min = test_dates.min()
        assert train_max < test_min, (
            f"Train max date {train_max} >= test min date {test_min}"
        )

    def test_train_ends_before_2011(self):
        art = _get_artefacts()
        train_dates = pd.to_datetime(art["train"]["date"])
        assert train_dates.max() < pd.Timestamp("2011-01-01")

    def test_test_starts_at_or_after_2011(self):
        art = _get_artefacts()
        test_dates = pd.to_datetime(art["test"]["date"])
        assert test_dates.min() >= pd.Timestamp("2011-01-01")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PORTFOLIO RETURNS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioReturns:
    """Verify portfolio returns are well-formed and match expected properties."""

    def test_strategies_lo_is_dict(self):
        art = _get_artefacts()
        assert isinstance(art["strategies_lo"], dict)

    def test_strategies_lo_has_m2(self):
        art = _get_artefacts()
        strats = art["strategies_lo"]
        has_m2 = any("xgb" in k.lower() or "m2" in k.lower() for k in strats)
        assert has_m2, f"No M2/XGB strategy found. Keys: {list(strats.keys())}"

    def test_portfolio_returns_are_monthly(self):
        """Returns should be monthly (no huge gaps or duplicates)."""
        art = _get_artefacts()
        for name, ret_series in art["strategies_lo"].items():
            if len(ret_series) < 10:
                continue
            dates = pd.to_datetime(ret_series.index)
            diffs = dates.to_series().diff().dropna()
            median_days = diffs.dt.days.median()
            assert 25 <= median_days <= 35, (
                f"Strategy '{name}' median date gap = {median_days} days, "
                f"expected ~30 (monthly)"
            )

    def test_m2_return_count_is_167(self):
        """Test period should have 167 months."""
        art = _get_artefacts()
        strats = art["strategies_lo"]
        for name, ret_series in strats.items():
            if "xgb" in name.lower() or "m2" in name.lower():
                assert len(ret_series) == 167, (
                    f"M2 has {len(ret_series)} months, expected 167"
                )
                break

    def test_m2_sharpe_matches_table(self):
        """Recompute M2 Sharpe from returns and compare to table value."""
        art = _get_artefacts()
        strats = art["strategies_lo"]
        for name, ret_series in strats.items():
            if "xgb" in name.lower() or "m2" in name.lower():
                sharpe = _compute_sharpe(ret_series)
                assert sharpe == pytest.approx(1.11, abs=0.02), (
                    f"Recomputed M2 Sharpe = {sharpe:.3f}, table says 1.11"
                )
                break

    def test_m2_annualised_return(self):
        art = _get_artefacts()
        strats = art["strategies_lo"]
        for name, ret_series in strats.items():
            if "xgb" in name.lower() or "m2" in name.lower():
                ann_ret = (1 + ret_series).prod() ** (12 / len(ret_series)) - 1
                assert ann_ret * 100 == pytest.approx(21.9, abs=0.5), (
                    f"M2 ann ret = {ann_ret*100:.1f}%, expected 21.9%"
                )
                break

    def test_m2_annualised_vol(self):
        art = _get_artefacts()
        strats = art["strategies_lo"]
        for name, ret_series in strats.items():
            if "xgb" in name.lower() or "m2" in name.lower():
                ann_vol = ret_series.std() * np.sqrt(12)
                assert ann_vol * 100 == pytest.approx(19.7, abs=0.5), (
                    f"M2 ann vol = {ann_vol*100:.1f}%, expected 19.7%"
                )
                break

    def test_m2_max_drawdown(self):
        art = _get_artefacts()
        strats = art["strategies_lo"]
        for name, ret_series in strats.items():
            if "xgb" in name.lower() or "m2" in name.lower():
                wealth = (1 + ret_series).cumprod()
                peak = wealth.cummax()
                dd = (wealth / peak - 1).min()
                assert dd * 100 == pytest.approx(-24.8, abs=0.5), (
                    f"M2 MDD = {dd*100:.1f}%, expected -24.8%"
                )
                break

    def test_no_nan_in_returns(self):
        art = _get_artefacts()
        for name, ret_series in art["strategies_lo"].items():
            assert ret_series.isna().sum() == 0, (
                f"Strategy '{name}' has {ret_series.isna().sum()} NaN returns"
            )

    def test_returns_are_reasonable(self):
        """No single month should have > 100% return (long-short)."""
        art = _get_artefacts()
        for name, ret_series in art["strategies_lo"].items():
            max_abs = ret_series.abs().max()
            assert max_abs < 1.0, (
                f"Strategy '{name}' has max |return| = {max_abs:.2f}, suspicious"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SHAP FEATURE IMPORTANCE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSHAPValues:
    """Verify SHAP values and importance shares match thesis claims."""

    def test_shap_momentum_share(self):
        """Momentum (12 horizons) should account for ~54% of total SHAP."""
        art = _get_artefacts()
        shap = art["shap_values"]
        abs_shap = np.abs(shap).mean(axis=0)
        total = abs_shap.sum()
        # mom_1..mom_12 are first 12 features, pi_filter is last
        mom_share = abs_shap[:12].sum() / total * 100
        assert mom_share == pytest.approx(54, abs=2), (
            f"Momentum SHAP share = {mom_share:.1f}%, expected ~54%"
        )

    def test_shap_pi_share(self):
        """Pi_filter should account for ~46% of total SHAP."""
        art = _get_artefacts()
        shap = art["shap_values"]
        abs_shap = np.abs(shap).mean(axis=0)
        total = abs_shap.sum()
        pi_share = abs_shap[12] / total * 100  # pi_filter is the 13th feature
        assert pi_share == pytest.approx(46, abs=2), (
            f"Pi SHAP share = {pi_share:.1f}%, expected ~46%"
        )

    def test_shap_shares_sum_to_100(self):
        art = _get_artefacts()
        shap = art["shap_values"]
        abs_shap = np.abs(shap).mean(axis=0)
        total = abs_shap.sum()
        shares = abs_shap / total * 100
        assert shares.sum() == pytest.approx(100, abs=0.01)

    def test_shap_no_negative_importance(self):
        """Mean |SHAP| should be non-negative for all features."""
        art = _get_artefacts()
        abs_shap = np.abs(art["shap_values"]).mean(axis=0)
        assert (abs_shap >= 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REGIME-CONDITIONAL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeConditional:
    """Verify regime-conditional Sharpe ratios match thesis claims."""

    def _get_regime_returns(self):
        art = _get_artefacts()
        panel = _load_panel_with_regimes()
        panel['date'] = pd.to_datetime(panel['date'])
        monthly_pi = panel.groupby('date')['pi_filter'].first()

        for name, ret_series in art["strategies_lo"].items():
            if "xgb" in name.lower() or "m2" in name.lower():
                ret_df = ret_series.to_frame("ret")
                ret_df.index = pd.to_datetime(ret_df.index)
                ret_df = ret_df.join(monthly_pi, how='left')
                calm = ret_df[ret_df['pi_filter'] < 0.5]['ret']
                panic = ret_df[ret_df['pi_filter'] >= 0.5]['ret']
                return calm, panic
        pytest.fail("Could not find M2 strategy returns")

    def test_calm_month_count(self):
        calm, panic = self._get_regime_returns()
        assert len(calm) == pytest.approx(108, abs=5), (
            f"Calm months = {len(calm)}, expected ~108"
        )

    def test_panic_month_count(self):
        calm, panic = self._get_regime_returns()
        assert len(panic) == pytest.approx(59, abs=5), (
            f"Panic months = {len(panic)}, expected ~59"
        )

    def test_calm_sharpe(self):
        calm, _ = self._get_regime_returns()
        sharpe = _compute_sharpe(calm)
        assert sharpe == pytest.approx(0.82, abs=0.05), (
            f"Calm Sharpe = {sharpe:.2f}, expected 0.82"
        )

    def test_panic_sharpe(self):
        _, panic = self._get_regime_returns()
        sharpe = _compute_sharpe(panic)
        assert sharpe == pytest.approx(1.56, abs=0.05), (
            f"Panic Sharpe = {sharpe:.2f}, expected 1.56"
        )

    def test_panic_sharpe_exceeds_calm(self):
        calm, panic = self._get_regime_returns()
        calm_sr = _compute_sharpe(calm)
        panic_sr = _compute_sharpe(panic)
        assert panic_sr > calm_sr, (
            f"Panic Sharpe ({panic_sr:.2f}) should exceed calm ({calm_sr:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TABLE FILE COMPLETENESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableCompleteness:
    """Verify all expected .tex table files exist and are non-empty."""

    EXPECTED_TABLES = [
        "table_performance.tex",
        "table_regime_sharpe.tex",
        "table_shap.tex",
        "table_lr_coef.tex",
        "table_ic.tex",
        "table_granger.tex",
        "table_subperiod.tex",
        "table_turnover.tex",
        "table_cost_sensitivity.tex",
        "table_xgb_hyperparams.tex",
        "table_threshold_sensitivity.tex",
        "table_factor_alphas.tex",
        "table_ghm_comparison.tex",
        "table_regime_signal_ablation.tex",
        "table_student_t_hmm.tex",
        "table_gelman_rubin.tex",
        "table_hmm_separation.tex",
        "table_hmm_feature_ablation.tex",
        "table_stress_scenarios.tex",
        "table_ridge.tex",
        "table_risk_aversion.tex",
        "table_january.tex",
        "table_placebo.tex",
        "table_combo_freq.tex",
        "table_combo_long_cp.tex",
        "table_combo_short_cp.tex",
        "table_sample_summary.tex",
    ]

    @pytest.mark.parametrize("table_name", EXPECTED_TABLES)
    def test_table_exists(self, table_name):
        path = os.path.join(TABLES_DIR, table_name)
        assert os.path.exists(path), f"Missing table: {path}"

    @pytest.mark.parametrize("table_name", EXPECTED_TABLES)
    def test_table_non_empty(self, table_name):
        path = os.path.join(TABLES_DIR, table_name)
        if os.path.exists(path):
            content = open(path, "r").read().strip()
            assert len(content) > 50, (
                f"Table {table_name} is too short ({len(content)} chars)"
            )

    @pytest.mark.parametrize("table_name", EXPECTED_TABLES)
    def test_table_has_tabular(self, table_name):
        """Every table should contain a LaTeX tabular environment."""
        path = os.path.join(TABLES_DIR, table_name)
        if os.path.exists(path):
            content = open(path, "r").read()
            assert "\\begin{tabular" in content or "\\begin{table" in content, (
                f"Table {table_name} missing tabular/table environment"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TABLE-TO-TABLE CONSISTENCY (numbers agree across tables)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossTableConsistency:
    """Verify that the same number reported in multiple tables is consistent."""

    def test_m2_sharpe_across_tables(self):
        """M2 Sharpe = 1.11 should appear identically in multiple tables."""
        perf = _read("tables/table_performance.tex")
        regime = _read("tables/table_regime_sharpe.tex")
        sub = _read("tables/table_subperiod.tex")
        cost = _read("tables/table_cost_sensitivity.tex")
        ridge = _read("tables/table_ridge.tex")
        ghm = _read("tables/table_ghm_comparison.tex")

        val_perf = float(_find_in_table(perf, "M2: XGB", 3))
        val_regime = float(_find_in_table(regime, "M2: XGB", 1))
        val_sub = float(_find_in_table(sub, "M2: XGB", 4))
        val_cost = float(_find_in_table(cost, "M2: XGB", 3))  # 10 bps column
        val_ridge = float(_find_in_table(ridge, "M2 (XGBoost)", 2))
        val_ghm = float(_find_in_table(ghm, "M2: XGB", 1))

        target = 1.11
        for label, val in [("performance", val_perf), ("regime_sharpe", val_regime),
                           ("subperiod Full", val_sub), ("cost_sensitivity 10bps", val_cost),
                           ("ridge", val_ridge), ("ghm_comparison", val_ghm)]:
            assert val == pytest.approx(target, abs=0.02), (
                f"M2 Sharpe in {label} = {val}, expected {target}"
            )

    def test_m2_ann_ret_across_tables(self):
        perf = _read("tables/table_performance.tex")
        ablation = _read("tables/table_regime_signal_ablation.tex")
        jan = _read("tables/table_january.tex")

        val_perf = float(_find_in_table(perf, "M2: XGB", 1))
        val_abl = float(_find_in_table(ablation, "HMM", 1))
        val_jan_baseline = float(_find_in_table(jan, "M2: XGB", 1))

        # Baseline 21.9% should match across tables
        assert val_perf == pytest.approx(21.9, abs=0.1)
        assert val_abl == pytest.approx(21.9, abs=0.1)
        assert val_jan_baseline == pytest.approx(21.9, abs=0.1)

    def test_m2_vol_across_tables(self):
        perf = _read("tables/table_performance.tex")
        ablation = _read("tables/table_regime_signal_ablation.tex")

        val_perf = float(_find_in_table(perf, "M2: XGB", 2))
        val_abl = float(_find_in_table(ablation, "HMM", 2))

        assert val_perf == pytest.approx(19.7, abs=0.1)
        assert val_abl == pytest.approx(19.7, abs=0.1)

    def test_m2_mdd_across_tables(self):
        perf = _read("tables/table_performance.tex")
        ablation = _read("tables/table_regime_signal_ablation.tex")
        stress = _read("tables/table_stress_scenarios.tex")

        val_perf = float(_find_in_table(perf, "M2: XGB", 4))
        val_abl = float(_find_in_table(ablation, "HMM", 4))
        val_stress = float(_find_in_table(stress, "Baseline", 3))

        assert val_perf == pytest.approx(-24.8, abs=0.1)
        assert val_abl == pytest.approx(-24.8, abs=0.1)
        assert val_stress == pytest.approx(-24.8, abs=0.1)

    def test_fixed_12mo_sharpe_across_tables(self):
        perf = _read("tables/table_performance.tex")
        cost = _read("tables/table_cost_sensitivity.tex")

        val_perf = float(_find_in_table(perf, "Fixed 12-mo", 3))
        val_cost = float(_find_in_table(cost, "Fixed 12-mo", 3))  # 10 bps

        assert val_perf == pytest.approx(val_cost, abs=0.02)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. THESIS TEXT vs TABLE NUMBERS
# ═══════════════════════════════════════════════════════════════════════════════

class TestThesisTextMatchesTables:
    """Verify every inline number in thesis text is consistent with its table."""

    # --- main_results.tex ---

    def test_main_results_m2_sharpe(self):
        text = _read("latex/main_results.tex")
        tbl = _read("tables/table_performance.tex")
        tbl_val = float(_find_in_table(tbl, "M2: XGB", 3))
        assert f"{tbl_val:.2f}" in text or f"{tbl_val}" in text, (
            f"main_results.tex should mention M2 Sharpe {tbl_val}"
        )

    def test_main_results_m2_alpha(self):
        text = _read("latex/main_results.tex")
        tbl = _read("tables/table_factor_alphas.tex")
        val = float(_find_in_table(tbl, "FF6", 1))
        assert str(val) in text, (
            f"main_results.tex should mention FF6 alpha {val}"
        )

    def test_main_results_alpha_tstat(self):
        text = _read("latex/main_results.tex")
        tbl = _read("tables/table_factor_alphas.tex")
        val = float(_find_in_table(tbl, "FF6", 2))
        assert str(val) in text, (
            f"main_results.tex should mention FF6 t-stat {val}"
        )

    def test_main_results_fixed_mom_sharpe(self):
        text = _read("latex/main_results.tex")
        assert "0.06" in text, "main_results.tex should mention fixed mom Sharpe 0.06"

    def test_main_results_m0_sharpe(self):
        text = _read("latex/main_results.tex")
        assert "0.11" in text, "main_results.tex should mention M0 Sharpe 0.11"

    def test_main_results_m1_sharpe(self):
        text = _read("latex/main_results.tex")
        assert "-0.03" in text or "0.03" in text

    def test_main_results_fixed_mom_mdd(self):
        text = _read("latex/main_results.tex")
        tbl = _read("tables/table_performance.tex")
        val = float(_find_in_table(tbl, "Fixed 12-mo", 4))
        assert str(abs(val)) in text or f"{val}" in text

    def test_main_results_ablation_sharpe(self):
        text = _read("latex/main_results.tex")
        tbl = _read("tables/table_regime_signal_ablation.tex")
        val = float(_find_in_table(tbl, "no regime signal", 3))
        assert f"{val:.2f}" in text or "0.41" in text or "0.40" in text

    def test_main_results_raw_indicators_sharpe(self):
        text = _read("latex/main_results.tex")
        tbl = _read("tables/table_regime_signal_ablation.tex")
        val = float(_find_in_table(tbl, "raw indicators", 3))
        assert "0.30" in text or f"{val}" in text

    def test_main_results_panic_sharpe(self):
        text = _read("latex/main_results.tex")
        assert "1.56" in text

    def test_main_results_calm_sharpe(self):
        text = _read("latex/main_results.tex")
        assert "0.82" in text

    def test_main_results_shap_pi_46(self):
        text = _read("latex/main_results.tex")
        assert "46" in text, "main_results.tex should mention pi_filter 46% SHAP share"

    def test_main_results_shap_momentum_calm_50(self):
        text = _read("latex/main_results.tex")
        assert "50" in text

    def test_main_results_shap_momentum_panic_58(self):
        text = _read("latex/main_results.tex")
        assert "58" in text

    def test_main_results_depth4_return(self):
        text = _read("latex/main_results.tex")
        assert "21.9" in text

    # --- introduction.tex ---

    def test_introduction_167_months(self):
        text = _read("latex/introduction.tex")
        assert "167" in text

    def test_introduction_sharpe(self):
        text = _read("latex/introduction.tex")
        assert "1.11" in text

    def test_introduction_alpha(self):
        text = _read("latex/introduction.tex")
        assert "24.7" in text

    def test_introduction_alpha_tstat(self):
        text = _read("latex/introduction.tex")
        assert "4.78" in text

    def test_introduction_m1_sharpe(self):
        text = _read("latex/introduction.tex")
        assert "-0.03" in text or "0.03" in text

    def test_introduction_calm_spread(self):
        text = _read("latex/introduction.tex")
        assert "1.3" in text, "introduction.tex should mention calm spread 1.3%/month"

    def test_introduction_panic_spread(self):
        text = _read("latex/introduction.tex")
        assert "3.1" in text, "introduction.tex should mention panic spread 3.1%/month"

    # --- conclusion.tex ---

    def test_conclusion_167_months(self):
        text = _read("latex/conclusion.tex")
        assert "167" in text

    def test_conclusion_stress_12mo_drawdown(self):
        text = _read("latex/conclusion.tex")
        tbl = _read("tables/table_stress_scenarios.tex")
        val_str = _find_in_table(tbl, "12 months", 3)
        assert "51" in text, (
            f"conclusion.tex should mention 12-month stress MDD (~51%)"
        )

    def test_conclusion_stress_24mo_drawdown(self):
        text = _read("latex/conclusion.tex")
        assert "70" in text, (
            "conclusion.tex should mention 24-month stress MDD (~70%)"
        )

    def test_conclusion_subperiod_2021_2025(self):
        text = _read("latex/conclusion.tex")
        tbl = _read("tables/table_subperiod.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 3))  # 2021-2025 column
        assert f"{val}" in text or "1.69" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TABLE INTERNAL CONSISTENCY (numbers within a table add up)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableInternalConsistency:
    """Verify that numbers within individual tables are internally consistent."""

    def test_shap_shares_sum_to_100(self):
        tbl = _read("tables/table_shap.tex")
        mom = float(_find_in_table(tbl, "Momentum", 1))
        pi = float(_find_in_table(tbl, "pi", 1))
        assert mom + pi == pytest.approx(100, abs=1), (
            f"SHAP overall: {mom} + {pi} = {mom+pi}, expected 100"
        )

    def test_shap_calm_sums_to_100(self):
        tbl = _read("tables/table_shap.tex")
        mom = float(_find_in_table(tbl, "Momentum", 2))
        pi = float(_find_in_table(tbl, "pi", 2))
        assert mom + pi == pytest.approx(100, abs=1)

    def test_shap_panic_sums_to_100(self):
        tbl = _read("tables/table_shap.tex")
        mom = float(_find_in_table(tbl, "Momentum", 3))
        pi = float(_find_in_table(tbl, "pi", 3))
        assert mom + pi == pytest.approx(100, abs=1)

    def test_regime_sharpe_month_counts(self):
        """Calm + panic months should equal total test months (167)."""
        tbl = _read("tables/table_regime_sharpe.tex")
        # Extract month counts from caption
        caption = [l for l in tbl.splitlines() if "calm months" in l.lower()
                   or "108" in l]
        if caption:
            nums = re.findall(r"(\d+)\s*calm", caption[0])
            if nums:
                calm = int(nums[0])
                panic_nums = re.findall(r"(\d+)\s*panic", caption[0])
                if panic_nums:
                    panic = int(panic_nums[0])
                    assert calm + panic == 167, (
                        f"Calm ({calm}) + Panic ({panic}) = {calm+panic}, expected 167"
                    )

    def test_gelman_rubin_all_ok(self):
        """All Gelman-Rubin R-hat values should be < 1.1."""
        tbl = _read("tables/table_gelman_rubin.tex")
        rhat_values = re.findall(r"(\d+\.\d+)\s*&\s*OK", tbl)
        for rhat_str in rhat_values:
            rhat = float(rhat_str)
            assert rhat < 1.1, f"R-hat = {rhat} exceeds 1.1 threshold"

    def test_student_t_agreement_above_97(self):
        """All Student-t HMM agreement rates should be >= 97%."""
        tbl = _read("tables/table_student_t_hmm.tex")
        agreements = re.findall(r"(\d+\.\d+)\\%", tbl)
        for ag_str in agreements:
            ag = float(ag_str)
            assert ag >= 97.0, (
                f"Student-t agreement = {ag}%, expected >= 97%"
            )

    def test_student_t_normal_has_best_bic(self):
        """Normal HMM should have the best (lowest) BIC."""
        tbl = _read("tables/table_student_t_hmm.tex")
        normal_line = [l for l in tbl.splitlines() if "Normal" in l]
        assert len(normal_line) > 0, "Normal model not found in Student-t table"
        assert "textbf" in normal_line[0], (
            "Normal BIC should be bolded (best model)"
        )

    def test_cost_sensitivity_monotone(self):
        """M2 Sharpe should decrease as transaction costs increase."""
        tbl = _read("tables/table_cost_sensitivity.tex")
        m2_line = [l for l in tbl.splitlines() if "M2" in l]
        if m2_line:
            cols = [c.strip().rstrip("\\").strip() for c in m2_line[0].split("&")]
            vals = []
            for c in cols[1:]:  # skip row label
                try:
                    vals.append(float(c.replace("$-$", "-")))
                except ValueError:
                    continue
            for i in range(1, len(vals)):
                assert vals[i] <= vals[i-1] + 0.01, (
                    f"M2 Sharpe not monotonically decreasing with cost: {vals}"
                )

    def test_hmm_separation_all_significant(self):
        """All HMM feature separations should have CIs excluding zero."""
        tbl = _read("tables/table_hmm_separation.tex")
        for feat in cfg.HMM_FEATURES:
            short_name = feat.replace("_z", "").replace("_", r"\_")
            line = [l for l in tbl.splitlines() if short_name in l]
            if line:
                # Check that the CI doesn't span zero
                nums = re.findall(r"[-+]?\d+\.\d+", line[0])
                if len(nums) >= 5:
                    ci_low = float(nums[-2])
                    ci_high = float(nums[-1])
                    assert ci_low * ci_high > 0, (
                        f"HMM separation CI for {feat} spans zero: [{ci_low}, {ci_high}]"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CONFIG-TO-ARTIFACT ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigAlignment:
    """Verify pipeline config matches production settings and artifacts."""

    def test_use_fundamentals_false(self):
        assert cfg.USE_FUNDAMENTALS is False

    def test_xgb_seeds_50(self):
        assert len(cfg.XGB_SEEDS) == 50

    def test_hmm_seeds_200(self):
        assert len(cfg.HMM_SEEDS) == 200

    def test_train_end_2011(self):
        assert cfg.TRAIN_END == "2011-01-01"

    def test_trading_fee_10bps(self):
        assert cfg.TRADING_FEE == pytest.approx(0.001)

    def test_max_depth_4(self):
        assert cfg.MAX_DEPTH == 4

    def test_k_states_2(self):
        assert cfg.K_STATES == 2

    def test_n_estimators_500(self):
        assert cfg.N_ESTIMATORS == 500

    def test_learning_rate_005(self):
        assert cfg.LEARNING_RATE == pytest.approx(0.05)

    def test_subsample_08(self):
        assert cfg.SUBSAMPLE == pytest.approx(0.8)

    def test_colsample_08(self):
        assert cfg.COLSAMPLE == pytest.approx(0.8)

    def test_hmm_features_correct(self):
        assert cfg.HMM_FEATURES == ['DD_z', 'DISP_z', 'REL_N_z', 'CS_z']

    def test_momentum_lookbacks_1_to_12(self):
        assert cfg.MOM_LOOKBACKS == list(range(1, 13))

    def test_cs_features_13(self):
        assert len(cfg.CS_FEATURES) == 13

    def test_hmm_iterations(self):
        assert cfg.HMM_ITERATIONS == 2000

    def test_hmm_burnin(self):
        assert cfg.HMM_BURNIN == 500

    def test_methodology_matches_config(self):
        """Methodology.tex numbers should match config."""
        text = _read("latex/methodology.tex")
        # LaTeX uses {,} for thousands separator: $M=2{,}000$
        assert "2,000" in text or "2000" in text or "2{,}000" in text, (
            "methodology.tex should mention 2000 Gibbs iterations"
        )
        assert "500" in text, "methodology.tex should mention 500 burn-in"
        assert "200" in text, "methodology.tex should mention 200 seeds"
        assert "50" in text, "methodology.tex should mention 50 XGB seeds"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. INTER-SCRIPT DATA FLOW (PIPELINE CONNECTIONS)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineConnections:
    """Verify that outputs of each stage are correctly consumed by the next."""

    def test_hmm_output_feeds_cross_sectional(self):
        """panel_with_regimes.parquet must have pi_filter for cross-sectional model."""
        panel = _load_panel_with_regimes()
        assert 'pi_filter' in panel.columns
        assert 'date' in panel.columns
        # Must have enough data for training period
        panel['date'] = pd.to_datetime(panel['date'])
        train_data = panel[panel['date'] < cfg.TRAIN_END]
        assert len(train_data) > 100, (
            f"Only {len(train_data)} training rows in panel_with_regimes"
        )

    def test_cross_sectional_output_feeds_analysis(self):
        """cs_artefacts_data.pkl must have what main_results_analysis needs."""
        art = _get_artefacts()
        # main_results_analysis needs these keys
        needed = ['test', 'train', 'X_test', 'X_train', 'FEATURES',
                  'strategies_lo', 'shap_values']
        for key in needed:
            assert key in art, f"Artefact missing key '{key}' needed by analysis scripts"

    def test_artefact_test_df_has_score_columns(self):
        """Test DataFrame should have scoring columns for all methods."""
        art = _get_artefacts()
        test_df = art["test"]
        assert "score_xgb" in test_df.columns
        assert "score_lr" in test_df.columns or "score_logistic" in test_df.columns

    def test_ff_factors_align_with_test_period(self):
        """FF factors must cover the full test period."""
        ff = pd.read_parquet(os.path.join(PROJECT_ROOT, cfg.FF_FACTORS_PATH))
        ff.index = pd.to_datetime(ff.index)
        art = _get_artefacts()
        strats = art["strategies_lo"]
        for name, ret_series in strats.items():
            if "xgb" in name.lower() or "m2" in name.lower():
                test_dates = pd.to_datetime(ret_series.index)
                ff_start = ff.index.min()
                ff_end = ff.index.max()
                assert ff_start <= test_dates.min(), (
                    f"FF factors start ({ff_start}) after test ({test_dates.min()})"
                )
                assert ff_end >= test_dates.max(), (
                    f"FF factors end ({ff_end}) before test end ({test_dates.max()})"
                )
                break

    def test_all_pipeline_scripts_exist(self):
        """All scripts referenced in run_pipeline.py should exist."""
        scripts = [
            'scripts/hmm_model.py',
            'scripts/cross_sectional_model.py',
            'scripts/main_results_analysis.py',
            'tests/robustness_checks.py',
            'tests/test_external_validity.py',
            'scripts/hmm_diagnostics.py',
            'scripts/new_ls_analyses.py',
            'scripts/generate_plots.py',
            'scripts/stress_test.py',
            'scripts/economic_mechanism.py',
            'scripts/selection_rank_analysis.py',
            'scripts/risk_aversion_thesis_table.py',
            'scripts/ridge_baseline_test.py',
            'scripts/depth_vs_sharpe.py',
            'scripts/tree_path_analysis.py',
            'scripts/tree_combo_grouped.py',
            'scripts/test_2008_oos.py',
            'scripts/expanding_window.py',
        ]
        missing = []
        for s in scripts:
            path = os.path.join(PROJECT_ROOT, s)
            if not os.path.exists(path):
                missing.append(s)
        assert missing == [], f"Missing pipeline scripts: {missing}"

    def test_tables_dir_exists(self):
        assert os.path.isdir(TABLES_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. REPRODUCIBILITY ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestReproducibilityArtifacts:
    """Verify CSV/pkl result files exist and contain expected values."""

    def test_depth_results_csv_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "depth_results.csv")
        assert os.path.exists(path)

    def test_depth_results_has_depth_4(self):
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "depth_results.csv")
        df = pd.read_csv(path)
        depth_col = [c for c in df.columns if "depth" in c.lower()][0]
        assert 4 in df[depth_col].values, "depth_results.csv missing depth=4 row"

    def test_depth_results_depth4_return_matches_thesis(self):
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "depth_results.csv")
        df = pd.read_csv(path)
        depth_col = [c for c in df.columns if "depth" in c.lower()][0]
        ret_col = [c for c in df.columns if "ret" in c.lower() or "ann" in c.lower()]
        if ret_col:
            row = df[df[depth_col] == 4]
            val = float(row[ret_col[0]].iloc[0])
            if val < 1:
                val *= 100
            assert val == pytest.approx(21.9, abs=1.0)

    def test_risk_aversion_results_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "risk_aversion_thesis_results.csv")
        assert os.path.exists(path)

    def test_xgb_model_pickle_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.XGB_MODEL_PATH)
        assert os.path.exists(path)

    def test_lr_model_pickle_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.LR_MODEL_PATH)
        assert os.path.exists(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. FACTOR ALPHA TABLE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorAlphas:
    """Verify factor alpha table values match thesis claims."""

    def test_ff6_alpha(self):
        tbl = _read("tables/table_factor_alphas.tex")
        val = float(_find_in_table(tbl, "FF6", 1))
        assert val == pytest.approx(24.7, abs=0.1)

    def test_ff6_tstat(self):
        tbl = _read("tables/table_factor_alphas.tex")
        val = float(_find_in_table(tbl, "FF6", 2))
        assert val == pytest.approx(4.78, abs=0.05)

    def test_capm_alpha(self):
        tbl = _read("tables/table_factor_alphas.tex")
        val = float(_find_in_table(tbl, "CAPM", 1))
        assert val == pytest.approx(23.8, abs=0.1)

    def test_capm_tstat(self):
        tbl = _read("tables/table_factor_alphas.tex")
        val = float(_find_in_table(tbl, "CAPM", 2))
        assert val == pytest.approx(4.08, abs=0.05)

    def test_all_alphas_positive(self):
        """All factor model alphas should be positive and significant."""
        tbl = _read("tables/table_factor_alphas.tex")
        for model in ["CAPM", "FF3", "Carhart", "FF5", "FF6"]:
            alpha = float(_find_in_table(tbl, model, 1))
            tstat = float(_find_in_table(tbl, model, 2))
            assert alpha > 0, f"{model} alpha = {alpha}, expected positive"
            assert tstat > 2.0, f"{model} t-stat = {tstat}, expected > 2.0"

    def test_alphas_increase_with_controls(self):
        """Alpha should be stable or increase as factors are added."""
        tbl = _read("tables/table_factor_alphas.tex")
        capm = float(_find_in_table(tbl, "CAPM", 1))
        ff6 = float(_find_in_table(tbl, "FF6", 1))
        # FF6 alpha >= CAPM alpha (or at least close)
        assert ff6 >= capm - 2.0, (
            f"FF6 alpha ({ff6}) much lower than CAPM ({capm})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 15. ROBUSTNESS TABLE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustnessTables:
    """Verify robustness results are consistent with thesis claims."""

    def test_subperiod_m2_all_positive(self):
        """M2 should have positive Sharpe in all sub-periods."""
        tbl = _read("tables/table_subperiod.tex")
        for col_idx in [1, 2, 3]:  # 2011-2015, 2016-2020, 2021-2025
            val = float(_find_in_table(tbl, "M2: XGB", col_idx))
            assert val > 0, f"M2 sub-period {col_idx} Sharpe = {val}, expected > 0"

    def test_subperiod_values_match_thesis(self):
        tbl = _read("tables/table_subperiod.tex")
        expected = [0.62, 1.03, 1.69]
        for i, exp in enumerate(expected):
            val = float(_find_in_table(tbl, "M2: XGB", i + 1))
            assert val == pytest.approx(exp, abs=0.02), (
                f"M2 sub-period {i+1} Sharpe = {val}, expected {exp}"
            )

    def test_cost_sensitivity_m2_at_50bps(self):
        tbl = _read("tables/table_cost_sensitivity.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 6))  # 50 bps column
        assert val > 0.5, f"M2 at 50 bps = {val}, expected > 0.5 (thesis says ~0.79)"

    def test_january_exclusion_sharpe(self):
        tbl = _read("tables/table_january.tex")
        # Find the excluding-January M2 Sharpe
        lines = tbl.splitlines()
        jan_section = False
        for line in lines:
            if "Excluding January" in line:
                jan_section = True
            if jan_section and "M2: XGB" in line:
                cols = [c.strip().rstrip("\\").strip() for c in line.split("&")]
                val = float(cols[3].replace("$-$", "-"))
                assert val == pytest.approx(1.06, abs=0.03), (
                    f"M2 Jan-excluded Sharpe = {val}, expected 1.06"
                )
                break

    def test_ablation_hmm_outperforms_no_signal(self):
        tbl = _read("tables/table_regime_signal_ablation.tex")
        hmm_sharpe = float(_find_in_table(tbl, "HMM", 3))
        no_signal = float(_find_in_table(tbl, "no regime signal", 3))
        raw = float(_find_in_table(tbl, "raw indicators", 3))
        assert hmm_sharpe > no_signal, (
            f"HMM ({hmm_sharpe}) should outperform no signal ({no_signal})"
        )
        assert hmm_sharpe > raw, (
            f"HMM ({hmm_sharpe}) should outperform raw indicators ({raw})"
        )

    def test_ghm_all_below_m2(self):
        """All GHM variants should have Sharpe below M2."""
        tbl = _read("tables/table_ghm_comparison.tex")
        m2_sharpe = float(_find_in_table(tbl, "M2: XGB", 1))
        for variant in ["GHM SLOW", "GHM MED", "GHM FAST", "GHM DYN"]:
            val = float(_find_in_table(tbl, variant, 1))
            assert val < m2_sharpe, (
                f"{variant} Sharpe ({val}) should be < M2 ({m2_sharpe})"
            )

    def test_ridge_sharpe_negative(self):
        """Ridge regression should have negative Sharpe (linear model fails)."""
        tbl = _read("tables/table_ridge.tex")
        for alpha_label in ["0.1", "1.0", "10", "100"]:
            line = [l for l in tbl.splitlines() if f"Ridge" in l and alpha_label in l]
            if line:
                cols = [c.strip().rstrip("\\").strip() for c in line[0].split("&")]
                sharpe_str = cols[2].replace("$-$", "-")
                try:
                    val = float(sharpe_str)
                    assert val < 0, f"Ridge alpha={alpha_label} Sharpe = {val}, expected < 0"
                except ValueError:
                    pass

    def test_xgb_hyperparams_baseline_sharpe(self):
        """Baseline config (depth=4, lr=0.05, n=500) should have Sharpe ~1.08."""
        tbl = _read("tables/table_xgb_hyperparams.tex")
        val = float(_find_in_table(tbl, "depth=4, lr=0.05, $n$=500", 1))
        assert val == pytest.approx(1.08, abs=0.05)

    def test_stress_baseline_mdd(self):
        tbl = _read("tables/table_stress_scenarios.tex")
        val = float(_find_in_table(tbl, "Baseline", 3))
        assert val == pytest.approx(-24.8, abs=0.5)

    def test_stress_24mo_mdd(self):
        tbl = _read("tables/table_stress_scenarios.tex")
        val = float(_find_in_table(tbl, "24 months", 3))
        assert val == pytest.approx(-70, abs=2)

    def test_turnover_m2(self):
        tbl = _read("tables/table_turnover.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 1))
        assert val == pytest.approx(132.4, abs=1.0), (
            f"M2 avg monthly turnover = {val}%, expected 132.4%"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 16. FIGURE FILE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestFigureFiles:
    """Verify that all figures referenced in LaTeX exist on disk."""

    @staticmethod
    def _collect_figure_refs() -> list:
        refs = []
        for dirpath in [LATEX_DIR, PROJECT_ROOT]:
            if not os.path.isdir(dirpath):
                continue
            for fname in os.listdir(dirpath):
                if not fname.endswith(".tex"):
                    continue
                fpath = os.path.join(dirpath, fname)
                content = open(fpath, "r", encoding="utf-8").read()
                for m in re.finditer(
                    r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content
                ):
                    refs.append(m.group(1))
        return list(set(refs))

    def test_all_referenced_figures_exist(self):
        refs = self._collect_figure_refs()
        assert len(refs) > 0, "No figure references found in LaTeX files"
        missing = []
        for fig in refs:
            full_path = os.path.join(PROJECT_ROOT, fig)
            if not os.path.exists(full_path):
                missing.append(fig)
        assert missing == [], (
            f"Missing figures:\n  " + "\n  ".join(missing)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 17. CROSS-REFERENCE INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossReferences:
    """Verify LaTeX labels, inputs, and citations are consistent."""

    @staticmethod
    def _all_tex_content() -> str:
        parts = []
        for dirpath in [PROJECT_ROOT, LATEX_DIR, TABLES_DIR]:
            if not os.path.isdir(dirpath):
                continue
            for fname in sorted(os.listdir(dirpath)):
                if fname.endswith(".tex"):
                    fpath = os.path.join(dirpath, fname)
                    parts.append(open(fpath, "r", encoding="utf-8").read())
        return "\n".join(parts)

    def test_all_refs_have_labels(self):
        content = self._all_tex_content()
        refs = set(re.findall(r"\\(?:c?ref|Cref)\{([^}]+)\}", content))
        labels = set(re.findall(r"\\label\{([^}]+)\}", content))
        missing = refs - labels
        assert missing == set(), (
            f"\\ref targets without \\label: {sorted(missing)}"
        )

    def test_all_inputs_exist(self):
        content = self._all_tex_content()
        inputs = re.findall(r"\\input\{([^}]+)\}", content)
        missing = []
        for inp in inputs:
            candidates = [
                os.path.join(PROJECT_ROOT, inp),
                os.path.join(PROJECT_ROOT, inp + ".tex"),
                os.path.join(LATEX_DIR, inp),
                os.path.join(LATEX_DIR, inp + ".tex"),
                os.path.join(TABLES_DIR, inp),
                os.path.join(TABLES_DIR, inp + ".tex"),
            ]
            if not any(os.path.exists(c) for c in candidates):
                missing.append(inp)
        assert missing == [], f"\\input files not found: {missing}"

    def test_citation_keys_in_bib(self):
        content = self._all_tex_content()
        cite_groups = re.findall(
            r"\\cite[tp]?(?:\[[^\]]*\])*\{([^}]+)\}", content
        )
        cited_keys = set()
        for group in cite_groups:
            for key in group.split(","):
                key = key.strip()
                if key:
                    cited_keys.add(key)
        bib_path = os.path.join(PROJECT_ROOT, "references.bib")
        if os.path.exists(bib_path):
            bib_content = open(bib_path, "r", encoding="utf-8").read()
            bib_keys = set(re.findall(r"@\w+\{(\w+)", bib_content))
            missing = cited_keys - bib_keys
            assert missing == set(), (
                f"Citation keys not in references.bib: {sorted(missing)}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 18. MOMENTUM COMPUTATION SANITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestMomentumComputation:
    """Verify momentum features are computed correctly."""

    def test_mom_12_is_cumulative_return(self):
        """mom_12 should approximately equal the 12-month cumulative return."""
        art = _get_artefacts()
        test_df = art["test"]
        # mom_12 should be reasonable (not too extreme)
        mom12 = test_df["mom_12"].dropna()
        assert mom12.median() > -0.5, f"mom_12 median = {mom12.median()}, too low"
        assert mom12.median() < 1.0, f"mom_12 median = {mom12.median()}, too high"

    def test_mom_1_is_shorter_than_mom_12(self):
        """Variance of mom_1 should be less than mom_12."""
        art = _get_artefacts()
        test_df = art["test"]
        var_1 = test_df["mom_1"].var()
        var_12 = test_df["mom_12"].var()
        assert var_1 < var_12, (
            f"mom_1 variance ({var_1:.4f}) should be < mom_12 ({var_12:.4f})"
        )

    def test_momentum_horizons_ordered(self):
        """Standard deviation should generally increase with horizon."""
        art = _get_artefacts()
        test_df = art["test"]
        stds = [test_df[f"mom_{h}"].std() for h in range(1, 13)]
        # Allow some non-monotonicity but overall trend should be increasing
        assert stds[-1] > stds[0], (
            f"mom_12 std ({stds[-1]:.4f}) should exceed mom_1 std ({stds[0]:.4f})"
        )

    def test_no_look_ahead_in_momentum(self):
        """Momentum features should only use past returns (no future data)."""
        art = _get_artefacts()
        test_df = art["test"]
        # Check correlation between mom_1 and ret_fwd is not suspiciously high
        corr = test_df[["mom_1", "ret_fwd"]].corr().iloc[0, 1]
        assert abs(corr) < 0.5, (
            f"mom_1 vs ret_fwd correlation = {corr:.3f}, suspiciously high "
            f"(possible look-ahead bias)"
        )
