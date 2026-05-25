"""
test_new_ls_analyses_helpers.py
================================
Direct logic tests for `scripts/new_ls_analyses.py` (Step 7 of the
pipeline). The script computes the headline factor-alpha tables for M2,
the D&M comparison, IC rotation, and Mom→M2 / M2→Mom spanning
regressions.

The factor-alpha numbers (CAPM α=23.4%, FF6 α=24.1% etc.) are quoted
verbatim in the thesis. Pinning the math here is the only logic-level
guard — `test_pipeline_technical.py` only checks the table output values.
"""

import ast
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

SCRIPT_PATH = os.path.join(REPO, 'scripts', 'new_ls_analyses.py')


def _extract_function(src_path, fn_name, namespace=None):
    """AST-extract a function definition and exec in an isolated namespace."""
    with open(src_path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            ns = namespace or {}
            ns.setdefault('np', np)
            ns.setdefault('pd', pd)
            ns.setdefault('sm', sm)
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, src_path, 'exec'), ns)
            return ns[fn_name]
    raise ValueError(f'{fn_name} not found in {src_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Source-level invariants — pin the spec the script implements
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceInvariants:

    @pytest.fixture(scope='class')
    def src(self):
        with open(SCRIPT_PATH) as f:
            return f.read()

    def test_uses_HAC_with_six_lags(self, src):
        """Newey-West with 6 lags is the thesis convention."""
        assert "cov_type='HAC'" in src
        assert "'maxlags': 6" in src

    def test_alpha_annualised_by_twelve(self, src):
        """Monthly OLS intercept × 12 = annualised alpha."""
        assert 'res.params[0] * 12' in src

    def test_factor_models_in_get_alphas(self, src):
        """Pin all 5 factor-model specifications."""
        for model in ['CAPM', 'FF3', 'Carhart', 'FF5', 'FF6']:
            assert f"('{model}'," in src or f"'{model}'" in src

    def test_ff_factor_columns_present(self, src):
        """FF columns the script consumes."""
        for col in ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RMW', 'CMA', 'RF']:
            assert f"'{col}'" in src

    def test_spanning_regression_runs_both_directions(self, src):
        """Both M2 ~ M1 and M1 ~ M2 spanning regressions are reported."""
        assert 'res_m2_on_m1' in src
        assert 'res_m1_on_m2' in src

    def test_ic_rotation_uses_spearman(self, src):
        """IC rotation uses Spearman rank correlation, not Pearson."""
        assert 'spearmanr' in src


# ═══════════════════════════════════════════════════════════════════════════════
# get_alphas — extract and test on synthetic factor-return panel
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetAlphas:

    @pytest.fixture(scope='class')
    def get_alphas(self):
        """Stub the file-scope `ff` dataframe + `print` to make the
        function self-contained for testing."""
        # Build a synthetic FF panel
        rng = np.random.default_rng(0)
        dates = pd.date_range('2011-01-31', periods=120, freq='ME')
        ff = pd.DataFrame({
            'Mkt-RF': rng.normal(0.005, 0.04, 120),
            'SMB':    rng.normal(0.001, 0.03, 120),
            'HML':    rng.normal(0.001, 0.03, 120),
            'RMW':    rng.normal(0.001, 0.02, 120),
            'CMA':    rng.normal(0.001, 0.02, 120),
            'UMD':    rng.normal(0.005, 0.05, 120),
            'RF':     0.0001,
        }, index=dates)
        ns = {'ff': ff}
        return _extract_function(SCRIPT_PATH, 'get_alphas', namespace=ns)

    def test_returns_dict_with_five_models(self, get_alphas):
        rng = np.random.default_rng(1)
        dates = pd.date_range('2011-01-31', periods=120, freq='ME')
        r = pd.Series(rng.normal(0.01, 0.04, 120), index=dates)
        out = get_alphas(r, 'test')
        assert set(out.keys()) == {'CAPM', 'FF3', 'Carhart', 'FF5', 'FF6'}

    def test_each_model_has_alpha_and_t(self, get_alphas):
        rng = np.random.default_rng(2)
        dates = pd.date_range('2011-01-31', periods=120, freq='ME')
        r = pd.Series(rng.normal(0.01, 0.04, 120), index=dates)
        out = get_alphas(r, 'test')
        for model, d in out.items():
            assert 'alpha' in d and 't' in d and 'p' in d
            assert np.isfinite(d['alpha'])
            assert np.isfinite(d['t'])

    def test_alpha_zero_when_returns_match_market(self, get_alphas):
        """If r is exactly proportional to Mkt-RF (no excess return),
        CAPM alpha should be ~0. The t-stat on alpha can be anything
        when residuals are floating-point near zero — we only assert
        the magnitude is small."""
        rng = np.random.default_rng(3)
        dates = pd.date_range('2011-01-31', periods=120, freq='ME')
        ns = {'ff': pd.DataFrame({
            'Mkt-RF': rng.normal(0.005, 0.04, 120),
            'SMB':    np.zeros(120),
            'HML':    np.zeros(120),
            'RMW':    np.zeros(120),
            'CMA':    np.zeros(120),
            'UMD':    np.zeros(120),
            'RF':     0.0,
        }, index=dates)}
        ga = _extract_function(SCRIPT_PATH, 'get_alphas', namespace=ns)
        r = pd.Series(0.5 * ns['ff']['Mkt-RF'].values, index=dates)
        out = ga(r, 'test')
        # CAPM alpha annualised should be ~0 (within 0.01 = 1pp)
        assert abs(out['CAPM']['alpha']) < 0.01

    def test_alpha_positive_when_constant_excess_return(self, get_alphas):
        """If r = 0.01 + 0.5 * Mkt-RF (a 1% per-month true alpha),
        CAPM alpha t-stat should be highly positive."""
        rng = np.random.default_rng(4)
        dates = pd.date_range('2011-01-31', periods=120, freq='ME')
        ns = {'ff': pd.DataFrame({
            'Mkt-RF': rng.normal(0.005, 0.04, 120),
            'SMB':    np.zeros(120),
            'HML':    np.zeros(120),
            'RMW':    np.zeros(120),
            'CMA':    np.zeros(120),
            'UMD':    np.zeros(120),
            'RF':     0.0,
        }, index=dates)}
        ga = _extract_function(SCRIPT_PATH, 'get_alphas', namespace=ns)
        # Add a tiny epsilon noise to avoid singular regressors
        eps = rng.normal(0, 1e-4, 120)
        r = pd.Series(0.01 + 0.5 * ns['ff']['Mkt-RF'].values + eps,
                      index=dates)
        out = ga(r, 'test')
        # Annualised alpha ≈ 12%
        assert out['CAPM']['alpha'] == pytest.approx(0.12, abs=0.02)
        # t-stat highly significant
        assert out['CAPM']['t'] > 4.0


# ═══════════════════════════════════════════════════════════════════════════════
# Residual diagnostic helpers — added 2026-05-25 for NW(6) assumption validation
# (Appendix H.3 residual diagnostics)
# ═══════════════════════════════════════════════════════════════════════════════


class TestResidualDiagnosticInvariants:
    """Source-level pins for the three new diagnostic helpers."""

    @pytest.fixture(scope='class')
    def src(self):
        with open(SCRIPT_PATH) as f:
            return f.read()

    def test_has_ljung_box_diagnostic(self, src):
        assert 'def ljung_box_diagnostic' in src

    def test_has_nw_lag_stability(self, src):
        assert 'def nw_lag_stability' in src

    def test_has_acf_plot_grid(self, src):
        assert 'def plot_residual_acf_grid' in src

    def test_lb_uses_statsmodels_acorr(self, src):
        assert 'acorr_ljungbox' in src

    def test_nw_lag_stability_sweeps_multiple_lags(self, src):
        """Pin the {3, 6, 12, 24} default sweep used in the appendix."""
        assert '(3, 6, 12, 24)' in src or '[3, 6, 12, 24]' in src

    def test_writes_residual_diagnostics_csv(self, src):
        """Diagnostic block must save the per-(strategy, model) results."""
        assert 'factor_residual_diagnostics.csv' in src

    def test_writes_acf_figure_to_thesis(self, src):
        """ACF figure for Appendix H.3 must land in plots/thesis/."""
        assert 'factor_residual_acf' in src
        assert 'PLOTS_THESIS_DIR' in src

    def test_writes_residual_diagnostics_tex_table(self, src):
        """LB table must be generated as a thesis-grade .tex file."""
        assert 'table_residual_diagnostics' in src


class TestLjungBoxDiagnostic:

    @pytest.fixture
    def lb_fn(self):
        ns = {'sm': sm}
        return _extract_function(SCRIPT_PATH, 'ljung_box_diagnostic',
                                 namespace=ns)

    def test_returns_keys_for_each_lag(self, lb_fn):
        rng = np.random.default_rng(0)
        resid = rng.normal(0, 0.04, 200)
        out = lb_fn(resid, lags=(6, 12))
        assert set(out.keys()) == {'LB6_stat', 'LB6_p', 'LB12_stat', 'LB12_p'}
        for v in out.values():
            assert np.isfinite(v)

    def test_white_noise_fails_to_reject(self, lb_fn):
        """For Gaussian white noise, LB should not strongly reject. Under H0
        p-values are uniform on (0,1), so ~5% of single-seed draws land below
        0.05 by chance — we use the looser p > 0.01 threshold (would fail
        only 1% of the time under H0) to make the test robust to seed."""
        rng = np.random.default_rng(42)
        resid = rng.normal(0, 0.04, 300)
        out = lb_fn(resid, lags=(6, 12))
        assert out['LB6_p'] > 0.01
        assert out['LB12_p'] > 0.01

    def test_ar1_residuals_reject(self, lb_fn):
        """Strongly autocorrelated AR(1) residuals (φ=0.5) should reject LB."""
        rng = np.random.default_rng(7)
        T = 400
        e = np.zeros(T)
        for t in range(1, T):
            e[t] = 0.5 * e[t-1] + rng.normal(0, 0.04)
        out = lb_fn(e, lags=(6, 12))
        assert out['LB6_p'] < 0.01
        assert out['LB12_p'] < 0.01


class TestNWLagStability:

    @pytest.fixture
    def nw_fn(self):
        ns = {'sm': sm}
        return _extract_function(SCRIPT_PATH, 'nw_lag_stability',
                                 namespace=ns)

    def test_returns_t_per_lag(self, nw_fn):
        rng = np.random.default_rng(0)
        T = 200
        y = 0.01 + rng.normal(0, 0.04, T)
        X = sm.add_constant(rng.normal(0, 0.04, (T, 2)))
        out = nw_fn(y, X, lags=(3, 6, 12, 24))
        assert set(out.keys()) == {'t_nw3', 't_nw6', 't_nw12', 't_nw24'}
        for v in out.values():
            assert np.isfinite(v)

    def test_t_stat_stable_under_white_residuals(self, nw_fn):
        """When residuals are white noise the t-stat on the intercept
        should be similar across NW lag choices."""
        rng = np.random.default_rng(0)
        T = 400
        y = 0.01 + rng.normal(0, 0.04, T)
        X = np.ones((T, 1))
        out = nw_fn(y, X, lags=(3, 6, 12, 24))
        ts = list(out.values())
        # Stability under white noise: range across lags should be small
        assert (max(ts) - min(ts)) < 1.0


class TestPlotResidualACFGrid:

    @pytest.fixture
    def plot_fn(self):
        # Provide plt and plot_acf inside the namespace so the extracted
        # function can find them.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
        ns = {'plt': plt, 'plot_acf': plot_acf}
        return _extract_function(SCRIPT_PATH, 'plot_residual_acf_grid',
                                 namespace=ns)

    def test_creates_file(self, plot_fn, tmp_path):
        rng = np.random.default_rng(0)
        residuals = {
            'A': rng.normal(0, 0.04, 150),
            'B': rng.normal(0, 0.04, 150),
            'C': rng.normal(0, 0.04, 150),
            'D': rng.normal(0, 0.04, 150),
        }
        save_path = tmp_path / 'test_acf.png'
        plot_fn(residuals, str(save_path), lags=24, suptitle='Test ACF')
        assert save_path.exists()
        # File should have non-trivial size (matplotlib PNG > 5 KB typically)
        assert save_path.stat().st_size > 5_000
