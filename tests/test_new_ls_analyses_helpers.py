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
