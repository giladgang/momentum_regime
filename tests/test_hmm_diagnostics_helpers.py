"""
test_hmm_diagnostics_helpers.py
================================
Direct logic tests for the helper functions inside
`scripts/hmm_diagnostics.py`. The script is module-level (everything
executes at import), so we extract each helper by AST and run it in an
isolated namespace — no heavy fits, no file I/O at test time.

Helpers covered
---------------
- gelman_rubin: R-hat convergence diagnostic
- log_emission: per-state log multivariate normal density
- forward_filter: causal filtered probabilities of a 2-state HMM

Why this layer matters
----------------------
hmm_diagnostics.py reports the convergence of the production HMM to the
thesis. If gelman_rubin or forward_filter regresses silently, the
table_gelman_rubin.tex would still write but with wrong numbers. Pinning
the math here is the only logic-level guard.
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

SCRIPT_PATH = os.path.join(REPO, 'scripts', 'hmm_diagnostics.py')


def _extract_function(src_path, fn_name, namespace=None):
    """AST-extract a top-level function definition and exec it in an
    isolated namespace. Returns the resulting callable."""
    with open(src_path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            ns = namespace or {}
            ns.setdefault('np', np)
            ns.setdefault('logsumexp', logsumexp)
            ns.setdefault('multivariate_normal', multivariate_normal)
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, src_path, 'exec'), ns)
            return ns[fn_name]
    raise ValueError(f'{fn_name} not found in {src_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# gelman_rubin
# ═══════════════════════════════════════════════════════════════════════════════

class TestGelmanRubin:

    @pytest.fixture(scope='class')
    def fn(self):
        return _extract_function(SCRIPT_PATH, 'gelman_rubin')

    def test_identical_chains_give_rhat_close_to_one(self, fn):
        """When all chains have identical samples, between-chain variance
        B=0 so var_hat = (1 - 1/n) * W, giving R-hat = sqrt(1 - 1/n).
        For n=1000 this is ~0.9995 — close to 1 but not exactly. The
        n→∞ limit is exactly 1."""
        rng = np.random.default_rng(0)
        n = 1000
        same = rng.normal(0, 1, n)
        chains = [same.copy() for _ in range(4)]
        rhat = fn(chains)
        expected = np.sqrt(1 - 1/n)
        assert rhat == pytest.approx(expected, abs=1e-9)

    def test_well_mixed_chains_close_to_one(self, fn):
        """Independent chains drawn from the same distribution → R-hat
        very close to 1 with enough samples."""
        rng = np.random.default_rng(1)
        chains = [rng.normal(0, 1, 5000) for _ in range(4)]
        rhat = fn(chains)
        assert rhat < 1.05, (
            f'well-mixed chains should give R-hat ≈ 1, got {rhat}'
        )

    def test_chains_with_different_means_flag_nonconvergence(self, fn):
        """When chains have very different means, R-hat >> 1.1."""
        rng = np.random.default_rng(2)
        chains = [rng.normal(loc, 0.1, 1000) for loc in (0, 5, -3, 8)]
        rhat = fn(chains)
        assert rhat > 1.1, (
            f'divergent chains should give R-hat > 1.1, got {rhat}'
        )

    def test_zero_within_chain_variance_returns_one(self, fn):
        """Edge case: if every chain is constant (W=0), function returns 1.0
        (avoids divide-by-zero — pinned to current contract)."""
        chains = [np.full(100, 1.0) for _ in range(4)]
        rhat = fn(chains)
        assert rhat == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# log_emission
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogEmission:

    @pytest.fixture(scope='class')
    def fn(self):
        return _extract_function(SCRIPT_PATH, 'log_emission',
                                 namespace={'K': 2})

    def test_returns_n_by_k_matrix(self, fn):
        Z = np.random.default_rng(0).normal(0, 1, (50, 4))
        mu = np.array([np.zeros(4), np.ones(4)])
        Sigma = np.array([np.eye(4), np.eye(4)])
        out = fn(Z, mu, Sigma)
        assert out.shape == (50, 2)
        assert np.isfinite(out).all()

    def test_log_density_higher_under_correct_state(self, fn):
        """A point at mu_0 should have higher log-density under state 0
        than under state 1 when state 1 is far away."""
        Z = np.zeros((1, 3))
        mu = np.array([np.zeros(3), np.full(3, 5.0)])
        Sigma = np.array([np.eye(3), np.eye(3)])
        out = fn(Z, mu, Sigma)
        assert out[0, 0] > out[0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# forward_filter (causal pi_filter)
# ═══════════════════════════════════════════════════════════════════════════════

class TestForwardFilter:

    @pytest.fixture(scope='class')
    def fn(self):
        ns = {'K': 2}
        # Pull dependent helper into the namespace too
        ns['log_emission'] = _extract_function(SCRIPT_PATH, 'log_emission',
                                                namespace={'K': 2})
        return _extract_function(SCRIPT_PATH, 'forward_filter', namespace=ns)

    def test_rows_sum_to_one(self, fn):
        """Filtered probabilities at each time t must form a valid
        distribution over states."""
        rng = np.random.default_rng(0)
        Z = rng.normal(0, 1, (60, 3))
        mu = np.array([np.zeros(3), np.full(3, 2.0)])
        Sigma = np.array([np.eye(3), np.eye(3)])
        P = np.array([[0.95, 0.05], [0.10, 0.90]])
        out = fn(Z, mu, Sigma, P)
        sums = out.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9)

    def test_output_in_unit_interval(self, fn):
        rng = np.random.default_rng(1)
        Z = rng.normal(0, 1, (60, 3))
        mu = np.array([np.zeros(3), np.full(3, 2.0)])
        Sigma = np.array([np.eye(3), np.eye(3)])
        P = np.array([[0.95, 0.05], [0.10, 0.90]])
        out = fn(Z, mu, Sigma, P)
        assert (out >= 0).all() and (out <= 1).all()

    def test_strong_state_signal_locks_assignment(self, fn):
        """When a sequence is far from state 0 and close to state 1,
        forward filter should assign high probability to state 1."""
        Z = np.full((30, 3), 5.0)  # all observations near state 1's mean
        mu = np.array([np.zeros(3), np.full(3, 5.0)])
        Sigma = np.array([np.eye(3), np.eye(3)])
        P = np.array([[0.95, 0.05], [0.10, 0.90]])
        out = fn(Z, mu, Sigma, P)
        # Skip first row (initial uniform prior dominates briefly)
        assert (out[5:, 1] > 0.9).all()
