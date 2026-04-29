"""Tests for scripts/bootstrap_helpers.py.

Run with:
    pytest scripts/test_bootstrap_helpers.py -v
"""

import numpy as np
import pytest

from bootstrap_helpers import block_bootstrap_sharpe, seed_stability


# ----------------------------------------------------------------------
# block_bootstrap_sharpe
# ----------------------------------------------------------------------
def test_bootstrap_returns_required_keys():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.01, 0.05, 30)
    result = block_bootstrap_sharpe(returns, block_size=6, n_reps=100, seed=42)
    for key in ('sharpe_point', 'sharpe_lo95', 'sharpe_hi95', 'n_reps_valid'):
        assert key in result, f'missing key {key}'


def test_bootstrap_point_matches_analytical():
    # Point estimate is just (mean/std)*sqrt(12) — independent of bootstrap.
    # Uses ddof=1 (sample std) to match the thesis Sharpe convention via
    # pandas.Series.std() — see landscape_subregime_analysis.py:124-127.
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.005, 0.0, 0.015, -0.02])
    expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(12)
    result = block_bootstrap_sharpe(returns, block_size=2, n_reps=100, seed=42)
    assert result['sharpe_point'] == pytest.approx(expected)


def test_bootstrap_block_size_eq_n_gives_tight_ci():
    # When block_size=n, there's only one valid block start (0). Every
    # bootstrap rep is identical to the original — CI degenerates to the
    # point estimate.
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    result = block_bootstrap_sharpe(returns, block_size=6, n_reps=200, seed=42)
    point = (returns.mean() / returns.std(ddof=1)) * np.sqrt(12)
    assert result['sharpe_lo95'] == pytest.approx(point)
    assert result['sharpe_hi95'] == pytest.approx(point)


def test_bootstrap_reproducibility_same_seed():
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.005, 0.0, 0.015, -0.02])
    r1 = block_bootstrap_sharpe(returns, block_size=2, n_reps=300, seed=42)
    r2 = block_bootstrap_sharpe(returns, block_size=2, n_reps=300, seed=42)
    assert r1['sharpe_lo95'] == r2['sharpe_lo95']
    assert r1['sharpe_hi95'] == r2['sharpe_hi95']
    assert r1['n_reps_valid'] == r2['n_reps_valid']


def test_bootstrap_different_seeds_give_different_results():
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.005, 0.0, 0.015, -0.02])
    r1 = block_bootstrap_sharpe(returns, block_size=2, n_reps=300, seed=42)
    r2 = block_bootstrap_sharpe(returns, block_size=2, n_reps=300, seed=99)
    # Should not be exactly equal (vanishingly small probability if both
    # use different RNG streams).
    assert (r1['sharpe_lo95'] != r2['sharpe_lo95']
            or r1['sharpe_hi95'] != r2['sharpe_hi95'])


def test_bootstrap_ci_brackets_point_for_normal_data():
    # 95% CI on a moderately-sized normal sample should bracket the point
    # estimate the vast majority of the time. Use a fixed seed for
    # determinism.
    rng = np.random.default_rng(0)
    returns = rng.normal(0.01, 0.05, 60)
    result = block_bootstrap_sharpe(returns, block_size=6, n_reps=2000, seed=42)
    assert result['sharpe_lo95'] <= result['sharpe_point'] <= result['sharpe_hi95']


def test_bootstrap_lo_below_hi():
    # CI must be ordered.
    rng = np.random.default_rng(1)
    returns = rng.normal(0.005, 0.04, 40)
    result = block_bootstrap_sharpe(returns, block_size=4, n_reps=500, seed=42)
    assert result['sharpe_lo95'] < result['sharpe_hi95']


def test_bootstrap_handles_zero_variance_returns():
    # All-zero returns: Sharpe undefined. Function should return NaN
    # gracefully, not error.
    returns = np.zeros(30)
    result = block_bootstrap_sharpe(returns, block_size=6, n_reps=100, seed=42)
    assert np.isnan(result['sharpe_point'])
    assert np.isnan(result['sharpe_lo95'])
    assert np.isnan(result['sharpe_hi95'])


def test_bootstrap_block_size_clamped_to_n():
    # If block_size > n, function should still work (e.g. clamp to n).
    returns = np.array([0.01, 0.02, 0.03, 0.04])
    result = block_bootstrap_sharpe(returns, block_size=10, n_reps=50, seed=42)
    point = (returns.mean() / returns.std(ddof=1)) * np.sqrt(12)
    # With effective block_size = n, CI degenerates (as in the explicit
    # block_size=n test).
    assert result['sharpe_lo95'] == pytest.approx(point)
    assert result['sharpe_hi95'] == pytest.approx(point)


# ----------------------------------------------------------------------
# seed_stability
# ----------------------------------------------------------------------
def test_seed_stability_returns_required_keys():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    out = seed_stability(X, k=2, seeds=[42, 99])
    for key in ('sorted_sizes_per_seed', 'pairwise_ari', 'mean_ari'):
        assert key in out, f'missing key {key}'


def test_seed_stability_same_seed_twice_gives_ari_1():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    out = seed_stability(X, k=2, seeds=[42, 42])
    assert len(out['pairwise_ari']) == 1
    assert out['pairwise_ari'][0] == pytest.approx(1.0)
    assert out['mean_ari'] == pytest.approx(1.0)


def test_seed_stability_well_separated_blobs_high_ari():
    # Three well-separated 5-d Gaussian blobs. Any reasonable KMeans run
    # should recover the same partition regardless of seed → ARI ~= 1.0.
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.standard_normal((30, 5)) + 10.0 * np.array([1, 0, 0, 0, 0]),
        rng.standard_normal((30, 5)) + 10.0 * np.array([0, 1, 0, 0, 0]),
        rng.standard_normal((30, 5)) + 10.0 * np.array([0, 0, 1, 0, 0]),
    ])
    out = seed_stability(X, k=3, seeds=[1, 2, 3, 4, 5])
    assert out['mean_ari'] > 0.95


def test_seed_stability_sizes_match_k():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    out = seed_stability(X, k=2, seeds=[42, 99, 7])
    for sizes in out['sorted_sizes_per_seed']:
        assert len(sizes) == 2
        assert sum(sizes) == 30


def test_seed_stability_ari_in_unit_range():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))
    out = seed_stability(X, k=2, seeds=[1, 2, 3])
    for ari in out['pairwise_ari']:
        # ARI is bounded above by 1.0; can be slightly negative for
        # worse-than-random partitions but won't go below ~-0.5 in practice.
        assert -1.0 <= ari <= 1.0
