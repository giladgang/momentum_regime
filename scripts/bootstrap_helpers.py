"""Statistical primitives for cluster robustness analysis.

Two functions:
  block_bootstrap_sharpe  — 95% CI on annualised Sharpe via moving-block
                            bootstrap. Block size approximates monthly
                            autocorrelation horizon (default 6).
  seed_stability          — run KMeans at multiple seeds, report sorted
                            cluster sizes per seed and pairwise ARI.

Tested in scripts/test_bootstrap_helpers.py.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def block_bootstrap_sharpe(returns, block_size=6, n_reps=5000, seed=42):
    """Block-bootstrap 95% CI for the annualised Sharpe ratio.

    Parameters
    ----------
    returns : array-like, shape (n,)
        Monthly returns. Assumes monthly frequency for sqrt(12) annualisation.
    block_size : int
        Length of each resampling block. Clamped to len(returns) if larger.
    n_reps : int
        Number of bootstrap repetitions.
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys:
        sharpe_point    point estimate of annualised Sharpe (NaN if undefined)
        sharpe_lo95     2.5th percentile of bootstrap Sharpe distribution
        sharpe_hi95     97.5th percentile
        n_reps_valid    number of reps that produced a finite Sharpe
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n == 0 or returns.std(ddof=1) == 0:
        return {
            'sharpe_point': float('nan'),
            'sharpe_lo95':  float('nan'),
            'sharpe_hi95':  float('nan'),
            'n_reps_valid': 0,
        }

    point = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(12))

    # Clamp block size and compute number of blocks needed.
    L = min(block_size, n)
    n_blocks = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)

    reps = np.empty(n_reps, dtype=float)
    # max(1, ...) guards against L == n where range collapses to [0, 1).
    upper = max(1, n - L + 1)
    for i in range(n_reps):
        starts = rng.integers(0, upper, size=n_blocks)
        rep = np.concatenate([returns[s:s + L] for s in starts])[:n]
        sd = rep.std(ddof=1)
        if sd == 0:
            reps[i] = np.nan
        else:
            reps[i] = (rep.mean() / sd) * np.sqrt(12)

    valid = reps[~np.isnan(reps)]
    if len(valid) == 0:
        return {
            'sharpe_point': point,
            'sharpe_lo95':  float('nan'),
            'sharpe_hi95':  float('nan'),
            'n_reps_valid': 0,
        }

    return {
        'sharpe_point': point,
        'sharpe_lo95':  float(np.percentile(valid, 2.5)),
        'sharpe_hi95':  float(np.percentile(valid, 97.5)),
        'n_reps_valid': int(len(valid)),
    }


def seed_stability(X, k, seeds, n_init=20):
    """KMeans seed-stability check.

    Refits KMeans at each seed; reports sorted cluster sizes per seed and
    pairwise Adjusted Rand Index between every pair of seed-label vectors.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    k : int
        Number of clusters.
    seeds : iterable of int
        Seeds to refit at.
    n_init : int
        KMeans n_init parameter (number of random initialisations per seed).

    Returns
    -------
    dict with keys:
        sorted_sizes_per_seed   list of tuples, one per seed (sorted ascending)
        pairwise_ari            list of floats, all C(len(seeds), 2) pairs
        mean_ari                mean of pairwise_ari (NaN if < 2 seeds)
    """
    seeds = list(seeds)
    label_sets = []
    sizes_per_seed = []
    for s in seeds:
        km = KMeans(n_clusters=k, random_state=s, n_init=n_init).fit(X)
        label_sets.append(km.labels_)
        sizes_per_seed.append(
            tuple(sorted(np.bincount(km.labels_, minlength=k).tolist()))
        )

    pairwise_ari = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            pairwise_ari.append(
                float(adjusted_rand_score(label_sets[i], label_sets[j]))
            )

    return {
        'sorted_sizes_per_seed': sizes_per_seed,
        'pairwise_ari': pairwise_ari,
        'mean_ari': float(np.mean(pairwise_ari)) if pairwise_ari else float('nan'),
    }
