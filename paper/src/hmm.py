"""Canonical HMM machinery for the applied study.

Everything between the K=2 constant and the paper-wrapper marker is a
VERBATIM port of experiments/2026-07-09-prod-budget-dd-vol-reln.py
(lines 62-168, the bit-validated harness that mirrors scripts/hmm_model.py:
Gibbs with NIW/Dirichlet priors, posterior-MEAN mu/Sigma/P, causal forward
filter, crisis-sign panic labeling). Enforced by
paper/tests/test_hmm_gate.py: same inputs + same seed => bit-identical pi.
"""
import os

import numpy as np
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp

from config import (HMM_PRIOR_M0, HMM_PRIOR_KAPPA0, HMM_PRIOR_NU0_OFF,
                    HMM_PRIOR_DIRICHLET_ALPHA)

K = 2


# ── HMM core (functionally identical to scripts/hmm_model.py) ────────────────

def log_emission(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)])


def ffbs(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K)); la[0] = np.log(0.5) + le[0]
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t - 1] + np.log(P[:, k]))
    la -= logsumexp(la, axis=1, keepdims=True)
    a = np.exp(la); s = np.zeros(n, dtype=int)
    s[n - 1] = np.random.choice(K, p=a[n - 1])
    for t in range(n - 2, -1, -1):
        p = a[t] * P[:, s[t + 1]]; p /= p.sum()
        s[t] = np.random.choice(K, p=p)
    return s


def forward_filter(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K)); la[0] = np.log(0.5) + le[0]
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t - 1] + np.log(P[:, k]))
    la -= logsumexp(la, axis=1, keepdims=True)
    return np.exp(la)


# Worker globals (loaded once per process)
_Z_TRAIN = None
_Z_FULL = None
_SIGNS = None
_N_ITER = None
_N_BURNIN = None


def _init_worker(z_train, z_full, signs, n_iter, n_burnin):
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'
    global _Z_TRAIN, _Z_FULL, _SIGNS, _N_ITER, _N_BURNIN
    _Z_TRAIN, _Z_FULL, _SIGNS = z_train, z_full, signs
    _N_ITER, _N_BURNIN = n_iter, n_burnin


def fit_seed(seed):
    """One full Gibbs chain -> seed-specific causal pi_filter on the full sample.
    Posterior-MEAN mu/Sigma/P (hmm_model.py convention), crisis-sign panic label."""
    np.random.seed(seed)
    Z_train, Z_full = _Z_TRAIN, _Z_FULL
    T, D = Z_train.shape

    m_0 = np.full(D, HMM_PRIOR_M0)
    kappa_0 = HMM_PRIOR_KAPPA0
    nu_0 = D + HMM_PRIOR_NU0_OFF
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.asarray(HMM_PRIOR_DIRICHLET_ALPHA, dtype=float)

    # init: DD below median (deeper drawdown) -> state 1 (hmm_model.py convention)
    states = (Z_train[:, 0] < np.median(Z_train[:, 0])).astype(int)
    mu = np.zeros((K, D)); Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
    P = np.array([[0.90, 0.10], [0.10, 0.90]])

    n_keep = _N_ITER - _N_BURNIN
    mu_d = np.zeros((n_keep, K, D))
    Sig_d = np.zeros((n_keep, K, D, D))
    P_d = np.zeros((n_keep, K, K))

    for m in range(_N_ITER):
        states = ffbs(Z_train, mu, Sigma, P)
        for k in range(K):
            idx = states == k
            n_k = int(idx.sum())
            if n_k < D + 2:
                continue
            x_bar = Z_train[idx].mean(axis=0)
            S_k = (Z_train[idx] - x_bar).T @ (Z_train[idx] - x_bar)
            kappa_n = kappa_0 + n_k
            m_n = (kappa_0 * m_0 + n_k * x_bar) / kappa_n
            nu_n = nu_0 + n_k
            Psi_n = Psi_0 + S_k + (kappa_0 * n_k / kappa_n) * np.outer(x_bar - m_0, x_bar - m_0)
            Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
            mu[k] = np.random.multivariate_normal(m_n, Sigma[k] / kappa_n)
        for i in range(K):
            counts = np.array([np.sum((states[:-1] == i) & (states[1:] == j))
                               for j in range(K)], dtype=float)
            P[i] = np.random.dirichlet(alpha_dir[i] + counts)
        if m >= _N_BURNIN:
            j = m - _N_BURNIN
            mu_d[j], Sig_d[j], P_d[j] = mu, Sigma, P

    mu_post, Sig_post, P_post = mu_d.mean(0), Sig_d.mean(0), P_d.mean(0)
    score0 = np.sum(_SIGNS * mu_post[0]); score1 = np.sum(_SIGNS * mu_post[1])
    panic = 1 if score1 >= score0 else 0
    pi = forward_filter(Z_full, mu_post, Sig_post, P_post)[:, panic]
    return seed, pi, panic, (score0, score1)


# ── paper wrappers (new code) ────────────────────────────────────────────────

def crisis_signs(Z_train, train_dates, crisis_windows):
    cmask = np.zeros(len(Z_train), dtype=bool)
    for s, e in crisis_windows:
        cmask |= ((train_dates >= np.datetime64(s))
                  & (train_dates <= np.datetime64(e)))
    assert cmask.sum() >= 20, f'only {cmask.sum()} crisis months in train'
    return np.array([1.0 if np.percentile(Z_train[cmask, j], 95)
                     >= np.percentile(Z_train[~cmask, j], 95) else -1.0
                     for j in range(Z_train.shape[1])])


def fit_pi(Z_train, Z_full, signs, seeds, workers, n_iter=2000, n_burnin=500):
    from multiprocessing import get_context
    ctx = get_context('spawn')
    out = {}
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(Z_train, Z_full, signs, n_iter, n_burnin)) as pool:
        for seed, pi, panic, _ in pool.imap_unordered(fit_seed, seeds,
                                                      chunksize=1):
            out[seed] = pi
    return np.vstack([out[s] for s in sorted(out)]).mean(axis=0)
