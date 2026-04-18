"""
hmm_model.py
============
2-state Bayesian Hidden Markov Model estimated via Gibbs sampling (MCMC).

Model
-----
Observation equation:
    z_t | s_t = k  ~  N(mu_k, Sigma_k)

Transition equation:
    P(s_t = j | s_{t-1} = i) = P[i, j]

where z_t = [DD_z, CS_z, DISP_z, REL_N_z] (standardized, pre-computed
in import_wrds.py). Sample starts 1990 (when stock-level data begins).

The regime s_t in {0, 1} is latent.

Train/test split
----------------
Gibbs sampler is fit on TRAIN only (1990-2010). Posterior mean parameters
(mu, Sigma, P) are then fixed and applied forward to compute pi_filter on the
full sample -- no look-ahead leakage.

Out-of-sample evaluation uses TEST months (2011-2025) only.

Gibbs sampler blocks (per iteration)
--------------------------------------
1. FFBS  : sample full hidden state path s_{1:T} via Forward-Filtering
           Backward-Sampling, given current (mu, Sigma, P).
2. NIW   : sample (mu_k, Sigma_k) per regime from Normal-Inverse-Wishart
           conjugate posterior, given current state assignment.
3. Dir   : sample each row of transition matrix P from Dirichlet
           conjugate posterior, given current state path.

Panic identification
--------------------
Crisis-calibrated sign correction: for each feature, compare 95th percentile
during known crises (dot-com, GFC) vs non-crisis periods. Features that rise
in crisis get sign +1, features that fall get sign -1. The state with the
higher sum of sign-corrected posterior means is identified as the panic state.

Outputs
-------
pi_smooth_train : P(s_t = panic | z_{1:T_train}) -- posterior average over MCMC draws
pi_smooth_test  : P(s_t = panic | z_{1:T_test})  -- average of FFBS draws with fixed params
pi_filter       : P(s_t = panic | z_{1:t})        -- causal signal for trading
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
import matplotlib.pyplot as plt

# ── Section 1: Load data and split ───────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HMM_FEATURES, HMM_SEEDS, HMM_ITERATIONS, HMM_BURNIN, K_STATES, HMM_START, TRAIN_END, PANEL_PATH

panel = pd.read_parquet(PANEL_PATH)

# Use features from config.py (selected via exhaustive search + validation)
features_z = HMM_FEATURES
panel = panel.dropna(subset=features_z).reset_index(drop=True)

# Train/test split from config
train_panel = panel[panel['date'] < TRAIN_END].reset_index(drop=True)
test_panel  = panel[panel['date'] >= TRAIN_END].reset_index(drop=True)

Z_train = train_panel[features_z].values.astype(float)
Z_test  = test_panel[features_z].values.astype(float)
Z_full  = panel[features_z].values.astype(float)

T_train, D = Z_train.shape
T_test      = len(Z_test)
T_full      = len(Z_full)

print(f"Train: {train_panel['date'].min().date()} → {train_panel['date'].max().date()}  ({T_train} months)")
print(f"Test:  {test_panel['date'].min().date()}  → {test_panel['date'].max().date()}  ({T_test} months)")

# ── Section 2: Priors ─────────────────────────────────────────────────────────

K = 2  # two hidden regimes: calm and panic

# Normal-Inverse-Wishart (NIW) prior for (μ_k, Σ_k)
m_0  = np.zeros(D)          # prior mean: zero, since features are z-scored
κ_0  = 0.01                 # prior strength on mean: nearly flat (0.01 pseudo-observations)
ν_0  = D + 2                # prior degrees of freedom: minimum valid value for a proper IW
Ψ_0  = np.eye(D) * (ν_0 - D - 1)  # prior scale matrix: chosen so E[Σ_k] = I (unit covariance)

# Dirichlet prior for each row of transition matrix P
# Row i gets α_dir[i]: diagonal element is 9 (persist), off-diagonal is 1.
α_dir = np.array([[9.0, 1.0],    # calm→calm 90%, calm→panic 10%
                   [1.0, 9.0]])   # panic→calm 10%, panic→panic 90%

# ── Section 3: Initialization helper ─────────────────────────────────────────

# Use first feature (DD) for initial state assignment -- below-median DD = panic
vol_idx = 0  # DD_z is always first; below median = more stress

def init_sampler(seed):
    """Initialize Gibbs sampler state for a given random seed."""
    np.random.seed(seed)
    # crude initial state assignment: above-median first feature -> state 1, below -> state 0
    states = (Z_train[:, vol_idx] > np.median(Z_train[:, vol_idx])).astype(int)
    mu_init    = np.zeros((K, D))
    Sigma_init = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu_init[k]    = Z_train[idx].mean(axis=0)
            Sigma_init[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
    P_init = np.array([[0.95, 0.05],
                        [0.10, 0.90]])
    return states, mu_init, Sigma_init, P_init

# ── Section 4: Gibbs sampler helper functions ─────────────────────────────────

def log_emission(Z, mu, Sigma):
    """Compute log N(z_t; μ_k, Σ_k) for all t and k. Returns (len(Z), K)."""
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)  # log P(z_t | s_t=k) for all t; allow_singular handles near-degenerate Σ in early iterations
        for k in range(K)
    ])


def ffbs(Z, mu, Sigma, P, init_prior=None):
    """
    Forward-Filtering Backward-Sampling.
    Returns a sampled state path of shape (len(Z),).

    init_prior : array of shape (K,) with prior P(s_0=k) before seeing data.
                 Defaults to uniform (0.5, 0.5). Pass the last train filter
                 state when running on test data to avoid cold-start bias.
    """
    n = len(Z)
    log_emit  = log_emission(Z, mu, Sigma)   # (T, K): log likelihood of each obs under each regime
    log_alpha = np.zeros((n, K))             # (T, K): will hold log filtered beliefs
    if init_prior is None:
        init_prior = np.full(K, 1.0 / K)    # uniform prior
    log_alpha[0] = np.log(init_prior) + log_emit[0]  # prior updated by first obs

    # Forward pass: sweep left to right, computing log P(s_t=k | z_{1:t}) at each step
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(  # Bayes update: likelihood × predicted prior
                log_alpha[t-1] + np.log(P[:, k])           # predicted prior: yesterday's belief × transition probs into k, summed over all previous states
            )

    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)  # normalize each row to sum to 1 in prob space
    alpha = np.exp(log_alpha)                                  # convert to probabilities for sampling

    # Backward pass: sweep right to left, sampling one state per step conditioned on the future
    s = np.zeros(n, dtype=int)
    s[n-1] = np.random.choice(K, p=alpha[n-1])    # anchor: draw last state from final filtered belief
    for t in range(n - 2, -1, -1):
        probs = alpha[t] * P[:, s[t+1]]  # weight filtered belief by transition prob into already-sampled next state
        probs /= probs.sum()             # normalize to valid probability vector
        s[t] = np.random.choice(K, p=probs)  # draw state at t conditioned on both past data and future state

    return s  # (T,) array of 0/1 regime labels — one complete sampled path


def sample_niw(Z, states, k):
    """
    Sample (μ_k, Σ_k) from Normal-Inverse-Wishart posterior.

    Given observations Z_k = {z_t : s_t = k}:
        κ_n = κ_0 + n_k
        m_n = (κ_0 * m_0 + n_k * x̄) / κ_n
        ν_n = ν_0 + n_k
        Ψ_n = Ψ_0 + S_k + (κ_0 n_k / κ_n) * (x̄ - m_0)(x̄ - m_0)^T
        Σ_k ~ IW(ν_n, Ψ_n)
        μ_k | Σ_k ~ N(m_n, Σ_k / κ_n)
    """
    idx = states == k   # boolean mask: which months belong to regime k this iteration
    Z_k = Z[idx]        # observations assigned to regime k
    n_k = len(Z_k)      # number of observations in regime k

    if n_k < D + 2:           # too few observations to form a valid scatter matrix
        return mu[k], Sigma[k]  # keep previous draw rather than crash

    x_bar = Z_k.mean(axis=0)               # sample mean of regime k: (D,)
    S_k   = (Z_k - x_bar).T @ (Z_k - x_bar)  # scatter matrix: (D,D) — sum of outer products

    κ_n = κ_0 + n_k                                        # posterior pseudo-count: prior + data
    m_n = (κ_0 * m_0 + n_k * x_bar) / κ_n                 # posterior mean: weighted average of prior and sample mean
    ν_n = ν_0 + n_k                                        # posterior degrees of freedom: prior + data count
    Ψ_n = (Ψ_0 + S_k +
           (κ_0 * n_k / κ_n) * np.outer(x_bar - m_0, x_bar - m_0))  # posterior scale: prior + scatter + penalty for mean shift

    Sigma_k = invwishart.rvs(df=ν_n, scale=Ψ_n)                   # draw covariance from IW posterior
    mu_k    = np.random.multivariate_normal(m_n, Sigma_k / κ_n)    # draw mean conditional on drawn covariance

    return mu_k, Sigma_k


def sample_P(states):
    """
    Sample transition matrix from Dirichlet posterior.
    Count n_ij = # transitions i→j, then row i ~ Dirichlet(α_dir + n_i).
    """
    P_new = np.zeros((K, K))   # will hold the new transition matrix
    for i in range(K):
        counts = np.array([
            np.sum((states[:-1] == i) & (states[1:] == j))  # count transitions i→j in sampled path
            for j in range(K)
        ], dtype=float)
        P_new[i] = np.random.dirichlet(α_dir[i] + counts)  # draw row i from Dirichlet posterior: prior[i] + transition counts
    return P_new


def forward_filter(Z, mu, Sigma, P):
    """
    Compute filtered probabilities P(s_t = k | z_{1:t}) with fixed parameters.
    Returns array of shape (len(Z), K).
    Causal — uses only data up to t at each step.
    """
    n = len(Z)
    log_emit  = log_emission(Z, mu, Sigma)   # (T, K): log likelihoods under each regime
    log_alpha = np.zeros((n, K))             # (T, K): log filtered beliefs
    log_alpha[0] = np.log(0.5) + log_emit[0]  # t=0: equal prior updated by first observation

    # same forward recursion as ffbs but no backward sampling — strictly causal
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k])  # predicted prior × likelihood
            )

    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)  # normalize rows
    return np.exp(log_alpha)  # return probabilities, not log-probs

# ── Section 5: Multi-seed Gibbs sampler with Bayesian model averaging ─────────
# Run the full Gibbs sampler with multiple seeds and average the resulting
# pi_filter across seeds. This reduces sensitivity to MCMC initialization
# and produces a more stable trading signal.
#
# Set RERUN_MCMC = False to skip the sampler and load saved draws from disk.
# Set RERUN_MCMC = True  to re-run the full sampler and overwrite saved draws.

RERUN_MCMC  = True
MCMC_CACHE  = 'data/mcmc_draws.npz'

SEEDS = HMM_SEEDS  # from config.py
N_SEEDS = len(SEEDS)

n_iter   = HMM_ITERATIONS   # total Gibbs iterations per seed (from config.py)
n_burnin = HMM_BURNIN       # iterations discarded before collecting draws (from config.py)
n_keep   = n_iter - n_burnin  # number of posterior draws saved per seed

import os

# Build crisis mask once (used for panic identification in each seed)
crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
train_dates = train_panel['date'].values
crisis_mask = np.zeros(T_train, dtype=bool)
for s, e in crisis_windows:
    crisis_mask |= ((train_dates >= np.datetime64(s)) & (train_dates <= np.datetime64(e)))

# Determine sign direction for each feature (data-driven, same for all seeds)
signs = np.zeros(D)
for j in range(D):
    p95_crisis = np.percentile(Z_train[crisis_mask, j], 95)
    p95_normal = np.percentile(Z_train[~crisis_mask, j], 95)
    signs[j] = 1.0 if p95_crisis >= p95_normal else -1.0

if RERUN_MCMC or not os.path.exists(MCMC_CACHE):
    print(f"\nRunning multi-seed Gibbs sampler ({N_SEEDS} seeds x {n_iter} iterations) ...")

    # Accumulators for averaging across seeds
    pi_filter_accum = np.zeros(T_full)
    pi_smooth_train_accum = np.zeros(T_train)
    pi_smooth_test_accum  = np.zeros(T_test)

    # Keep draws from first seed for diagnostics (trace plots, ESS, posterior summary)
    state_draws = None
    mu_draws    = None
    Sigma_draws = None
    P_draws     = None
    mu_trace    = None
    P_trace     = None

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed_idx+1}/{N_SEEDS} (seed={seed}) ...")
        states, mu, Sigma, P = init_sampler(seed)

        # Pre-allocate for this seed
        sd = np.zeros((n_keep, T_train), dtype=int)
        md = np.zeros((n_keep, K, D))
        Sd = np.zeros((n_keep, K, D, D))
        Pd = np.zeros((n_keep, K, K))
        mt = np.zeros((n_iter, K, D))
        pt = np.zeros((n_iter, K, K))

        for m in range(n_iter):
            if m % 500 == 0:
                print(f"    Iteration {m}/{n_iter}")
            states = ffbs(Z_train, mu, Sigma, P)
            for k in range(K):
                mu[k], Sigma[k] = sample_niw(Z_train, states, k)
            P = sample_P(states)
            mt[m] = mu
            pt[m] = P
            if m >= n_burnin:
                idx = m - n_burnin
                sd[idx] = states
                md[idx] = mu
                Sd[idx] = Sigma
                Pd[idx] = P

        # Identify panic state for this seed using crisis-calibrated sign correction
        mu_post_seed    = md.mean(axis=0)
        Sigma_post_seed = Sd.mean(axis=0)
        P_post_seed     = Pd.mean(axis=0)

        score0 = np.sum(signs * mu_post_seed[0])
        score1 = np.sum(signs * mu_post_seed[1])
        panic_seed = 1 if score1 >= score0 else 0
        calm_seed  = 1 - panic_seed

        # Filtered probability (full sample, causal)
        filtered_seed = forward_filter(Z_full, mu_post_seed, Sigma_post_seed, P_post_seed)
        pi_filter_seed = filtered_seed[:, panic_seed]
        pi_filter_accum += pi_filter_seed

        # Smoothed probability (train: from MCMC draws)
        pi_smooth_train_seed = (sd == panic_seed).mean(axis=0)
        pi_smooth_train_accum += pi_smooth_train_seed

        # Smoothed probability (test: FFBS with fixed params)
        last_train = pi_filter_seed[T_train - 1]
        init_prior = np.zeros(K)
        init_prior[panic_seed] = last_train
        init_prior[calm_seed]  = 1.0 - last_train
        smooth_test_seed = np.zeros((200, T_test), dtype=int)
        for i in range(200):
            smooth_test_seed[i] = ffbs(Z_test, mu_post_seed, Sigma_post_seed, P_post_seed,
                                       init_prior=init_prior)
        pi_smooth_test_accum += (smooth_test_seed == panic_seed).mean(axis=0)

        print(f"    Panic=state {panic_seed}, score0={score0:.3f}, score1={score1:.3f}")

        # Save first seed's draws for diagnostics
        if seed_idx == 0:
            state_draws = sd
            mu_draws    = md
            Sigma_draws = Sd
            P_draws     = Pd
            mu_trace    = mt
            P_trace     = pt
            panic_state = panic_seed
            calm_state  = calm_seed

    # Average across seeds
    pi_filter_full  = pi_filter_accum / N_SEEDS
    pi_smooth_train = pi_smooth_train_accum / N_SEEDS
    pi_smooth_test  = pi_smooth_test_accum / N_SEEDS

    np.savez(MCMC_CACHE,
             state_draws=state_draws, mu_draws=mu_draws,
             Sigma_draws=Sigma_draws, P_draws=P_draws,
             mu_trace=mu_trace, P_trace=P_trace,
             pi_filter_full=pi_filter_full,
             pi_smooth_train=pi_smooth_train,
             pi_smooth_test=pi_smooth_test,
             panic_state=np.array(panic_state),
             calm_state=np.array(calm_state))
    print(f"\nMulti-seed sampler complete. Averaged pi_filter saved to {MCMC_CACHE}")

else:
    cache = np.load(MCMC_CACHE)
    state_draws     = cache['state_draws']
    mu_draws        = cache['mu_draws']
    Sigma_draws     = cache['Sigma_draws']
    P_draws         = cache['P_draws']
    mu_trace        = cache['mu_trace']
    P_trace         = cache['P_trace']
    pi_filter_full  = cache['pi_filter_full']
    pi_smooth_train = cache['pi_smooth_train']
    pi_smooth_test  = cache['pi_smooth_test']
    panic_state     = int(cache['panic_state'])
    calm_state      = int(cache['calm_state'])
    print(f"Loaded cached MCMC draws from {MCMC_CACHE} (set RERUN_MCMC=True to re-run)")

# ── Section 6: Panic identification summary ──────────────────────────────────

print(f"\nPanic identification (crisis-calibrated sign correction):")
print(f"  Feature signs: {dict(zip(features_z, signs))}")
print(f"  Panic regime = state {panic_state} (from first seed, consistent across seeds)")

# ── Section 7: Posterior mean parameters (from first seed, for diagnostics) ───

mu_post    = mu_draws.mean(axis=0)
Sigma_post = Sigma_draws.mean(axis=0)
P_post     = P_draws.mean(axis=0)

# ── Section 8: Validate alignment ─────────────────────────────────────────────

assert len(pi_filter_full) == len(panel), \
    f"Length mismatch: pi_filter_full ({len(pi_filter_full)}) != panel ({len(panel)})"

pi_filter_train = pi_filter_full[:T_train]
pi_filter_test  = pi_filter_full[T_train:]

# ── Section 11: Attach to panel and save ──────────────────────────────────────

panel = panel.copy()
panel['pi_filter'] = pi_filter_full

# pi_smooth: MCMC-based for train, fixed-param FFBS for test
pi_smooth_full = np.concatenate([pi_smooth_train, pi_smooth_test])
panel['pi_smooth'] = pi_smooth_full

# pi_next: one-step-ahead predicted panic probability — use to adjust positions ahead of time
panel['pi_next'] = (P_post[calm_state,  panic_state] * (1 - pi_filter_full) +
                    P_post[panic_state, panic_state] *      pi_filter_full)

panel.to_parquet('data/panel_with_regimes.parquet', index=False)
print("Saved: data/panel_with_regimes.parquet")

# ── Section 11b: Convergence diagnostics ──────────────────────────────────────
# Two checks:
#   Trace plots  — visual check that parameters mix freely and don't get stuck
#   ESS          — effective sample size: how many independent draws the chain is worth
#                  after accounting for autocorrelation between consecutive draws

def compute_ess(chain):
    """
    Compute Effective Sample Size for a 1D chain of post-burnin draws.
    ESS = n / (1 + 2 * sum of autocorrelations until they become negligible).
    Low ESS means consecutive draws are highly correlated — sampler is moving slowly.
    """
    n = len(chain)
    chain = chain - chain.mean()
    # autocorrelation at each lag via FFT
    acf_full = np.fft.irfft(np.abs(np.fft.rfft(chain, n=2*n))**2)[:n]
    acf_full /= acf_full[0]
    # sum positive autocorrelations until first negative lag (Geyer's rule)
    ess_sum = 0.0
    for lag in range(1, n):
        if acf_full[lag] < 0:
            break
        ess_sum += acf_full[lag]
    return n / (1 + 2 * ess_sum)

# ── Trace plots ────────────────────────────────────────────────────────────────
# One row per feature showing mu_panic and mu_calm across all 2000 iterations.
# A healthy chain looks like white noise around a stable mean.
# A stuck chain flatlines or drifts — means more iterations or tuning needed.

regime_labels = ['Calm', 'Panic']
feature_display_names = {
    'DD_z': 'DD (Drawdown)',
    'CS_z': 'CS (Credit Spread)',
    'DISP_z': 'DISP (Return Dispersion)',
    'REL_N_z': 'REL_N (Market Participation)',
}
fig, axes = plt.subplots(D + 1, 1, figsize=(14, 3 * (D + 1)), sharex=True)

for j, feat in enumerate(features_z):
    axes[j].plot(mu_trace[:, 0, j], color='steelblue', linewidth=0.6, alpha=0.8, label='Calm')
    axes[j].plot(mu_trace[:, 1, j], color='crimson',   linewidth=0.6, alpha=0.8, label='Panic')
    axes[j].axvline(n_burnin, color='black', linewidth=1, linestyle='--', label='Burn-in end' if j == 0 else None)
    axes[j].set_ylabel(feature_display_names.get(feat, feat), fontsize=9)
    if j == 0:
        axes[j].legend(fontsize=8, loc='upper right')

# bottom panel: transition matrix diagonal (persistence of each regime)
axes[D].plot(P_trace[:, 0, 0], color='steelblue', linewidth=0.6, alpha=0.8, label=r'$P_{00}$ (calm persistence)')
axes[D].plot(P_trace[:, 1, 1], color='crimson',   linewidth=0.6, alpha=0.8, label=r'$P_{11}$ (panic persistence)')
axes[D].axvline(n_burnin, color='black', linewidth=1, linestyle='--')
axes[D].set_ylabel('Persistence probability', fontsize=9)
axes[D].set_ylim(0.5, 1.0)
axes[D].set_xlabel('Iteration', fontsize=9)
axes[D].legend(fontsize=8, loc='upper right')

fig.suptitle('Trace plots - dashed line = end of burn-in', fontsize=11)
plt.tight_layout()
fig.savefig('convergence_trace.png', dpi=150)
plt.close(fig)
print("Saved: convergence_trace.png")

# ── ESS ───────────────────────────────────────────────────────────────────────
# Computed on post-burnin draws only.
# Rule of thumb: ESS > 100 is acceptable, > 400 is good.

ess_records = []
for j, feat in enumerate(features_z):
    for k, label in enumerate(regime_labels):
        ess = compute_ess(mu_draws[:, k, j])
        ess_records.append((f"mu_{label}[{feat}]", ess))

for k, label in enumerate(regime_labels):
    ess = compute_ess(P_draws[:, k, k])
    ess_records.append((f"P({label}→{label})", ess))

ess_results = [e for _, e in ess_records]
min_ess     = min(ess_results)

# ── Section 12: Posterior summary ─────────────────────────────────────────────
# Compute all results first, then print one clean summary at the end.

# ── μ: compute separation and significance ────────────────────────────────────
mu_results = {}
for j, feat in enumerate(features_z):
    calm_mean  = mu_draws[:, calm_state,  j].mean()
    panic_mean = mu_draws[:, panic_state, j].mean()
    sep_draws  = mu_draws[:, panic_state, j] - mu_draws[:, calm_state, j]
    lo, hi     = np.percentile(sep_draws, [2.5, 97.5])
    sig        = (lo > 0 or hi < 0)
    mu_results[feat] = dict(calm=calm_mean, panic=panic_mean,
                            sep=sep_draws.mean(), lo=lo, hi=hi, sig=sig)
check1_passed = all(abs(v['sep']) > 0.3 for v in mu_results.values())

# ── P: compute persistence and significance ───────────────────────────────────
p_results = {}
for k, label in enumerate(['calm', 'panic']):
    stay_draws = P_draws[:, k, k]
    lo, hi     = np.percentile(stay_draws, [2.5, 97.5])
    p_results[label] = dict(mean=stay_draws.mean(), lo=lo, hi=hi, sig=(lo > 0.5))

# ── Σ: compute significant panic correlations ─────────────────────────────────
corr_results = []
for j1 in range(D):
    for j2 in range(j1+1, D):
        cov_draws  = Sigma_draws[:, panic_state, j1, j2]
        std1_draws = np.sqrt(Sigma_draws[:, panic_state, j1, j1])
        std2_draws = np.sqrt(Sigma_draws[:, panic_state, j2, j2])
        corr_draws = cov_draws / (std1_draws * std2_draws)
        mean_corr  = corr_draws.mean()
        if abs(mean_corr) < 0.15:
            continue
        lo, hi = np.percentile(corr_draws, [2.5, 97.5])
        sig    = (lo > 0 or hi < 0)
        corr_results.append(dict(f1=features_z[j1], f2=features_z[j2],
                                 mean=mean_corr, lo=lo, hi=hi, sig=sig))

# ── Check 2: crisis alignment ─────────────────────────────────────────────────
crisis_check = [
    ('Dot-com (train)',    '2000-03-01', '2002-10-01', train_panel, pi_smooth_train),
    ('GFC (train)',        '2007-10-01', '2009-06-01', train_panel, pi_smooth_train),
    ('COVID (test)',       '2020-02-01', '2020-05-01', test_panel,  pi_smooth_test),
    ('Non-crisis (train)', '2003-01-01', '2006-12-01', train_panel, pi_smooth_train),
]
check2_results = {}
for label, start, end, p, pi in crisis_check:
    mask = (p['date'] >= start) & (p['date'] <= end)
    if mask.any():
        avg    = pi[mask.values].mean()
        passed = (label != 'Non-crisis (train)' and avg > 0.5) or \
                 (label == 'Non-crisis (train)' and avg < 0.3)
        check2_results[label] = (avg, passed)
check2_passed = all(v[1] for v in check2_results.values())

# ── Compute realized regimes and pi_next (for plots) ─────────────────────────
realized_test = (pi_smooth_test > 0.5).astype(int)
p_calm_to_panic  = P_post[calm_state,  panic_state]
p_panic_to_panic = P_post[panic_state, panic_state]
pi_next_test = (p_calm_to_panic  * (1 - pi_filter_test) +
                p_panic_to_panic * pi_filter_test)
pred   = pi_next_test[:-1]
actual = realized_test[1:]

# ── Print consolidated output ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  POSTERIOR PARAMETERS  (posterior mean  [95% CI]  sig=CI excludes null)")
print("=" * 70)

print(f"\n  μ — regime means  (significant = panic mean ≠ calm mean)")
print(f"  {'Feature':<12} {'Calm':>8} {'Panic':>8} {'Sep':>8}  {'95% CI':>20}  Sig")
print("  " + "-" * 64)
for feat, r in mu_results.items():
    sig_str = 'YES' if r['sig'] else 'NO'
    print(f"  {feat:<12} {r['calm']:>8.3f} {r['panic']:>8.3f} {r['sep']:>+8.3f}"
          f"  [{r['lo']:>+7.3f}, {r['hi']:>+7.3f}]  {sig_str}")

print(f"\n  P — regime persistence  (significant = P(stay) > 0.5)")
print(f"  {'Regime':<12} {'P(stay)':>8} {'Dur (mo)':>10}  {'95% CI':>20}  Sig")
print("  " + "-" * 58)
for label, r in p_results.items():
    dur     = 1 / (1 - r['mean'])
    sig_str = 'YES' if r['sig'] else 'NO'
    print(f"  {label:<12} {r['mean']:>8.3f} {dur:>10.1f}  [{r['lo']:>7.3f}, {r['hi']:>7.3f}]  {sig_str}")

if corr_results:
    print(f"\n  Σ_panic — significant feature correlations  (|mean corr| > 0.15)")
    print(f"  {'Feature pair':<22} {'Corr':>8}  {'95% CI':>20}  Sig")
    print("  " + "-" * 58)
    for r in corr_results:
        sig_str = 'YES' if r['sig'] else 'NO'
        print(f"  {r['f1']+' / '+r['f2']:<22} {r['mean']:>+8.3f}  [{r['lo']:>+7.3f}, {r['hi']:>+7.3f}]  {sig_str}")

print(f"\n  Crisis alignment  (panic > 0.5 in crisis, < 0.3 in calm)")
print(f"  {'Period':<24} {'Avg π':>8}  Pass")
print("  " + "-" * 40)
for label, (avg, passed) in check2_results.items():
    print(f"  {label:<24} {avg:>8.3f}  {'YES' if passed else 'NO'}")

print(f"\n  MCMC convergence  (ESS > 100 = acceptable, post-burnin draws = {n_keep})")
print(f"  {'Parameter':<22} {'ESS':>8}  Status")
print("  " + "-" * 38)
for name, ess in ess_records:
    print(f"  {name:<22} {ess:>8.0f}  {'ok' if ess >= 100 else 'LOW — check trace'}")
print(f"  Minimum ESS: {min_ess:.0f}  ({'ok' if min_ess >= 100 else 'WARNING: low — increase n_iter'})")

print("\n" + "=" * 70)
print("  VALIDATION SUMMARY")
print("=" * 70)
c1 = "PASS" if check1_passed  else "FAIL"
c2 = "PASS" if check2_passed  else "FAIL"
c3 = "PASS" if min_ess >= 100 else "FAIL"
print(f"  {'1. Regime separation':<40} {c1}  Panic/calm means are distinct")
print(f"  {'2. Crisis alignment':<40} {c2}  Flags known crises correctly")
print(f"  {'3. MCMC convergence (min ESS>=100)':<40} {c3}  Sampler mixed well")
n_passed = sum([check1_passed, check2_passed, min_ess >= 100])
print("=" * 70)
print(f"  {n_passed}/3 passed." + ("" if n_passed < 3 else "  Model looks good."))

# ── Section 13: Diagnostic plots ─────────────────────────────────────────────

crises = [
    ('1973-10-01', '1974-12-01', 'Oil shock'),
    ('1987-10-01', '1987-12-01', 'Black Monday'),
    ('2000-03-01', '2002-10-01', 'Dot-com'),
    ('2007-10-01', '2009-06-01', 'GFC'),
    ('2020-02-01', '2020-05-01', 'COVID'),
]

def shade_crises(ax, label=False):
    for start, end, lbl in crises:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color='grey')
        if label:
            ax.text(pd.Timestamp(start), 0.97, lbl, fontsize=7, color='grey',
                    va='top', transform=ax.get_xaxis_transform())

dates = panel['date']

# Plot A: filtered panic probability (single panel)
fig, ax = plt.subplots(figsize=(14, 4))

ax.fill_between(dates, pi_filter_full, alpha=0.5, color='crimson')
ax.axvline(pd.Timestamp('2011-01-01'), color='black', linewidth=1,
           linestyle='--', label='Train/test split')
ax.set_ylabel('$\\pi_t^{\\mathrm{filter}}$', fontsize=11)
ax.set_ylim(0, 1)
ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':')
ax.legend(fontsize=8, loc='upper left')
shade_crises(ax, label=True)
ax.set_xlabel('Date')
ax.set_xlim(dates.min(), dates.max())

plt.tight_layout()
fig.savefig('regime_probabilities.png', dpi=150)
plt.close(fig)

# Plot B: raw features coloured by regime — split into two charts
regime_label = (pi_smooth_full > 0.5).astype(int)
colors = np.where(regime_label == 1, 'crimson', 'steelblue')

# Chart B1: HMM features (DD, CS, DISP, REL_N)
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for ax, feat_z, feat_raw in zip(axes,
    ['DD_z', 'CS_z', 'DISP_z', 'REL_N_z'],
    ['DD', 'CS', 'DISP', 'REL_N']):
    ax.scatter(dates, panel[feat_raw], c=colors, s=4, alpha=0.7)
    ax.set_ylabel(feat_z, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
    ax.axvline(pd.Timestamp('2011-01-01'), color='black', linewidth=1, linestyle='--')
    shade_crises(ax)
axes[-1].set_xlabel('Date')
fig.suptitle('HMM Features (DD, CS, DISP, REL_N) coloured by regime (red=panic, blue=calm)', fontsize=11)
plt.tight_layout()
fig.savefig('regime_features_1.png', dpi=150)
plt.close(fig)

print("\nPlots saved: regime_probabilities.png, regime_features_1.png")

# Plot C: predicted vs realized panic probability (test period only)
# pi_next[t]      = one-step-ahead predicted P(panic at t+1) using pi_filter[t]
# pi_smooth_test  = realized regime (hindsight smoother on test data)
# Gap between lines shows where the filter leads, lags, or misses transitions.

test_dates = test_panel['date'].values

fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(test_dates[1:], pi_next_test[:-1],  color='steelblue', linewidth=1.2,
        label='Predicted P(panic next month) — pi_filter based')
ax.plot(test_dates,     pi_smooth_test,      color='crimson',   linewidth=1.2,
        alpha=0.8, label='Realized regime — pi_smooth_test')

ax.fill_between(test_dates[1:],
                pi_next_test[:-1], pi_smooth_test[1:],
                alpha=0.15, color='grey', label='Prediction error')

ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':')
ax.set_ylim(0, 1)
ax.set_ylabel('P(panic)', fontsize=9)
ax.set_xlabel('Date')
ax.legend(fontsize=8, loc='upper left')

# shade test-period crises only
for start, end, lbl in crises:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e >= test_panel['date'].min():
        ax.axvspan(s, e, alpha=0.12, color='grey')
        ax.text(s, 0.97, lbl, fontsize=7, color='grey',
                va='top', transform=ax.get_xaxis_transform())

ax.set_title('Test period: predicted vs realized panic probability\n'
             'Blue leads red = filter anticipates transitions. Blue lags = filter is slow.',
             fontsize=10)
plt.tight_layout()
fig.savefig('transition_accuracy.png', dpi=150)
plt.close(fig)
print("Saved: transition_accuracy.png")

# Plot D: test-period regime signal with missed predictions flagged
# ----------------------------------------------------------------
# pi_filter_test  = current belief (what you act on)
# realized_test   = actual regime next month (smoother > 0.5)
# Missed          = pi_filter said calm (<0.5) but next month was panic
# False alarm     = pi_filter said panic (>0.5) but next month was calm

test_events = [
    ('2011-08-01', '2011-10-01', 'EU debt crisis'),
    ('2015-08-01', '2015-09-01', 'China crash'),
    ('2018-10-01', '2018-12-01', 'Rate fear'),
    ('2020-02-01', '2020-05-01', 'COVID'),
    ('2022-01-01', '2022-10-01', 'Rate hikes'),
]

fig, ax = plt.subplots(figsize=(16, 5))

# filled area: current regime belief
ax.fill_between(test_dates, pi_filter_test, alpha=0.4, color='steelblue',
                label='pi_filter (current belief)')
ax.plot(test_dates, pi_filter_test, color='steelblue', linewidth=0.8)

# realized regime line
ax.plot(test_dates, pi_smooth_test, color='crimson', linewidth=1.2,
        alpha=0.7, label='Realized regime (smoother)')

# 0.5 threshold
ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':')

# flag missed panics: filter calm but realized was panic next month
missed = (pi_filter_test[:-1] < 0.5) & (actual == 1)
ax.scatter(test_dates[:-1][missed], pi_filter_test[:-1][missed],
           marker='v', color='red', s=60, zorder=5,
           label='Missed panic (filter calm, realized panic)')

# flag false alarms: filter panic but realized was calm next month
false_alarm = (pi_filter_test[:-1] >= 0.5) & (actual == 0)
ax.scatter(test_dates[:-1][false_alarm], pi_filter_test[:-1][false_alarm],
           marker='^', color='orange', s=60, zorder=5,
           label='False alarm (filter panic, realized calm)')

# shade economic events
for start, end, lbl in test_events:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s >= pd.Timestamp(test_panel['date'].min()):
        ax.axvspan(s, e, alpha=0.10, color='gold')
        ax.text(s, 1.01, lbl, fontsize=7, color='goldenrod',
                va='bottom', rotation=20, transform=ax.get_xaxis_transform())

ax.set_ylim(0, 1)
ax.set_ylabel('P(panic)', fontsize=9)
ax.set_xlabel('Date')
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.set_title('Test period regime signal — red▼ = missed panic, orange▲ = false alarm, gold = economic events',
             fontsize=10)
plt.tight_layout()
fig.savefig('regime_signal_test.png', dpi=150)
plt.close(fig)
print("Saved: regime_signal_test.png")
