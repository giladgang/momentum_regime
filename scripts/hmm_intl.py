"""
hmm_intl.py
===========
Regional Bayesian 2-state HMM (Gibbs / FFBS) for UK or Japan, mirroring the
US production HMM in `scripts/hmm_model.py` but using a regional 4-feature
set in which BANK_REL_z replaces CS_z (no Moody's BAA-AAA equivalent for
either market — see INTL_VALIDATION_PLAN.md for the design rationale).

Self-contained: does NOT import `hmm_model.py` so importing it does not
trigger a US HMM run. Identical math, identical priors (from config.py).

Usage
-----
    python scripts/hmm_intl.py --region UK
    python scripts/hmm_intl.py --region JP
    python scripts/hmm_intl.py --region UK --smoke      # 5 seeds, 500 iter
    python scripts/hmm_intl.py --region UK --seeds 50   # custom seed count

Outputs
-------
    data/{region}_panel_with_regimes.parquet  (regime probs joined to market panel)
    data/{region}_mcmc_draws.npz              (first-seed draws for diagnostics)

Console output reports:
    - Train / test split sizes
    - Per-seed panic-state identification (which of {0,1} got tagged panic)
    - Top crisis-month panic probabilities for sanity check
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (HMM_PRIOR_M0, HMM_PRIOR_KAPPA0, HMM_PRIOR_NU0_OFF,
                    HMM_PRIOR_DIRICHLET_ALPHA, TRAIN_END)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--region', choices=['UK', 'JP'], required=True)
parser.add_argument('--seeds', type=int, default=200,
                    help='Number of MCMC seeds (production = 200).')
parser.add_argument('--iterations', type=int, default=2000,
                    help='Gibbs iterations per seed (production = 2000).')
parser.add_argument('--burnin', type=int, default=500,
                    help='Burnin iterations to discard (production = 500).')
parser.add_argument('--smoke', action='store_true',
                    help='Smoke test: 5 seeds, 500 iter, 100 burnin (~1-2 min).')
parser.add_argument('--workers', type=int, default=1,
                    help='Parallel worker processes for the seed loop '
                         '(default 1 = serial). On 8-core machine use 6 to '
                         'leave 2 cores free.')
args = parser.parse_args()

if args.smoke:
    args.seeds = 5
    args.iterations = 500
    args.burnin = 100

REGION = args.region
INPUT  = f'data/{REGION.lower()}_market_panel.parquet'
OUT_PANEL = f'data/{REGION.lower()}_panel_with_regimes.parquet'
OUT_MCMC  = f'data/{REGION.lower()}_mcmc_draws.npz'

REGION_FEATURES = ['DD_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z']

# Region-specific crisis windows (used only for panic-state sign correction
# during training — the Gibbs sampler itself is unsupervised).
CRISIS_WINDOWS = {
    'UK': [
        ('2000-03-01', '2002-10-01'),  # dot-com
        ('2007-10-01', '2009-06-01'),  # global financial crisis
    ],
    'JP': [
        ('1990-01-01', '1992-12-01'),  # bubble collapse
        ('1997-11-01', '1998-12-01'),  # LTCB / Yamaichi failures
        ('2001-09-01', '2003-04-01'),  # NPL crisis nadir
    ],
}[REGION]

# ── Load and split ────────────────────────────────────────────────────────────

print(f"=== Regional HMM: {REGION} ===")
print(f"Input: {INPUT}")
print(f"Features: {REGION_FEATURES}")
print(f"Seeds: {args.seeds} | Iterations: {args.iterations} | Burnin: {args.burnin}")

market = pd.read_parquet(INPUT)
market = market.dropna(subset=REGION_FEATURES).reset_index(drop=True)
market['date'] = pd.to_datetime(market['date'])

train_end_dt = pd.Timestamp(TRAIN_END)
train_panel = market[market['date'] < train_end_dt].reset_index(drop=True)
test_panel  = market[market['date'] >= train_end_dt].reset_index(drop=True)

Z_train = train_panel[REGION_FEATURES].values.astype(float)
Z_test  = test_panel [REGION_FEATURES].values.astype(float)
Z_full  = market    [REGION_FEATURES].values.astype(float)

T_train, D = Z_train.shape
T_test     = len(Z_test)
T_full     = len(Z_full)
K = 2

print(f"Train: {train_panel['date'].min().date()} -> {train_panel['date'].max().date()} ({T_train} months)")
print(f"Test:  {test_panel['date'].min().date()}  -> {test_panel['date'].max().date()}  ({T_test} months)")

# ── Priors (same as US production from config.py) ────────────────────────────

m_0  = np.full(D, HMM_PRIOR_M0)
κ_0  = HMM_PRIOR_KAPPA0
ν_0  = D + HMM_PRIOR_NU0_OFF
Ψ_0  = np.eye(D) * (ν_0 - D - 1)
α_dir = np.asarray(HMM_PRIOR_DIRICHLET_ALPHA, dtype=float)

# ── Crisis mask + sign correction (regional crisis windows) ──────────────────

train_dates = train_panel['date'].values
crisis_mask = np.zeros(T_train, dtype=bool)
for s, e in CRISIS_WINDOWS:
    crisis_mask |= ((train_dates >= np.datetime64(s)) & (train_dates <= np.datetime64(e)))

print(f"Crisis windows ({REGION}): {CRISIS_WINDOWS}")
print(f"  Crisis months in train: {int(crisis_mask.sum())} / {T_train}")

signs = np.zeros(D)
for j in range(D):
    p95_crisis = np.percentile(Z_train[crisis_mask, j], 95)
    p95_normal = np.percentile(Z_train[~crisis_mask, j], 95)
    signs[j] = 1.0 if p95_crisis >= p95_normal else -1.0
print(f"  Feature signs (crisis-aligned): "
      + ', '.join(f"{f}={int(s):+d}" for f, s in zip(REGION_FEATURES, signs)))

# ── Sampler helpers ──────────────────────────────────────────────────────────

def init_sampler(seed):
    np.random.seed(seed)
    states = (Z_train[:, 0] > np.median(Z_train[:, 0])).astype(int)  # DD_z is index 0
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


def log_emission(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])


def ffbs(Z, mu, Sigma, P, init_prior=None):
    n = len(Z)
    log_emit  = log_emission(Z, mu, Sigma)
    log_alpha = np.zeros((n, K))
    if init_prior is None:
        init_prior = np.full(K, 1.0 / K)
    log_alpha[0] = np.log(init_prior) + log_emit[0]
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(log_alpha[t-1] + np.log(P[:, k]))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    alpha = np.exp(log_alpha)
    s = np.zeros(n, dtype=int)
    s[n-1] = np.random.choice(K, p=alpha[n-1])
    for t in range(n - 2, -1, -1):
        probs = alpha[t] * P[:, s[t+1]]
        probs /= probs.sum()
        s[t] = np.random.choice(K, p=probs)
    return s


def sample_niw(Z, states, k, mu_prev, Sigma_prev):
    idx = states == k
    Z_k = Z[idx]
    n_k = len(Z_k)
    if n_k < D + 2:
        return mu_prev, Sigma_prev
    x_bar = Z_k.mean(axis=0)
    S_k   = (Z_k - x_bar).T @ (Z_k - x_bar)
    κ_n = κ_0 + n_k
    m_n = (κ_0 * m_0 + n_k * x_bar) / κ_n
    ν_n = ν_0 + n_k
    Ψ_n = Ψ_0 + S_k + (κ_0 * n_k / κ_n) * np.outer(x_bar - m_0, x_bar - m_0)
    Sigma_k = invwishart.rvs(df=ν_n, scale=Ψ_n)
    mu_k    = np.random.multivariate_normal(m_n, Sigma_k / κ_n)
    return mu_k, Sigma_k


def sample_P(states):
    P_new = np.zeros((K, K))
    for i in range(K):
        counts = np.array([np.sum((states[:-1] == i) & (states[1:] == j))
                           for j in range(K)], dtype=float)
        P_new[i] = np.random.dirichlet(α_dir[i] + counts)
    return P_new


def forward_filter(Z, mu, Sigma, P):
    n = len(Z)
    log_emit  = log_emission(Z, mu, Sigma)
    log_alpha = np.zeros((n, K))
    log_alpha[0] = np.log(0.5) + log_emit[0]
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(log_alpha[t-1] + np.log(P[:, k]))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)


# ── Multi-seed Gibbs ─────────────────────────────────────────────────────────

SEEDS = list(range(1, args.seeds + 1))
n_iter, n_burnin = args.iterations, args.burnin
n_keep = n_iter - n_burnin
n_test_draws = 100 if args.smoke else 200


def _run_one_seed(seed_payload):
    """Run a single Gibbs seed end-to-end. Returns a dict suitable for
    aggregation. Used by both serial and parallel paths.

    Reads module globals (Z_train, Z_test, Z_full, signs, K, D, T_train,
    T_test, T_full, n_iter, n_burnin, n_test_draws, n_keep). When called
    from a fork()-based Pool, those globals were set up in the parent
    before fork; child inherits them.
    """
    seed_idx, seed, save_full_draws, verbose = seed_payload
    if verbose:
        print(f"\n  Seed {seed_idx+1}/{len(SEEDS)} (seed={seed}) ...", flush=True)
    states, mu, Sigma, P = init_sampler(seed)

    sd = np.zeros((n_keep, T_train), dtype=int)
    md = np.zeros((n_keep, K, D))
    Sd = np.zeros((n_keep, K, D, D))
    Pd = np.zeros((n_keep, K, K))

    for m in range(n_iter):
        if verbose and (n_iter <= 200 or m % max(1, n_iter // 4) == 0):
            print(f"    [seed={seed}] iter {m}/{n_iter}", flush=True)
        states = ffbs(Z_train, mu, Sigma, P)
        for k in range(K):
            mu[k], Sigma[k] = sample_niw(Z_train, states, k, mu[k], Sigma[k])
        P = sample_P(states)
        if m >= n_burnin:
            idx = m - n_burnin
            sd[idx] = states
            md[idx] = mu
            Sd[idx] = Sigma
            Pd[idx] = P

    mu_post_seed    = md.mean(axis=0)
    Sigma_post_seed = Sd.mean(axis=0)
    P_post_seed     = Pd.mean(axis=0)

    score0 = float(np.sum(signs * mu_post_seed[0]))
    score1 = float(np.sum(signs * mu_post_seed[1]))
    panic_seed = 1 if score1 >= score0 else 0
    calm_seed  = 1 - panic_seed

    filtered_seed = forward_filter(Z_full, mu_post_seed, Sigma_post_seed, P_post_seed)
    pi_filter_seed = filtered_seed[:, panic_seed]

    pi_smooth_train_seed = (sd == panic_seed).mean(axis=0)

    last_train = pi_filter_seed[T_train - 1]
    init_prior = np.zeros(K)
    init_prior[panic_seed] = last_train
    init_prior[calm_seed]  = 1.0 - last_train
    smooth_test_seed = np.zeros((n_test_draws, T_test), dtype=int)
    for i in range(n_test_draws):
        smooth_test_seed[i] = ffbs(Z_test, mu_post_seed, Sigma_post_seed,
                                   P_post_seed, init_prior=init_prior)
    pi_smooth_test_seed = (smooth_test_seed == panic_seed).mean(axis=0)

    if verbose:
        print(f"    [seed={seed}] panic=state {panic_seed}  "
              f"score0={score0:.3f} score1={score1:.3f}", flush=True)

    return {
        'seed_idx': seed_idx,
        'seed': seed,
        'panic_seed': panic_seed,
        'calm_seed': calm_seed,
        'pi_filter_seed': pi_filter_seed,
        'pi_smooth_train_seed': pi_smooth_train_seed,
        'pi_smooth_test_seed': pi_smooth_test_seed,
        # Full draws only for the first seed (used for diagnostics output)
        'sd': sd if save_full_draws else None,
        'md': md if save_full_draws else None,
        'Sd': Sd if save_full_draws else None,
        'Pd': Pd if save_full_draws else None,
    }


pi_filter_accum       = np.zeros(T_full)
pi_smooth_train_accum = np.zeros(T_train)
pi_smooth_test_accum  = np.zeros(T_test)

state_draws = mu_draws = Sigma_draws = P_draws = None
panic_first = calm_first = None


def _accumulate(result):
    global pi_filter_accum, pi_smooth_train_accum, pi_smooth_test_accum
    global state_draws, mu_draws, Sigma_draws, P_draws
    global panic_first, calm_first
    pi_filter_accum       += result['pi_filter_seed']
    pi_smooth_train_accum += result['pi_smooth_train_seed']
    pi_smooth_test_accum  += result['pi_smooth_test_seed']
    if result['seed_idx'] == 0:
        state_draws = result['sd']
        mu_draws    = result['md']
        Sigma_draws = result['Sd']
        P_draws     = result['Pd']
        panic_first = result['panic_seed']
        calm_first  = result['calm_seed']


print(f"\nRunning {len(SEEDS)} seeds x {n_iter} iter ({n_burnin} burnin) "
      f"with {args.workers} worker(s) ...")

# First seed always saves full draws for the npz diagnostics file; later
# seeds only return summary fields. Verbose printing only for serial mode
# (interleaved parallel output is hard to read).
verbose = (args.workers == 1)
seed_payloads = [(idx, seed, idx == 0, verbose)
                 for idx, seed in enumerate(SEEDS)]

if args.workers > 1:
    from multiprocessing import get_context
    ctx = get_context('fork')  # fork inherits module globals — required
    print(f"  [parallel] dispatching {len(SEEDS)} seeds across "
          f"{args.workers} workers ...", flush=True)
    n_done = 0
    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_run_one_seed, seed_payloads):
            _accumulate(result)
            n_done += 1
            print(f"  [{n_done:3d}/{len(SEEDS)}] seed={result['seed']} "
                  f"panic=state {result['panic_seed']}", flush=True)
else:
    for payload in seed_payloads:
        _accumulate(_run_one_seed(payload))

# ── Aggregate across seeds ──────────────────────────────────────────────────

N_SEEDS = len(SEEDS)
pi_filter_full  = pi_filter_accum / N_SEEDS
pi_smooth_train = pi_smooth_train_accum / N_SEEDS
pi_smooth_test  = pi_smooth_test_accum / N_SEEDS
pi_smooth_full  = np.concatenate([pi_smooth_train, pi_smooth_test])

# ── Save ────────────────────────────────────────────────────────────────────

market_out = market.copy()
market_out['pi_filter']  = pi_filter_full
market_out['pi_smooth']  = pi_smooth_full
market_out['pi_next']    = np.r_[pi_filter_full[1:], np.nan]  # 1-month-ahead view of pi_filter

os.makedirs('data', exist_ok=True)
market_out.to_parquet(OUT_PANEL, index=False)
np.savez(OUT_MCMC,
         state_draws=state_draws,
         mu_draws=mu_draws,
         Sigma_draws=Sigma_draws,
         P_draws=P_draws,
         panic_state=panic_first,
         features=REGION_FEATURES)

# ── Sanity report ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"  {REGION} HMM COMPLETE")
print("=" * 60)
print(f"  Output: {OUT_PANEL} ({len(market_out)} rows)")
print(f"  MCMC:   {OUT_MCMC}")
print(f"  pi_filter range: [{pi_filter_full.min():.3f}, {pi_filter_full.max():.3f}]")
print(f"  pi_filter mean:  {pi_filter_full.mean():.3f}")
print(f"  Months above pi_filter > 0.5: {int((pi_filter_full > 0.5).sum())} / {T_full}")

print(f"\n  Top 10 panic-probability months (pi_filter):")
top = market_out.nlargest(10, 'pi_filter')[['date', 'pi_filter', 'DD_z', 'BANK_REL_z']]
print(top.to_string(index=False))
