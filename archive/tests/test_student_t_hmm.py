"""
test_student_t_hmm.py
=====================
Robustness check: re-estimate the 2-state HMM with multivariate Student-t
emissions instead of Normal emissions.

Strategy: data augmentation (Geweke 1993 / Liu 1994).
    z_t | s_t=k, w_t  ~  N(mu_k, Sigma_k / w_t)
    w_t | s_t=k        ~  Gamma(nu_k/2, nu_k/2)

Marginalizing over w_t gives a multivariate Student-t with nu_k degrees of
freedom. The augmentation preserves conjugacy: conditional on w_t, the
emission is Normal, so we can still use NIW updates (with weighted data).

We compare:
  1. Log-likelihood (in-sample and out-of-sample)
  2. Regime classification agreement with the Normal HMM
  3. Downstream portfolio performance (Sharpe ratio of the regime signal)
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invwishart, gamma
from scipy.special import logsumexp, gammaln
import warnings
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────────

panel = pd.read_parquet('data/panel.parquet')
features_z = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']
panel = panel.dropna(subset=features_z).reset_index(drop=True)

train_panel = panel[panel['date'] < '2011-01-01'].reset_index(drop=True)
test_panel  = panel[panel['date'] >= '2011-01-01'].reset_index(drop=True)

Z_train = train_panel[features_z].values.astype(float)
Z_test  = test_panel[features_z].values.astype(float)
Z_full  = panel[features_z].values.astype(float)

T_train, D = Z_train.shape
T_test     = len(Z_test)
T_full     = len(Z_full)
K = 2

print(f"Train: {T_train} months, Test: {T_test} months, D={D} features")

# ── Priors (same as Normal HMM) ─────────────────────────────────────────────

m_0  = np.zeros(D)
kappa_0 = 0.01
nu_0 = D + 2
Psi_0 = np.eye(D) * (nu_0 - D - 1)

alpha_dir = np.array([[9.0, 1.0],
                       [1.0, 9.0]])

# Prior on Student-t degrees of freedom: we'll try fixed values and also
# estimate via grid search on marginal likelihood
NU_GRID = [3, 5, 7, 10, 15, 20, 30, 50, 100]  # nu=100 ≈ Normal

# ── Helper functions ─────────────────────────────────────────────────────────

def log_mvt(z, mu, Sigma, nu):
    """Log-density of multivariate Student-t with nu df, location mu, scale Sigma."""
    D = len(mu)
    diff = z - mu
    L = np.linalg.cholesky(Sigma)
    solved = np.linalg.solve(L, diff.T).T  # shape (n, D) or (D,)
    if solved.ndim == 1:
        maha = np.dot(solved, solved)
    else:
        maha = np.sum(solved ** 2, axis=1)

    log_det = 2 * np.sum(np.log(np.diag(L)))
    const = (gammaln((nu + D) / 2) - gammaln(nu / 2)
             - D / 2 * np.log(nu * np.pi) - 0.5 * log_det)
    return const - (nu + D) / 2 * np.log(1 + maha / nu)


def log_emission_t(Z, mu, Sigma, nu):
    """Log Student-t density for all t and k. Returns (T, K)."""
    return np.column_stack([
        log_mvt(Z, mu[k], Sigma[k], nu)
        for k in range(K)
    ])


def log_emission_normal_weighted(Z, mu, Sigma, w):
    """Log N(z_t; mu_k, Sigma_k/w_t) for the augmented model."""
    T = len(Z)
    log_e = np.zeros((T, K))
    for k in range(K):
        for t in range(T):
            cov_t = Sigma[k] / w[t]
            log_e[t, k] = multivariate_normal.logpdf(Z[t], mean=mu[k], cov=cov_t,
                                                      allow_singular=True)
    return log_e


def ffbs(Z, log_emit, P, init_prior=None):
    """Forward-Filtering Backward-Sampling on precomputed log-emissions."""
    n = len(Z)
    log_alpha = np.zeros((n, K))
    if init_prior is None:
        init_prior = np.full(K, 1.0 / K)
    log_alpha[0] = np.log(init_prior + 1e-300) + log_emit[0]

    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t - 1] + np.log(P[:, k] + 1e-300)
            )

    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    alpha = np.exp(log_alpha)

    s = np.zeros(n, dtype=int)
    s[n - 1] = np.random.choice(K, p=alpha[n - 1])
    for t in range(n - 2, -1, -1):
        probs = alpha[t] * P[:, s[t + 1]]
        probs /= probs.sum()
        s[t] = np.random.choice(K, p=probs)
    return s


def sample_niw_weighted(Z, states, w, k):
    """
    NIW posterior update with observation weights w_t (from t-augmentation).
    Weighted sufficient stats:
        n_eff = sum(w_t for s_t=k)
        x_bar = sum(w_t * z_t) / n_eff
        S_k   = sum(w_t * (z_t - x_bar)(z_t - x_bar)^T)
    """
    idx = states == k
    Z_k = Z[idx]
    w_k = w[idx]
    n_k = len(Z_k)

    if n_k < D + 2:
        return None, None

    n_eff = np.sum(w_k)
    x_bar = np.sum(w_k[:, None] * Z_k, axis=0) / n_eff
    diff  = Z_k - x_bar
    S_k   = (w_k[:, None] * diff).T @ diff

    kappa_n = kappa_0 + n_eff
    m_n     = (kappa_0 * m_0 + n_eff * x_bar) / kappa_n
    nu_n    = nu_0 + n_k  # integer count, not weighted
    Psi_n   = (Psi_0 + S_k +
               (kappa_0 * n_eff / kappa_n) * np.outer(x_bar - m_0, x_bar - m_0))

    # Ensure Psi_n is symmetric
    Psi_n = 0.5 * (Psi_n + Psi_n.T)

    Sigma_k = invwishart.rvs(df=nu_n, scale=Psi_n)
    mu_k    = np.random.multivariate_normal(m_n, Sigma_k / kappa_n)
    return mu_k, Sigma_k


def sample_weights(Z, states, mu, Sigma, nu):
    """
    Sample augmentation weights w_t ~ Gamma((nu+D)/2, (nu + maha_t)/2).
    Conditional on (s_t, mu, Sigma), w_t is Gamma-distributed.
    """
    T = len(Z)
    w = np.ones(T)
    for t in range(T):
        k = states[t]
        diff = Z[t] - mu[k]
        try:
            L = np.linalg.cholesky(Sigma[k])
            solved = np.linalg.solve(L, diff)
            maha = np.dot(solved, solved)
        except np.linalg.LinAlgError:
            maha = diff @ np.linalg.solve(Sigma[k] + 1e-6 * np.eye(D), diff)

        shape = (nu + D) / 2
        rate  = (nu + maha) / 2
        w[t]  = np.random.gamma(shape, 1.0 / rate)
    return w


def sample_P(states):
    """Sample transition matrix from Dirichlet posterior."""
    P_new = np.zeros((K, K))
    for i in range(K):
        counts = np.array([
            np.sum((states[:-1] == i) & (states[1:] == j))
            for j in range(K)
        ], dtype=float)
        P_new[i] = np.random.dirichlet(alpha_dir[i] + counts)
    return P_new


def forward_filter_t(Z, mu, Sigma, P, nu):
    """Forward filter with Student-t emissions (fixed params, causal)."""
    n = len(Z)
    log_emit = log_emission_t(Z, mu, Sigma, nu)
    log_alpha = np.zeros((n, K))
    log_alpha[0] = np.log(0.5) + log_emit[0]

    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t - 1] + np.log(P[:, k] + 1e-300)
            )

    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)


# ── Panic identification (same as Normal HMM) ───────────────────────────────

crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
train_dates = train_panel['date'].values
crisis_mask = np.zeros(T_train, dtype=bool)
for s, e in crisis_windows:
    crisis_mask |= ((train_dates >= np.datetime64(s)) & (train_dates <= np.datetime64(e)))

signs = np.zeros(D)
for j in range(D):
    p95_crisis = np.percentile(Z_train[crisis_mask, j], 95)
    p95_normal = np.percentile(Z_train[~crisis_mask, j], 95)
    signs[j] = 1.0 if p95_crisis >= p95_normal else -1.0


# ── Run Student-t HMM for each nu on the grid ───────────────────────────────

vol_idx = features_z.index('VOL_z')
SEED = 2201
N_ITER = 2000
N_BURNIN = 500
N_KEEP = N_ITER - N_BURNIN

# Load Normal HMM results for comparison
normal_draws = np.load('data/mcmc_draws.npz')
pi_filter_normal = normal_draws['pi_filter_full']
panic_state_normal = int(normal_draws['panic_state'])

results = []

for nu in NU_GRID:
    print(f"\n{'='*70}")
    print(f"  Student-t HMM with nu = {nu}")
    print(f"{'='*70}")

    np.random.seed(SEED)

    # Initialize states
    states = (Z_train[:, vol_idx] > np.median(Z_train[:, vol_idx])).astype(int)
    mu    = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k]    = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)

    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    w = np.ones(T_train)  # initial weights = 1 (equivalent to Normal)

    # Storage for post-burnin draws
    mu_draws    = np.zeros((N_KEEP, K, D))
    Sigma_draws = np.zeros((N_KEEP, K, D, D))
    P_draws     = np.zeros((N_KEEP, K, K))
    state_draws = np.zeros((N_KEEP, T_train), dtype=int)

    for m in range(N_ITER):
        if m % 500 == 0:
            print(f"  Iteration {m}/{N_ITER}")

        # Step 1: Sample weights (t-augmentation)
        w = sample_weights(Z_train, states, mu, Sigma, nu)

        # Step 2: FFBS with weighted emissions
        log_emit = log_emission_normal_weighted(Z_train, mu, Sigma, w)
        states = ffbs(Z_train, log_emit, P)

        # Step 3: NIW update with weighted data
        for k in range(K):
            mu_k, Sigma_k = sample_niw_weighted(Z_train, states, w, k)
            if mu_k is not None:
                mu[k], Sigma[k] = mu_k, Sigma_k

        # Step 4: Transition matrix
        P = sample_P(states)

        # Store post-burnin
        if m >= N_BURNIN:
            idx = m - N_BURNIN
            mu_draws[idx]    = mu
            Sigma_draws[idx] = Sigma
            P_draws[idx]     = P
            state_draws[idx] = states

    # ── Posterior summary ────────────────────────────────────────────────────

    mu_post    = mu_draws.mean(axis=0)
    Sigma_post = Sigma_draws.mean(axis=0)
    P_post     = P_draws.mean(axis=0)

    # Identify panic state
    score0 = np.sum(signs * mu_post[0])
    score1 = np.sum(signs * mu_post[1])
    panic = 1 if score1 >= score0 else 0
    calm  = 1 - panic

    print(f"  Panic = state {panic} (score0={score0:.3f}, score1={score1:.3f})")
    print(f"  P_post = {P_post}")
    print(f"  mu_panic = {mu_post[panic]}")
    print(f"  mu_calm  = {mu_post[calm]}")

    # Filtered probability (full sample)
    pi_filter = forward_filter_t(Z_full, mu_post, Sigma_post, P_post, nu)[:, panic]

    # ── Comparison metrics ───────────────────────────────────────────────────

    # 1. Regime agreement with Normal HMM
    regime_t = (pi_filter > 0.5).astype(int)
    regime_n = (pi_filter_normal > 0.5).astype(int)
    agreement = np.mean(regime_t == regime_n)

    # 2. Correlation of pi_filter
    corr = np.corrcoef(pi_filter, pi_filter_normal)[0, 1]

    # 3. In-sample log-likelihood (train)
    ll_train = 0.0
    for t in range(T_train):
        ll_t = logsumexp([
            log_mvt(Z_train[t], mu_post[k], Sigma_post[k], nu) + np.log(0.5)
            for k in range(K)
        ])
        ll_train += ll_t

    # 4. Out-of-sample log-likelihood (test)
    ll_test = 0.0
    for t in range(T_test):
        ll_t = logsumexp([
            log_mvt(Z_test[t], mu_post[k], Sigma_post[k], nu) + np.log(0.5)
            for k in range(K)
        ])
        ll_test += ll_t

    # 5. BIC (train)
    n_params = K * D + K * D * (D + 1) // 2 + K * (K - 1)  # means + covariances + transitions
    bic = n_params * np.log(T_train) - 2 * ll_train

    # 6. Persistence
    persistence_calm  = P_post[calm, calm]
    persistence_panic = P_post[panic, panic]

    print(f"\n  Log-lik (train): {ll_train:.1f}")
    print(f"  Log-lik (test):  {ll_test:.1f}")
    print(f"  BIC:             {bic:.1f}")
    print(f"  Agreement with Normal HMM: {agreement:.1%}")
    print(f"  Correlation with Normal pi_filter: {corr:.4f}")
    print(f"  Persistence: calm={persistence_calm:.3f}, panic={persistence_panic:.3f}")

    results.append({
        'nu': nu,
        'll_train': ll_train,
        'll_test': ll_test,
        'bic': bic,
        'agreement': agreement,
        'correlation': corr,
        'persistence_calm': persistence_calm,
        'persistence_panic': persistence_panic,
        'mu_panic': mu_post[panic].tolist(),
        'mu_calm': mu_post[calm].tolist(),
    })

# ── Final comparison table ───────────────────────────────────────────────────

print(f"\n{'='*90}")
print("COMPARISON: Student-t HMM across degrees of freedom")
print(f"{'='*90}")
print(f"{'nu':>5}  {'LL(train)':>10}  {'LL(test)':>10}  {'BIC':>10}  "
      f"{'Agreement':>10}  {'Corr':>6}  {'P(calm)':>8}  {'P(panic)':>8}")
print(f"{'-'*85}")

for r in results:
    print(f"{r['nu']:>5}  {r['ll_train']:>10.1f}  {r['ll_test']:>10.1f}  {r['bic']:>10.1f}  "
          f"{r['agreement']:>10.1%}  {r['correlation']:>6.4f}  "
          f"{r['persistence_calm']:>8.3f}  {r['persistence_panic']:>8.3f}")

# Also compute Normal HMM log-likelihood for comparison
mu_normal    = normal_draws['mu_draws'].mean(axis=0)
Sigma_normal = normal_draws['Sigma_draws'].mean(axis=0)

ll_train_normal = 0.0
for t in range(T_train):
    ll_t = logsumexp([
        multivariate_normal.logpdf(Z_train[t], mean=mu_normal[k], cov=Sigma_normal[k])
        + np.log(0.5)
        for k in range(K)
    ])
    ll_train_normal += ll_t

ll_test_normal = 0.0
for t in range(T_test):
    ll_t = logsumexp([
        multivariate_normal.logpdf(Z_test[t], mean=mu_normal[k], cov=Sigma_normal[k])
        + np.log(0.5)
        for k in range(K)
    ])
    ll_test_normal += ll_t

n_params_normal = K * D + K * D * (D + 1) // 2 + K * (K - 1)
bic_normal = n_params_normal * np.log(T_train) - 2 * ll_train_normal

print(f"{'-'*85}")
print(f"{'N/A':>5}  {ll_train_normal:>10.1f}  {ll_test_normal:>10.1f}  {bic_normal:>10.1f}  "
      f"{'(baseline)':>10}  {'1.000':>6}  {'—':>8}  {'—':>8}   ← Normal HMM")

# Best by BIC
best = min(results, key=lambda r: r['bic'])
print(f"\nBest Student-t by BIC: nu={best['nu']} (BIC={best['bic']:.1f})")
print(f"Normal HMM BIC: {bic_normal:.1f}")
delta = bic_normal - best['bic']
print(f"BIC improvement of Student-t over Normal: {delta:.1f}")

if delta > 10:
    print("→ Very strong evidence for Student-t emissions")
elif delta > 6:
    print("→ Strong evidence for Student-t emissions")
elif delta > 2:
    print("→ Positive evidence for Student-t emissions")
else:
    print("→ No meaningful improvement — Normal emissions are adequate")

print(f"\nRegime agreement between best Student-t and Normal: {best['agreement']:.1%}")
print(f"Correlation of pi_filter: {best['correlation']:.4f}")
if best['agreement'] > 0.90:
    print("→ The two models largely agree on regime classification")
    print("→ Normal emissions are a defensible choice for this application")
