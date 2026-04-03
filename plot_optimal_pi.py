"""
Plot pi_filter for the optimal combo (DD+TERM+CPS+ADR) vs baseline (DD+VOL+CS+LVIX)
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')

# ── Data loading ────────────────────────────────────────────────────────────

panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
panel['year_month'] = panel['date'].dt.to_period('M')

stock_data = pd.read_parquet('data/crsp_msf_raw.parquet')
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['year_month'] = stock_data['date'].dt.to_period('M')

train_mask = panel['date'] < '2011-01-01'

# Compute needed features
import pandas_datareader.data as web

if 'TERM' not in panel.columns:
    t10 = web.DataReader('DGS10', 'fred', start='1970-01-01', end='2025-12-31')
    t2 = web.DataReader('DGS2', 'fred', start='1970-01-01', end='2025-12-31')
    term = (t10['DGS10'] - t2['DGS2']).resample('ME').mean().to_frame('TERM')
    term.index = term.index.to_period('M')
    term_df = term.reset_index(); term_df.columns = ['year_month', 'TERM']
    panel = panel.merge(term_df, on='year_month', how='left')

if 'CP_SPREAD' not in panel.columns:
    cp = web.DataReader('DCPF3M', 'fred', start='1970-01-01', end='2025-12-31')
    tbill = web.DataReader('DTB3', 'fred', start='1970-01-01', end='2025-12-31')
    cp_spread = (cp['DCPF3M'] - tbill['DTB3']).resample('ME').mean().to_frame('CP_SPREAD')
    cp_spread.index = cp_spread.index.to_period('M')
    cp_df = cp_spread.reset_index(); cp_df.columns = ['year_month', 'CP_SPREAD']
    panel = panel.merge(cp_df, on='year_month', how='left')

if 'ADR' not in panel.columns:
    adr = stock_data.groupby('year_month').apply(
        lambda g: (g['ret_adj'] > 0).mean(), include_groups=False
    ).reset_index()
    adr.columns = ['year_month', 'ADR']
    panel = panel.merge(adr, on='year_month', how='left')

# Standardize
for col in ['TERM', 'CP_SPREAD', 'ADR']:
    zcol = f'{col}_z'
    if zcol not in panel.columns:
        vals = panel[col].astype(float)
        mu = vals[train_mask].dropna().mean()
        sd = vals[train_mask].dropna().std()
        if sd > 0:
            panel[zcol] = (vals - mu) / sd

# ── HMM functions ──────────────────────────────────────────────────────────

K = 2

def log_emission(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])

def ffbs(Z, mu, Sigma, P):
    n = len(Z)
    log_emit = log_emission(Z, mu, Sigma)
    log_alpha = np.zeros((n, K))
    log_alpha[0] = np.log(0.5) + log_emit[0]
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    alpha = np.exp(log_alpha)
    s = np.zeros(n, dtype=int)
    s[n-1] = np.random.choice(K, p=alpha[n-1])
    for t in range(n - 2, -1, -1):
        probs = alpha[t] * P[:, s[t+1]]
        probs /= probs.sum()
        s[t] = np.random.choice(K, p=probs)
    return s

def forward_filter(Z, mu, Sigma, P):
    n = len(Z)
    log_emit = log_emission(Z, mu, Sigma)
    log_alpha = np.zeros((n, K))
    log_alpha[0] = np.log(0.5) + log_emit[0]
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)

def fit_and_filter(feat_list, label):
    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01']
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']
    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)

    np.random.seed(2201)
    T, D = Z_tr.shape

    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    init_idx = feat_list.index('VOL_z') if 'VOL_z' in feat_list else 0
    states = (Z_tr[:, init_idx] > np.median(Z_tr[:, init_idx])).astype(int)

    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_tr[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_tr[idx].T) + 1e-6 * np.eye(D)
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    for m in range(2000):
        states = ffbs(Z_tr, mu, Sigma, P)
        for k in range(K):
            Z_k = Z_tr[states == k]
            n_k = len(Z_k)
            if n_k < D + 2:
                continue
            x_bar = Z_k.mean(axis=0)
            S_k = (Z_k - x_bar).T @ (Z_k - x_bar)
            kappa_n = kappa_0 + n_k
            m_n = (kappa_0 * m_0 + n_k * x_bar) / kappa_n
            nu_n = nu_0 + n_k
            Psi_n = Psi_0 + S_k + (kappa_0 * n_k / kappa_n) * np.outer(x_bar - m_0, x_bar - m_0)
            try:
                Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
                mu[k] = np.random.multivariate_normal(m_n, Sigma[k] / kappa_n)
            except:
                pass
        for i in range(K):
            counts = np.array([np.sum((states[:-1] == i) & (states[1:] == j))
                               for j in range(K)], dtype=float)
            P[i] = np.random.dirichlet(alpha_dir[i] + counts)

    mu_post = mu.copy()
    Z_full = np.vstack([Z_tr, Z_te])
    filtered = forward_filter(Z_full, mu_post, Sigma, P)

    # Crisis-calibrated sign correction + L2
    crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
    crisis_mask = np.zeros(T, dtype=bool)
    hmm_dates = sub_train['date'].values
    for s, e in crisis_windows:
        crisis_mask |= ((hmm_dates >= np.datetime64(s)) & (hmm_dates <= np.datetime64(e)))

    if crisis_mask.sum() > 5:
        signs = np.zeros(D)
        for j in range(D):
            signs[j] = 1.0 if np.percentile(Z_tr[crisis_mask, j], 95) >= np.percentile(Z_tr[~crisis_mask, j], 95) else -1.0
        score0 = np.sum(signs * mu_post[0])
        score1 = np.sum(signs * mu_post[1])
        panic_state = 1 if score1 >= score0 else 0
    else:
        panic_state = 1 if np.sum(mu_post[1]**2) >= np.sum(mu_post[0]**2) else 0

    pi_filter = filtered[:, panic_state]
    dates = sub['date'].values

    print(f"  {label}: T_train={T}, T_total={len(dates)}, panic_state={panic_state}")
    return dates, pi_filter


# ── Fit both combos ────────────────────────────────────────────────────────

print("Fitting HMMs ...")
dates_opt, pi_opt = fit_and_filter(
    ['DD_z', 'TERM_z', 'CP_SPREAD_z', 'ADR_z'], 'DD+TERM+CPS+ADR (Optimal)')
dates_bl, pi_bl = fit_and_filter(
    ['DD_z', 'VOL_z', 'CS_z', 'LVIX_z'], 'DD+VOL+CS+LVIX (Baseline)')

# ── Plot ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Crisis shading periods
crises = [
    ('2000-03-01', '2002-10-01', 'Dot-com'),
    ('2007-10-01', '2009-06-01', 'GFC'),
    ('2020-02-01', '2020-05-01', 'COVID'),
    ('2022-01-01', '2022-10-01', '2022\nBear'),
]

for ax_idx, (dates, pi, title, color) in enumerate([
    (dates_opt, pi_opt, r'Optimal: DD + TERM + CPS + ADR  (XGB Sharpe = 1.16)', '#2196F3'),
    (dates_bl, pi_bl, r'Baseline: DD + VOL + CS + LVIX  (XGB Sharpe = 0.76)', '#FF9800'),
]):
    ax = axes[ax_idx]
    ax.fill_between(dates, 0, pi, alpha=0.4, color=color)
    ax.plot(dates, pi, color=color, linewidth=0.8)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(pd.Timestamp('2011-01-01'), color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(pd.Timestamp('2010-06-01'), 0.95, 'Train', ha='right', fontsize=9, alpha=0.6)
    ax.text(pd.Timestamp('2011-06-01'), 0.95, 'Test', ha='left', fontsize=9, alpha=0.6)

    for cs, ce, clabel in crises:
        ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), alpha=0.15, color='red')
        mid = pd.Timestamp(cs) + (pd.Timestamp(ce) - pd.Timestamp(cs)) / 2
        ax.text(mid, 1.02, clabel, ha='center', va='bottom', fontsize=7, color='red', alpha=0.8)

    ax.set_ylabel(r'$\pi_t^{\mathrm{filter}}$ (panic probability)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(dates[0], dates[-1])

axes[1].set_xlabel('Date')
plt.suptitle('HMM Filtered Panic Probability: Optimal vs Baseline Feature Set',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plots/pi_filter_optimal_vs_baseline.png', dpi=200, bbox_inches='tight')
print("\nSaved: plots/pi_filter_optimal_vs_baseline.png")
