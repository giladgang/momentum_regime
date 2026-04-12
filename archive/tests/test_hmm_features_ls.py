"""
test_hmm_features.py
====================
Exhaustive search over all C(11,4)=330 four-feature HMM combinations,
using the EXACT same HMM settings as hmm_model.py (2000 iter, 500 burn-in,
same priors, VOL_z panic identification).

Two-pass approach:
  Pass 1: Run HMM for all 330 combos, record quality metrics
          (ESS, regime separation, crisis alignment)
  Pass 2: For combos that pass quality checks, run full cross-sectional
          portfolio test (M0 formula, M1 LR, M2 XGB)

Quality gates (Pass 1 -> Pass 2):
  - Min ESS >= 50 across mu parameters
  - At least 2 features with significant regime separation (95% CI excludes 0)
  - Crisis alignment: avg pi > 0.5 during GFC
  - Non-crisis: avg pi < 0.4
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.preprocessing import StandardScaler
import warnings, time
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  HMM FEATURE SEARCH (LONG-SHORT) — FULL PIPELINE MATCH")
print("=" * 80)

panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
panel['year_month'] = panel['date'].dt.to_period('M')

stock_data = pd.read_parquet('data/crsp_msf_raw.parquet')
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['year_month'] = stock_data['date'].dt.to_period('M')

train_mask = panel['date'] < '2011-01-01'

# ── Compute candidate features not already in panel ─────────────────────────

print("\nComputing candidate features ...")
# Use precomputed extra features instead of pandas_datareader (distutils removed in Python 3.12)
try:
    extra = pd.read_pickle('data/extra_hmm_features.pkl')
    panel = panel.merge(extra, on='year_month', how='left')
    print("  Loaded precomputed ADR, SKEW, TERM from data/extra_hmm_features.pkl")
except:
    pass

class _DummyWeb:
    def DataReader(self, *a, **kw):
        raise RuntimeError("Using precomputed features instead")
web = _DummyWeb()

# TERM: 10Y - 2Y Treasury
if 'TERM' not in panel.columns:
    try:
        t10 = web.DataReader('DGS10', 'fred', start='1970-01-01', end='2025-12-31')
        t2 = web.DataReader('DGS2', 'fred', start='1970-01-01', end='2025-12-31')
        term = (t10['DGS10'] - t2['DGS2']).resample('ME').mean().to_frame('TERM')
        term.index = term.index.to_period('M')
        term_df = term.reset_index()
        term_df.columns = ['year_month', 'TERM']
        panel = panel.merge(term_df, on='year_month', how='left')
        print("  TERM: OK")
    except Exception as e:
        print(f"  TERM: FAILED ({e})")
else:
    print("  TERM: already in panel")

# CP_SPREAD: Commercial Paper minus T-bill
if 'CP_SPREAD' not in panel.columns:
    try:
        cp = web.DataReader('DCPF3M', 'fred', start='1970-01-01', end='2025-12-31')
        tbill = web.DataReader('DTB3', 'fred', start='1970-01-01', end='2025-12-31')
        cp_spread = (cp['DCPF3M'] - tbill['DTB3']).resample('ME').mean().to_frame('CP_SPREAD')
        cp_spread.index = cp_spread.index.to_period('M')
        cp_df = cp_spread.reset_index()
        cp_df.columns = ['year_month', 'CP_SPREAD']
        panel = panel.merge(cp_df, on='year_month', how='left')
        print(f"  CP_SPREAD: OK ({panel['CP_SPREAD'].notna().sum()} months)")
    except Exception as e:
        print(f"  CP_SPREAD: FAILED ({e})")
else:
    print("  CP_SPREAD: already in panel")

# HY_OAS
if 'HY_OAS' not in panel.columns:
    try:
        hy = web.DataReader('BAMLH0A0HYM2', 'fred', start='1970-01-01', end='2025-12-31')
        hy = hy.resample('ME').mean()
        hy.columns = ['HY_OAS']
        hy.index = hy.index.to_period('M')
        hy_df = hy.reset_index()
        hy_df.columns = ['year_month', 'HY_OAS']
        panel = panel.merge(hy_df, on='year_month', how='left')
        print("  HY_OAS: OK")
    except Exception as e:
        print(f"  HY_OAS: FAILED ({e})")
else:
    print("  HY_OAS: already in panel")

# DISP: cross-sectional return dispersion
if 'DISP' not in panel.columns:
    disp = stock_data.groupby('year_month')['ret_adj'].std().reset_index()
    disp.columns = ['year_month', 'DISP']
    disp['DISP'] = np.log(disp['DISP'])
    panel = panel.merge(disp, on='year_month', how='left')
    print("  DISP: OK")
else:
    print("  DISP: already in panel")

# ADR: advance-decline ratio
if 'ADR' not in panel.columns:
    adr = stock_data.groupby('year_month').apply(
        lambda g: (g['ret_adj'] > 0).mean(), include_groups=False
    ).reset_index()
    adr.columns = ['year_month', 'ADR']
    panel = panel.merge(adr, on='year_month', how='left')
    print("  ADR: OK")
else:
    print("  ADR: already in panel")

# SKEW: cross-sectional skewness
if 'SKEW' not in panel.columns:
    skew_feat = stock_data.groupby('year_month')['ret_adj'].skew().reset_index()
    skew_feat.columns = ['year_month', 'SKEW']
    panel = panel.merge(skew_feat, on='year_month', how='left')
    print("  SKEW: OK")
else:
    print("  SKEW: already in panel")

# REL_N: relative number of stocks
if 'REL_N' not in panel.columns:
    n_stocks = stock_data.groupby('year_month')['permno'].nunique().reset_index()
    n_stocks.columns = ['year_month', 'N_STOCKS']
    n_stocks = n_stocks.sort_values('year_month')
    n_stocks['N_STOCKS_MA12'] = n_stocks['N_STOCKS'].rolling(12).mean()
    n_stocks['REL_N'] = np.log(n_stocks['N_STOCKS'] / n_stocks['N_STOCKS_MA12'])
    panel = panel.merge(n_stocks[['year_month', 'REL_N']], on='year_month', how='left')
    print("  REL_N: OK")
else:
    print("  REL_N: already in panel")

# ── Standardize all candidate features ───────────────────────────────────────

CANDIDATE_RAW = ['TERM', 'CP_SPREAD', 'HY_OAS', 'DISP', 'ADR', 'SKEW', 'REL_N']
for col in CANDIDATE_RAW:
    zcol = f'{col}_z'
    if col in panel.columns and zcol not in panel.columns:
        vals = panel[col].astype(float)
        mu = vals[train_mask].dropna().mean()
        sd = vals[train_mask].dropna().std()
        if sd > 0 and not np.isnan(sd):
            panel[zcol] = (vals - mu) / sd

# ── Feature pool ─────────────────────────────────────────────────────────────

FEATURE_POOL = {
    'DD_z':         'DD (Market drawdown)',
    'VOL_z':        'VOL (Realised volatility)',
    'CS_z':         'CS (Credit spread BAA-AAA)',
    'LVIX_z':       'LVIX (Log VIX)',
    'TERM_z':       'TERM (Yield curve 10Y-2Y)',
    'CP_SPREAD_z':  'CPS (Funding stress)',
    'HY_OAS_z':     'HY_OAS (High-yield spread)',
    'DISP_z':       'DISP (Return dispersion)',
    'ADR_z':        'ADR (Advance-decline ratio)',
    'SKEW_z':       'SKEW (Return skewness)',
    'REL_N_z':      'REL_N (Relative participation)',
}

available = {k: v for k, v in FEATURE_POOL.items() if k in panel.columns}
FEAT_NAMES = list(available.keys())

print(f"\n  Feature pool: {len(FEAT_NAMES)} features")
for f, d in available.items():
    n_valid = panel[f].notna().sum()
    dates = panel.loc[panel[f].notna(), 'date']
    print(f"    {f:<16} {d:<35} ({n_valid} months, {dates.min().date()} - {dates.max().date()})")

all_combos = list(combinations(FEAT_NAMES, 4))
baseline_combo = ('DD_z', 'VOL_z', 'CS_z', 'LVIX_z')
print(f"\n  Total 4-feature combinations: {len(all_combos)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS — exact match to hmm_model.py
# ═══════════════════════════════════════════════════════════════════════════════

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


def fit_hmm(Z_train, Z_test, feat_list, n_iter=2000, n_burnin=500, seed=2201):
    """
    Fit 2-state HMM with EXACT same settings as hmm_model.py.
    Returns posterior parameters, pi_filter, and quality metrics.
    """
    np.random.seed(seed)
    T, D = Z_train.shape

    # Priors — matching hmm_model.py exactly
    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    # Initialize using VOL_z if available, else first feature
    if 'VOL_z' in feat_list:
        init_idx = feat_list.index('VOL_z')
    else:
        init_idx = 0
    states = (Z_train[:, init_idx] > np.median(Z_train[:, init_idx])).astype(int)

    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    n_keep = n_iter - n_burnin
    mu_draws = np.zeros((n_keep, K, D))
    Sigma_draws = np.zeros((n_keep, K, D, D))
    P_draws = np.zeros((n_keep, K, K))

    for m in range(n_iter):
        states = ffbs(Z_train, mu, Sigma, P)
        for k in range(K):
            Z_k = Z_train[states == k]
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
        if m >= n_burnin:
            idx = m - n_burnin
            mu_draws[idx] = mu
            Sigma_draws[idx] = Sigma
            P_draws[idx] = P

    mu_post = mu_draws.mean(axis=0)
    Sigma_post = Sigma_draws.mean(axis=0)
    P_post = P_draws.mean(axis=0)

    # Panic identification: L2 norm on crisis-calibrated sign-corrected means
    # For each feature, check 95th percentile during crisis vs non-crisis.
    # Features that go UP in panic keep sign +1, features that go DOWN get -1.
    # Then apply L2 norm on sign-corrected posterior means to identify panic state.
    Z_full = np.vstack([Z_train, Z_test])
    filtered = forward_filter(Z_full, mu_post, Sigma_post, P_post)

    # Build crisis mask from training dates
    crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
    crisis_mask = np.zeros(T, dtype=bool)
    sub_hmm = panel[['date'] + feat_list].dropna()
    sub_hmm = sub_hmm[sub_hmm['date'] >= '1990-01-01']
    sub_hmm_train = sub_hmm[sub_hmm['date'] < '2011-01-01']
    hmm_dates = sub_hmm_train['date'].values
    if len(hmm_dates) == T:
        for s, e in crisis_windows:
            crisis_mask |= ((hmm_dates >= np.datetime64(s)) & (hmm_dates <= np.datetime64(e)))

    if crisis_mask.sum() > 5:
        # Determine sign direction for each feature
        signs = np.zeros(D)
        for j in range(D):
            p95_crisis = np.percentile(Z_train[crisis_mask, j], 95)
            p95_normal = np.percentile(Z_train[~crisis_mask, j], 95)
            signs[j] = 1.0 if p95_crisis >= p95_normal else -1.0
        # Sign-correct the posterior means, then use L2 norm
        corrected_0 = signs * mu_post[0]
        corrected_1 = signs * mu_post[1]
        l2_0 = np.sqrt(np.sum(corrected_0 ** 2))
        l2_1 = np.sqrt(np.sum(corrected_1 ** 2))
        # Panic state = the one with higher L2 in the sign-corrected space
        # But we also need directionality: panic should have positive sign-corrected means
        score0 = np.sum(corrected_0)
        score1 = np.sum(corrected_1)
        panic_state = 1 if score1 >= score0 else 0
    else:
        # Fallback: plain L2 norm
        panic_state = 1 if np.sum(mu_post[1]**2) >= np.sum(mu_post[0]**2) else 0

    pi_filter = filtered[:, panic_state]

    # ── Quality metrics ──────────────────────────────────────────────────────

    # ESS for mu parameters
    def ess(x):
        n = len(x)
        if n < 10 or np.std(x) == 0:
            return n
        acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
        acf = acf[n-1:] / acf[n-1]
        cutoff = next((i for i in range(1, len(acf)) if acf[i] < 0.05), len(acf))
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        return max(1, n / tau)

    min_ess = float('inf')
    for k_idx in range(K):
        for d_idx in range(D):
            e = ess(mu_draws[:, k_idx, d_idx])
            min_ess = min(min_ess, e)

    # Regime separation significance
    n_sig = 0
    for j in range(D):
        delta = mu_draws[:, panic_state, j] - mu_draws[:, 1-panic_state, j]
        ci_lo, ci_hi = np.percentile(delta, [2.5, 97.5])
        if ci_lo > 0 or ci_hi < 0:
            n_sig += 1

    # Crisis alignment (train only: GFC)
    train_dates_full = panel.loc[panel['date'] < '2011-01-01', 'date'].values
    pi_train = pi_filter[:T]

    gfc_mask = np.zeros(T, dtype=bool)
    non_crisis_mask = np.zeros(T, dtype=bool)

    # Need actual dates for this combo's train data
    sub = panel[['date'] + feat_list].dropna()
    sub = sub[sub['date'] >= '1990-01-01']
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_dates = sub_train['date'].values

    if len(sub_dates) == T:
        gfc_mask = (sub_dates >= np.datetime64('2007-10-01')) & (sub_dates <= np.datetime64('2009-06-01'))
        non_crisis_mask = (sub_dates >= np.datetime64('2003-01-01')) & (sub_dates <= np.datetime64('2006-12-01'))

    gfc_pi = pi_train[gfc_mask].mean() if gfc_mask.sum() > 0 else 0
    non_crisis_pi = pi_train[non_crisis_mask].mean() if non_crisis_mask.sum() > 0 else 1

    # COVID alignment (test)
    sub_test = sub[sub['date'] >= '2011-01-01']
    test_dates = sub_test['date'].values
    T_test = len(sub_test)
    pi_test = pi_filter[T:T+T_test]
    covid_mask = (test_dates >= np.datetime64('2020-02-01')) & (test_dates <= np.datetime64('2020-05-01'))
    covid_pi = pi_test[covid_mask].mean() if covid_mask.sum() > 0 else 0

    return {
        'mu_post': mu_post,
        'panic_state': panic_state,
        'pi_filter': pi_filter,
        'pi_train': pi_train,
        'pi_test': pi_test,
        'min_ess': min_ess,
        'n_sig': n_sig,
        'gfc_pi': gfc_pi,
        'non_crisis_pi': non_crisis_pi,
        'covid_pi': covid_pi,
        'T_train': T,
        'T_test': T_test,
        'dates_all': sub['date'].values,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 1: HMM QUALITY SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(f"  PASS 1: HMM quality screen — {len(all_combos)} combinations, 2000 iterations each")
print("=" * 80)

QUALITY_GATES = {
    'min_ess': 50,       # minimum ESS across all mu parameters
    'n_sig': 2,          # at least N features with significant separation
    'gfc_pi': 0.5,       # avg pi during GFC must exceed this
    'non_crisis_pi': 0.4, # avg pi during non-crisis must be below this
}

screen_results = []
t_start = time.time()

for i, combo in enumerate(all_combos):
    feat_list = list(combo)
    short_name = "+".join(f.replace("_z", "").replace("CP_SPREAD", "CPS") for f in feat_list)

    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01']
    if len(sub) < 200:
        continue

    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']

    if len(sub_train) < 150 or len(sub_test) < 50:
        continue

    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)

    try:
        result = fit_hmm(Z_tr, Z_te, feat_list)

        passed = (result['min_ess'] >= QUALITY_GATES['min_ess'] and
                  result['n_sig'] >= QUALITY_GATES['n_sig'] and
                  result['gfc_pi'] >= QUALITY_GATES['gfc_pi'] and
                  result['non_crisis_pi'] <= QUALITY_GATES['non_crisis_pi'])

        screen_results.append({
            'combo': feat_list,
            'name': short_name,
            'min_ess': result['min_ess'],
            'n_sig': result['n_sig'],
            'gfc_pi': result['gfc_pi'],
            'non_crisis_pi': result['non_crisis_pi'],
            'covid_pi': result['covid_pi'],
            'passed': passed,
        })

        status = "PASS" if passed else "fail"
        if (i + 1) % 10 == 0 or passed:
            elapsed = time.time() - t_start
            n_passed = sum(1 for r in screen_results if r['passed'])
            print(f"  [{i+1:3d}/{len(all_combos)}] {short_name:40s}  "
                  f"ESS={result['min_ess']:6.0f}  sig={result['n_sig']}  "
                  f"GFC={result['gfc_pi']:.2f}  calm={result['non_crisis_pi']:.2f}  "
                  f"COVID={result['covid_pi']:.2f}  {status}  "
                  f"[{elapsed:.0f}s, {n_passed} passed]")
    except Exception as e:
        screen_results.append({
            'combo': feat_list, 'name': short_name,
            'min_ess': 0, 'n_sig': 0, 'gfc_pi': 0, 'non_crisis_pi': 1,
            'covid_pi': 0, 'passed': False,
        })

elapsed = time.time() - t_start
df_screen = pd.DataFrame(screen_results)
n_passed = df_screen['passed'].sum()
print(f"\n  Pass 1 complete: {elapsed:.0f}s")
print(f"  Tested: {len(df_screen)}  |  Passed: {n_passed}")

# Show all passing combos
if n_passed > 0:
    df_pass = df_screen[df_screen['passed']].sort_values('min_ess', ascending=False)
    print(f"\n  Passing combinations:")
    for _, row in df_pass.iterrows():
        print(f"    {row['name']:40s}  ESS={row['min_ess']:6.0f}  sig={row['n_sig']}  "
              f"GFC={row['gfc_pi']:.2f}  calm={row['non_crisis_pi']:.2f}  COVID={row['covid_pi']:.2f}")

# Also show top 10 by ESS regardless of pass/fail
print(f"\n  Top 20 by min ESS (regardless of pass/fail):")
df_top = df_screen.sort_values('min_ess', ascending=False).head(20)
for _, row in df_top.iterrows():
    status = "PASS" if row['passed'] else "fail"
    print(f"    {row['name']:40s}  ESS={row['min_ess']:6.0f}  sig={row['n_sig']}  "
          f"GFC={row['gfc_pi']:.2f}  calm={row['non_crisis_pi']:.2f}  "
          f"COVID={row['covid_pi']:.2f}  {status}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 2: PORTFOLIO TEST (only for passing combos)
# ═══════════════════════════════════════════════════════════════════════════════

# Always include baseline even if it somehow didn't pass
pass_combos = [r['combo'] for r in screen_results if r['passed']]
if list(baseline_combo) not in pass_combos:
    pass_combos.append(list(baseline_combo))

if len(pass_combos) == 0:
    print("\n  No combos passed quality gates. Exiting.")
    exit(0)

print("\n" + "=" * 80)
print(f"  PASS 2: Portfolio test — {len(pass_combos)} combinations")
print("=" * 80)

# ── Prepare stock data for cross-sectional model ────────────────────────────

print("\n  Preparing stock data ...")
stocks_raw = stock_data.copy()
stocks_raw = stocks_raw[stocks_raw['shrcd'].isin([10, 11])]
stocks_raw = stocks_raw[stocks_raw['exchcd'].isin([1, 2, 3])]
stocks_raw = stocks_raw[stocks_raw['prc'].abs() > 1.0]

# Market equity
stocks_raw['me'] = stocks_raw['prc'].abs() * stocks_raw['shrout'] / 1000
stocks_raw['log_me'] = np.log(stocks_raw['me'].clip(lower=1e-6))

# Momentum lookbacks
stocks_raw['_log_ret'] = np.log1p(stocks_raw['ret_adj'].clip(lower=-0.999))
stocks_raw['_log_ret_s1'] = stocks_raw.groupby('permno')['_log_ret'].shift(1)
for lb in range(1, 13):
    roll_sum = (
        stocks_raw.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks_raw[f'mom_{lb}'] = np.expm1(roll_sum)

# Forward return
stocks_raw['ret_fwd'] = stocks_raw.groupby('permno')['ret_adj'].shift(-1)

MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth', 'gross_profit_a']
CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me'] + FUND_FEATURES
ALL_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me'] + FUND_FEATURES

TRADING_FEE = 0.001


def long_only_port(df_test, score_col, fee=TRADING_FEE):
    monthly, prev_weights = [], {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0:
            continue
        total_me = longs['me'].sum()
        new_w = (longs.set_index('permno')['me'] / total_me).to_dict()
        turnover = sum(abs(new_w.get(p, 0) - prev_weights.get(p, 0))
                       for p in set(new_w) | set(prev_weights)) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_w
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']



def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly, prev_lw, prev_sw = [], {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0)) for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0)) for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']


def compute_metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 12:
        return 0, 0, 0, 0
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


def run_portfolio_test(pi_df):
    stocks_with_pi = stocks_raw.merge(pi_df[['date', 'pi_filter']], on='date', how='left')
    stocks_with_pi['pi_filter'] = stocks_with_pi['pi_filter'].ffill()
    df_cs = stocks_with_pi.dropna(subset=['ret_fwd'] + MOM_FEATURES + ['pi_filter', 'log_me']).copy()
    train_cs = df_cs[df_cs['date'] < '2011-01-01'].copy()
    test_cs = df_cs[df_cs['date'] >= '2011-01-01'].copy()
    if len(train_cs) < 1000 or len(test_cs) < 1000:
        return None

    results = {}

    # M0: Formula
    pi_by_month = test_cs.groupby('date')['pi_filter'].first()
    lb_by_month = np.clip(np.round(12 - 11 * pi_by_month).astype(int), 1, 12)
    test_cs['score_m0'] = np.nan
    for date, lb in lb_by_month.to_dict().items():
        mask = test_cs['date'] == date
        test_cs.loc[mask, 'score_m0'] = test_cs.loc[mask, f'mom_{lb}']
    r_m0 = long_short_port(test_cs, 'score_m0')
    if len(r_m0) > 12:
        results['M0'] = compute_metrics(r_m0)

    # M1: LR
    train_imp = train_cs.copy()
    test_imp = test_cs.copy()
    for col in FUND_FEATURES:
        med = train_imp[col].median()
        train_imp[col] = train_imp[col].fillna(med)
        test_imp[col] = test_imp[col].fillna(med)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(train_imp[ALL_FEATURES].values.astype(float))
    X_te_s = scaler.transform(test_imp[ALL_FEATURES].values.astype(float))
    train_imp['above_med'] = train_imp.groupby('date')['ret_fwd'].transform(
        lambda x: (x > x.median()).astype(int))
    lr = LR_sklearn(max_iter=1000, solver='lbfgs', random_state=42)
    lr.fit(X_tr_s, train_imp['above_med'].values)
    test_imp['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
    r_lr = long_short_port(test_imp, 'score_lr')
    if len(r_lr) > 12:
        results['M1_LR'] = compute_metrics(r_lr)

    # M2: XGB
    try:
        from xgboost import XGBRegressor
        X_tr_xgb = train_cs[ALL_FEATURES].values.astype(float)
        X_te_xgb = test_cs[ALL_FEATURES].values.astype(float)
        xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_tr_xgb, train_cs['ret_fwd'].values.astype(float))
        test_cs['score_xgb'] = xgb.predict(X_te_xgb)
        r_xgb = long_short_port(test_cs, 'score_xgb')
        if len(r_xgb) > 12:
            results['M2_XGB'] = compute_metrics(r_xgb)
    except:
        pass

    return results


# ── Run portfolio tests ──────────────────────────────────────────────────────

portfolio_results = {}
t_start = time.time()

for i, feat_list in enumerate(pass_combos):
    short_name = "+".join(f.replace("_z", "").replace("CP_SPREAD", "CPS") for f in feat_list)
    is_baseline = set(feat_list) == set(baseline_combo)
    print(f"\n  [{i+1}/{len(pass_combos)}] {short_name}" +
          (" (BASELINE)" if is_baseline else ""), end=" ... ", flush=True)

    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01']
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']
    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)

    result = fit_hmm(Z_tr, Z_te, feat_list)

    pi_df = pd.DataFrame({'date': sub['date'].values, 'pi_filter': result['pi_filter']})
    port_res = run_portfolio_test(pi_df)

    if port_res:
        portfolio_results[short_name] = port_res
        for method, (ar, av, sr, mdd) in port_res.items():
            print(f"{method}:SR={sr:.3f}", end="  ")
        print()
    else:
        print("FAILED")

elapsed = time.time() - t_start
print(f"\n  Pass 2 complete in {elapsed:.0f}s")

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  RESULTS SUMMARY")
print("=" * 80)
print(f"\n  {'Combination':40s}  {'M0 SR':>7s}  {'M1 SR':>7s}  {'M2 SR':>7s}")
print("  " + "-" * 70)

for name, res in sorted(portfolio_results.items(),
                         key=lambda x: x[1].get('M2_XGB', (0,0,0,0))[2], reverse=True):
    m0_sr = res.get('M0', (0,0,0,0))[2]
    m1_sr = res.get('M1_LR', (0,0,0,0))[2]
    m2_sr = res.get('M2_XGB', (0,0,0,0))[2]
    bl = " (BASELINE)" if name == "+".join(f.replace("_z", "") for f in baseline_combo) else ""
    print(f"  {name+bl:40s}  {m0_sr:7.3f}  {m1_sr:7.3f}  {m2_sr:7.3f}")

print("\n" + "=" * 80)
