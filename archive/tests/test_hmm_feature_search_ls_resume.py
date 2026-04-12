"""
test_hmm_feature_search_ls.py
=============================
Exhaustive search over C(9,4)=126 four-feature HMM combinations for long-short.
Uses the EXACT same panic identification and quality gates as hmm_model.py.

Pass 1: HMM quality screen (1 seed, 2000 iter per combo)
Pass 2: Portfolio test (5 HMM seeds averaged, 10 XGB seeds averaged, mom+pi only, L/S)
"""

import numpy as np
import pandas as pd
import pickle, warnings, time
from itertools import combinations
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRADING_FEE, TRAIN_END, N_ESTIMATORS, MAX_DEPTH,
                    LEARNING_RATE, SUBSAMPLE, COLSAMPLE, MOM_FEATURES)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  HMM FEATURE SEARCH — LONG-SHORT (FULL PIPELINE MATCH)")
print("=" * 80)

t_start_total = time.time()

panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
panel['year_month'] = panel['date'].dt.to_period('M')
train_mask = panel['date'] < TRAIN_END

# Load extra features (ADR, SKEW, TERM)
extra = pd.read_pickle('data/extra_hmm_features.pkl')
panel = panel.merge(extra, on='year_month', how='left')

# Standardize extra features using training-period stats only
for col in ['ADR', 'SKEW', 'TERM']:
    if col in panel.columns:
        vals = panel[col].astype(float)
        mu, sd = vals[train_mask].dropna().mean(), vals[train_mask].dropna().std()
        if sd > 0:
            panel[f'{col}_z'] = (vals - mu) / sd

# Stock data for portfolio construction
stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks[stocks['shrcd'].isin([10, 11]) & stocks['exchcd'].isin([1, 2, 3]) & (stocks['prc'].abs() > 1)]
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

# Momentum features
MOM_LBS = list(range(1, 13))
stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)
for lb in MOM_LBS:
    roll_sum = (stocks.groupby('permno', sort=False)['_log_ret_s1']
                .rolling(lb, min_periods=lb).sum()
                .reset_index(level='permno', drop=True).sort_index())
    stocks[f'mom_{lb}'] = np.expm1(roll_sum)
stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)
stocks['log_me'] = np.log(stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan))
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

# Merge pi_filter placeholder (will be replaced per combo)
stocks = stocks.dropna(subset=['ret_fwd'] + MOM_FEATURES).copy().reset_index(drop=True)
train_stocks = stocks[stocks['date'] < TRAIN_END].copy()
test_stocks = stocks[stocks['date'] >= TRAIN_END].copy()

print(f"  Train stocks: {len(train_stocks):,}  |  Test stocks: {len(test_stocks):,}")

# Feature pool
FEATURE_POOL = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z', 'CS_z', 'LVIX_z', 'ADR_z', 'SKEW_z', 'TERM_z']
available = [f for f in FEATURE_POOL if f in panel.columns and panel[f].notna().sum() > 200]

print(f"\n  Feature pool: {len(available)} features")
for f in available:
    n_valid = panel[f].notna().sum()
    dates = panel.loc[panel[f].notna(), 'date']
    print(f"    {f:<12s} ({n_valid} months, {dates.min().date()} - {dates.max().date()})")

all_combos = list(combinations(available, 4))
print(f"\n  Total 4-feature combinations: {len(all_combos)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS (exact match to hmm_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

K = 2

def log_emission(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)])

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

def fit_hmm(Z_train, Z_full, feat_list, train_dates, n_iter=2000, n_burnin=500, seed=42):
    """
    Fit 2-state HMM with EXACT same settings and panic identification as hmm_model.py.
    Returns dict with pi_filter, quality metrics, and panic identification details.
    """
    np.random.seed(seed)
    T, D = Z_train.shape

    # Priors (exact match to hmm_model.py)
    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    # Initialize: use first feature for initial state assignment
    # (above median = state 1, below = state 0)
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
            mu_draws[m - n_burnin] = mu

    mu_post = mu_draws.mean(axis=0)
    Sigma_post = Sigma  # use last draw for filtering
    P_post = P

    # ── Panic identification (EXACT hmm_model.py method) ──
    # Step 1: determine sign direction for each feature using crisis windows
    crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
    crisis_mask = np.zeros(T, dtype=bool)
    for s, e in crisis_windows:
        crisis_mask |= ((train_dates >= np.datetime64(s)) & (train_dates <= np.datetime64(e)))

    signs = np.zeros(D)
    if crisis_mask.sum() > 5:
        for j in range(D):
            p95_crisis = np.percentile(Z_train[crisis_mask, j], 95)
            p95_normal = np.percentile(Z_train[~crisis_mask, j], 95)
            signs[j] = 1.0 if p95_crisis >= p95_normal else -1.0
    else:
        signs = np.ones(D)

    # Step 2: sign-corrected score to identify panic state
    score0 = np.sum(signs * mu_post[0])
    score1 = np.sum(signs * mu_post[1])
    panic_state = 1 if score1 >= score0 else 0

    # ── Filter full sample ──
    Z_full_data = Z_full
    filtered = forward_filter(Z_full_data, mu_post, Sigma_post, P_post)
    pi_filter = filtered[:, panic_state]

    # ── Quality metrics ──

    # ESS
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

    return {
        'pi_filter': pi_filter,
        'mu_post': mu_post,
        'panic_state': panic_state,
        'signs': signs,
        'min_ess': min_ess,
        'n_sig': n_sig,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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

def compute_sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 12 or r.std() == 0:
        return 0
    return r.mean() / r.std() * np.sqrt(12)

# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 1: HMM QUALITY SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(f"  PASS 1: HMM quality screen — {len(all_combos)} combinations")
print("=" * 80)

# Quality gates
QUALITY_GATES = {
    'min_ess': 50,
    'n_sig': 2,
    'gfc_pi_min': 0.5,
    'covid_pi_min': 0.5,
    'dotcom_pi_min': 0.3,
    'non_crisis_pi_max': 0.4,
    'calm_2013_2019_pi_max': 0.4,
}

print(f"\n  Quality gates:")
for k, v in QUALITY_GATES.items():
    print(f"    {k}: {v}")

# Crisis/calm period definitions
VALIDATION_PERIODS = [
    ('Dot-com',     '2000-03-01', '2002-10-01', 'crisis',  QUALITY_GATES['dotcom_pi_min']),
    ('GFC',         '2007-10-01', '2009-06-01', 'crisis',  QUALITY_GATES['gfc_pi_min']),
    ('Non-crisis',  '2003-01-01', '2006-12-31', 'calm',    QUALITY_GATES['non_crisis_pi_max']),
    ('COVID',       '2020-02-01', '2020-05-31', 'crisis',  QUALITY_GATES['covid_pi_min']),
    ('Calm 2013-19','2013-01-01', '2019-12-31', 'calm',    QUALITY_GATES['calm_2013_2019_pi_max']),
]

screen_results = []
t_pass1 = time.time()

for i, combo in enumerate(all_combos):
    feat_list = list(combo)
    short_name = "+".join(f.replace("_z", "") for f in feat_list)

    # Prepare HMM data
    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
    if len(sub) < 200:
        continue

    sub_train = sub[sub['date'] < TRAIN_END]
    sub_test = sub[sub['date'] >= TRAIN_END]
    if len(sub_train) < 150 or len(sub_test) < 50:
        continue

    Z_tr = sub_train[feat_list].values.astype(float)
    Z_full = sub[feat_list].values.astype(float)
    train_dates = sub_train['date'].values
    all_dates = sub['date'].values
    T_train = len(sub_train)

    try:
        result = fit_hmm(Z_tr, Z_full, feat_list, train_dates)
    except Exception as e:
        continue

    # Compute pi for validation periods
    pi_full = result['pi_filter']
    period_results = {}
    all_passed = True

    for pname, pstart, pend, ptype, threshold in VALIDATION_PERIODS:
        mask = (all_dates >= np.datetime64(pstart)) & (all_dates <= np.datetime64(pend))
        if mask.sum() == 0:
            period_results[pname] = (np.nan, True)
            continue
        mean_pi = pi_full[mask].mean()
        if ptype == 'crisis':
            passed = mean_pi >= threshold
        else:
            passed = mean_pi <= threshold
        period_results[pname] = (mean_pi, passed)
        if not passed:
            all_passed = False

    # Check MCMC quality
    mcmc_passed = (result['min_ess'] >= QUALITY_GATES['min_ess'] and
                   result['n_sig'] >= QUALITY_GATES['n_sig'])

    overall_passed = mcmc_passed and all_passed

    screen_results.append({
        'combo': feat_list,
        'name': short_name,
        'min_ess': result['min_ess'],
        'n_sig': result['n_sig'],
        'periods': period_results,
        'passed': overall_passed,
    })

    status = "PASS" if overall_passed else "fail"
    period_str = "  ".join(f"{k}={v[0]:.2f}" for k, v in period_results.items() if not np.isnan(v[0]))
    n_passed = sum(1 for r in screen_results if r['passed'])

    if (i + 1) % 10 == 0 or overall_passed:
        print(f"  [{i+1:>3d}/{len(all_combos)}] {short_name:<35s}  "
              f"ESS={result['min_ess']:>6.0f}  sig={result['n_sig']}  "
              f"{period_str}  {status}  [{time.time()-t_pass1:.0f}s, {n_passed} passed]")

elapsed_pass1 = time.time() - t_pass1
n_passed_total = sum(1 for r in screen_results if r['passed'])
print(f"\n  Pass 1 complete: {elapsed_pass1:.0f}s")
print(f"  Tested: {len(screen_results)}  |  Passed: {n_passed_total}")

if n_passed_total == 0:
    print("\n  No combos passed quality gates. Exiting.")
    exit(0)

# Show all passing combos
pass_combos = [r for r in screen_results if r['passed']]
print(f"\n  Passing combinations ({len(pass_combos)}):")
for r in pass_combos:
    period_str = "  ".join(f"{k}={v[0]:.2f}" for k, v in r['periods'].items() if not np.isnan(v[0]))
    print(f"    {r['name']:<35s}  ESS={r['min_ess']:>6.0f}  sig={r['n_sig']}  {period_str}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 2: PORTFOLIO TEST (survivors only)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(f"  PASS 2: Portfolio test — {len(pass_combos)} combinations")
print(f"  5 HMM seeds averaged, 10 XGB seeds averaged, mom+pi only, L/S")
print("=" * 80)

HMM_SEEDS_SEARCH = [42, 2201, 1337, 314, 789]
XGB_SEEDS_SEARCH = [42, 123, 456, 789, 999, 2201, 1337, 314, 7777, 55]
REDUCED_FEATURES = MOM_FEATURES + ['pi_filter']


DONE_COMBOS = {'DD+VOL+DISP+REL_N', 'DD+VOL+DISP+CS', 'DD+VOL+DISP+ADR', 'DD+VOL+DISP+SKEW', 'DD+VOL+DISP+TERM'}
DONE_RESULTS = [
    {'name': 'DD+VOL+DISP+REL_N', 'combo': ['DD_z','VOL_z','DISP_z','REL_N_z'], 'sh_lr': -0.098, 'sh_xgb': 0.758, 'min_ess': 526, 'n_sig': 4},
    {'name': 'DD+VOL+DISP+CS', 'combo': ['DD_z','VOL_z','DISP_z','CS_z'], 'sh_lr': -0.055, 'sh_xgb': 0.385, 'min_ess': 137, 'n_sig': 4},
    {'name': 'DD+VOL+DISP+ADR', 'combo': ['DD_z','VOL_z','DISP_z','ADR_z'], 'sh_lr': -0.061, 'sh_xgb': 0.613, 'min_ess': 89, 'n_sig': 4},
    {'name': 'DD+VOL+DISP+SKEW', 'combo': ['DD_z','VOL_z','DISP_z','SKEW_z'], 'sh_lr': -0.075, 'sh_xgb': 0.322, 'min_ess': 254, 'n_sig': 4},
    {'name': 'DD+VOL+DISP+TERM', 'combo': ['DD_z','VOL_z','DISP_z','TERM_z'], 'sh_lr': -0.079, 'sh_xgb': 0.911, 'min_ess': 232, 'n_sig': 3},
]
portfolio_results = list(DONE_RESULTS)
t_pass2 = time.time()

for idx, screen_result in enumerate(pass_combos):
    feat_list = screen_result['combo']
    short_name = screen_result['name']
    t0 = time.time()

    # Prepare HMM data
    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
    sub_train = sub[sub['date'] < TRAIN_END]
    Z_tr = sub_train[feat_list].values.astype(float)
    Z_full = sub[feat_list].values.astype(float)
    train_dates = sub_train['date'].values
    all_dates = sub['date'].values

    # Average pi_filter across 5 HMM seeds
    pi_accum = np.zeros(len(Z_full))
    for hmm_seed in HMM_SEEDS_SEARCH:
        result = fit_hmm(Z_tr, Z_full, feat_list, train_dates, seed=hmm_seed)
        pi_accum += result['pi_filter']
    pi_avg = pi_accum / len(HMM_SEEDS_SEARCH)

    # Merge pi_filter into stock data
    pi_df = pd.DataFrame({'date': all_dates, 'pi_filter': pi_avg})

    tr = train_stocks.drop(columns=['pi_filter'], errors='ignore').merge(pi_df, on='date', how='left')
    tr['pi_filter'] = tr['pi_filter'].ffill().fillna(0.5)
    te = test_stocks.drop(columns=['pi_filter'], errors='ignore').merge(pi_df, on='date', how='left')
    te['pi_filter'] = te['pi_filter'].ffill().fillna(0.5)

    X_train = tr[REDUCED_FEATURES].values.astype(float)
    X_test = te[REDUCED_FEATURES].values.astype(float)
    y_train = tr['ret_fwd'].values.astype(float)

    # M1: Logistic Regression
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(imp.fit_transform(X_train))
    X_te_s = scaler.transform(imp.transform(X_test))
    tr_c = tr.copy()
    tr_c['above_med'] = tr_c.groupby('date')['ret_fwd'].transform(lambda x: (x >= x.median()).astype(int))
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_tr_s, tr_c['above_med'].values)
    te['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
    r_lr = long_short_port(te, 'score_lr')
    sh_lr = compute_sharpe(r_lr)

    # M2: XGBoost ensemble (average predictions across 10 seeds)
    xgb_preds = np.zeros(len(X_test))
    for xgb_seed in XGB_SEEDS_SEARCH:
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                           learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE, tree_method='hist',
                           random_state=xgb_seed, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_preds += xgb.predict(X_test)
    xgb_preds /= len(XGB_SEEDS_SEARCH)

    te['score_xgb'] = xgb_preds
    r_xgb = long_short_port(te, 'score_xgb')
    sh_xgb = compute_sharpe(r_xgb)

    elapsed = time.time() - t0
    portfolio_results.append({
        'name': short_name,
        'combo': feat_list,
        'sh_lr': sh_lr,
        'sh_xgb': sh_xgb,
        'min_ess': screen_result['min_ess'],
        'n_sig': screen_result['n_sig'],
    })

    print(f"  [{idx+1:>3d}/{len(pass_combos)}] {short_name:<35s}  "
          f"M1:SR={sh_lr:.3f}  M2:SR={sh_xgb:.3f}  [{elapsed:.0f}s]")

# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

# Sort by M2 XGB Sharpe
portfolio_results.sort(key=lambda x: -x['sh_xgb'])

print("\n" + "=" * 80)
print("  FINAL RANKINGS (by M2 XGB Sharpe)")
print("=" * 80)

print(f"\n  {'Rank':>4s}  {'Features':<35s}  {'M1 LR':>7s}  {'M2 XGB':>7s}  {'ESS':>6s}  {'Sig':>4s}")
print(f"  {'-'*70}")

for rank, r in enumerate(portfolio_results, 1):
    marker = ''
    if r['name'] == 'DD+VOL+DISP+REL_N':
        marker = ' <-- L/O winner'
    print(f"  {rank:>4d}  {r['name']:<35s}  {r['sh_lr']:>7.3f}  {r['sh_xgb']:>7.3f}  "
          f"{r['min_ess']:>6.0f}  {r['n_sig']:>4d}{marker}")

# Save results
df_results = pd.DataFrame(portfolio_results)
df_results.to_csv('hmm_feature_search_ls_final.csv', index=False)
print(f"\nSaved: hmm_feature_search_ls_final.csv")

# Print top 5 details
print("\n  TOP 5:")
for i, r in enumerate(portfolio_results[:5], 1):
    print(f"    {i}. {r['name']}: M2 Sharpe = {r['sh_xgb']:.3f}")

elapsed_total = time.time() - t_start_total
print(f"\nTotal time: {elapsed_total/60:.1f} minutes")
