"""
test_hmm_ablation.py
====================
Feature importance ablation for HMM inputs.

Two analyses:
1. Single-feature HMMs: run pipeline with each feature alone
   -> which feature contributes most on its own
2. Leave-one-out: drop one feature at a time from the 5-feature combo
   -> which feature's removal hurts most

Uses DD, VOL, CS, DISP, REL_N (the 5-feature set).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.preprocessing import StandardScaler
import warnings, time
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  HMM FEATURE ABLATION STUDY")
print("=" * 80)

panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
panel['year_month'] = panel['date'].dt.to_period('M')

stock_data = pd.read_parquet('data/crsp_msf_raw.parquet')
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['year_month'] = stock_data['date'].dt.to_period('M')

train_mask = panel['date'] < '2011-01-01'

print("\nComputing features ...")

if 'DISP' not in panel.columns:
    disp = stock_data.groupby('year_month')['ret_adj'].std().reset_index()
    disp.columns = ['year_month', 'DISP']
    disp['DISP'] = np.log(disp['DISP'])
    panel = panel.merge(disp, on='year_month', how='left')
    print("  DISP: OK")

if 'REL_N' not in panel.columns:
    n_stocks = stock_data.groupby('year_month')['permno'].nunique().reset_index()
    n_stocks.columns = ['year_month', 'N_STOCKS']
    n_stocks = n_stocks.sort_values('year_month')
    n_stocks['N_STOCKS_MA12'] = n_stocks['N_STOCKS'].rolling(12).mean()
    n_stocks['REL_N'] = np.log(n_stocks['N_STOCKS'] / n_stocks['N_STOCKS_MA12'])
    panel = panel.merge(n_stocks[['year_month', 'REL_N']], on='year_month', how='left')
    print("  REL_N: OK")

for col in ['DISP', 'REL_N']:
    zcol = f'{col}_z'
    if col in panel.columns and zcol not in panel.columns:
        vals = panel[col].astype(float)
        mu = vals[train_mask].dropna().mean()
        sd = vals[train_mask].dropna().std()
        if sd > 0 and not np.isnan(sd):
            panel[zcol] = (vals - mu) / sd

BASE_5 = ['DD_z', 'VOL_z', 'CS_z', 'DISP_z', 'REL_N_z']

# Single-feature tests
SINGLE = {f'Single: {f.replace("_z","")}': [f] for f in BASE_5}

# Leave-one-out tests
LOO = {}
for f in BASE_5:
    remaining = [x for x in BASE_5 if x != f]
    LOO[f'LOO: drop {f.replace("_z","")}'] = remaining

# Full 5-feature
FULL = {'Full 5F: DD+VOL+CS+DISP+REL_N': BASE_5}

ALL_TESTS = {}
ALL_TESTS.update(SINGLE)
ALL_TESTS.update(LOO)
ALL_TESTS.update(FULL)

SEEDS = [2201, 42, 1337]  # 3 seeds for speed

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

K = 2

def log_emission(Z, mu, Sigma):
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])

def ffbs(Z, mu, Sigma, P):
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
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
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
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


def fit_hmm(Z_train, Z_test, feat_list, dates_train, seed=2201):
    np.random.seed(seed)
    if Z_train.ndim == 1:
        Z_train = Z_train.reshape(-1, 1)
        Z_test = Z_test.reshape(-1, 1)
    T, D = Z_train.shape

    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    init_idx = feat_list.index('VOL_z') if 'VOL_z' in feat_list else 0
    states = (Z_train[:, init_idx] > np.median(Z_train[:, init_idx])).astype(int)

    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            if D > 1:
                Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
            else:
                Sigma[k] = np.array([[Z_train[idx].var() + 1e-6]])
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    n_iter, n_burnin = 2000, 500
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
            if D > 1:
                S_k = (Z_k - x_bar).T @ (Z_k - x_bar)
            else:
                S_k = np.array([[(Z_k.ravel() - x_bar[0]) @ (Z_k.ravel() - x_bar[0])]])
            kappa_n = kappa_0 + n_k
            m_n = (kappa_0 * m_0 + n_k * x_bar) / kappa_n
            nu_n = nu_0 + n_k
            Psi_n = Psi_0 + S_k + (kappa_0 * n_k / kappa_n) * np.outer(x_bar - m_0, x_bar - m_0)
            try:
                Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
                if D == 1:
                    Sigma[k] = np.atleast_2d(Sigma[k])
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
    if D > 1:
        Sigma_post = np.array([np.cov(Z_train[states == k].T) + 1e-6 * np.eye(D)
                               for k in range(K)])
    else:
        Sigma_post = np.array([[[Z_train[states == k].var() + 1e-6]] for k in range(K)])

    Z_full = np.vstack([Z_train, Z_test])
    filtered = forward_filter(Z_full, mu_post, Sigma_post, P)

    # Crisis-calibrated sign correction
    crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
    crisis_mask = np.zeros(T, dtype=bool)
    for s, e in crisis_windows:
        crisis_mask |= ((dates_train >= np.datetime64(s)) & (dates_train <= np.datetime64(e)))

    if crisis_mask.sum() > 5:
        signs = np.zeros(D)
        for j in range(D):
            signs[j] = 1.0 if np.percentile(Z_train[crisis_mask, j], 95) >= np.percentile(Z_train[~crisis_mask, j], 95) else -1.0
        score0 = np.sum(signs * mu_post[0])
        score1 = np.sum(signs * mu_post[1])
        panic_state = 1 if score1 >= score0 else 0
    else:
        panic_state = 1 if np.sum(mu_post[1]**2) >= np.sum(mu_post[0]**2) else 0

    pi_filter = filtered[:, panic_state]

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
            min_ess = min(min_ess, ess(mu_draws[:, k_idx, d_idx]))

    return {
        'pi_filter': pi_filter,
        'min_ess': min_ess,
        'mu_post': mu_post,
        'panic_state': panic_state,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK DATA PREP
# ═══════════════════════════════════════════════════════════════════════════════

print("\nPreparing stock data ...")
stocks_raw = stock_data.copy()
stocks_raw = stocks_raw[stocks_raw['shrcd'].isin([10, 11])]
stocks_raw = stocks_raw[stocks_raw['exchcd'].isin([1, 2, 3])]
stocks_raw = stocks_raw[stocks_raw['prc'].abs() > 1.0]
stocks_raw['me'] = stocks_raw['prc'].abs() * stocks_raw['shrout'] / 1000
stocks_raw['log_me'] = np.log(stocks_raw['me'].clip(lower=1e-6))
stocks_raw['_log_ret'] = np.log1p(stocks_raw['ret_adj'].clip(lower=-0.999))
stocks_raw['_log_ret_s1'] = stocks_raw.groupby('permno')['_log_ret'].shift(1)
for lb in range(1, 13):
    roll_sum = (
        stocks_raw.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks_raw[f'mom_{lb}'] = np.expm1(roll_sum)
stocks_raw['ret_fwd'] = stocks_raw.groupby('permno')['ret_adj'].shift(-1)

MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth', 'gross_profit_a']
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


def run_xgb(pi_df):
    stocks_with_pi = stocks_raw.merge(pi_df[['date', 'pi_filter']], on='date', how='left')
    stocks_with_pi['pi_filter'] = stocks_with_pi['pi_filter'].ffill()
    df_cs = stocks_with_pi.dropna(subset=['ret_fwd'] + MOM_FEATURES + ['pi_filter', 'log_me']).copy()
    train_cs = df_cs[df_cs['date'] < '2011-01-01'].copy()
    test_cs = df_cs[df_cs['date'] >= '2011-01-01'].copy()
    if len(train_cs) < 1000 or len(test_cs) < 1000:
        return None
    from xgboost import XGBRegressor
    X_tr = train_cs[ALL_FEATURES].values.astype(float)
    X_te = test_cs[ALL_FEATURES].values.astype(float)
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, train_cs['ret_fwd'].values.astype(float))
    test_cs['score_xgb'] = xgb.predict(X_te)
    r_xgb = long_only_port(test_cs, 'score_xgb')
    if len(r_xgb) > 12:
        return compute_metrics(r_xgb)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(f"  RUNNING {len(ALL_TESTS)} TESTS x {len(SEEDS)} SEEDS")
print("=" * 80)

all_results = {}
t_global = time.time()

for test_name, feat_list in ALL_TESTS.items():
    print(f"\n  {test_name}  (features: {', '.join(f.replace('_z','') for f in feat_list)})")

    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01']
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']
    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)
    dates_train = sub_train['date'].values
    dates_all = sub['date'].values

    seed_sharpes = []
    for seed in SEEDS:
        t0 = time.time()
        hmm_res = fit_hmm(Z_tr, Z_te, feat_list, dates_train, seed=seed)
        pi_df = pd.DataFrame({'date': dates_all, 'pi_filter': hmm_res['pi_filter']})
        xgb_res = run_xgb(pi_df)
        elapsed = time.time() - t0

        if xgb_res:
            sr = xgb_res[2]
            seed_sharpes.append(sr)
            print(f"    seed={seed:5d}  SR={sr:.3f}  ESS={hmm_res['min_ess']:.0f}  ({elapsed:.0f}s)")
        else:
            print(f"    seed={seed:5d}  FAILED  ({elapsed:.0f}s)")

    if seed_sharpes:
        all_results[test_name] = {
            'mean_sr': np.mean(seed_sharpes),
            'std_sr': np.std(seed_sharpes),
            'min_sr': np.min(seed_sharpes),
            'max_sr': np.max(seed_sharpes),
            'features': feat_list,
        }

elapsed_total = time.time() - t_global
print(f"\n  Total time: {elapsed_total:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SINGLE-FEATURE RESULTS (which feature matters most alone)")
print("=" * 80)
print(f"\n  {'Test':30s}  {'Mean SR':>8s}  {'Std SR':>7s}  {'Min':>6s}  {'Max':>6s}")
print("  " + "-" * 60)
for name, res in sorted(all_results.items(), key=lambda x: x[1]['mean_sr'], reverse=True):
    if name.startswith('Single'):
        print(f"  {name:30s}  {res['mean_sr']:8.3f}  {res['std_sr']:7.3f}  "
              f"{res['min_sr']:6.3f}  {res['max_sr']:6.3f}")

print("\n" + "=" * 80)
print("  LEAVE-ONE-OUT RESULTS (which feature's removal hurts most)")
print("=" * 80)

full_sr = all_results.get('Full 5F: DD+VOL+CS+DISP+REL_N', {}).get('mean_sr', 0)
print(f"\n  Full 5F baseline: Mean SR = {full_sr:.3f}")
print(f"\n  {'Test':30s}  {'Mean SR':>8s}  {'Drop':>7s}  {'Std SR':>7s}")
print("  " + "-" * 60)
for name, res in sorted(all_results.items(), key=lambda x: x[1]['mean_sr']):
    if name.startswith('LOO'):
        drop = full_sr - res['mean_sr']
        print(f"  {name:30s}  {res['mean_sr']:8.3f}  {drop:+7.3f}  {res['std_sr']:7.3f}")

print("\n  Interpretation:")
print("  - Largest positive 'Drop' = most important feature (removing it hurts most)")
print("  - Negative 'Drop' = feature is actually hurting performance")

print("\n" + "=" * 80)
print("  FULL COMPARISON")
print("=" * 80)
print(f"\n  {'Test':40s}  {'Mean SR':>8s}  {'Std SR':>7s}")
print("  " + "-" * 60)
for name, res in sorted(all_results.items(), key=lambda x: x[1]['mean_sr'], reverse=True):
    print(f"  {name:40s}  {res['mean_sr']:8.3f}  {res['std_sr']:7.3f}")

print("\n" + "=" * 80)
