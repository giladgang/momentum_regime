"""
test_hmm_feature_ablation.py
============================
Test individual HMM input features: build a single-feature HMM for each,
then run M1 (LR) and M2 (XGB) to see which features produce a strong
enough regime signal for downstream portfolio performance.

Features tested: DD, VOL, DISP, REL_N (the 4 used in the main model)
Also tests: full 4-feature HMM (baseline)
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings, time
warnings.filterwarnings('ignore')

print("=" * 80)
print("  HMM SINGLE-FEATURE ABLATION: M1 (LR) + M2 (XGB)")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

# Momentum
MOM_LBS = list(range(1, 13))
stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)
for lb in MOM_LBS:
    roll_sum = (
        stocks.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks[f'mom_{lb}'] = np.expm1(roll_sum)
stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)
stocks['log_me'] = np.log(
    stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth',
                 'gross_profit_a', 'log_me']
FEATURES_WITH_PI = MOM_FEATURES + ['pi_filter'] + FUND_FEATURES
CORE_DROP = MOM_FEATURES + ['log_me', 'ret_fwd']

TRADING_FEE = 0.001
K = 2

print(f"  Stocks: {len(stocks):,} rows")

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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


def fit_hmm(Z_train, Z_full, dates_train, seed=42):
    np.random.seed(seed)
    if Z_train.ndim == 1:
        Z_train = Z_train.reshape(-1, 1)
        Z_full = Z_full.reshape(-1, 1)
    T_tr, D = Z_train.shape

    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    # Init by median split on first feature
    states = (Z_train[:, 0] > np.median(Z_train[:, 0])).astype(int)
    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D) if D > 1 \
                else np.array([[Z_train[idx].var() + 1e-6]])
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    n_iter, n_burnin = 2000, 500
    for m in range(n_iter):
        states = ffbs(Z_train, mu, Sigma, P)
        for k in range(K):
            Z_k = Z_train[states == k]
            n_k = len(Z_k)
            if n_k < D + 2:
                continue
            x_bar = Z_k.mean(axis=0)
            S_k = (Z_k - x_bar).T @ (Z_k - x_bar) if D > 1 \
                else np.array([[(Z_k.ravel() - x_bar[0]) @ (Z_k.ravel() - x_bar[0])]])
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

    # Posterior means (use last iteration as approximation for speed)
    mu_post = mu.copy()
    Sigma_post = Sigma.copy()

    filtered = forward_filter(Z_full, mu_post, Sigma_post, P)

    # Crisis-calibrated sign correction
    crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
    crisis_mask = np.zeros(T_tr, dtype=bool)
    for s, e in crisis_windows:
        crisis_mask |= ((dates_train >= np.datetime64(s)) & (dates_train <= np.datetime64(e)))

    if crisis_mask.sum() > 5:
        signs = np.zeros(D)
        for j in range(D):
            signs[j] = 1.0 if np.percentile(Z_train[crisis_mask, j], 95) >= \
                np.percentile(Z_train[~crisis_mask, j], 95) else -1.0
        panic_state = 1 if np.sum(signs * mu_post[1]) >= np.sum(signs * mu_post[0]) else 0
    else:
        panic_state = 1 if np.sum(mu_post[1]**2) >= np.sum(mu_post[0]**2) else 0

    return filtered[:, panic_state]


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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


def run_m1_m2(stocks_with_pi):
    """Run both LR (M1) and XGB (M2) with the given pi_filter signal."""
    df = stocks_with_pi.dropna(subset=CORE_DROP).copy().reset_index(drop=True)
    train = df[df['date'] < '2011-01-01'].copy()
    test  = df[df['date'] >= '2011-01-01'].copy()

    if len(train) < 1000 or len(test) < 1000:
        return None, None

    results = {}

    # ── M1: Logistic Regression ──
    scaler = StandardScaler()
    X_tr = train[FEATURES_WITH_PI].values.astype(float)
    X_te = test[FEATURES_WITH_PI].values.astype(float)

    # Impute NaN with column median (training set)
    for j in range(X_tr.shape[1]):
        col_median = np.nanmedian(X_tr[:, j])
        X_tr[np.isnan(X_tr[:, j]), j] = col_median
        X_te[np.isnan(X_te[:, j]), j] = col_median

    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Binary target: above cross-sectional median
    train['above_med'] = train.groupby('date')['ret_fwd'].transform(
        lambda x: (x >= x.median()).astype(int))
    y_cls = train['above_med'].values

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_cls)
    test['score_lr'] = lr.predict_proba(X_te_s)[:, 1]

    r_lr = long_only_port(test, 'score_lr')
    if len(r_lr) > 12:
        results['M1_LR'] = compute_metrics(r_lr)

    # ── M2: XGBoost ──
    xgb = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42, verbosity=0
    )
    xgb.fit(train[FEATURES_WITH_PI].values.astype(float),
            train['ret_fwd'].values.astype(float))
    test['score_xgb'] = xgb.predict(test[FEATURES_WITH_PI].values.astype(float))

    r_xgb = long_only_port(test, 'score_xgb')
    if len(r_xgb) > 12:
        results['M2_XGB'] = compute_metrics(r_xgb)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

HMM_FEATURES = {
    'DD only':   ['DD_z'],
    'VOL only':  ['VOL_z'],
    'DISP only': ['DISP_z'],
    'REL_N only': ['REL_N_z'],
    'Full 4F (DD+VOL+DISP+REL_N)': ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z'],
}

# Also add the existing pi_filter as baseline (no re-estimation needed)
all_results = {}

# Baseline: use existing pi_filter
print("\n" + "─" * 60)
print("  Baseline: existing pi_filter (full 4F HMM, multi-seed averaged)")
t0 = time.time()
stocks_base = stocks.merge(panel[['date', 'pi_filter']], on='date', how='left', suffixes=('_old', ''))
stocks_base['pi_filter'] = stocks_base['pi_filter'].ffill()
if 'pi_filter_old' in stocks_base.columns:
    stocks_base.drop(columns=['pi_filter_old'], inplace=True)
res = run_m1_m2(stocks_base)
elapsed = time.time() - t0
print(f"  ({elapsed:.0f}s)")
if res:
    all_results['Baseline (multi-seed 4F)'] = res

# Now test each single-feature HMM
sub_panel = panel[['date', 'DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].dropna().reset_index(drop=True)
sub_panel = sub_panel[sub_panel['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
train_panel = sub_panel[sub_panel['date'] < '2011-01-01']
dates_train = train_panel['date'].values
dates_all = sub_panel['date'].values

for test_name, feat_list in HMM_FEATURES.items():
    print(f"\n{'─' * 60}")
    print(f"  {test_name}")
    t0 = time.time()

    Z_tr = train_panel[feat_list].values.astype(float)
    Z_full = sub_panel[feat_list].values.astype(float)

    # Run HMM with 3 seeds and average
    pi_seeds = []
    for seed in [42, 2201, 1337]:
        pi = fit_hmm(Z_tr, Z_full, dates_train, seed=seed)
        pi_seeds.append(pi)
    pi_avg = np.mean(pi_seeds, axis=0)

    pi_df = pd.DataFrame({'date': dates_all, 'pi_filter': pi_avg})

    # Merge into stocks
    stocks_test = stocks.copy()
    if 'pi_filter' in stocks_test.columns:
        stocks_test.drop(columns=['pi_filter'], inplace=True)
    stocks_test = stocks_test.merge(pi_df, on='date', how='left')
    stocks_test['pi_filter'] = stocks_test['pi_filter'].ffill()

    res = run_m1_m2(stocks_test)
    elapsed = time.time() - t0

    if res:
        all_results[test_name] = res
        for method, m in res.items():
            print(f"  {method}: Ann.Ret={m[0]:.1%}  Vol={m[1]:.1%}  SR={m[2]:.3f}  MDD={m[3]:.1%}")
    print(f"  ({elapsed:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SUMMARY: HMM INPUT FEATURE ABLATION")
print("=" * 80)

print(f"\n  {'HMM Input':<35s}  {'M1 LR':>10s}  {'M2 XGB':>10s}")
print(f"  {'':35s}  {'Sharpe':>10s}  {'Sharpe':>10s}")
print("  " + "─" * 60)

for name in ['Baseline (multi-seed 4F)', 'Full 4F (DD+VOL+DISP+REL_N)',
             'DD only', 'VOL only', 'DISP only', 'REL_N only']:
    if name in all_results:
        res = all_results[name]
        sr_lr = f"{res['M1_LR'][2]:.3f}" if 'M1_LR' in res else "  N/A"
        sr_xgb = f"{res['M2_XGB'][2]:.3f}" if 'M2_XGB' in res else "  N/A"
        print(f"  {name:<35s}  {sr_lr:>10s}  {sr_xgb:>10s}")

print("\n  Interpretation:")
print("  - Baseline uses the production pi_filter (multi-seed, 4-feature)")
print("  - Full 4F re-estimates with single seed (slight difference expected)")
print("  - Single features show which HMM input drives downstream performance")
print("\n" + "=" * 80)
