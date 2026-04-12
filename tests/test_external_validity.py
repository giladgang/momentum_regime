"""
test_external_validity.py
=========================
Four validation batteries:
1. Alternative train/test splits (2005, 2008, 2015)
2. Expanding-window rolling re-estimation
3. Placebo regime signals (shuffled, random noise, VIX-based)
4. Economics vs flexible prediction (XGB w/ vs w/o pi, SHAP comparison,
   kitchen-sink test with a random feature)
"""

import numpy as np
import pandas as pd
import pickle, warnings, time, sys, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PORTFOLIO_TYPE, TRADING_FEE as CFG_TRADING_FEE, TRAIN_END,
                    HMM_FEATURES, ALT_SPLITS, N_PLACEBO_RUNS,
                    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
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

# Merge pi_filter
stocks = stocks.merge(panel[['date', 'pi_filter']], on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth',
                 'gross_profit_a', 'log_me']
FEATURES = MOM_FEATURES + ['pi_filter'] + FUND_FEATURES
FEATURES_NO_PI = MOM_FEATURES + FUND_FEATURES
CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me']

TRADING_FEE = CFG_TRADING_FEE

df = stocks.dropna(subset=['ret_fwd'] + CORE_FEATURES).copy().reset_index(drop=True)
print(f"  Total obs: {len(df):,}")

# HMM data
sub_panel = panel[['date', 'DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].dropna().reset_index(drop=True)
sub_panel = sub_panel[sub_panel['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
Z_all = sub_panel[['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].values.astype(float)
dates_all_hmm = sub_panel['date'].values

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def long_only_port(df_test, score_col, fee=TRADING_FEE):
    monthly, prev_weights = [], {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0: continue
        total_me = longs['me'].sum()
        new_w = (longs.set_index('permno')['me'] / total_me).to_dict()
        turnover = sum(abs(new_w.get(p, 0) - prev_weights.get(p, 0))
                       for p in set(new_w) | set(prev_weights)) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_w
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
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

def build_port(df_test, score_col, fee=TRADING_FEE):
    """Dispatcher: calls long_only_port or long_short_port based on config."""
    if PORTFOLIO_TYPE == 'long_short':
        return long_short_port(df_test, score_col, fee=fee)
    else:
        return long_only_port(df_test, score_col, fee=fee)

def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 6: return 0, 0, 0, 0
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd

def run_pipeline(train_df, test_df, features, label=''):
    """Train LR + XGB on train_df, evaluate on test_df. Return Sharpe dict."""
    X_tr = train_df[features].values.astype(float)
    X_te = test_df[features].values.astype(float)
    y_tr = train_df['ret_fwd'].values.astype(float)

    for j in range(X_tr.shape[1]):
        col_median = np.nanmedian(X_tr[:, j])
        X_tr[np.isnan(X_tr[:, j]), j] = col_median
        X_te[np.isnan(X_te[:, j]), j] = col_median

    # LR
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    train_df = train_df.copy()
    train_df['above_med'] = train_df.groupby('date')['ret_fwd'].transform(
        lambda x: (x >= x.median()).astype(int))
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_tr_s, train_df['above_med'].values)
    test_df = test_df.copy()
    test_df['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
    r_lr = build_port(test_df, 'score_lr')
    _, _, sh_lr, _ = metrics(r_lr)

    # XGB
    xgb = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method='hist', random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    test_df['score_xgb'] = xgb.predict(X_te)
    r_xgb = build_port(test_df, 'score_xgb')
    ret_xgb, vol_xgb, sh_xgb, mdd_xgb = metrics(r_xgb)

    return {'sh_lr': sh_lr, 'sh_xgb': sh_xgb, 'ret_xgb': ret_xgb,
            'vol_xgb': vol_xgb, 'n_test_months': test_df['date'].nunique(),
            'xgb_model': xgb, 'test_df': test_df, 'X_te': X_te, 'features': features}


# ═══════════════════════════════════════════════════════════════════════════════
#  HMM HELPER (for rolling re-estimation)
# ═══════════════════════════════════════════════════════════════════════════════

def log_emission_k2(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(2)
    ])

def forward_filter_k2(Z, mu, Sigma, P):
    n = len(Z)
    log_emit = log_emission_k2(Z, mu, Sigma)
    log_alpha = np.zeros((n, 2))
    log_alpha[0] = np.log(0.5) + log_emit[0]
    for t in range(1, n):
        for k in range(2):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)

def ffbs_k2(Z, mu, Sigma, P):
    n = len(Z)
    log_emit = log_emission_k2(Z, mu, Sigma)
    log_alpha = np.zeros((n, 2))
    log_alpha[0] = np.log(0.5) + log_emit[0]
    for t in range(1, n):
        for k in range(2):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    alpha = np.exp(log_alpha)
    s = np.zeros(n, dtype=int)
    s[n-1] = np.random.choice(2, p=alpha[n-1])
    for t in range(n - 2, -1, -1):
        probs = alpha[t] * P[:, s[t+1]]
        probs /= probs.sum()
        s[t] = np.random.choice(2, p=probs)
    return s

def fit_hmm_k2(Z_train, Z_full, seed=42, n_iter=2000):
    np.random.seed(seed)
    T_tr, D = Z_train.shape
    m_0 = np.zeros(D); kappa_0 = 0.01; nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9., 1.], [1., 9.]])

    med = np.median(Z_train[:, 0])
    states = (Z_train[:, 0] < med).astype(int)
    mu = np.zeros((2, D)); Sigma = np.array([np.eye(D)] * 2)
    for k in range(2):
        idx = states == k
        if idx.sum() > D+1:
            mu[k] = Z_train[idx].mean(0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6*np.eye(D)
    P = np.array([[0.95, 0.05], [0.05, 0.95]])

    for m in range(n_iter):
        states = ffbs_k2(Z_train, mu, Sigma, P)
        for k in range(2):
            Z_k = Z_train[states == k]; n_k = len(Z_k)
            if n_k < D+2: continue
            x_bar = Z_k.mean(0); S_k = (Z_k - x_bar).T @ (Z_k - x_bar)
            kappa_n = kappa_0 + n_k; m_n = (kappa_0*m_0 + n_k*x_bar)/kappa_n
            nu_n = nu_0 + n_k
            Psi_n = Psi_0 + S_k + (kappa_0*n_k/kappa_n)*np.outer(x_bar-m_0, x_bar-m_0)
            try:
                Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
                mu[k] = np.random.multivariate_normal(m_n, Sigma[k]/kappa_n)
            except: pass
        for i in range(2):
            counts = np.array([np.sum((states[:-1]==i)&(states[1:]==j)) for j in range(2)], dtype=float)
            P[i] = np.random.dirichlet(alpha_dir[i] + counts)

    filtered = forward_filter_k2(Z_full, mu, Sigma, P)
    # Panic = state with lower DD mean
    panic_state = np.argmin(mu[:, 0])
    return filtered[:, panic_state]


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1: ALTERNATIVE TRAIN/TEST SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TEST 1: ALTERNATIVE TRAIN/TEST SPLITS")
print("=" * 90)

splits = {
    '1990-2004 / 2005-2025': '2005-01-01',
    '1990-2007 / 2008-2025': '2008-01-01',
    '1990-2010 / 2011-2025 (baseline)': '2011-01-01',
    '1990-2014 / 2015-2025': '2015-01-01',
}

print(f"\n  {'Split':<38s}  {'N test mo':>9s}  {'M1 Sharpe':>10s}  {'M2 Sharpe':>10s}  {'M2 Ret':>8s}  {'M2 Vol':>8s}")
print("  " + "-" * 90)

for name, cutoff in splits.items():
    train_s = df[df['date'] < cutoff].copy()
    test_s = df[df['date'] >= cutoff].copy()
    res = run_pipeline(train_s, test_s, FEATURES)
    print(f"  {name:<38s}  {res['n_test_months']:>9d}  {res['sh_lr']:>10.3f}  {res['sh_xgb']:>10.3f}  "
          f"{res['ret_xgb']:>7.1%}  {res['vol_xgb']:>7.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2: EXPANDING-WINDOW ROLLING RE-ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TEST 2: EXPANDING-WINDOW RE-ESTIMATION (retrain HMM + ML every 5 years)")
print("=" * 90)

# Windows: train up to cutoff, test next 5 years (or to end)
windows = [
    ('1990-2004', '2005-01-01', '2010-01-01'),
    ('1990-2009', '2010-01-01', '2015-01-01'),
    ('1990-2014', '2015-01-01', '2020-01-01'),
    ('1990-2019', '2020-01-01', '2026-01-01'),
]

all_rolling_returns_lr = []
all_rolling_returns_xgb = []

print(f"\n  {'Window':<20s}  {'Train end':>10s}  {'Test':>12s}  {'M1 Sharpe':>10s}  {'M2 Sharpe':>10s}")
print("  " + "-" * 70)

for train_label, test_start, test_end in windows:
    train_end = test_start

    # Re-estimate HMM
    hmm_train_mask = dates_all_hmm < np.datetime64(train_end)
    Z_tr_w = Z_all[hmm_train_mask]
    pi_w = fit_hmm_k2(Z_tr_w, Z_all, seed=42, n_iter=2000)

    # Replace pi_filter in stock data
    pi_df = pd.DataFrame({'date': sub_panel['date'].values, 'pi_filter_w': pi_w})
    df_w = df.copy()
    df_w = df_w.merge(pi_df, on='date', how='left')
    df_w['pi_filter_w'] = df_w['pi_filter_w'].ffill()
    df_w['pi_filter'] = df_w['pi_filter_w']

    train_w = df_w[(df_w['date'] < test_start)].copy()
    test_w = df_w[(df_w['date'] >= test_start) & (df_w['date'] < test_end)].copy()

    if len(test_w) < 100:
        continue

    res = run_pipeline(train_w, test_w, FEATURES)

    test_period = f"{test_start[:4]}-{test_end[:4]}"
    print(f"  {train_label:<20s}  {train_end[:10]:>10s}  {test_period:>12s}  "
          f"{res['sh_lr']:>10.3f}  {res['sh_xgb']:>10.3f}")

# Also compute the combined rolling Sharpe by concatenating all windows
print("\n  (Concatenated rolling results would require stitching monthly returns)")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3: PLACEBO REGIME SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TEST 3: PLACEBO REGIME SIGNALS")
print("=" * 90)

train_base = df[df['date'] < TRAIN_END].copy()
test_base = df[df['date'] >= TRAIN_END].copy()

# 3a: Baseline (real pi_filter)
res_base = run_pipeline(train_base, test_base, FEATURES)
print(f"\n  {'Signal':<35s}  {'M1 Sharpe':>10s}  {'M2 Sharpe':>10s}")
print("  " + "-" * 60)
print(f"  {'Real pi_filter (baseline)':<35s}  {res_base['sh_lr']:>10.3f}  {res_base['sh_xgb']:>10.3f}")

# 3b: No pi_filter at all
res_no_pi = run_pipeline(train_base, test_base, FEATURES_NO_PI)
print(f"  {'No pi_filter':<35s}  {res_no_pi['sh_lr']:>10.3f}  {res_no_pi['sh_xgb']:>10.3f}")

# 3c: Shuffled pi_filter (breaks temporal structure, preserves marginal distribution)
np.random.seed(42)
pi_monthly = df[['date', 'pi_filter']].drop_duplicates('date').copy()
shuffled_pi = pi_monthly['pi_filter'].values.copy()
np.random.shuffle(shuffled_pi)
pi_monthly['pi_shuffled'] = shuffled_pi

# Run 5 shuffles to get mean + std
shuffle_results = []
for shuf_seed in range(5):
    np.random.seed(shuf_seed)
    shuf = pi_monthly['pi_filter'].values.copy()
    np.random.shuffle(shuf)
    pi_monthly[f'pi_shuf_{shuf_seed}'] = shuf
    df_shuf = df.copy()
    df_shuf = df_shuf.merge(pi_monthly[['date', f'pi_shuf_{shuf_seed}']], on='date', how='left')
    df_shuf['pi_filter'] = df_shuf[f'pi_shuf_{shuf_seed}'].ffill()
    train_shuf = df_shuf[df_shuf['date'] < TRAIN_END].copy()
    test_shuf = df_shuf[df_shuf['date'] >= TRAIN_END].copy()
    res_shuf = run_pipeline(train_shuf, test_shuf, FEATURES)
    shuffle_results.append(res_shuf['sh_xgb'])

print(f"  {'Shuffled pi (mean +/- std, 5 runs)':<35s}  {'--':>10s}  {np.mean(shuffle_results):>7.3f} +/- {np.std(shuffle_results):.3f}")

# 3d: Random noise (uniform [0,1])
noise_results = []
for noise_seed in range(5):
    np.random.seed(noise_seed + 100)
    pi_monthly['pi_noise'] = np.random.uniform(0, 1, len(pi_monthly))
    df_noise = df.copy()
    df_noise = df_noise.merge(pi_monthly[['date', 'pi_noise']], on='date', how='left')
    df_noise['pi_filter'] = df_noise['pi_noise'].ffill()
    train_noise = df_noise[df_noise['date'] < TRAIN_END].copy()
    test_noise = df_noise[df_noise['date'] >= TRAIN_END].copy()
    res_noise = run_pipeline(train_noise, test_noise, FEATURES)
    noise_results.append(res_noise['sh_xgb'])

print(f"  {'Random noise (mean +/- std, 5 runs)':<35s}  {'--':>10s}  {np.mean(noise_results):>7.3f} +/- {np.std(noise_results):.3f}")

# 3e: Inverted pi_filter (1 - pi)
df_inv = df.copy()
df_inv['pi_filter'] = 1.0 - df_inv['pi_filter']
train_inv = df_inv[df_inv['date'] < TRAIN_END].copy()
test_inv = df_inv[df_inv['date'] >= TRAIN_END].copy()
res_inv = run_pipeline(train_inv, test_inv, FEATURES)
print(f"  {'Inverted pi (1 - pi_filter)':<35s}  {res_inv['sh_lr']:>10.3f}  {res_inv['sh_xgb']:>10.3f}")

# 3f: VIX-based regime (if available)
try:
    if 'LVIX_z' in panel.columns:
        # Use log VIX z-score as regime proxy
        vix_pi = panel[['date', 'LVIX_z']].dropna().drop_duplicates('date')
        # Convert to [0,1] via sigmoid
        vix_pi['pi_vix'] = 1.0 / (1.0 + np.exp(-vix_pi['LVIX_z']))
        df_vix = df.copy()
        df_vix = df_vix.drop(columns=['pi_filter'])
        df_vix = df_vix.merge(vix_pi[['date', 'pi_vix']], on='date', how='left')
        df_vix['pi_filter'] = df_vix['pi_vix'].ffill()
        train_vix = df_vix[df_vix['date'] < '2011-01-01'].copy()
        test_vix = df_vix[df_vix['date'] >= '2011-01-01'].copy()
        res_vix = run_pipeline(train_vix, test_vix, FEATURES)
        print(f"  {'VIX-based regime (sigmoid of LVIX_z)':<35s}  {res_vix['sh_lr']:>10.3f}  {res_vix['sh_xgb']:>10.3f}")
    else:
        print(f"  {'VIX-based regime':<35s}  {'(LVIX_z not available)':>25s}")
except Exception as e:
    print(f"  VIX test error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4: ECONOMICS VS FLEXIBLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("  TEST 4: ECONOMICS VS FLEXIBLE PREDICTION")
print("=" * 90)

# 4a: XGB with vs without pi_filter -- does the Sharpe change?
print(f"\n  Part A: Does pi_filter improve XGB Sharpe ratio?")
print(f"  {'Config':<35s}  {'M2 Sharpe':>10s}  {'M2 Ret':>8s}  {'M2 Vol':>8s}")
print("  " + "-" * 65)

res_with = run_pipeline(train_base, test_base, FEATURES)
res_without = run_pipeline(train_base, test_base, FEATURES_NO_PI)
print(f"  {'XGB with pi_filter':<35s}  {res_with['sh_xgb']:>10.3f}  {res_with['ret_xgb']:>7.1%}  {res_with['vol_xgb']:>7.1%}")
print(f"  {'XGB without pi_filter':<35s}  {res_without['sh_xgb']:>10.3f}  {res_without['ret_xgb']:>7.1%}  {res_without['vol_xgb']:>7.1%}")

# 4b: Kitchen sink -- add a random feature. Does XGB benefit equally from any extra feature?
print(f"\n  Part B: Kitchen-sink test (does any extra feature help equally?)")
kitchen_results = []
for ks_seed in range(10):
    np.random.seed(ks_seed + 200)
    # Generate a persistent random feature (random walk at monthly level, same for all stocks)
    monthly_dates = df[['date']].drop_duplicates().sort_values('date')
    monthly_dates['random_feat'] = np.cumsum(np.random.randn(len(monthly_dates))) * 0.1
    df_ks = df.copy()
    df_ks = df_ks.merge(monthly_dates, on='date', how='left')
    features_ks = FEATURES_NO_PI + ['random_feat']
    train_ks = df_ks[df_ks['date'] < '2011-01-01'].copy()
    test_ks = df_ks[df_ks['date'] >= '2011-01-01'].copy()
    res_ks = run_pipeline(train_ks, test_ks, features_ks)
    kitchen_results.append(res_ks['sh_xgb'])

print(f"  {'XGB + random walk feature (10 runs)':<35s}  mean={np.mean(kitchen_results):.3f} +/- {np.std(kitchen_results):.3f}")
print(f"  {'XGB + real pi_filter':<35s}  {res_with['sh_xgb']:.3f}")
print(f"  {'XGB without extra feature':<35s}  {res_without['sh_xgb']:.3f}")

# 4c: SHAP comparison -- does removing pi_filter change WHICH momentum horizons matter?
print(f"\n  Part C: SHAP feature importance comparison (with vs without pi_filter)")
try:
    import shap

    # With pi
    xgb_with = res_with['xgb_model']
    X_te_with = res_with['X_te']
    explainer_with = shap.TreeExplainer(xgb_with)
    shap_with = explainer_with.shap_values(X_te_with)
    mean_shap_with = np.abs(shap_with).mean(axis=0)
    feat_imp_with = dict(zip(FEATURES, mean_shap_with))

    # Without pi
    xgb_without = res_without['xgb_model']
    X_te_without = res_without['X_te']
    explainer_without = shap.TreeExplainer(xgb_without)
    shap_without = explainer_without.shap_values(X_te_without)
    mean_shap_without = np.abs(shap_without).mean(axis=0)
    feat_imp_without = dict(zip(FEATURES_NO_PI, mean_shap_without))

    print(f"\n  {'Feature':<15s}  {'With pi':>10s}  {'Without pi':>12s}  {'Change':>8s}")
    print("  " + "-" * 50)
    for feat in MOM_FEATURES:
        v_with = feat_imp_with.get(feat, 0)
        v_without = feat_imp_without.get(feat, 0)
        change = v_without - v_with
        print(f"  {feat:<15s}  {v_with:>10.4f}  {v_without:>12.4f}  {change:>+8.4f}")

    # pi_filter itself
    print(f"  {'pi_filter':<15s}  {feat_imp_with.get('pi_filter', 0):>10.4f}  {'--':>12s}  {'--':>8s}")

    # Key question: does mom_1 importance increase without pi?
    mom1_with = feat_imp_with.get('mom_1', 0)
    mom1_without = feat_imp_without.get('mom_1', 0)
    print(f"\n  Key diagnostic: mom_1 importance {mom1_with:.4f} -> {mom1_without:.4f} when pi removed")
    print(f"  If mom_1 importance INCREASES without pi, the model compensates by using")
    print(f"  mom_1 uniformly rather than conditionally -- supporting the 'context variable' interpretation.")

except Exception as e:
    print(f"  SHAP comparison error: {e}")


# 4d: Regime-conditional SHAP -- does pi_filter's SHAP vary by regime?
print(f"\n  Part D: Does pi_filter's SHAP importance differ by regime?")
try:
    test_df_with = res_with['test_df']
    pi_test = test_df_with.groupby('date')['pi_filter'].first()

    # Map each test observation to its month's pi_filter
    test_dates = test_df_with['date'].values
    unique_dates = sorted(test_df_with['date'].unique())
    date_to_pi = dict(zip(pi_test.index, pi_test.values))
    obs_pi = np.array([date_to_pi.get(d, 0.5) for d in test_dates])

    calm_mask = obs_pi < 0.5
    panic_mask = obs_pi >= 0.5

    pi_idx = FEATURES.index('pi_filter')
    shap_pi_calm = np.abs(shap_with[calm_mask, pi_idx]).mean()
    shap_pi_panic = np.abs(shap_with[panic_mask, pi_idx]).mean()

    print(f"  pi_filter mean |SHAP| in calm:  {shap_pi_calm:.4f}  (n={calm_mask.sum():,})")
    print(f"  pi_filter mean |SHAP| in panic: {shap_pi_panic:.4f}  (n={panic_mask.sum():,})")
    print(f"  Ratio (panic/calm): {shap_pi_panic/shap_pi_calm:.2f}x")

except Exception as e:
    print(f"  Error: {e}")


print("\n" + "=" * 90)
print("  ALL TESTS COMPLETE")
print("=" * 90)
