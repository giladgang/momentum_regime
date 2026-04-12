"""
robustness_checks.py
====================
Run 7 robustness checks:
1. Sub-period analysis (2011-2015, 2016-2020, 2021-2025)
2. Turnover by strategy
3. Transaction cost sensitivity (breakeven cost)
4. XGBoost hyperparameter sensitivity
5. Regime threshold sensitivity (pi = 0.25, 0.5, 0.75)
6. Skip-month momentum
7. K=3 state HMM
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
                    HMM_FEATURES, K_STATES_ROBUSTNESS, COST_LEVELS_BPS,
                    PI_THRESHOLDS, XGB_CONFIGS, SUB_PERIODS,
                    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE,
                    XGB_SEEDS)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading data ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

strats_lo = artefacts['strategies_lo']
r_mkt = artefacts['r_mkt']

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
CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me']

TRADING_FEE = CFG_TRADING_FEE

df = stocks.dropna(subset=['ret_fwd'] + CORE_FEATURES).copy().reset_index(drop=True)
train = df[df['date'] < TRAIN_END].copy()
test = df[df['date'] >= TRAIN_END].copy()

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def long_short_port(df_test, score_col, fee=TRADING_FEE, return_turnover=False):
    monthly, prev_weights = [], {}
    turnovers = []
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
        monthly.append({'date': date, 'ret': r_gross - fee * turnover, 'ret_gross': r_gross})
        turnovers.append(turnover)
        prev_weights = new_w
    if not monthly:
        return pd.Series(dtype=float), 0.0
    result = pd.DataFrame(monthly).set_index('date')
    avg_turnover = np.mean(turnovers)
    if return_turnover:
        return result['ret'], avg_turnover
    return result['ret']


def long_short_port(df_test, score_col, fee=TRADING_FEE, return_turnover=False):
    monthly, prev_lw, prev_sw = [], {}, {}
    turnovers = []
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
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts), 'ret_gross': r_long - r_short})
        turnovers.append(tl + ts)
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return (pd.Series(dtype=float), 0.0) if return_turnover else pd.Series(dtype=float)
    result = pd.DataFrame(monthly).set_index('date')
    avg_turnover = np.mean(turnovers)
    if return_turnover:
        return result['ret'], avg_turnover
    return result['ret']


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 6:
        return 0, 0, 0, 0
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECK 1: SUB-PERIOD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CHECK 1: SUB-PERIOD ANALYSIS")
print("=" * 80)

# Recompute strategies as long-short from test scores
test_art = artefacts['test'].copy()
all_strats = {
    'Market':          r_mkt,
    'Fixed 12-mo':     long_short_port(test_art, 'score_mom12'),
    'M1: LR':          long_short_port(test_art, 'score_lr'),
    'M2: XGB':         long_short_port(test_art, 'score_xgb'),
}

periods = [
    # Sub-periods from config.py (excludes 'Full')
    *[(n, s, e) for n, s, e in SUB_PERIODS if n != 'Full'],
    ('Full',      '2011-01-01', '2026-01-01'),
]

print(f"\n  {'Strategy':<15s}", end='')
for pname, _, _ in periods:
    print(f"  {pname:>12s}", end='')
print()
print("  " + "-" * 70)

for sname, r in all_strats.items():
    print(f"  {sname:<15s}", end='')
    for pname, start, end in periods:
        r_sub = r[(r.index >= start) & (r.index < end)]
        if len(r_sub) < 6:
            print(f"  {'N/A':>12s}", end='')
        else:
            _, _, sh, _ = metrics(r_sub)
            print(f"  {sh:>12.3f}", end='')
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECK 2: TURNOVER BY STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CHECK 2: TURNOVER BY STRATEGY")
print("=" * 80)

# Need to recompute with turnover tracking
X_train = train[FEATURES].values.astype(float)
X_test = test[FEATURES].values.astype(float)
y_train = train['ret_fwd'].values.astype(float)

# LR
scaler = StandardScaler()
train['above_med'] = train.groupby('date')['ret_fwd'].transform(
    lambda x: (x >= x.median()).astype(int))

for j in range(X_train.shape[1]):
    col_median = np.nanmedian(X_train[:, j])
    X_train[np.isnan(X_train[:, j]), j] = col_median
    X_test[np.isnan(X_test[:, j]), j] = col_median

X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tr_s, train['above_med'].values)
test_c = test.copy()
test_c['score_lr'] = lr.predict_proba(X_te_s)[:, 1]

xgb = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    tree_method='hist', random_state=42, verbosity=0)
xgb.fit(train[FEATURES].values.astype(float), y_train)
test_c['score_xgb'] = xgb.predict(test[FEATURES].values.astype(float))

# Fixed momentum scores
test_c['score_mom12'] = test_c.groupby('date')['mom_12'].rank(pct=True)
test_c['score_mom1'] = test_c.groupby('date')['mom_1'].rank(pct=True)

turnover_strats = {
    'Fixed 12-mo': 'score_mom12',
    'Fixed 1-mo': 'score_mom1',
    'M1: LR': 'score_lr',
    'M2: XGB': 'score_xgb',
}

print(f"\n  {'Strategy':<15s}  {'Avg Monthly TO':>15s}  {'Ann. TO':>10s}")
print("  " + "-" * 45)
for name, col in turnover_strats.items():
    _, avg_to = long_short_port(test_c, col, return_turnover=True)
    print(f"  {name:<15s}  {avg_to:>14.1%}  {avg_to*12:>9.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECK 3: TRANSACTION COST SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CHECK 3: TRANSACTION COST SENSITIVITY")
print("=" * 80)

cost_levels = [0, 5, 10, 20, 30, 50]  # in bps

print(f"\n  {'Strategy':<15s}", end='')
for c in cost_levels:
    print(f"  {c:>6d}bps", end='')
print()
print("  " + "-" * 70)

for name, col in turnover_strats.items():
    print(f"  {name:<15s}", end='')
    for c in cost_levels:
        r, _ = long_short_port(test_c, col, fee=c/10000, return_turnover=True)
        _, _, sh, _ = metrics(r)
        print(f"  {sh:>9.3f}", end='')
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECK 4: XGBOOST HYPERPARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CHECK 4: XGBOOST HYPERPARAMETER SENSITIVITY")
print("=" * 80)

X_tr_raw = train[FEATURES].values.astype(float)
X_te_raw = test[FEATURES].values.astype(float)
y_tr_raw = train['ret_fwd'].values.astype(float)

configs = [
    ('depth=3, lr=0.05, n=500', {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 500}),
    ('depth=4, lr=0.05, n=500', {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500}),  # baseline
    ('depth=5, lr=0.05, n=500', {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 500}),
    ('depth=6, lr=0.05, n=500', {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500}),
    ('depth=4, lr=0.01, n=500', {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 500}),
    ('depth=4, lr=0.10, n=500', {'max_depth': 4, 'learning_rate': 0.10, 'n_estimators': 500}),
    ('depth=4, lr=0.05, n=200', {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 200}),
    ('depth=4, lr=0.05, n=1000', {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 1000}),
]

print(f"\n  {'Config':<30s}  {'Sharpe':>8s}  {'Ann.Ret':>8s}  {'Vol':>8s}")
print("  " + "-" * 60)

for config_name, params in configs:
    model = XGBRegressor(
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=42, verbosity=0,
        **params
    )
    model.fit(X_tr_raw, y_tr_raw)
    test_hp = test.copy()
    test_hp['score'] = model.predict(X_te_raw)
    r = long_short_port(test_hp, 'score')
    ar, av, sh, _ = metrics(r)
    marker = " <-- baseline" if "depth=4, lr=0.05, n=500" in config_name else ""
    print(f"  {config_name:<30s}  {sh:>8.3f}  {ar:>7.1%}  {av:>7.1%}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECK 5: REGIME THRESHOLD SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CHECK 5: REGIME THRESHOLD SENSITIVITY")
print("=" * 80)

r_m1 = long_short_port(test_art, 'score_lr')
r_m2 = long_short_port(test_art, 'score_xgb')
r_mom = long_short_port(test_art, 'score_mom12')

pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

thresholds = [0.25, 0.50, 0.75]

print(f"\n  Threshold  |  N_calm  N_panic  |  Mkt_calm  Mkt_panic  |  M1_calm  M1_panic  |  M2_calm  M2_panic")
print("  " + "-" * 105)

for thresh in thresholds:
    common = r_mkt.index.intersection(pi_monthly.index)
    pi_vals = pi_monthly.loc[common, 'pi_filter']
    calm_mask = pi_vals < thresh
    panic_mask = pi_vals >= thresh

    n_calm = calm_mask.sum()
    n_panic = panic_mask.sum()

    calm_dates = pi_vals[calm_mask].index
    panic_dates = pi_vals[panic_mask].index

    def sharpe_subset(r, dates):
        r_sub = r[r.index.isin(dates)]
        if len(r_sub) < 6:
            return np.nan
        return r_sub.mean() / r_sub.std() * np.sqrt(12) if r_sub.std() > 0 else 0

    mkt_c = sharpe_subset(r_mkt, calm_dates)
    mkt_p = sharpe_subset(r_mkt, panic_dates)
    m1_c = sharpe_subset(r_m1, calm_dates)
    m1_p = sharpe_subset(r_m1, panic_dates)
    m2_c = sharpe_subset(r_m2, calm_dates)
    m2_p = sharpe_subset(r_m2, panic_dates)

    print(f"  pi>{thresh:.2f}    |  {n_calm:>5d}  {n_panic:>6d}   |  {mkt_c:>8.3f}  {mkt_p:>9.3f}  |  {m1_c:>7.3f}  {m1_p:>8.3f}  |  {m2_c:>7.3f}  {m2_p:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECK 6: SKIP-MONTH MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CHECK 6: SKIP-MONTH MOMENTUM (exclude most recent month)")
print("=" * 80)

# Recompute momentum with skip-month: cumulative return from t-k to t-2 (skip t-1)
stocks2 = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks2['date'] = pd.to_datetime(stocks2['date'])
stocks2 = stocks2[stocks2['shrcd'].isin([10, 11])]
stocks2 = stocks2[stocks2['exchcd'].isin([1, 2, 3])]
stocks2 = stocks2[stocks2['prc'].abs() > 1.0]
stocks2 = stocks2.sort_values(['permno', 'date']).reset_index(drop=True)

stocks2['_log_ret'] = np.log1p(stocks2['ret_adj'].clip(lower=-0.999))
# Shift by 2 instead of 1 to skip the most recent month
stocks2['_log_ret_s2'] = stocks2.groupby('permno')['_log_ret'].shift(2)
for lb in MOM_LBS:
    roll_sum = (
        stocks2.groupby('permno', sort=False)['_log_ret_s2']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks2[f'mom_{lb}'] = np.expm1(roll_sum)
stocks2.drop(columns=['_log_ret', '_log_ret_s2'], inplace=True)
stocks2['log_me'] = np.log(
    stocks2.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)
stocks2['ret_fwd'] = stocks2.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))
stocks2 = stocks2.merge(panel[['date', 'pi_filter']], on='date', how='left')
stocks2['pi_filter'] = stocks2['pi_filter'].ffill()

df2 = stocks2.dropna(subset=['ret_fwd'] + CORE_FEATURES).copy().reset_index(drop=True)
train2 = df2[df2['date'] < TRAIN_END].copy()
test2 = df2[df2['date'] >= TRAIN_END].copy()

X_tr2 = train2[FEATURES].values.astype(float)
X_te2 = test2[FEATURES].values.astype(float)
y_tr2 = train2['ret_fwd'].values.astype(float)

for j in range(X_tr2.shape[1]):
    col_median = np.nanmedian(X_tr2[:, j])
    X_tr2[np.isnan(X_tr2[:, j]), j] = col_median
    X_te2[np.isnan(X_te2[:, j]), j] = col_median

# LR
scaler2 = StandardScaler()
X_tr2_s = scaler2.fit_transform(X_tr2)
X_te2_s = scaler2.transform(X_te2)

train2['above_med'] = train2.groupby('date')['ret_fwd'].transform(
    lambda x: (x >= x.median()).astype(int))

lr2 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr2.fit(X_tr2_s, train2['above_med'].values)
test2['score_lr'] = lr2.predict_proba(X_te2_s)[:, 1]

# XGB
xgb2 = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     tree_method='hist', random_state=42, verbosity=0)
xgb2.fit(X_tr2, y_tr2)
test2['score_xgb'] = xgb2.predict(X_te2)

# Fixed 12-mo
test2['score_mom12'] = test2.groupby('date')['mom_12'].rank(pct=True)

print(f"\n  {'Strategy':<20s}  {'Sharpe (skip)':>14s}  {'Sharpe (base)':>14s}")
print("  " + "-" * 55)

for name, col, base_sh in [
    ('Fixed 12-mo', 'score_mom12', 0.70),
    ('M1: LR', 'score_lr', 1.038),
    ('M2: XGB', 'score_xgb', 1.007),
]:
    r = long_short_port(test2, col)
    _, _, sh, _ = metrics(r)
    print(f"  {name:<20s}  {sh:>14.3f}  {base_sh:>14.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECK 7: K=3 STATE HMM
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  CHECK 7: K=3 STATE HMM")
print("=" * 80)

K3 = 3

sub_panel = panel[['date', 'DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].dropna().reset_index(drop=True)
sub_panel = sub_panel[sub_panel['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
train_panel = sub_panel[sub_panel['date'] < '2011-01-01']
dates_train = train_panel['date'].values
dates_all = sub_panel['date'].values

Z_tr = train_panel[['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].values.astype(float)
Z_full = sub_panel[['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].values.astype(float)
T_tr, D = Z_tr.shape


def log_emission_k3(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K3)
    ])

def forward_filter_k3(Z, mu, Sigma, P):
    n = len(Z)
    log_emit = log_emission_k3(Z, mu, Sigma)
    log_alpha = np.zeros((n, K3))
    log_alpha[0] = np.log(1.0/K3) + log_emit[0]
    for t in range(1, n):
        for k in range(K3):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)

def ffbs_k3(Z, mu, Sigma, P):
    n = len(Z)
    log_emit = log_emission_k3(Z, mu, Sigma)
    log_alpha = np.zeros((n, K3))
    log_alpha[0] = np.log(1.0/K3) + log_emit[0]
    for t in range(1, n):
        for k in range(K3):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    alpha = np.exp(log_alpha)
    s = np.zeros(n, dtype=int)
    s[n-1] = np.random.choice(K3, p=alpha[n-1])
    for t in range(n - 2, -1, -1):
        probs = alpha[t] * P[:, s[t+1]]
        probs /= probs.sum()
        s[t] = np.random.choice(K3, p=probs)
    return s

def fit_hmm_k3(Z_train, Z_full, dates_train, seed=42):
    np.random.seed(seed)
    T_tr, D = Z_train.shape

    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.ones((K3, K3)) + 8 * np.eye(K3)  # persistence prior

    # Init by tertile split
    q33 = np.percentile(Z_train[:, 0], 33)
    q66 = np.percentile(Z_train[:, 0], 66)
    states = np.where(Z_train[:, 0] < q33, 0, np.where(Z_train[:, 0] < q66, 1, 2))

    mu = np.zeros((K3, D))
    Sigma = np.array([np.eye(D)] * K3)
    for k in range(K3):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
    P = np.ones((K3, K3)) * 0.05 + 0.85 * np.eye(K3)
    P /= P.sum(axis=1, keepdims=True)

    n_iter = 2000
    for m in range(n_iter):
        states = ffbs_k3(Z_train, mu, Sigma, P)
        for k in range(K3):
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
        for i in range(K3):
            counts = np.array([np.sum((states[:-1] == i) & (states[1:] == j))
                               for j in range(K3)], dtype=float)
            P[i] = np.random.dirichlet(alpha_dir[i] + counts)

    filtered = forward_filter_k3(Z_full, mu, Sigma, P)

    # Identify panic state: highest mean stress during known crises
    crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
    crisis_mask = np.zeros(T_tr, dtype=bool)
    for s, e in crisis_windows:
        crisis_mask |= ((dates_train >= np.datetime64(s)) & (dates_train <= np.datetime64(e)))

    crisis_probs = filtered[:T_tr][crisis_mask].mean(axis=0)
    panic_state = np.argmax(crisis_probs)

    # Return max stress probability (sum of non-calm states, or just panic state)
    return filtered[:, panic_state], mu, filtered

t0 = time.time()
pi_seeds = []
for seed in [42, 2201, 1337]:
    pi, mu_post, filt_full = fit_hmm_k3(Z_tr, Z_full, dates_train, seed=seed)
    pi_seeds.append(pi)
pi_3state = np.mean(pi_seeds, axis=0)
print(f"  K=3 HMM fitted in {time.time()-t0:.0f}s")

# Report regime means from last seed
print(f"\n  Regime means (last seed):")
for k in range(K3):
    print(f"    State {k}: DD={mu_post[k,0]:+.2f}  VOL={mu_post[k,1]:+.2f}  "
          f"DISP={mu_post[k,2]:+.2f}  REL_N={mu_post[k,3]:+.2f}")

# Feed into M1 and M2
pi_3s_df = pd.DataFrame({'date': dates_all, 'pi_filter': pi_3state})
stocks3 = df.copy()
stocks3 = stocks3.drop(columns=['pi_filter'])
stocks3 = stocks3.merge(pi_3s_df, on='date', how='left')
stocks3['pi_filter'] = stocks3['pi_filter'].ffill()

train3 = stocks3[stocks3['date'] < '2011-01-01'].copy()
test3 = stocks3[stocks3['date'] >= '2011-01-01'].copy()

X_tr3 = train3[FEATURES].values.astype(float)
X_te3 = test3[FEATURES].values.astype(float)
y_tr3 = train3['ret_fwd'].values.astype(float)

for j in range(X_tr3.shape[1]):
    col_median = np.nanmedian(X_tr3[:, j])
    X_tr3[np.isnan(X_tr3[:, j]), j] = col_median
    X_te3[np.isnan(X_te3[:, j]), j] = col_median

# LR
scaler3 = StandardScaler()
X_tr3_s = scaler3.fit_transform(X_tr3)
X_te3_s = scaler3.transform(X_te3)
train3['above_med'] = train3.groupby('date')['ret_fwd'].transform(
    lambda x: (x >= x.median()).astype(int))
lr3 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr3.fit(X_tr3_s, train3['above_med'].values)
test3['score_lr'] = lr3.predict_proba(X_te3_s)[:, 1]
r_lr3 = long_short_port(test3, 'score_lr')

# XGB
xgb3 = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     tree_method='hist', random_state=42, verbosity=0)
xgb3.fit(X_tr3, y_tr3)
test3['score_xgb'] = xgb3.predict(X_te3)
r_xgb3 = long_short_port(test3, 'score_xgb')

print(f"\n  {'Model':<10s}  {'K=2 Sharpe':>12s}  {'K=3 Sharpe':>12s}")
print("  " + "-" * 40)
_, _, sh_lr3, _ = metrics(r_lr3)
_, _, sh_xgb3, _ = metrics(r_xgb3)
print(f"  {'M1: LR':<10s}  {'1.038':>12s}  {sh_lr3:>12.3f}")
print(f"  {'M2: XGB':<10s}  {'1.007':>12s}  {sh_xgb3:>12.3f}")


print("\n" + "=" * 80)
print("  ALL CHECKS COMPLETE")
print("=" * 80)
