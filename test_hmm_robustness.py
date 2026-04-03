"""
test_hmm_robustness.py
======================
Robustness analysis for HMM feature selection.

1. Sub-period stability: split test (2011-2017 / 2018-2025), compute Sharpe
   in each half for the top N combos from test_hmm_features.py
2. Bootstrap multiple testing correction: simulate null distribution of
   "best Sharpe among M combos" to check if top result is significant
3. Ensemble: average pi_filter from stable combos, run final portfolio test
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.preprocessing import StandardScaler
import warnings, time
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP COMBOS FROM FEATURE SEARCH (hardcoded from test_hmm_features.py results)
# ═══════════════════════════════════════════════════════════════════════════════

TOP_COMBOS = [
    ['DD_z', 'CS_z', 'DISP_z', 'SKEW_z'],           # 1.160
    ['DD_z', 'TERM_z', 'CP_SPREAD_z', 'REL_N_z'],    # 1.158
    ['DD_z', 'TERM_z', 'CP_SPREAD_z', 'ADR_z'],      # 1.143
    ['DD_z', 'VOL_z', 'ADR_z', 'REL_N_z'],           # 1.141
    ['DD_z', 'CS_z', 'DISP_z', 'REL_N_z'],           # 1.139
    ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z'],          # 1.131
    ['DD_z', 'HY_OAS_z', 'DISP_z', 'REL_N_z'],      # 1.127
    ['DD_z', 'DISP_z', 'SKEW_z', 'REL_N_z'],         # 1.127
    ['CS_z', 'CP_SPREAD_z', 'SKEW_z', 'REL_N_z'],    # 1.103
    ['VOL_z', 'CS_z', 'LVIX_z', 'SKEW_z'],           # 1.099
    ['DD_z', 'VOL_z', 'LVIX_z', 'SKEW_z'],           # 1.094
    ['DD_z', 'DISP_z', 'ADR_z', 'SKEW_z'],           # 1.087
    ['CS_z', 'CP_SPREAD_z', 'HY_OAS_z', 'DISP_z'],   # 1.065
    ['DD_z', 'VOL_z', 'CS_z', 'LVIX_z'],             # 0.852 (BASELINE)
]

BASELINE_COMBO = ['DD_z', 'VOL_z', 'CS_z', 'LVIX_z']

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  HMM ROBUSTNESS ANALYSIS")
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
import pandas_datareader.data as web

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

if 'CP_SPREAD' not in panel.columns:
    try:
        cp = web.DataReader('DCPF3M', 'fred', start='1970-01-01', end='2025-12-31')
        tbill = web.DataReader('DTB3', 'fred', start='1970-01-01', end='2025-12-31')
        cp_spread = (cp['DCPF3M'] - tbill['DTB3']).resample('ME').mean().to_frame('CP_SPREAD')
        cp_spread.index = cp_spread.index.to_period('M')
        cp_df = cp_spread.reset_index()
        cp_df.columns = ['year_month', 'CP_SPREAD']
        panel = panel.merge(cp_df, on='year_month', how='left')
        print("  CP_SPREAD: OK")
    except Exception as e:
        print(f"  CP_SPREAD: FAILED ({e})")

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

if 'DISP' not in panel.columns:
    disp = stock_data.groupby('year_month')['ret_adj'].std().reset_index()
    disp.columns = ['year_month', 'DISP']
    disp['DISP'] = np.log(disp['DISP'])
    panel = panel.merge(disp, on='year_month', how='left')
    print("  DISP: OK")

if 'ADR' not in panel.columns:
    adr = stock_data.groupby('year_month').apply(
        lambda g: (g['ret_adj'] > 0).mean(), include_groups=False
    ).reset_index()
    adr.columns = ['year_month', 'ADR']
    panel = panel.merge(adr, on='year_month', how='left')
    print("  ADR: OK")

if 'SKEW' not in panel.columns:
    skew_feat = stock_data.groupby('year_month')['ret_adj'].skew().reset_index()
    skew_feat.columns = ['year_month', 'SKEW']
    panel = panel.merge(skew_feat, on='year_month', how='left')
    print("  SKEW: OK")

if 'REL_N' not in panel.columns:
    n_stocks = stock_data.groupby('year_month')['permno'].nunique().reset_index()
    n_stocks.columns = ['year_month', 'N_STOCKS']
    n_stocks = n_stocks.sort_values('year_month')
    n_stocks['N_STOCKS_MA12'] = n_stocks['N_STOCKS'].rolling(12).mean()
    n_stocks['REL_N'] = np.log(n_stocks['N_STOCKS'] / n_stocks['N_STOCKS_MA12'])
    panel = panel.merge(n_stocks[['year_month', 'REL_N']], on='year_month', how='left')
    print("  REL_N: OK")

# Standardize
CANDIDATE_RAW = ['TERM', 'CP_SPREAD', 'HY_OAS', 'DISP', 'ADR', 'SKEW', 'REL_N']
for col in CANDIDATE_RAW:
    zcol = f'{col}_z'
    if col in panel.columns and zcol not in panel.columns:
        vals = panel[col].astype(float)
        mu = vals[train_mask].dropna().mean()
        sd = vals[train_mask].dropna().std()
        if sd > 0 and not np.isnan(sd):
            panel[zcol] = (vals - mu) / sd

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS (same as test_hmm_features.py)
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
    np.random.seed(seed)
    T, D = Z_train.shape

    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

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
    Sigma_post = np.array([np.cov(Z_train[states == k].T) + 1e-6 * np.eye(D)
                           for k in range(K)])
    P_post = P.copy()

    # Panic identification: L2 + crisis-calibrated sign correction
    Z_full = np.vstack([Z_train, Z_test])
    filtered = forward_filter(Z_full, mu_post, Sigma_post, P_post)

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
        signs = np.zeros(D)
        for j in range(D):
            p95_crisis = np.percentile(Z_train[crisis_mask, j], 95)
            p95_normal = np.percentile(Z_train[~crisis_mask, j], 95)
            signs[j] = 1.0 if p95_crisis >= p95_normal else -1.0
        score0 = np.sum(signs * mu_post[0])
        score1 = np.sum(signs * mu_post[1])
        panic_state = 1 if score1 >= score0 else 0
    else:
        panic_state = 1 if np.sum(mu_post[1]**2) >= np.sum(mu_post[0]**2) else 0

    pi_filter = filtered[:, panic_state]

    sub_all = panel[['date'] + feat_list].dropna()
    sub_all = sub_all[sub_all['date'] >= '1990-01-01']

    return {
        'pi_filter': pi_filter,
        'dates': sub_all['date'].values,
        'T_train': T,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK DATA PREPARATION (same as test_hmm_features.py)
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


def run_xgb_test(pi_df, period='full'):
    """Run XGB portfolio test. period='full', 'early' (2011-2017), 'late' (2018-2025)"""
    stocks_with_pi = stocks_raw.merge(pi_df[['date', 'pi_filter']], on='date', how='left')
    stocks_with_pi['pi_filter'] = stocks_with_pi['pi_filter'].ffill()
    df_cs = stocks_with_pi.dropna(subset=['ret_fwd'] + MOM_FEATURES + ['pi_filter', 'log_me']).copy()

    train_cs = df_cs[df_cs['date'] < '2011-01-01'].copy()

    if period == 'full':
        test_cs = df_cs[df_cs['date'] >= '2011-01-01'].copy()
    elif period == 'early':
        test_cs = df_cs[(df_cs['date'] >= '2011-01-01') & (df_cs['date'] < '2018-01-01')].copy()
    elif period == 'late':
        test_cs = df_cs[df_cs['date'] >= '2018-01-01'].copy()

    if len(train_cs) < 1000 or len(test_cs) < 1000:
        return None, None

    try:
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
            return compute_metrics(r_xgb), r_xgb
    except:
        pass
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: FIT HMMs FOR ALL TOP COMBOS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(f"  STEP 1: Fit HMMs for {len(TOP_COMBOS)} combos")
print("=" * 80)

hmm_results = {}
t_start = time.time()

for i, feat_list in enumerate(TOP_COMBOS):
    short_name = "+".join(f.replace("_z", "").replace("CP_SPREAD", "CPS") for f in feat_list)
    is_bl = set(feat_list) == set(BASELINE_COMBO)
    print(f"  [{i+1}/{len(TOP_COMBOS)}] {short_name}" +
          (" (BASELINE)" if is_bl else ""), end=" ... ", flush=True)

    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01']
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']
    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)

    result = fit_hmm(Z_tr, Z_te, feat_list)
    pi_df = pd.DataFrame({'date': result['dates'], 'pi_filter': result['pi_filter']})
    hmm_results[short_name] = {'feat_list': feat_list, 'pi_df': pi_df}
    print(f"done ({time.time() - t_start:.0f}s)")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: SUB-PERIOD STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  STEP 2: Sub-period stability (2011-2017 vs 2018-2025)")
print("=" * 80)

stability_results = {}

for name, data in hmm_results.items():
    pi_df = data['pi_df']
    print(f"\n  {name}:")

    metrics_full, r_full = run_xgb_test(pi_df, 'full')
    metrics_early, r_early = run_xgb_test(pi_df, 'early')
    metrics_late, r_late = run_xgb_test(pi_df, 'late')

    if metrics_full and metrics_early and metrics_late:
        sr_full = metrics_full[2]
        sr_early = metrics_early[2]
        sr_late = metrics_late[2]
        stability_results[name] = {
            'sr_full': sr_full,
            'sr_early': sr_early,
            'sr_late': sr_late,
            'r_full': r_full,
            'stable': sr_early > 0.5 and sr_late > 0.5,
        }
        stable_tag = "STABLE" if stability_results[name]['stable'] else "UNSTABLE"
        print(f"    Full: {sr_full:.3f}  |  2011-17: {sr_early:.3f}  |  2018-25: {sr_late:.3f}  -> {stable_tag}")
    else:
        print(f"    FAILED")

# Summary table
print("\n  Sub-period stability summary:")
print(f"  {'Combo':40s}  {'Full':>7s}  {'2011-17':>7s}  {'2018-25':>7s}  {'Status':>8s}")
print("  " + "-" * 75)
for name, res in sorted(stability_results.items(), key=lambda x: x[1]['sr_full'], reverse=True):
    tag = "STABLE" if res['stable'] else "UNSTABLE"
    is_bl = "(BL)" if name == "DD+VOL+CS+LVIX" else ""
    print(f"  {name+is_bl:40s}  {res['sr_full']:7.3f}  {res['sr_early']:7.3f}  "
          f"{res['sr_late']:7.3f}  {tag:>8s}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: BOOTSTRAP MULTIPLE TESTING CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  STEP 3: Bootstrap multiple testing correction")
print("=" * 80)

# Collect monthly return series for all combos
all_returns = {}
for name, res in stability_results.items():
    if res['r_full'] is not None:
        all_returns[name] = res['r_full']

if len(all_returns) > 1:
    # Align all return series to same dates
    ret_df = pd.DataFrame(all_returns)
    ret_df = ret_df.dropna()
    n_months = len(ret_df)
    n_combos = len(ret_df.columns)

    # Observed best Sharpe
    observed_sharpes = ret_df.mean() / ret_df.std() * np.sqrt(12)
    observed_best = observed_sharpes.max()
    observed_best_name = observed_sharpes.idxmax()

    print(f"\n  Observed best Sharpe: {observed_best:.3f} ({observed_best_name})")
    print(f"  Testing against null: {n_combos} combos, {n_months} months")

    # Block bootstrap (block size = 12 months) under the null
    # Null hypothesis: all combos have the same expected Sharpe
    # We subtract each combo's mean and resample
    N_BOOT = 10000
    BLOCK_SIZE = 12
    np.random.seed(42)

    ret_centered = ret_df - ret_df.mean()  # center under null
    n_blocks = n_months // BLOCK_SIZE + 1

    boot_best_sharpes = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        # Draw random blocks
        block_starts = np.random.randint(0, n_months - BLOCK_SIZE + 1, size=n_blocks)
        boot_idx = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in block_starts])[:n_months]
        boot_returns = ret_centered.values[boot_idx]
        # Compute Sharpe for each combo in this bootstrap sample
        boot_sharpes = boot_returns.mean(axis=0) / boot_returns.std(axis=0) * np.sqrt(12)
        boot_best_sharpes[b] = np.nanmax(boot_sharpes)

    # p-value: fraction of bootstrap samples where best null Sharpe >= observed best
    p_value = np.mean(boot_best_sharpes >= observed_best)
    ci_95 = np.percentile(boot_best_sharpes, 95)

    print(f"\n  Bootstrap results ({N_BOOT} replications, block size {BLOCK_SIZE}):")
    print(f"    Null distribution of max Sharpe: mean={np.mean(boot_best_sharpes):.3f}, "
          f"95th pct={ci_95:.3f}")
    print(f"    Observed best: {observed_best:.3f}")
    print(f"    p-value (multiple testing adjusted): {p_value:.4f}")
    if p_value < 0.05:
        print(f"    -> SIGNIFICANT at 5% level: the best combo is unlikely due to chance")
    else:
        print(f"    -> NOT significant at 5%: could be due to testing {n_combos} combos")

    # Also report Bonferroni-adjusted individual p-values
    print(f"\n  Bonferroni correction:")
    # Individual bootstrap p-value for each combo's Sharpe
    boot_all_sharpes = np.zeros((N_BOOT, n_combos))
    for b in range(N_BOOT):
        block_starts = np.random.randint(0, n_months - BLOCK_SIZE + 1, size=n_blocks)
        boot_idx = np.concatenate([np.arange(s, s + BLOCK_SIZE) for s in block_starts])[:n_months]
        boot_returns = ret_df.values[boot_idx]  # NOT centered -- testing if Sharpe > 0
        boot_all_sharpes[b] = boot_returns.mean(axis=0) / boot_returns.std(axis=0) * np.sqrt(12)

    for col_idx, col_name in enumerate(ret_df.columns):
        individual_ci = np.percentile(boot_all_sharpes[:, col_idx], [2.5, 97.5])
        print(f"    {col_name:40s}  Sharpe={observed_sharpes.iloc[col_idx]:.3f}  "
              f"95% CI=[{individual_ci[0]:.3f}, {individual_ci[1]:.3f}]")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: ENSEMBLE STABLE COMBOS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  STEP 4: Ensemble of stable combos")
print("=" * 80)

stable_combos = [name for name, res in stability_results.items() if res['stable']]
print(f"\n  Stable combos: {len(stable_combos)}")
for name in stable_combos:
    res = stability_results[name]
    print(f"    {name:40s}  Full={res['sr_full']:.3f}  Early={res['sr_early']:.3f}  Late={res['sr_late']:.3f}")

if len(stable_combos) >= 2:
    # Collect pi_filter series from stable combos and average them
    pi_series = []
    for name in stable_combos:
        pi_df = hmm_results[name]['pi_df'].copy()
        pi_df = pi_df.rename(columns={'pi_filter': f'pi_{name}'})
        pi_series.append(pi_df)

    # Merge all on date
    merged = pi_series[0]
    for df in pi_series[1:]:
        merged = merged.merge(df, on='date', how='outer')

    pi_cols = [c for c in merged.columns if c.startswith('pi_')]
    merged['pi_filter'] = merged[pi_cols].mean(axis=1)
    ensemble_pi = merged[['date', 'pi_filter']].dropna()

    print(f"\n  Ensemble pi_filter: {len(ensemble_pi)} months")
    print(f"  Mean pi: {ensemble_pi['pi_filter'].mean():.3f}")
    print(f"  Correlation matrix of component pi_filters:")
    corr = merged[pi_cols].corr()
    print(corr.to_string())

    # Run portfolio tests with ensemble pi
    print("\n  Running portfolio tests with ensemble pi_filter ...")

    metrics_full, r_full = run_xgb_test(ensemble_pi, 'full')
    metrics_early, r_early = run_xgb_test(ensemble_pi, 'early')
    metrics_late, r_late = run_xgb_test(ensemble_pi, 'late')

    if metrics_full:
        print(f"\n  ENSEMBLE RESULTS (XGB):")
        print(f"    Full:    Sharpe={metrics_full[2]:.3f}  Return={metrics_full[0]:.1%}  Vol={metrics_full[1]:.1%}  MaxDD={metrics_full[3]:.1%}")
        if metrics_early:
            print(f"    2011-17: Sharpe={metrics_early[2]:.3f}")
        if metrics_late:
            print(f"    2018-25: Sharpe={metrics_late[2]:.3f}")

    # Also run LR and M0
    stocks_with_pi = stocks_raw.merge(ensemble_pi[['date', 'pi_filter']], on='date', how='left')
    stocks_with_pi['pi_filter'] = stocks_with_pi['pi_filter'].ffill()
    df_cs = stocks_with_pi.dropna(subset=['ret_fwd'] + MOM_FEATURES + ['pi_filter', 'log_me']).copy()
    train_cs = df_cs[df_cs['date'] < '2011-01-01'].copy()
    test_cs = df_cs[df_cs['date'] >= '2011-01-01'].copy()

    # M0 formula
    pi_by_month = test_cs.groupby('date')['pi_filter'].first()
    lb_by_month = np.clip(np.round(12 - 11 * pi_by_month).astype(int), 1, 12)
    test_cs['score_m0'] = np.nan
    for date, lb in lb_by_month.to_dict().items():
        mask = test_cs['date'] == date
        test_cs.loc[mask, 'score_m0'] = test_cs.loc[mask, f'mom_{lb}']
    r_m0 = long_only_port(test_cs, 'score_m0')
    if len(r_m0) > 12:
        m0_metrics = compute_metrics(r_m0)
        print(f"    M0:      Sharpe={m0_metrics[2]:.3f}")

    # M1 LR
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
    r_lr = long_only_port(test_imp, 'score_lr')
    if len(r_lr) > 12:
        lr_metrics = compute_metrics(r_lr)
        print(f"    M1 LR:   Sharpe={lr_metrics[2]:.3f}")

elif len(stable_combos) == 1:
    print(f"\n  Only 1 stable combo -- no ensemble needed, just use: {stable_combos[0]}")
else:
    print(f"\n  No stable combos found -- consider relaxing stability threshold")
    # Try with threshold 0.3 instead of 0.5
    relaxed = [name for name, res in stability_results.items()
               if res['sr_early'] > 0.3 and res['sr_late'] > 0.3]
    if relaxed:
        print(f"  With relaxed threshold (>0.3): {len(relaxed)} combos")
        for name in relaxed:
            res = stability_results[name]
            print(f"    {name:40s}  Full={res['sr_full']:.3f}  Early={res['sr_early']:.3f}  Late={res['sr_late']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  FINAL COMPARISON")
print("=" * 80)

# Baseline single-combo results
if 'DD+VOL+CS+LVIX' in stability_results:
    bl = stability_results['DD+VOL+CS+LVIX']
    print(f"\n  Baseline (DD+VOL+CS+LVIX):  Full={bl['sr_full']:.3f}  "
          f"Early={bl['sr_early']:.3f}  Late={bl['sr_late']:.3f}")

# Best single combo
if stability_results:
    best_name = max(stability_results, key=lambda x: stability_results[x]['sr_full'])
    best = stability_results[best_name]
    print(f"  Best single combo ({best_name}):  Full={best['sr_full']:.3f}  "
          f"Early={best['sr_early']:.3f}  Late={best['sr_late']:.3f}")

print(f"\n  Ensemble: see above")
print("\n" + "=" * 80)
