"""
test_hmm_combinations.py
========================
Exhaustive search over all C(11,4) = 330 combinations of 4 features
for the 2-state HMM. Two-pass approach:
  Pass 1: Fit HMM (fast, reduced iterations) → evaluate regime quality
  Pass 2: Top N combinations → full cross-sectional portfolio test (M0/M1/M2)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.preprocessing import StandardScaler
import warnings, time
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  EXHAUSTIVE 4-FEATURE HMM SEARCH")
print("=" * 80)

# ── Load panel and compute candidate features ───────────────────────────────

panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
panel['year_month'] = panel['date'].dt.to_period('M')

stock_data = pd.read_parquet('data/crsp_msf_raw.parquet')
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['year_month'] = stock_data['date'].dt.to_period('M')

train_mask = panel['date'] < '2011-01-01'

# ── Candidate features ──────────────────────────────────────────────────────

print("\nComputing candidate features ...")
import pandas_datareader.data as web

# TERM: 10Y - 2Y Treasury
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

# TED spread
try:
    ted = web.DataReader('TEDRATE', 'fred', start='1970-01-01', end='2025-12-31')
    ted = ted.resample('ME').mean()
    ted.columns = ['TED']
    ted.index = ted.index.to_period('M')
    ted_df = ted.reset_index()
    ted_df.columns = ['year_month', 'TED']
    panel = panel.merge(ted_df, on='year_month', how='left')
    print(f"  TED: OK")
except Exception as e:
    print(f"  TED: FAILED ({e})")

# HY_OAS
try:
    hy = web.DataReader('BAMLH0A0HYM2', 'fred', start='1970-01-01', end='2025-12-31')
    hy = hy.resample('ME').mean()
    hy.columns = ['HY_OAS']
    hy.index = hy.index.to_period('M')
    hy_df = hy.reset_index()
    hy_df.columns = ['year_month', 'HY_OAS']
    panel = panel.merge(hy_df, on='year_month', how='left')
    print(f"  HY_OAS: OK")
except Exception as e:
    print(f"  HY_OAS: FAILED ({e})")

# DISP: cross-sectional return dispersion
disp = stock_data.groupby('year_month')['ret_adj'].std().reset_index()
disp.columns = ['year_month', 'DISP']
disp['DISP'] = np.log(disp['DISP'])
panel = panel.merge(disp, on='year_month', how='left')
print("  DISP: OK")

# ADR: advance-decline ratio
adr = stock_data.groupby('year_month').apply(
    lambda g: (g['ret_adj'] > 0).mean(), include_groups=False
).reset_index()
adr.columns = ['year_month', 'ADR']
panel = panel.merge(adr, on='year_month', how='left')
print("  ADR: OK")

# SKEW: cross-sectional skewness
skew_feat = stock_data.groupby('year_month')['ret_adj'].skew().reset_index()
skew_feat.columns = ['year_month', 'SKEW']
panel = panel.merge(skew_feat, on='year_month', how='left')
print("  SKEW: OK")

# REL_N: relative number of stocks
n_stocks = stock_data.groupby('year_month')['permno'].nunique().reset_index()
n_stocks.columns = ['year_month', 'N_STOCKS']
n_stocks = n_stocks.sort_values('year_month')
n_stocks['N_STOCKS_MA12'] = n_stocks['N_STOCKS'].rolling(12).mean()
n_stocks['REL_N'] = np.log(n_stocks['N_STOCKS'] / n_stocks['N_STOCKS_MA12'])
panel = panel.merge(n_stocks[['year_month', 'REL_N']], on='year_month', how='left')
print("  REL_N: OK")

# ── Standardize all features ────────────────────────────────────────────────

CANDIDATE_RAW = ['TERM', 'TED', 'HY_OAS', 'DISP', 'ADR', 'SKEW', 'REL_N']

for col in CANDIDATE_RAW:
    if col in panel.columns:
        vals = panel[col].astype(float)
        mu = vals[train_mask].mean()
        sd = vals[train_mask].std()
        if sd > 0 and not np.isnan(sd):
            panel[f'{col}_z'] = (vals - mu) / sd

# ── Define the full feature pool ─────────────────────────────────────────────

FEATURE_POOL = {
    'DD_z':     'DD (Market drawdown)',
    'VOL_z':    'VOL (Realised volatility)',
    'CS_z':     'CS (Credit spread)',
    'LVIX_z':   'LVIX (Log VIX)',
    'TERM_z':   'TERM (Yield curve 10Y-2Y)',
    'TED_z':    'TED (Interbank stress)',
    'HY_OAS_z': 'HY_OAS (High-yield spread)',
    'DISP_z':   'DISP (Return dispersion)',
    'ADR_z':    'ADR (Advance-decline ratio)',
    'SKEW_z':   'SKEW (Return skewness)',
    'REL_N_z':  'REL_N (Relative participation)',
}

# Remove features not available in panel
available = {k: v for k, v in FEATURE_POOL.items() if k in panel.columns}
FEAT_NAMES = list(available.keys())
FEAT_DESCS = available

print(f"\n  Feature pool: {len(FEAT_NAMES)} features")
for f, d in FEAT_DESCS.items():
    n_valid = panel[f].notna().sum()
    print(f"    {f:<12} {d:<35} ({n_valid} months)")

all_combos = list(combinations(FEAT_NAMES, 4))
print(f"\n  Total 4-feature combinations: C({len(FEAT_NAMES)},4) = {len(all_combos)}")

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

K = 2

def log_emission(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])

def ffbs(Z, mu, Sigma, P, init_prior=None):
    n = len(Z)
    log_emit = log_emission(Z, mu, Sigma)
    log_alpha = np.zeros((n, K))
    if init_prior is None:
        init_prior = np.full(K, 1.0 / K)
    log_alpha[0] = np.log(init_prior + 1e-300) + log_emit[0]
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


def fit_hmm_fast(Z_train, Z_test, n_iter=600, n_burnin=200, seed=2201):
    """Fast HMM fit for screening."""
    np.random.seed(seed)
    T, D = Z_train.shape
    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    # Init: split by first feature median
    states = (Z_train[:, 0] > np.median(Z_train[:, 0])).astype(int)
    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
    P_mat = np.array([[0.95, 0.05], [0.10, 0.90]])

    n_keep = n_iter - n_burnin
    mu_draws = np.zeros((n_keep, K, D))
    Sigma_draws = np.zeros((n_keep, K, D, D))
    P_draws = np.zeros((n_keep, K, K))

    for m in range(n_iter):
        states = ffbs(Z_train, mu, Sigma, P_mat)
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
                pass  # keep previous draw
        for i in range(K):
            counts = np.array([np.sum((states[:-1] == i) & (states[1:] == j))
                               for j in range(K)], dtype=float)
            P_mat[i] = np.random.dirichlet(alpha_dir[i] + counts)
        if m >= n_burnin:
            idx = m - n_burnin
            mu_draws[idx] = mu
            Sigma_draws[idx] = Sigma
            P_draws[idx] = P_mat

    mu_post = mu_draws.mean(axis=0)
    Sigma_post = Sigma_draws.mean(axis=0)
    P_post = P_draws.mean(axis=0)

    # Identify panic: state with larger L2 norm of mean vector
    panic_state = 1 if np.sum(mu_post[1]**2) >= np.sum(mu_post[0]**2) else 0

    Z_full = np.vstack([Z_train, Z_test])
    filtered = forward_filter(Z_full, mu_post, Sigma_post, P_post)
    pi_filter = filtered[:, panic_state]

    return {
        'mu_post': mu_post, 'Sigma_post': Sigma_post, 'P_post': P_post,
        'panic_state': panic_state,
        'pi_filter': pi_filter,
        'pi_filter_train': pi_filter[:len(Z_train)],
        'pi_filter_test': pi_filter[len(Z_train):],
    }


# ── Crisis ground truth ─────────────────────────────────────────────────────

CRISIS_PERIODS = [
    ('2000-03-01', '2002-10-01'),
    ('2007-10-01', '2009-06-01'),
    ('2020-02-01', '2021-12-01'),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 1: FAST HMM SCREEN (all 330 combinations)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(f"  PASS 1: Screening all {len(all_combos)} combinations (fast HMM, 600 iter)")
print("=" * 80)

screen_results = []
t_start = time.time()

for i, combo in enumerate(all_combos):
    feat_list = list(combo)
    short_name = "+".join(f.replace("_z", "") for f in feat_list)

    # Prepare data
    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01']  # LVIX starts 1990
    if len(sub) < 200:
        continue

    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']

    if len(sub_train) < 150 or len(sub_test) < 50:
        continue

    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)
    dates_all = sub['date'].values

    try:
        result = fit_hmm_fast(Z_tr, Z_te)
        pi_f = result['pi_filter']

        # Crisis detection AUC
        is_crisis = np.zeros(len(dates_all), dtype=bool)
        for start, end in CRISIS_PERIODS:
            mask = (dates_all >= pd.Timestamp(start)) & (dates_all <= pd.Timestamp(end))
            is_crisis |= mask

        auc = roc_auc_score(is_crisis.astype(float), pi_f)
        brier = brier_score_loss(is_crisis.astype(float), pi_f)
        crisis_pi = pi_f[is_crisis].mean()
        calm_pi = pi_f[~is_crisis].mean()
        contrast = crisis_pi - calm_pi

        # COVID out-of-sample detection
        dates_test = sub_test['date'].values
        pi_test = result['pi_filter_test']
        covid_mask = (dates_test >= pd.Timestamp('2020-02-01')) & \
                     (dates_test <= pd.Timestamp('2021-12-01'))
        covid_det = (pi_test[covid_mask] > 0.5).mean() if covid_mask.sum() > 0 else 0

        # Panic fraction in test
        frac_panic = (pi_test > 0.5).mean()

        # Regime separation
        mu = result['mu_post']
        ps = result['panic_state']
        sep = np.abs(mu[ps] - mu[1-ps]).mean()

        screen_results.append({
            'combo': feat_list,
            'name': short_name,
            'auc': auc,
            'brier': brier,
            'crisis_pi': crisis_pi,
            'calm_pi': calm_pi,
            'contrast': contrast,
            'covid_det': covid_det,
            'frac_panic': frac_panic,
            'sep': sep,
            'n_train': len(Z_tr),
        })

    except Exception as e:
        pass

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(all_combos) - i - 1)
        print(f"  {i+1}/{len(all_combos)} done  "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

elapsed_total = time.time() - t_start
print(f"\n  Pass 1 complete: {len(screen_results)} valid combinations "
      f"in {elapsed_total:.0f}s")

# ── Sort and display top results ─────────────────────────────────────────

df_screen = pd.DataFrame(screen_results)
df_screen = df_screen.sort_values('auc', ascending=False).reset_index(drop=True)

print(f"\n  TOP 30 by AUC (crisis detection):\n")
print(f"  {'Rank':<5} {'Features':<45} {'AUC':>6} {'Brier':>6} "
      f"{'Cris↑':>6} {'Calm↓':>6} {'Δ':>6} {'COV%':>5} {'Sep':>5}")
print("  " + "-" * 100)

for idx, row in df_screen.head(30).iterrows():
    is_baseline = set(row['combo']) == {'DD_z', 'VOL_z', 'CS_z', 'LVIX_z'}
    marker = " ◄ BASELINE" if is_baseline else ""
    print(f"  {idx+1:<5} {row['name']:<45} {row['auc']:>6.3f} {row['brier']:>6.3f} "
          f"{row['crisis_pi']:>6.3f} {row['calm_pi']:>6.3f} "
          f"{row['contrast']:>+6.3f} {row['covid_det']*100:>5.0f} "
          f"{row['sep']:>5.2f}{marker}")

# Show baseline rank
baseline_mask = df_screen['combo'].apply(
    lambda x: set(x) == {'DD_z', 'VOL_z', 'CS_z', 'LVIX_z'})
if baseline_mask.any():
    baseline_rank = df_screen[baseline_mask].index[0] + 1
    print(f"\n  Baseline (DD+VOL+CS+LVIX) rank: {baseline_rank} / {len(df_screen)}")

# ── Bottom 10 (worst) ───────────────────────────────────────────────────

print(f"\n  BOTTOM 10 (worst AUC):\n")
for idx, row in df_screen.tail(10).iterrows():
    print(f"  {idx+1:<5} {row['name']:<45} {row['auc']:>6.3f} "
          f"{row['contrast']:>+6.3f} {row['covid_det']*100:>5.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 2: FULL PORTFOLIO TEST (top N combinations)
# ═══════════════════════════════════════════════════════════════════════════════

TOP_N = 15  # test top 15 + baseline

# Always include baseline
top_combos = df_screen.head(TOP_N)['combo'].tolist()
baseline_combo = ['DD_z', 'VOL_z', 'CS_z', 'LVIX_z']
if baseline_combo not in top_combos:
    top_combos.append(baseline_combo)

print(f"\n" + "=" * 80)
print(f"  PASS 2: Full portfolio test on top {len(top_combos)} combinations")
print("=" * 80)

# ── Load stock data ──────────────────────────────────────────────────────

print("\n  Loading stock data ...")
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw.sort_values(['permno', 'date']).reset_index(drop=True)
stocks_raw = stocks_raw[stocks_raw['shrcd'].isin([10, 11])]
stocks_raw = stocks_raw[stocks_raw['exchcd'].isin([1, 2, 3])]
stocks_raw = stocks_raw[stocks_raw['prc'].abs() > 1.0].reset_index(drop=True)

print("  Computing momentum signals ...")
stocks_raw['_log_ret'] = np.log1p(stocks_raw['ret_adj'].clip(lower=-0.999))
stocks_raw['_log_ret_s1'] = stocks_raw.groupby('permno')['_log_ret'].shift(1)
for lb in range(1, 13):
    roll_sum = (
        stocks_raw.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks_raw[f'mom_{lb}'] = np.expm1(roll_sum)
stocks_raw.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

stocks_raw['log_me'] = np.log(
    stocks_raw.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan))
stocks_raw['ret_fwd'] = stocks_raw.groupby('permno')['ret_adj'].transform(
    lambda x: x.shift(-1))

MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]
ALL_FEATURES = MOM_FEATURES + [
    'pi_filter', 'bm', 'roe', 'earnings_growth',
    'leverage', 'asset_growth', 'gross_profit_a', 'log_me'
]
CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me']
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


def run_portfolio_test(pi_df):
    """Run M0/M1/M2 with given pi_filter and return metrics dict."""
    stocks_with_pi = stocks_raw.merge(pi_df[['date', 'pi_filter']], on='date', how='left')
    stocks_with_pi['pi_filter'] = stocks_with_pi['pi_filter'].ffill()
    df_cs = stocks_with_pi.dropna(subset=['ret_fwd'] + CORE_FEATURES).copy().reset_index(drop=True)
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
    r_m0 = long_only_port(test_cs, 'score_m0')
    if len(r_m0) > 12:
        results['M0'] = compute_metrics(r_m0)

    # M1: LR
    train_imp = train_cs.copy()
    test_imp = test_cs.copy()
    fund_cols = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth', 'gross_profit_a']
    for col in fund_cols:
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
        r_xgb = long_only_port(test_cs, 'score_xgb')
        if len(r_xgb) > 12:
            results['M2_XGB'] = compute_metrics(r_xgb)
    except:
        pass

    return results


# ── Run portfolio tests ──────────────────────────────────────────────────

portfolio_results = {}
t_start = time.time()

for i, feat_list in enumerate(top_combos):
    short_name = "+".join(f.replace("_z", "") for f in feat_list)
    is_baseline = set(feat_list) == set(baseline_combo)
    print(f"\n  [{i+1}/{len(top_combos)}] {short_name}" +
          (" (BASELINE)" if is_baseline else ""), end=" ... ", flush=True)

    # Fit HMM (full iterations for portfolio test)
    sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
    sub = sub[sub['date'] >= '1990-01-01']
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']
    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)

    result = fit_hmm_fast(Z_tr, Z_te, n_iter=1000, n_burnin=300, seed=2201)

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

# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  FINAL RESULTS: Portfolio Sharpe by HMM feature combination")
print("=" * 80)

all_methods = sorted(set(m for res in portfolio_results.values() for m in res))

# Sort by M1_LR Sharpe
sorted_port = sorted(portfolio_results.items(),
                      key=lambda x: x[1].get('M1_LR', (0,0,0,0))[2], reverse=True)

print(f"\n  {'Rank':<5} {'HMM Features':<45}", end="")
for m in all_methods:
    print(f" {m+' SR':>9} {m+' Ret':>8} {m+' Vol':>8} {m+' MDD':>8}", end="")
print()
print("  " + "-" * (50 + len(all_methods) * 36))

for rank, (name, methods) in enumerate(sorted_port, 1):
    is_baseline = name == "DD+VOL+CS+LVIX"
    marker = " ◄" if is_baseline else ""
    print(f"  {rank:<5} {name:<45}", end="")
    for m in all_methods:
        if m in methods:
            ar, av, sr, mdd = methods[m]
            print(f" {sr:>9.3f} {ar:>7.1%} {av:>7.1%} {mdd:>7.1%}", end="")
        else:
            print(f" {'--':>9} {'--':>8} {'--':>8} {'--':>8}", end="")
    print(marker)

# ── Best by each method ──────────────────────────────────────────────────

print(f"\n  BEST COMBINATION BY METHOD:")
print("  " + "-" * 60)
for m in all_methods:
    best_name, best_sr = None, -999
    for name, methods in portfolio_results.items():
        if m in methods:
            sr = methods[m][2]
            if sr > best_sr:
                best_sr = sr
                best_name = name
    if best_name:
        ar, av, sr, mdd = portfolio_results[best_name][m]
        print(f"  {m:<10}  {best_name:<40}  SR={sr:.3f}  Ret={ar:.1%}  Vol={av:.1%}")

# Baseline comparison
bl_name = "DD+VOL+CS+LVIX"
if bl_name in portfolio_results:
    print(f"\n  BASELINE COMPARISON:")
    print("  " + "-" * 60)
    for m in all_methods:
        if m in portfolio_results[bl_name]:
            bl_sr = portfolio_results[bl_name][m][2]
            best_name, best_sr = None, -999
            for name, methods in portfolio_results.items():
                if m in methods and methods[m][2] > best_sr:
                    best_sr = methods[m][2]
                    best_name = name
            delta = best_sr - bl_sr
            print(f"  {m:<10}  Baseline SR={bl_sr:.3f}  Best SR={best_sr:.3f} "
                  f"({best_name})  Δ={delta:+.3f}")

# ── Plot ─────────────────────────────────────────────────────────────────

if len(sorted_port) > 1:
    fig, axes = plt.subplots(1, len(all_methods), figsize=(7 * len(all_methods), 8))
    if len(all_methods) == 1:
        axes = [axes]

    for ax_idx, method in enumerate(all_methods):
        names_plot, sharpes_plot = [], []
        for name, methods in sorted_port:
            if method in methods:
                names_plot.append(name)
                sharpes_plot.append(methods[method][2])

        colors = ['gold' if 'DD+VOL+CS+LVIX' == n else 'steelblue' for n in names_plot]
        axes[ax_idx].barh(names_plot, sharpes_plot, color=colors, alpha=0.85,
                          edgecolor='white')
        axes[ax_idx].set_xlabel('Sharpe Ratio', fontsize=10)
        axes[ax_idx].set_title(f'{method}\n(gold = current baseline)', fontsize=11)
        axes[ax_idx].invert_yaxis()
        for i, v in enumerate(sharpes_plot):
            axes[ax_idx].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7)

    plt.suptitle('Portfolio Sharpe: all tested 4-feature HMM combinations\n'
                 'Same cross-sectional models (M0/M1/M2), different regime signals',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig('hmm_combo_portfolio_comparison.png', dpi=150)
    plt.close(fig)
    print("\n  Saved: hmm_combo_portfolio_comparison.png")

print("\n" + "=" * 80)
print("  ALL DONE")
print("=" * 80)
