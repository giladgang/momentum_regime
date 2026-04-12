"""
test_hmm_pca.py
===============
PCA-based HMM: group 11 candidate features into 3 categories
(volatility, credit, equity decline), extract PC1 from each,
feed 3 composite factors into the HMM, run full portfolio test.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.preprocessing import StandardScaler
import warnings, time
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  HMM PCA FACTOR TEST")
print("=" * 80)

panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
panel['year_month'] = panel['date'].dt.to_period('M')

stock_data = pd.read_parquet('data/crsp_msf_raw.parquet')
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['year_month'] = stock_data['date'].dt.to_period('M')

train_mask = panel['date'] < '2011-01-01'

# ── Compute candidate features ─────────────────────────────────────────────

print("\nComputing features ...")
import pandas_datareader.data as web

if 'TERM' not in panel.columns:
    try:
        t10 = web.DataReader('DGS10', 'fred', start='1970-01-01', end='2025-12-31')
        t2 = web.DataReader('DGS2', 'fred', start='1970-01-01', end='2025-12-31')
        term = (t10['DGS10'] - t2['DGS2']).resample('ME').mean().to_frame('TERM')
        term.index = term.index.to_period('M')
        term_df = term.reset_index(); term_df.columns = ['year_month', 'TERM']
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
        cp_df = cp_spread.reset_index(); cp_df.columns = ['year_month', 'CP_SPREAD']
        panel = panel.merge(cp_df, on='year_month', how='left')
        print("  CP_SPREAD: OK")
    except Exception as e:
        print(f"  CP_SPREAD: FAILED ({e})")

if 'HY_OAS' not in panel.columns:
    try:
        hy = web.DataReader('BAMLH0A0HYM2', 'fred', start='1970-01-01', end='2025-12-31')
        hy = hy.resample('ME').mean(); hy.columns = ['HY_OAS']
        hy.index = hy.index.to_period('M')
        hy_df = hy.reset_index(); hy_df.columns = ['year_month', 'HY_OAS']
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

# Standardize all raw features
for col in ['TERM', 'CP_SPREAD', 'HY_OAS', 'DISP', 'ADR', 'SKEW', 'REL_N']:
    zcol = f'{col}_z'
    if col in panel.columns and zcol not in panel.columns:
        vals = panel[col].astype(float)
        mu = vals[train_mask].dropna().mean()
        sd = vals[train_mask].dropna().std()
        if sd > 0 and not np.isnan(sd):
            panel[zcol] = (vals - mu) / sd

# ═══════════════════════════════════════════════════════════════════════════════
#  PCA FACTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  PCA FACTOR CONSTRUCTION")
print("=" * 80)

# Define groups
GROUPS = {
    'VOL_factor': ['VOL_z', 'LVIX_z', 'DISP_z'],
    'CREDIT_factor': ['CS_z', 'HY_OAS_z', 'CP_SPREAD_z', 'TERM_z'],
    'EQUITY_factor': ['DD_z', 'ADR_z', 'SKEW_z', 'REL_N_z'],
}

# Use only months where ALL features are available
all_feats = []
for feats in GROUPS.values():
    all_feats.extend(feats)
all_feats = list(set(all_feats))

# Build market-level dataframe (one row per month)
market = panel[['date', 'year_month'] + all_feats].drop_duplicates('year_month').dropna()
market = market.sort_values('date').reset_index(drop=True)
market_train = market[market['date'] < '2011-01-01']
market_test = market[market['date'] >= '2011-01-01']

print(f"\n  Months with all features: {len(market)} (train={len(market_train)}, test={len(market_test)})")
print(f"  Date range: {market['date'].min().date()} to {market['date'].max().date()}")

# Fit PCA on training data for each group
pca_models = {}
for factor_name, feat_list in GROUPS.items():
    available = [f for f in feat_list if f in market.columns]
    if len(available) < 2:
        print(f"  {factor_name}: only {len(available)} features available, skipping PCA")
        continue

    pca = PCA(n_components=1)
    X_train = market_train[available].values
    pca.fit(X_train)

    # Extract PC1 for all data
    X_all = market[available].values
    pc1 = pca.transform(X_all).ravel()

    # Check sign: PC1 should increase during crises
    # Use GFC as reference
    gfc_mask = ((market['date'] >= '2007-10-01') & (market['date'] <= '2009-06-01')).values
    non_crisis = ((market['date'] >= '2003-01-01') & (market['date'] <= '2006-12-01')).values
    if gfc_mask.sum() > 0 and non_crisis.sum() > 0:
        if pc1[gfc_mask].mean() < pc1[non_crisis].mean():
            pc1 = -pc1
            pca.components_ = -pca.components_

    # Standardize using training stats
    pc1_train_mean = pc1[:len(market_train)].mean()
    pc1_train_std = pc1[:len(market_train)].std()
    pc1_z = (pc1 - pc1_train_mean) / pc1_train_std

    market[factor_name] = pc1_z
    pca_models[factor_name] = pca

    var_explained = pca.explained_variance_ratio_[0]
    loadings = pca.components_[0]
    print(f"\n  {factor_name} (variance explained: {var_explained:.1%}):")
    for feat, loading in zip(available, loadings):
        print(f"    {feat:16s}  loading={loading:+.3f}")

FACTOR_NAMES = list(GROUPS.keys())

# Recreate train/test splits with factors included
market_train = market[market['date'] < '2011-01-01'].copy()
market_test = market[market['date'] >= '2011-01-01'].copy()

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS
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


def fit_hmm_and_filter(Z_train, Z_test, dates_train, seed=2201):
    np.random.seed(seed)
    T, D = Z_train.shape

    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    # Initialize by first feature
    states = (Z_train[:, 0] > np.median(Z_train[:, 0])).astype(int)

    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
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

    Z_full = np.vstack([Z_train, Z_test])
    filtered = forward_filter(Z_full, mu_post, Sigma_post, P)

    # Panic identification: crisis-calibrated sign correction
    # All PCA factors are already oriented so that higher = more stress
    # So panic state = higher sum of means
    score0 = np.sum(mu_post[0])
    score1 = np.sum(mu_post[1])
    panic_state = 1 if score1 >= score0 else 0

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

    # Crisis alignment
    gfc_mask = ((dates_train >= np.datetime64('2007-10-01')) &
                (dates_train <= np.datetime64('2009-06-01')))
    non_crisis = ((dates_train >= np.datetime64('2003-01-01')) &
                  (dates_train <= np.datetime64('2006-12-01')))
    gfc_pi = pi_filter[:T][gfc_mask].mean() if gfc_mask.sum() > 0 else 0
    non_crisis_pi = pi_filter[:T][non_crisis].mean() if non_crisis.sum() > 0 else 1

    return {
        'pi_filter': pi_filter,
        'min_ess': min_ess,
        'mu_post': mu_post,
        'panic_state': panic_state,
        'gfc_pi': gfc_pi,
        'non_crisis_pi': non_crisis_pi,
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


def run_full_test(pi_df):
    stocks_with_pi = stocks_raw.merge(pi_df[['date', 'pi_filter']], on='date', how='left')
    stocks_with_pi['pi_filter'] = stocks_with_pi['pi_filter'].ffill()
    df_cs = stocks_with_pi.dropna(subset=['ret_fwd'] + MOM_FEATURES + ['pi_filter', 'log_me']).copy()
    train_cs = df_cs[df_cs['date'] < '2011-01-01'].copy()
    test_cs = df_cs[df_cs['date'] >= '2011-01-01'].copy()
    if len(train_cs) < 1000 or len(test_cs) < 1000:
        return None

    results = {}

    # M0 Formula
    pi_by_month = test_cs.groupby('date')['pi_filter'].first()
    lb_by_month = np.clip(np.round(12 - 11 * pi_by_month).astype(int), 1, 12)
    test_cs['score_m0'] = np.nan
    for date, lb in lb_by_month.to_dict().items():
        mask = test_cs['date'] == date
        test_cs.loc[mask, 'score_m0'] = test_cs.loc[mask, f'mom_{lb}']
    r_m0 = long_only_port(test_cs, 'score_m0')
    if len(r_m0) > 12:
        results['M0'] = compute_metrics(r_m0)

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
        results['M1_LR'] = compute_metrics(r_lr)

    # M2 XGB
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
            results['M2_XGB'] = compute_metrics(r_xgb)
    except:
        pass

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN HMM WITH PCA FACTORS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  HMM WITH 3 PCA FACTORS")
print("=" * 80)

Z_train = market_train[FACTOR_NAMES].values.astype(float)
Z_test = market_test[FACTOR_NAMES].values.astype(float)
dates_train = market_train['date'].values
dates_all = market['date'].values

SEEDS = [2201, 42, 1337, 7777, 31415]

print(f"\n  Running {len(SEEDS)} seeds ...")

seed_results = []
for seed in SEEDS:
    t0 = time.time()
    hmm_res = fit_hmm_and_filter(Z_train, Z_test, dates_train, seed=seed)
    elapsed_hmm = time.time() - t0

    pi_df = pd.DataFrame({'date': dates_all, 'pi_filter': hmm_res['pi_filter']})

    port_res = run_full_test(pi_df)
    elapsed_total = time.time() - t0

    if port_res:
        m0_sr = port_res.get('M0', (0,0,0,0))[2]
        lr_sr = port_res.get('M1_LR', (0,0,0,0))[2]
        xgb_sr = port_res.get('M2_XGB', (0,0,0,0))[2]
        seed_results.append({
            'seed': seed,
            'xgb_sr': xgb_sr,
            'lr_sr': lr_sr,
            'm0_sr': m0_sr,
            'ess': hmm_res['min_ess'],
            'gfc_pi': hmm_res['gfc_pi'],
            'non_crisis_pi': hmm_res['non_crisis_pi'],
        })
        print(f"    seed={seed:5d}  M0={m0_sr:.3f}  LR={lr_sr:.3f}  XGB={xgb_sr:.3f}  "
              f"ESS={hmm_res['min_ess']:.0f}  GFC={hmm_res['gfc_pi']:.2f}  "
              f"calm={hmm_res['non_crisis_pi']:.2f}  ({elapsed_total:.0f}s)")

        # Print regime means for first seed
        if seed == SEEDS[0]:
            print(f"\n    Regime means (panic state={hmm_res['panic_state']}):")
            for j, fname in enumerate(FACTOR_NAMES):
                mu_calm = hmm_res['mu_post'][1 - hmm_res['panic_state'], j]
                mu_panic = hmm_res['mu_post'][hmm_res['panic_state'], j]
                print(f"      {fname:20s}  calm={mu_calm:+.3f}  panic={mu_panic:+.3f}  "
                      f"delta={mu_panic - mu_calm:+.3f}")
            print()
    else:
        print(f"    seed={seed:5d}  FAILED ({elapsed_total:.0f}s)")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  PCA FACTOR HMM RESULTS")
print("=" * 80)

if seed_results:
    xgb_sharpes = [r['xgb_sr'] for r in seed_results]
    lr_sharpes = [r['lr_sr'] for r in seed_results]
    ess_vals = [r['ess'] for r in seed_results]

    print(f"\n  XGB Sharpe:  mean={np.mean(xgb_sharpes):.3f}  std={np.std(xgb_sharpes):.3f}  "
          f"min={np.min(xgb_sharpes):.3f}  max={np.max(xgb_sharpes):.3f}")
    print(f"  LR Sharpe:   mean={np.mean(lr_sharpes):.3f}  std={np.std(lr_sharpes):.3f}")
    print(f"  Mean ESS:    {np.mean(ess_vals):.0f}")

    print(f"\n  COMPARISON:")
    print(f"    PCA 3-factor:        XGB={np.mean(xgb_sharpes):.3f} (std={np.std(xgb_sharpes):.3f})")
    print(f"    DD+VOL+DISP+REL_N:   XGB=1.116 (std=0.062)  [from seed stability test]")
    print(f"    Baseline:            XGB=0.843 (std=0.070)  [from seed stability test]")

print("\n" + "=" * 80)
