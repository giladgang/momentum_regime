"""
test_multistate_hmm.py
======================
Comprehensive analysis of K-state HMMs (K=2,3,4,5).
- Tests whether K=3 XGB improvement is robust across seeds
- Examines regime interpretations for K>2
- Reports state means, frequencies, and downstream portfolio Sharpe ratios
"""

import numpy as np
import pandas as pd
import pickle, warnings, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA (same as robustness_checks.py)
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

# Merge pi_filter (will be replaced per K)
stocks = stocks.merge(panel[['date', 'pi_filter']], on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth',
                 'gross_profit_a', 'log_me']
FEATURES = MOM_FEATURES + ['pi_filter'] + FUND_FEATURES
CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me']

TRADING_FEE = 0.001

df = stocks.dropna(subset=['ret_fwd'] + CORE_FEATURES).copy().reset_index(drop=True)
train_base = df[df['date'] < '2011-01-01'].copy()
test_base = df[df['date'] >= '2011-01-01'].copy()

# HMM data
sub_panel = panel[['date', 'DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].dropna().reset_index(drop=True)
sub_panel = sub_panel[sub_panel['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
train_panel = sub_panel[sub_panel['date'] < '2011-01-01']
dates_train = train_panel['date'].values
dates_all = sub_panel['date'].values

Z_tr = train_panel[['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].values.astype(float)
Z_full = sub_panel[['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']].values.astype(float)
T_tr, D = Z_tr.shape

print(f"  Train stocks: {len(train_base):,}  |  Test stocks: {len(test_base):,}")
print(f"  HMM train months: {T_tr}  |  HMM total months: {len(Z_full)}")


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


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 6: return 0, 0, 0, 0
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


# ═══════════════════════════════════════════════════════════════════════════════
#  K-STATE HMM
# ═══════════════════════════════════════════════════════════════════════════════

def log_emission(Z, mu, Sigma, K):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])


def forward_filter(Z, mu, Sigma, P, K):
    n = len(Z)
    log_emit = log_emission(Z, mu, Sigma, K)
    log_alpha = np.zeros((n, K))
    log_alpha[0] = np.log(1.0 / K) + log_emit[0]
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)


def ffbs(Z, mu, Sigma, P, K):
    n = len(Z)
    log_emit = log_emission(Z, mu, Sigma, K)
    log_alpha = np.zeros((n, K))
    log_alpha[0] = np.log(1.0 / K) + log_emit[0]
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


def fit_hmm_k(Z_train, Z_full, dates_train, K, seed=42, n_iter=2000):
    """Fit a K-state Gaussian HMM via Gibbs sampling."""
    np.random.seed(seed)
    T_tr, D = Z_train.shape

    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.ones((K, K)) + 8 * np.eye(K)

    # Initialise by quantile split on DD
    quantiles = np.linspace(0, 100, K + 1)
    boundaries = np.percentile(Z_train[:, 0], quantiles)
    states = np.zeros(T_tr, dtype=int)
    for k in range(K):
        if k < K - 1:
            mask = (Z_train[:, 0] >= boundaries[k]) & (Z_train[:, 0] < boundaries[k+1])
        else:
            mask = Z_train[:, 0] >= boundaries[k]
        states[mask] = k

    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)

    P = np.ones((K, K)) * (0.05 / (K - 1)) + 0.95 * np.eye(K)
    P /= P.sum(axis=1, keepdims=True)

    for m in range(n_iter):
        states = ffbs(Z_train, mu, Sigma, P, K)
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

    filtered = forward_filter(Z_full, mu, Sigma, P, K)

    # Identify panic state by crisis overlap
    crisis_windows = [('2000-03-01', '2002-10-01'), ('2007-10-01', '2009-06-01')]
    crisis_mask = np.zeros(T_tr, dtype=bool)
    for s, e in crisis_windows:
        crisis_mask |= ((dates_train >= np.datetime64(s)) & (dates_train <= np.datetime64(e)))
    crisis_probs = filtered[:T_tr][crisis_mask].mean(axis=0)
    panic_state = np.argmax(crisis_probs)

    # Sort states by mean DD (ascending = most stressed first)
    dd_means = mu[:, 0]
    state_order = np.argsort(dd_means)  # lowest DD (most stressed) first

    return filtered, mu, P, panic_state, state_order


def build_pi_and_evaluate(filtered, panic_state, K):
    """Build pi_filter from filtered probs and evaluate M1/M2."""
    pi_panic = filtered[:, panic_state]

    pi_df = pd.DataFrame({'date': dates_all, 'pi_filter': pi_panic})
    stocks_k = df.copy()
    stocks_k = stocks_k.drop(columns=['pi_filter'])
    stocks_k = stocks_k.merge(pi_df, on='date', how='left')
    stocks_k['pi_filter'] = stocks_k['pi_filter'].ffill()

    train_k = stocks_k[stocks_k['date'] < '2011-01-01'].copy()
    test_k = stocks_k[stocks_k['date'] >= '2011-01-01'].copy()

    X_tr = train_k[FEATURES].values.astype(float)
    X_te = test_k[FEATURES].values.astype(float)
    y_tr = train_k['ret_fwd'].values.astype(float)

    for j in range(X_tr.shape[1]):
        col_median = np.nanmedian(X_tr[:, j])
        X_tr[np.isnan(X_tr[:, j]), j] = col_median
        X_te[np.isnan(X_te[:, j]), j] = col_median

    # LR
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    train_k['above_med'] = train_k.groupby('date')['ret_fwd'].transform(
        lambda x: (x >= x.median()).astype(int))
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_tr_s, train_k['above_med'].values)
    test_k['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
    r_lr = long_only_port(test_k, 'score_lr')

    # XGB
    xgb = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method='hist', random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    test_k['score_xgb'] = xgb.predict(X_te)
    r_xgb = long_only_port(test_k, 'score_xgb')

    _, _, sh_lr, _ = metrics(r_lr)
    _, vol_lr, _, mdd_lr = metrics(r_lr)
    ret_lr = (1 + r_lr).prod() ** (12 / len(r_lr)) - 1

    _, _, sh_xgb, _ = metrics(r_xgb)
    _, vol_xgb, _, mdd_xgb = metrics(r_xgb)
    ret_xgb = (1 + r_xgb).prod() ** (12 / len(r_xgb)) - 1

    return sh_lr, sh_xgb, ret_lr, ret_xgb, vol_lr, vol_xgb, mdd_lr, mdd_xgb


def build_multi_pi_and_evaluate(filtered, state_order, K):
    """Feed ALL K state probabilities as features (replace single pi_filter with K columns)."""
    # Replace pi_filter with the K filtered probabilities
    # Use states ordered by stress level (state_order[0] = most stressed)
    pi_df = pd.DataFrame({'date': dates_all})
    for rank, state_idx in enumerate(state_order):
        pi_df[f'pi_state_{rank}'] = filtered[:, state_idx]

    stocks_k = df.copy()
    stocks_k = stocks_k.drop(columns=['pi_filter'])
    stocks_k = stocks_k.merge(pi_df, on='date', how='left')
    for rank in range(K):
        stocks_k[f'pi_state_{rank}'] = stocks_k[f'pi_state_{rank}'].ffill()

    # Modify feature list: replace pi_filter with K state probabilities
    features_multi = MOM_FEATURES + [f'pi_state_{r}' for r in range(K)] + FUND_FEATURES

    train_k = stocks_k[stocks_k['date'] < '2011-01-01'].copy()
    test_k = stocks_k[stocks_k['date'] >= '2011-01-01'].copy()

    X_tr = train_k[features_multi].values.astype(float)
    X_te = test_k[features_multi].values.astype(float)
    y_tr = train_k['ret_fwd'].values.astype(float)

    for j in range(X_tr.shape[1]):
        col_median = np.nanmedian(X_tr[:, j])
        X_tr[np.isnan(X_tr[:, j]), j] = col_median
        X_te[np.isnan(X_te[:, j]), j] = col_median

    # XGB only (LR won't use these well anyway)
    xgb = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method='hist', random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    test_k['score_xgb'] = xgb.predict(X_te)
    r_xgb = long_only_port(test_k, 'score_xgb')
    _, _, sh_xgb, _ = metrics(r_xgb)
    ret_xgb = (1 + r_xgb).prod() ** (12 / len(r_xgb)) - 1
    vol_xgb = r_xgb.std() * np.sqrt(12)

    return sh_xgb, ret_xgb, vol_xgb


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN: TEST K=2,3,4,5
# ═══════════════════════════════════════════════════════════════════════════════

SEEDS = [42, 2201, 1337, 7, 999]
K_VALUES = [2, 3, 4, 5]

print("\n" + "=" * 100)
print("  MULTI-STATE HMM ANALYSIS (K=2,3,4,5)")
print("=" * 100)

all_results = {}  # K -> list of (seed, sh_lr, sh_xgb, ...)

for K in K_VALUES:
    print(f"\n{'─' * 80}")
    print(f"  K = {K}")
    print(f"{'─' * 80}")

    t0 = time.time()
    seed_results = []
    multi_pi_sharpes = []

    for seed in SEEDS:
        filtered, mu, P, panic_state, state_order = fit_hmm_k(
            Z_tr, Z_full, dates_train, K=K, seed=seed, n_iter=2000)

        # Method A: single panic probability -> pi_filter
        sh_lr, sh_xgb, ret_lr, ret_xgb, vol_lr, vol_xgb, mdd_lr, mdd_xgb = \
            build_pi_and_evaluate(filtered, panic_state, K)
        seed_results.append({
            'seed': seed, 'sh_lr': sh_lr, 'sh_xgb': sh_xgb,
            'ret_xgb': ret_xgb, 'vol_xgb': vol_xgb, 'mdd_xgb': mdd_xgb,
        })

        # Method B: all K state probabilities as features (XGB only)
        sh_xgb_multi, ret_xgb_multi, vol_xgb_multi = \
            build_multi_pi_and_evaluate(filtered, state_order, K)
        multi_pi_sharpes.append(sh_xgb_multi)

    elapsed = time.time() - t0
    print(f"  Fitted {len(SEEDS)} seeds in {elapsed:.0f}s\n")

    # Report state means from last seed
    print(f"  Regime means (last seed, sorted by DD):")
    for rank, state_idx in enumerate(state_order):
        label = "stressed" if rank == 0 else ("calm" if rank == K-1 else f"intermediate-{rank}")
        pct_time = filtered[:T_tr, state_idx].mean() * 100
        print(f"    State {rank} ({label:>15s}, {pct_time:4.1f}% of train): "
              f"DD={mu[state_idx,0]:+.2f}  VOL={mu[state_idx,1]:+.2f}  "
              f"DISP={mu[state_idx,2]:+.2f}  REL_N={mu[state_idx,3]:+.2f}")

    # Transition matrix
    print(f"\n  Transition matrix (last seed):")
    P_sorted = P[state_order][:, state_order]
    header = "    " + "".join([f"  {'S'+str(k):>6s}" for k in range(K)])
    print(header)
    for i in range(K):
        row = f"    S{i}"
        for j in range(K):
            row += f"  {P_sorted[i,j]:6.3f}"
        print(row)

    # Per-seed results
    print(f"\n  Per-seed results (single panic prob -> pi_filter):")
    print(f"  {'Seed':>6s}  {'M1 Sharpe':>10s}  {'M2 Sharpe':>10s}  {'M2 Ret':>8s}  {'M2 Vol':>8s}  {'M2 MDD':>8s}")
    print(f"  {'─' * 60}")
    for r in seed_results:
        print(f"  {r['seed']:>6d}  {r['sh_lr']:>10.3f}  {r['sh_xgb']:>10.3f}  "
              f"{r['ret_xgb']:>7.1%}  {r['vol_xgb']:>7.1%}  {r['mdd_xgb']:>7.1%}")

    # Averages
    avg_lr = np.mean([r['sh_lr'] for r in seed_results])
    std_lr = np.std([r['sh_lr'] for r in seed_results])
    avg_xgb = np.mean([r['sh_xgb'] for r in seed_results])
    std_xgb = np.std([r['sh_xgb'] for r in seed_results])
    print(f"\n  Mean +/- Std:")
    print(f"    M1: {avg_lr:.3f} +/- {std_lr:.3f}")
    print(f"    M2: {avg_xgb:.3f} +/- {std_xgb:.3f}")

    # Multi-pi results
    avg_multi = np.mean(multi_pi_sharpes)
    std_multi = np.std(multi_pi_sharpes)
    print(f"\n  M2 with ALL {K} state probs as features:")
    print(f"    Per seed: {['%.3f' % s for s in multi_pi_sharpes]}")
    print(f"    Mean +/- Std: {avg_multi:.3f} +/- {std_multi:.3f}")

    all_results[K] = {
        'seed_results': seed_results,
        'avg_lr': avg_lr, 'std_lr': std_lr,
        'avg_xgb': avg_xgb, 'std_xgb': std_xgb,
        'avg_multi': avg_multi, 'std_multi': std_multi,
        'multi_sharpes': multi_pi_sharpes,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 100)
print("  SUMMARY: SHARPE RATIOS BY NUMBER OF HMM STATES")
print("=" * 100)

print(f"\n  {'K':>3s}  {'M1 (mean)':>10s}  {'M1 (std)':>9s}  {'M2 single-pi':>13s}  {'M2 (std)':>9s}  {'M2 multi-pi':>12s}  {'M2m (std)':>10s}")
print(f"  {'─' * 75}")
for K in K_VALUES:
    r = all_results[K]
    print(f"  {K:>3d}  {r['avg_lr']:>10.3f}  {r['std_lr']:>9.3f}  "
          f"{r['avg_xgb']:>13.3f}  {r['std_xgb']:>9.3f}  "
          f"{r['avg_multi']:>12.3f}  {r['std_multi']:>10.3f}")

print(f"\n  Notes:")
print(f"  - 'single-pi': only panic state probability fed as pi_filter (same as baseline approach)")
print(f"  - 'multi-pi': all K state probabilities fed as separate features to XGB")
print(f"  - Each result averaged over {len(SEEDS)} seeds; std reflects cross-seed variation")
print(f"  - K=2 baseline from main paper: M1=1.038, M2=1.007")

print("\n" + "=" * 100)
print("  DONE")
print("=" * 100)
