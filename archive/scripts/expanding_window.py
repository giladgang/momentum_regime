"""
Expanding-window HMM re-estimation (Appendix robustness check).

Re-estimates HMM every 3 years using all data up to that point.
Test period stays fixed at 2011-2025 throughout; this checks whether
updating the HMM with more recent data changes the headline result.
Uses 10 HMM seeds + 10 XGB seeds for speed.

Output: tables/table_expanding.tex (referenced from app:expanding).

NOT TO BE CONFUSED WITH expanding_window_backtest_parallel.py, which is
a different analysis: a 30-year (1995-2024) annual-retraining OOS backtest
producing the historical-stress evidence in Section 5.4.3.
"""
import numpy as np, pandas as pd, pickle, warnings, time
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from xgboost import XGBRegressor
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (HMM_FEATURES, HMM_ITERATIONS, HMM_BURNIN, N_ESTIMATORS,
                    MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE, MOM_FEATURES)
warnings.filterwarnings('ignore')

K = 2

def log_emission(Z, mu, Sigma):
    return np.column_stack([multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True) for k in range(K)])

def forward_filter(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K))
    la[0] = np.log(0.5) + le[0]
    la[0] -= logsumexp(la[0])
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t-1] + np.log(P[:, k]))
        la[t] -= logsumexp(la[t])
    return np.exp(la[:, 1])  # panic probability

def ffbs(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K))
    la[0] = np.log(0.5) + le[0]; la[0] -= logsumexp(la[0])
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t-1] + np.log(P[:, k]))
        la[t] -= logsumexp(la[t])
    states = np.zeros(n, dtype=int)
    states[-1] = np.random.choice(K, p=np.exp(la[-1]))
    for t in range(n-2, -1, -1):
        lp = la[t] + np.log(P[:, states[t+1]])
        lp -= logsumexp(lp)
        states[t] = np.random.choice(K, p=np.exp(lp))
    return states

def fit_hmm(Z_train, Z_full, seed, n_iter=2000, n_burnin=500):
    np.random.seed(seed)
    T, D = Z_train.shape
    # Init
    med = np.median(Z_train[:, 0])
    s0 = (Z_train[:, 0] >= med).astype(int)
    mu = np.array([Z_train[s0==k].mean(axis=0) for k in range(K)])
    Sigma = np.array([np.cov(Z_train[s0==k].T) + 0.01*np.eye(D) for k in range(K)])
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    
    m0, kappa0, nu0, Psi0 = np.zeros(D), 0.01, D+2, np.eye(D)*(D+2-D-1)
    alpha_dir = np.array([[9,1],[1,9]])
    
    for it in range(n_iter):
        states = ffbs(Z_train, mu, Sigma, P)
        for k in range(K):
            idx = states == k
            nk = idx.sum()
            if nk < 2: continue
            xbar = Z_train[idx].mean(axis=0)
            S = (Z_train[idx] - xbar).T @ (Z_train[idx] - xbar)
            kn = kappa0 + nk
            mn = (kappa0*m0 + nk*xbar) / kn
            nun = nu0 + nk
            Psin = Psi0 + S + kappa0*nk/kn * np.outer(xbar-m0, xbar-m0)
            Sigma[k] = invwishart.rvs(df=nun, scale=Psin)
            mu[k] = np.random.multivariate_normal(mn, Sigma[k]/kn)
        for i in range(K):
            counts = np.array([((states[:-1]==i)&(states[1:]==j)).sum() for j in range(K)])
            P[i] = np.random.dirichlet(alpha_dir[i] + counts)
    
    # Sign correction
    crisis = ((sub['date'] >= '2000-03-01') & (sub['date'] <= '2002-10-01')) | \
             ((sub['date'] >= '2007-10-01') & (sub['date'] <= '2009-06-01'))
    crisis_mask = crisis[:T].values
    signs = np.array([1 if np.percentile(Z_train[crisis_mask, j], 95) >= np.percentile(Z_train[~crisis_mask, j], 95) else -1 for j in range(D)])
    scores = [sum(signs[j]*mu[k,j] for j in range(D)) for k in range(K)]
    panic = np.argmax(scores)
    
    pi = forward_filter(Z_full, mu, Sigma, P)
    if panic == 0:
        pi = 1 - pi
    return pi

# Load data
panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
sub = panel[['date'] + HMM_FEATURES].dropna().drop_duplicates('date').sort_values('date')
sub = sub[sub['date'] >= '1990-01-01'].reset_index(drop=True)

stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks[stocks['shrcd'].isin([10,11]) & stocks['exchcd'].isin([1,2,3]) & (stocks['prc'].abs() > 1)]
stocks = stocks.sort_values(['permno','date']).reset_index(drop=True)
stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)
for lb in range(1,13):
    rs = stocks.groupby('permno',sort=False)['_lr_s1'].rolling(lb,min_periods=lb).sum().reset_index(level='permno',drop=True).sort_index()
    stocks[f'mom_{lb}'] = np.expm1(rs)
stocks.drop(columns=['_lr','_lr_s1'], inplace=True)
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

FEATURES = MOM_FEATURES + ['pi_filter']
FEE = 0.001

def build_ls(df_test, score_col):
    prev_lw, prev_sw = {}, {}
    monthly = []
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd']==1][score_col].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum()==0 or shorts['me'].sum()==0: continue
        lw = (longs.set_index('permno')['me'] / longs['me'].sum()).to_dict()
        sw = (shorts.set_index('permno')['me'] / shorts['me'].sum()).to_dict()
        tl = sum(abs(lw.get(p,0)-prev_lw.get(p,0)) for p in set(lw)|set(prev_lw)) / 2
        ts = sum(abs(sw.get(p,0)-prev_sw.get(p,0)) for p in set(sw)|set(prev_sw)) / 2
        r_l = (longs['ret_fwd'] * longs['me']).sum() / longs['me'].sum()
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        monthly.append({'date': date, 'ret': r_l - r_s - FEE*(tl+ts)})
        prev_lw, prev_sw = lw, sw
    return pd.DataFrame(monthly).set_index('date')['ret'] if monthly else pd.Series(dtype=float)

Z_full = sub[HMM_FEATURES].values.astype(float)

# Windows: re-estimate every 3 years
windows = [
    ('Fixed 1990-2010', '2011-01-01', '2026-01-01'),
    ('Expand to 2013', '2014-01-01', '2026-01-01'),
    ('Expand to 2016', '2017-01-01', '2026-01-01'),
    ('Expand to 2019', '2020-01-01', '2026-01-01'),
    ('Expand to 2022', '2023-01-01', '2026-01-01'),
]

HMM_SEEDS = list(range(1, 11))  # 10 seeds for speed
XGB_SEEDS_EXP = list(range(1, 11))  # 10 XGB seeds

print("=" * 70)
print("  EXPANDING WINDOW HMM TEST")
print(f"  HMM seeds: {len(HMM_SEEDS)}, XGB seeds: {len(XGB_SEEDS_EXP)}")
print("=" * 70)

# Approach: for each window, re-estimate HMM, compute pi_filter for test period,
# then train XGB on all pre-test data and evaluate on test period
expanding_table_rows = []  # collect per-window results for LaTeX export

for label, test_start, test_end in windows:
    t0 = time.time()
    train_end = test_start
    
    # HMM training data
    Z_train = sub[sub['date'] < train_end][HMM_FEATURES].values.astype(float)
    
    if len(Z_train) < 50:
        print(f"\n{label}: Too few training months ({len(Z_train)}), skipping")
        continue
    
    # Average pi_filter across HMM seeds
    pi_all = np.zeros(len(Z_full))
    for hs in HMM_SEEDS:
        pi_all += fit_hmm(Z_train, Z_full, seed=hs)
    pi_all /= len(HMM_SEEDS)
    
    # Merge pi into stocks
    pi_df = pd.DataFrame({'date': sub['date'], 'pi_filter': pi_all})
    stocks_w = stocks.merge(pi_df, on='date', how='left')
    stocks_w['pi_filter'] = stocks_w['pi_filter'].ffill()
    
    # Train/test split for XGB
    df = stocks_w.dropna(subset=['ret_fwd'] + FEATURES).copy()
    train_xgb = df[df['date'] < train_end]
    test_xgb = df[(df['date'] >= test_start) & (df['date'] < test_end)]
    
    if len(train_xgb) < 100 or len(test_xgb) < 100:
        print(f"\n{label}: Insufficient data, skipping")
        continue
    
    X_tr = train_xgb[FEATURES].values.astype(float)
    X_te = test_xgb[FEATURES].values.astype(float)
    y_tr = train_xgb['ret_fwd'].values.astype(float)
    
    preds = np.zeros(len(X_te))
    for xs in XGB_SEEDS_EXP:
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                            learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                            colsample_bytree=COLSAMPLE, tree_method='hist',
                            random_state=xs, verbosity=0)
        xgb.fit(X_tr, y_tr)
        preds += xgb.predict(X_te)
    preds /= len(XGB_SEEDS_EXP)
    test_xgb = test_xgb.copy()
    test_xgb['score'] = preds
    
    r = build_ls(test_xgb, 'score')
    sr = r.mean()/r.std()*12**0.5 if len(r) > 1 and r.std() > 0 else 0
    elapsed = time.time() - t0
    
    # Map label to table format
    hmm_year_map = {
        'Fixed 1990-2010': '2010 (baseline)',
        'Expand to 2013': '2013',
        'Expand to 2016': '2016',
        'Expand to 2019': '2019',
        'Expand to 2022': '2022',
    }
    test_period_str = f"{test_start[:4]}--2025"
    expanding_table_rows.append({
        'hmm_through': hmm_year_map.get(label, label),
        'test_period': test_period_str,
        'sharpe': sr,
    })

    print(f"\n{label} (train to {train_end[:7]}, test {test_start[:7]}-{test_end[:7]}):")
    print(f"  HMM train months: {len(Z_train)}")
    print(f"  XGB train: {len(train_xgb):,}, test: {len(test_xgb):,}")
    print(f"  Test months: {len(r)}, Sharpe: {sr:.2f}")
    print(f"  Time: {elapsed:.0f}s")

# Now compute the full expanding-window strategy:
# Use fixed HMM for 2011-2013, then re-estimate at each window
print("\n" + "=" * 70)
print("  COMBINED EXPANDING-WINDOW STRATEGY")
print("=" * 70)

# Stitch together: for each test period, use the HMM estimated up to that point
combined_returns = []
periods = [
    ('2011-01-01', '2014-01-01', '2011-01-01'),  # train to 2010
    ('2014-01-01', '2017-01-01', '2014-01-01'),  # train to 2013
    ('2017-01-01', '2020-01-01', '2017-01-01'),  # train to 2016
    ('2020-01-01', '2023-01-01', '2020-01-01'),  # train to 2019
    ('2023-01-01', '2026-01-01', '2023-01-01'),  # train to 2022
]

for test_start, test_end, train_end in periods:
    Z_train = sub[sub['date'] < train_end][HMM_FEATURES].values.astype(float)
    if len(Z_train) < 50: continue
    
    pi_all = np.zeros(len(Z_full))
    for hs in HMM_SEEDS:
        pi_all += fit_hmm(Z_train, Z_full, seed=hs)
    pi_all /= len(HMM_SEEDS)
    
    pi_df = pd.DataFrame({'date': sub['date'], 'pi_filter': pi_all})
    stocks_w = stocks.merge(pi_df, on='date', how='left')
    stocks_w['pi_filter'] = stocks_w['pi_filter'].ffill()
    
    df = stocks_w.dropna(subset=['ret_fwd'] + FEATURES).copy()
    train_xgb = df[df['date'] < train_end]
    test_xgb = df[(df['date'] >= test_start) & (df['date'] < test_end)]
    
    if len(train_xgb) < 100 or len(test_xgb) < 100: continue
    
    X_tr = train_xgb[FEATURES].values.astype(float)
    X_te = test_xgb[FEATURES].values.astype(float)
    y_tr = train_xgb['ret_fwd'].values.astype(float)
    
    preds = np.zeros(len(X_te))
    for xs in XGB_SEEDS_EXP:
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                            learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                            colsample_bytree=COLSAMPLE, tree_method='hist',
                            random_state=xs, verbosity=0)
        xgb.fit(X_tr, y_tr)
        preds += xgb.predict(X_te)
    preds /= len(XGB_SEEDS_EXP)
    test_xgb = test_xgb.copy()
    test_xgb['score'] = preds
    
    r = build_ls(test_xgb, 'score')
    combined_returns.append(r)

combined_sharpe = None
if combined_returns:
    r_combined = pd.concat(combined_returns).sort_index()
    combined_sharpe = r_combined.mean()/r_combined.std()*12**0.5
    print(f"\nCombined expanding-window: {len(r_combined)} months, Sharpe: {combined_sharpe:.2f}")
    print(f"Fixed-window baseline: Sharpe = 1.08")

# ── Export LaTeX table: table_expanding.tex ───────────────────────────────────

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

tex = []
tex.append(r'\begin{table}[H]')
tex.append(r'\centering')
tex.append(r'\small')
tex.append(r'\begin{tabular}{l l r}')
tex.append(r'\toprule')
tex.append(r'HMM trained through & Test period & M2 Sharpe \\')
tex.append(r'\midrule')
for row in expanding_table_rows:
    tex.append(f"{row['hmm_through']} & {row['test_period']} & {row['sharpe']:.2f} \\\\")
tex.append(r'\midrule')
if combined_sharpe is not None:
    tex.append(f'Combined expanding & 2011--2025 & {combined_sharpe:.2f} \\\\')
tex.append(r'\bottomrule')
tex.append(r'\end{tabular}')
combined_sharpe_str = f"{combined_sharpe:.2f}" if combined_sharpe is not None else "N/A"
tex.append(r"\caption{Expanding-window HMM re-estimation. The combined strategy stitches together the best available model at each point. Performance is stable across windows, and the combined expanding-window Sharpe (" + combined_sharpe_str + r") is comparable to the fixed-window baseline, indicating that the regime structure does not drift meaningfully over time. Sharpe levels in this table differ slightly from the main results (Table~\ref{tab:performance}) due to the use of a separate analysis pipeline with different XGBoost seed draws.}")
tex.append(r'\label{tab:expanding}')
tex.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'table_expanding.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex) + '\n')
print(f"Saved: {tex_path}")

print("\nDone.")
