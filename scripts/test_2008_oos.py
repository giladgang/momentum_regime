"""
True out-of-sample test on the 2008-2009 crash.
Train HMM on 1990-1999, train XGB on 1990-1999, test on 2000-2010.
This puts the dot-com bust AND the GFC in the test period.
"""
import numpy as np, pandas as pd, pickle, warnings, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from xgboost import XGBRegressor
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
    return np.exp(la[:, 1])

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

def fit_hmm(Z_train, Z_full, seed, n_iter=2000, n_burnin=500, crisis_mask=None):
    np.random.seed(seed)
    T, D = Z_train.shape
    med = np.median(Z_train[:, 0])
    s0 = (Z_train[:, 0] < med).astype(int)  # DD below median = panic
    mu = np.array([Z_train[s0==k].mean(axis=0) if (s0==k).sum() > 1 else np.zeros(D) for k in range(K)])
    Sigma = np.array([np.cov(Z_train[s0==k].T) + 0.01*np.eye(D) if (s0==k).sum() > 2 else np.eye(D) for k in range(K)])
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
    
    # Sign correction using crisis windows in training data only
    if crisis_mask is not None and crisis_mask.sum() > 0:
        signs = np.array([1 if np.percentile(Z_train[crisis_mask, j], 95) >= np.percentile(Z_train[~crisis_mask, j], 95) else -1 for j in range(D)])
    else:
        # If no crisis in training, use DD direction (DD is negative during stress)
        signs = np.array([-1 if j == 0 else 1 for j in range(D)])
    
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
HMM_SEEDS = list(range(1, 21))  # 20 seeds
XGB_SEEDS = list(range(1, 21))  # 20 seeds

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

print("=" * 70)
print("  TRUE OUT-OF-SAMPLE TEST: 2008-2009 CRASH")
print(f"  HMM seeds: {len(HMM_SEEDS)}, XGB seeds: {len(XGB_SEEDS)}")
print("=" * 70)

# Test 1: Train 1990-1999, test 2000-2010 (dot-com + GFC both OOS)
configs = [
    ('Train 1990-1999, Test 2000-2010', '2000-01-01', '2011-01-01', None),
    ('Train 1990-2004, Test 2005-2010', '2005-01-01', '2011-01-01', True),
    ('Train 1990-2006, Test 2007-2010', '2007-01-01', '2011-01-01', True),
]

for label, test_start, test_end, use_dotcom_crisis in configs:
    t0 = time.time()
    train_end = test_start
    
    Z_train = sub[sub['date'] < train_end][HMM_FEATURES].values.astype(float)
    
    # Crisis mask for sign correction (only crises in training period)
    train_dates = sub[sub['date'] < train_end]['date']
    if use_dotcom_crisis:
        crisis_mask = ((train_dates >= '2000-03-01') & (train_dates <= '2002-10-01')).values
    else:
        # No crisis in 1990-1999 training -- use 1998 LTCM/Russia crisis as reference
        crisis_mask = ((train_dates >= '1998-07-01') & (train_dates <= '1998-10-01')).values
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  HMM train: {len(Z_train)} months")
    print(f"{'='*60}")
    
    # Average pi_filter across HMM seeds
    pi_all = np.zeros(len(Z_full))
    for hs in HMM_SEEDS:
        pi_all += fit_hmm(Z_train, Z_full, seed=hs, crisis_mask=crisis_mask)
    pi_all /= len(HMM_SEEDS)
    
    # Merge pi into stocks
    pi_df = pd.DataFrame({'date': sub['date'], 'pi_filter': pi_all})
    stocks_w = stocks.merge(pi_df, on='date', how='left')
    stocks_w['pi_filter'] = stocks_w['pi_filter'].ffill()
    
    df = stocks_w.dropna(subset=['ret_fwd'] + FEATURES).copy()
    train_xgb = df[df['date'] < train_end]
    test_xgb = df[(df['date'] >= test_start) & (df['date'] < test_end)]
    
    print(f"  XGB train: {len(train_xgb):,} obs, test: {len(test_xgb):,} obs")
    
    X_tr = train_xgb[FEATURES].values.astype(float)
    X_te = test_xgb[FEATURES].values.astype(float)
    y_tr = train_xgb['ret_fwd'].values.astype(float)
    
    # Remove NaN rows
    mask_tr = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
    mask_te = ~np.isnan(X_te).any(axis=1)
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
    X_te = X_te[mask_te]
    test_xgb = test_xgb[mask_te].copy()
    
    preds = np.zeros(len(X_te))
    for xs in XGB_SEEDS:
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                            learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                            colsample_bytree=COLSAMPLE, tree_method='hist',
                            random_state=xs, verbosity=0)
        xgb.fit(X_tr, y_tr)
        preds += xgb.predict(X_te)
    preds /= len(XGB_SEEDS)
    test_xgb['score'] = preds
    
    r = build_ls(test_xgb, 'score')
    
    if len(r) > 1 and r.std() > 0:
        sr = r.mean()/r.std()*12**0.5
        ann_ret = ((1+r).prod())**(12/len(r)) - 1
        mdd = ((1+r).cumprod() / (1+r).cumprod().cummax() - 1).min()
        
        print(f"\n  RESULTS:")
        print(f"  Test months: {len(r)}")
        print(f"  Sharpe: {sr:.2f}")
        print(f"  Ann Return: {ann_ret*100:.1f}%")
        print(f"  MDD: {mdd*100:.1f}%")
        
        # GFC-specific performance (Oct 2007 - Mar 2009)
        gfc = r[(r.index >= '2007-10-01') & (r.index <= '2009-06-01')]
        if len(gfc) > 0:
            gfc_cum = (1+gfc).prod() - 1
            print(f"\n  GFC period (Oct 2007 - Jun 2009):")
            print(f"  Months: {len(gfc)}, Cumulative: {gfc_cum*100:+.1f}%")
            print(f"  Monthly mean: {gfc.mean()*100:.2f}%")
        
        # 2009 rebound (Mar-Dec 2009) -- this is where traditional momentum crashes
        rebound = r[(r.index >= '2009-03-01') & (r.index <= '2009-12-01')]
        if len(rebound) > 0:
            reb_cum = (1+rebound).prod() - 1
            print(f"\n  2009 Rebound (Mar-Dec 2009):")
            print(f"  Months: {len(rebound)}, Cumulative: {reb_cum*100:+.1f}%")
            print(f"  (This is where traditional momentum crashed -73%)")
    else:
        print(f"  Insufficient data or zero variance")
    
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.0f}s")

# Also test fixed momentum for comparison
print(f"\n{'='*60}")
print(f"  COMPARISON: Fixed 12-month momentum (2000-2010)")
print(f"{'='*60}")
df_all = stocks.dropna(subset=['ret_fwd', 'mom_12']).copy()
test_mom = df_all[(df_all['date'] >= '2000-01-01') & (df_all['date'] < '2011-01-01')]
r_mom = build_ls(test_mom, 'mom_12')
if len(r_mom) > 1:
    sr_mom = r_mom.mean()/r_mom.std()*12**0.5
    mdd_mom = ((1+r_mom).cumprod() / (1+r_mom).cumprod().cummax() - 1).min()
    print(f"  Sharpe: {sr_mom:.2f}, MDD: {mdd_mom*100:.1f}%")
    gfc_mom = r_mom[(r_mom.index >= '2009-03-01') & (r_mom.index <= '2009-12-01')]
    if len(gfc_mom) > 0:
        print(f"  2009 Rebound: {(1+gfc_mom).prod()*100-100:+.1f}%")

print("\nDone.")
