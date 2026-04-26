"""
Diagnose CRRA counterintuitive result: higher gamma -> higher portfolio vol.

Checks:
  1. sigma alignment: train['sigma'][i] == trailing-12m std of ret_adj for that (permno, date)?
  2. Target sanity at gamma=0 and gamma=10
  3. Correlation corr(pred, sigma) across gamma
  4. Long-leg sigma vs short-leg sigma at each gamma
  5. Realized beta (L, S) vs market to confirm (beta_L - beta_S) story
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import pickle, warnings
warnings.filterwarnings('ignore')

EPS = 0.01
SIGMA_FLOOR = 0.01
TRADING_FEE = 0.001

print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

train = artefacts['train'].copy()
test  = artefacts['test'].copy()
X_train = artefacts['X_train']
X_test  = artefacts['X_test']
FEATURES = artefacts['FEATURES']
print(f"  Train {len(train):,} | Test {len(test):,} | Features {len(FEATURES)}")

stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw.sort_values(['permno', 'date'])
stocks_raw['trail_var'] = (
    stocks_raw.groupby('permno')['ret_adj']
    .transform(lambda x: x.shift(1).rolling(12, min_periods=6).var())
)
tv_map = stocks_raw.set_index(['permno', 'date'])['trail_var']

train['trail_var'] = train.set_index(['permno', 'date']).index.map(
    lambda idx: tv_map.get(idx, np.nan))
test['trail_var'] = test.set_index(['permno', 'date']).index.map(
    lambda idx: tv_map.get(idx, np.nan))

tv_median = train['trail_var'].median()
train['trail_var'] = train['trail_var'].fillna(tv_median)
test['trail_var']  = test['trail_var'].fillna(tv_median)

train['sigma'] = np.sqrt(np.maximum(train['trail_var'].values, SIGMA_FLOOR**2))
test['sigma']  = np.sqrt(np.maximum(test['trail_var'].values,  SIGMA_FLOOR**2))

# ── 1. sigma alignment spot-check ─────────────────────────────────────────
print("\n[1] sigma alignment spot-check")
sample = train.sample(5, random_state=1)[['permno', 'date', 'sigma']]
for _, row in sample.iterrows():
    p, d, s_cached = row['permno'], row['date'], row['sigma']
    hist = stocks_raw[(stocks_raw['permno'] == p) & (stocks_raw['date'] < d)].tail(12)
    s_recomp = np.sqrt(max(hist['ret_adj'].var(), SIGMA_FLOOR**2))
    hit = "ok" if abs(s_cached - s_recomp) / max(s_recomp, 1e-6) < 1e-3 else "MISMATCH"
    print(f"  permno={p} date={d.date()} sigma_cached={s_cached:.4f} "
          f"sigma_recomp={s_recomp:.4f} [{hit}]")

# ── 2. target sanity ──────────────────────────────────────────────────────
def crra(r, sigma, gamma, eps=EPS):
    return np.sign(r) * np.log1p(np.abs(r) / eps) - gamma * np.log(sigma)

print("\n[2] target distribution")
for g in [0.0, 1.0, 10.0]:
    y = crra(train['ret_fwd'].values, train['sigma'].values, g)
    print(f"  gamma={g:>4.1f}: mean={y.mean():+.3f} std={y.std():.3f} "
          f"corr(y, -log_sigma)={np.corrcoef(y, -np.log(train['sigma']))[0,1]:+.3f} "
          f"corr(y, ret_fwd)={np.corrcoef(y, train['ret_fwd'])[0,1]:+.3f}")

# ── 3 & 4. train single-seed model per gamma, check composition ──────────
print("\n[3-4] single-seed XGB per gamma: pred-sigma corr + leg composition")
r_train = train['ret_fwd'].values
sigma_train = train['sigma'].values
log_sigma_test = np.log(test['sigma'].values)

results = []
for gamma in [0.0, 1.0, 5.0, 10.0]:
    y_tr = crra(r_train, sigma_train, gamma)
    model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         tree_method='hist', random_state=1, verbosity=0)
    model.fit(X_train, y_tr)
    preds = model.predict(X_test)

    corr_ps = np.corrcoef(preds, log_sigma_test)[0, 1]
    corr_pr = np.corrcoef(preds, test['ret_fwd'].values)[0, 1]

    tmp = test.copy()
    tmp['score'] = preds
    leg_sigmas_L, leg_sigmas_S = [], []
    leg_rets_L, leg_rets_S = [], []
    for date, grp in tmp.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score'].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp['score'] >= hi]
        shorts = grp[grp['score'] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
        leg_sigmas_L.append((longs['sigma']  * longs['me']).sum()  / longs['me'].sum())
        leg_sigmas_S.append((shorts['sigma'] * shorts['me']).sum() / shorts['me'].sum())
        leg_rets_L.append((longs['ret_fwd']  * longs['me']).sum()  / longs['me'].sum())
        leg_rets_S.append((shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum())
    rL = np.array(leg_rets_L); rS = np.array(leg_rets_S)
    ls = rL - rS

    print(f"  gamma={gamma:>4.1f} | corr(pred, log_sigma)={corr_ps:+.3f}  "
          f"corr(pred, ret_fwd)={corr_pr:+.3f}")
    print(f"           | <sigma_L>={np.mean(leg_sigmas_L):.3f}  <sigma_S>={np.mean(leg_sigmas_S):.3f}  "
          f"ratio S/L={np.mean(leg_sigmas_S)/np.mean(leg_sigmas_L):.2f}")
    print(f"           | vol(L)={rL.std()*np.sqrt(12):.3f}  vol(S)={rS.std()*np.sqrt(12):.3f}  "
          f"vol(L-S)={ls.std()*np.sqrt(12):.3f}  corr(L,S)={np.corrcoef(rL,rS)[0,1]:+.3f}")
    results.append({
        'gamma': gamma, 'sigma_L': np.mean(leg_sigmas_L), 'sigma_S': np.mean(leg_sigmas_S),
        'vol_L': rL.std()*np.sqrt(12), 'vol_S': rS.std()*np.sqrt(12),
        'vol_LS': ls.std()*np.sqrt(12),
    })

# ── 5. realized beta of each leg ──────────────────────────────────────────
print("\n[5] realized beta of L and S legs (regress leg returns on equal-wt mkt)")
# crude market proxy: equal-weight mean return across all test stocks each month
mkt = test.groupby('date')['ret_fwd'].mean()
for gamma in [0.0, 10.0]:
    y_tr = crra(r_train, sigma_train, gamma)
    model = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         tree_method='hist', random_state=1, verbosity=0)
    model.fit(X_train, y_tr)
    tmp = test.copy()
    tmp['score'] = model.predict(X_test)
    dates, rL_list, rS_list = [], [], []
    for date, grp in tmp.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score'].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp['score'] >= hi]; shorts = grp[grp['score'] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
        dates.append(date)
        rL_list.append((longs['ret_fwd']*longs['me']).sum()/longs['me'].sum())
        rS_list.append((shorts['ret_fwd']*shorts['me']).sum()/shorts['me'].sum())
    df = pd.DataFrame({'rL': rL_list, 'rS': rS_list}, index=dates).join(mkt.rename('mkt'))
    bL = np.polyfit(df['mkt'], df['rL'], 1)[0]
    bS = np.polyfit(df['mkt'], df['rS'], 1)[0]
    print(f"  gamma={gamma:>4.1f}: beta_L={bL:+.3f}  beta_S={bS:+.3f}  "
          f"beta_L - beta_S = {bL-bS:+.3f}")
