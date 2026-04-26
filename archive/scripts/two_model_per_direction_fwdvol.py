"""
Per-direction utility with Model 2 predicting FORWARD STOCK VOLATILITY.

Model 1: XGB predicts r_fwd                -> r_hat
Model 2: XGB predicts sigma_fwd_12m        -> sigma_hat
            target = std(ret_adj[t+1 ... t+12])   -- stock's realized vol over next year
            features = main features + trailing sigma (persistence input)

Per-direction Denis utility:
    util_long_i  = +sign(r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_hat_i)
    util_short_i = -sign(r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_hat_i)

Portfolio:
    direction = argmax(util_long, util_short)
    conviction = max(util_long, util_short)
    Long:  top decile of conviction among r_hat > 0
    Short: top decile of conviction among r_hat < 0
    Value-weighted by me, 10 bps transaction fee.

Test run: 5 seeds, gammas = [0, 0.5, 1, 2, 3, 10].
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import pickle, warnings, time
warnings.filterwarnings('ignore')

EPS = 0.01
SIGMA_FLOOR = 0.01
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 10.0]
N_SEEDS = 5
TRADING_FEE = 0.001

# ── Load artefacts ────────────────────────────────────────────────────────
print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

train = artefacts['train'].copy()
test  = artefacts['test'].copy()
X_train = artefacts['X_train']
X_test  = artefacts['X_test']
FEATURES = artefacts['FEATURES']
print(f"  Train {len(train):,} | Test {len(test):,} | Features {len(FEATURES)}")

# ── Build trailing sigma (feature) and forward sigma (Model-2 target) ────
print("Computing trailing sigma and forward sigma_12m ...")
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw.sort_values(['permno', 'date'])

# trailing 12m std of past returns (known at t) -- used as Model-2 feature and as persistence baseline
stocks_raw['trail_std_12'] = (
    stocks_raw.groupby('permno')['ret_adj']
    .transform(lambda x: x.shift(1).rolling(12, min_periods=6).std())
)

# forward sigma: std of ret_adj[t+1 .. t+12]
#   rolling(12).std() at index i = std([i-11 .. i])
#   groupby.shift(-12) aligns so row i holds std([i+1 .. i+12])
rolling_std = stocks_raw.groupby('permno')['ret_adj'].transform(
    lambda x: x.rolling(12, min_periods=6).std())
stocks_raw['sigma_fwd_12m'] = stocks_raw.groupby('permno', group_keys=False).apply(
    lambda g: rolling_std.loc[g.index].shift(-12))

trail_map = stocks_raw.set_index(['permno', 'date'])['trail_std_12']
fwd_map   = stocks_raw.set_index(['permno', 'date'])['sigma_fwd_12m']

train['trail_sigma']   = train.set_index(['permno','date']).index.map(lambda k: trail_map.get(k, np.nan))
train['sigma_fwd_12m'] = train.set_index(['permno','date']).index.map(lambda k: fwd_map.get(k, np.nan))
test['trail_sigma']    = test.set_index(['permno','date']).index.map(lambda k: trail_map.get(k, np.nan))

# Fill NaN trailing sigma with median so X_train2 / X_test2 are complete
tv_med = train['trail_sigma'].median()
train['trail_sigma'] = train['trail_sigma'].fillna(tv_med)
test['trail_sigma']  = test['trail_sigma'].fillna(tv_med)
train['trail_sigma'] = np.maximum(train['trail_sigma'], SIGMA_FLOOR)
test['trail_sigma']  = np.maximum(test['trail_sigma'],  SIGMA_FLOOR)

X_train2 = np.column_stack([X_train, train['trail_sigma'].values])
X_test2  = np.column_stack([X_test,  test['trail_sigma'].values])

# ── Model 1: r_hat ────────────────────────────────────────────────────────
print(f"\n[Model 1] Training {N_SEEDS} XGBs to predict r_fwd ...")
t0 = time.time()
r_hat_test = np.zeros(len(X_test))
for seed in range(1, N_SEEDS + 1):
    m = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     tree_method='hist', random_state=seed, verbosity=0)
    m.fit(X_train, train['ret_fwd'].values)
    r_hat_test += m.predict(X_test)
r_hat_test /= N_SEEDS
print(f"  elapsed {time.time()-t0:.1f}s | "
      f"corr(r_hat, r_fwd) test = {np.corrcoef(r_hat_test, test['ret_fwd'])[0,1]:+.3f} | "
      f"fraction r_hat>0 = {(r_hat_test>0).mean():.1%}")

# ── Model 2: sigma_hat (forward realized vol) ────────────────────────────
mask_tr = train['sigma_fwd_12m'].notna().values
print(f"\n[Model 2] Training {N_SEEDS} XGBs to predict sigma_fwd_12m ... "
      f"(usable train rows: {mask_tr.sum():,} / {len(train):,})")
t0 = time.time()
y_sigma_train = train['sigma_fwd_12m'].values[mask_tr]
X_sigma_train = X_train2[mask_tr]
sigma_hat_test = np.zeros(len(X_test2))
for seed in range(1, N_SEEDS + 1):
    m = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     tree_method='hist', random_state=seed, verbosity=0)
    m.fit(X_sigma_train, y_sigma_train)
    sigma_hat_test += m.predict(X_test2)
sigma_hat_test /= N_SEEDS
sigma_hat_test = np.maximum(sigma_hat_test, SIGMA_FLOOR)
print(f"  elapsed {time.time()-t0:.1f}s | sigma_hat: "
      f"mean={sigma_hat_test.mean():.3f} median={np.median(sigma_hat_test):.3f} "
      f"p5={np.quantile(sigma_hat_test,0.05):.3f} p95={np.quantile(sigma_hat_test,0.95):.3f}")
print(f"  corr(sigma_hat, trail_sigma)_test = "
      f"{np.corrcoef(sigma_hat_test, test['trail_sigma'])[0,1]:+.3f}")

test = test.copy()
test['r_hat']     = r_hat_test
test['sigma_hat'] = sigma_hat_test

# ── Per-direction utility + portfolio ─────────────────────────────────────
def per_direction_util(r_hat, sigma, gamma, eps=EPS):
    ret_term = np.log1p(np.abs(r_hat) / eps)
    sig_pen  = -gamma * np.log(sigma)
    return (np.sign(r_hat) * ret_term + sig_pen,
           -np.sign(r_hat) * ret_term + sig_pen)

def portfolio(df, gamma, fee=TRADING_FEE):
    ul, us = per_direction_util(df['r_hat'].values, df['sigma_hat'].values, gamma)
    d = df.copy()
    d['util_long']  = ul
    d['util_short'] = us
    d['direction']  = np.where(ul > us, 1, -1)
    d['conviction'] = np.maximum(ul, us)

    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in d.groupby('date'):
        cand_L = grp[grp['direction'] == +1]
        cand_S = grp[grp['direction'] == -1]
        if len(cand_L) < 10 or len(cand_S) < 10:
            continue
        nyse_L = cand_L[cand_L['exchcd'] == 1]['conviction'].dropna()
        nyse_S = cand_S[cand_S['exchcd'] == 1]['conviction'].dropna()
        if len(nyse_L) < 5 or len(nyse_S) < 5:
            continue
        hi_L = nyse_L.quantile(0.90)
        hi_S = nyse_S.quantile(0.90)
        longs  = cand_L[cand_L['conviction'] >= hi_L]
        shorts = cand_S[cand_S['conviction'] >= hi_S]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lw = (longs.set_index('permno')['me']  / longs['me'].sum()).to_dict()
        sw = (shorts.set_index('permno')['me'] / shorts['me'].sum()).to_dict()
        tl = sum(abs(lw.get(p,0)-prev_lw.get(p,0)) for p in set(lw)|set(prev_lw)) / 2
        ts = sum(abs(sw.get(p,0)-prev_sw.get(p,0)) for p in set(sw)|set(prev_sw)) / 2
        r_l = (longs['ret_fwd']  * longs['me']).sum()  / longs['me'].sum()
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        sig_l = (longs['sigma_hat']  * longs['me']).sum()  / longs['me'].sum()
        sig_s = (shorts['sigma_hat'] * shorts['me']).sum() / shorts['me'].sum()
        monthly.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts),
                        'sigL': sig_l, 'sigS': sig_s,
                        'nL': len(longs), 'nS': len(shorts)})
        prev_lw, prev_sw = lw, sw
    return pd.DataFrame(monthly).set_index('date')

def metrics(r):
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, mdd

print("\n=== Per-direction utility, Model 2 predicts forward sigma_12m ===")
print(f"{'gamma':>6} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MDD':>8} "
      f"{'<sigL>':>8} {'<sigS>':>8} {'S/L':>6} {'<nL>':>5} {'<nS>':>5}")
results = []
for g in GAMMAS:
    pf = portfolio(test, gamma=g)
    ann_ret, ann_vol, sharpe, mdd = metrics(pf['ret'])
    sL, sS = pf['sigL'].mean(), pf['sigS'].mean()
    nL, nS = pf['nL'].mean(), pf['nS'].mean()
    results.append({'gamma': g, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
                    'sharpe': sharpe, 'mdd': mdd,
                    'sigma_L': sL, 'sigma_S': sS, 'nL': nL, 'nS': nS})
    print(f"{g:>6.1f} {ann_ret:>+7.1%} {ann_vol:>7.1%} {sharpe:>8.2f} {mdd:>+7.1%} "
          f"{sL:>8.3f} {sS:>8.3f} {sS/sL:>6.2f} {nL:>5.0f} {nS:>5.0f}")

res_df = pd.DataFrame(results)
res_df.to_csv('results/two_model_per_direction_fwdvol.csv', index=False, float_format='%.4f')
print("\nSaved: results/two_model_per_direction_fwdvol.csv")
