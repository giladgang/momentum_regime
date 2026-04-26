"""
Per-direction utility with TRAILING sigma (actual stock vol, not conditional variance).

Step 1 — return prediction only:
    Model 1 XGB -> r_hat

Step 2 — use observed trailing 12-month sigma as the stock's volatility:
    sigma_i = trailing realized vol at time t (no second model)

Step 3 — per-direction Denis utility, SAME sigma penalty on both sides:
    util_long_i  = +sign(r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_i)
    util_short_i = -sign(r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_i)

Step 4 — portfolio:
    direction = argmax(util_long, util_short)   (equiv to sign(r_hat))
    conviction = max(util_long, util_short)
    Long:  top decile of conviction among r_hat > 0
    Short: top decile of conviction among r_hat < 0
    Value-weighted by me, 10 bps transaction cost.

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
test['trail_var']  = test.set_index(['permno',  'date']).index.map(
    lambda idx: tv_map.get(idx, np.nan))
tv_med = train['trail_var'].median()
train['trail_var'] = train['trail_var'].fillna(tv_med)
test['trail_var']  = test['trail_var'].fillna(tv_med)
train['sigma'] = np.sqrt(np.maximum(train['trail_var'].values, SIGMA_FLOOR**2))
test['sigma']  = np.sqrt(np.maximum(test['trail_var'].values,  SIGMA_FLOOR**2))

# ── Model 1: r_hat (same as before) ──────────────────────────────────────
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

test = test.copy()
test['r_hat'] = r_hat_test

# ── Per-direction utility + portfolio ─────────────────────────────────────
def per_direction_util(r_hat, sigma, gamma, eps=EPS):
    ret_term = np.log1p(np.abs(r_hat) / eps)
    sig_pen  = -gamma * np.log(sigma)
    return (np.sign(r_hat) * ret_term + sig_pen,
           -np.sign(r_hat) * ret_term + sig_pen)

def portfolio(df, gamma, fee=TRADING_FEE):
    ul, us = per_direction_util(df['r_hat'].values, df['sigma'].values, gamma)
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
        sig_l = (longs['sigma']  * longs['me']).sum()  / longs['me'].sum()
        sig_s = (shorts['sigma'] * shorts['me']).sum() / shorts['me'].sum()
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

print("\n=== Per-direction utility with trailing sigma: gamma sweep ===")
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
res_df.to_csv('results/two_model_per_direction_trailsig.csv', index=False, float_format='%.4f')
print("\nSaved: results/two_model_per_direction_trailsig.csv")
