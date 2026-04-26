"""
Two-model CRRA (smoke test).

Model 1: XGB predicts r_fwd  ->  r_hat
Model 2: XGB predicts (r_fwd - r_hat)^2  ->  sigma_hat (conditional vol)

Portfolio:
  - Direction: long top decile of r_hat, short bottom decile of r_hat
  - Sizing within each leg: w_i ∝ sigma_hat_i^(-gamma), renormalized
  - gamma controls *concentration toward low-vol names within the leg*,
    NOT the long/short decision. gamma=0 collapses to equal weight.

Test run: 5 seeds per model, gammas = [0, 1, 3, 10].
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pickle, warnings, time
warnings.filterwarnings('ignore')

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

# trailing sigma (same source as before) -- feature input for Model 2
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
train['trail_sigma'] = np.sqrt(np.maximum(train['trail_var'].values, SIGMA_FLOOR**2))
test['trail_sigma']  = np.sqrt(np.maximum(test['trail_var'].values,  SIGMA_FLOOR**2))

# Augmented feature set for Model 2 (adds trailing sigma as input)
X_train2 = np.column_stack([X_train, train['trail_sigma'].values])
X_test2  = np.column_stack([X_test,  test['trail_sigma'].values])

# ── Model 1: r_hat ────────────────────────────────────────────────────────
print(f"\n[Model 1] Training {N_SEEDS} XGBs to predict r_fwd ...")
t0 = time.time()
r_hat_train = np.zeros(len(X_train))
r_hat_test  = np.zeros(len(X_test))
for seed in range(1, N_SEEDS + 1):
    m = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=seed, verbosity=0
    )
    m.fit(X_train, train['ret_fwd'].values)
    r_hat_train += m.predict(X_train)
    r_hat_test  += m.predict(X_test)
r_hat_train /= N_SEEDS
r_hat_test  /= N_SEEDS
print(f"  elapsed {time.time()-t0:.1f}s | corr(r_hat, r_fwd)_train = "
      f"{np.corrcoef(r_hat_train, train['ret_fwd'])[0,1]:+.3f} | "
      f"test = {np.corrcoef(r_hat_test, test['ret_fwd'])[0,1]:+.3f}")

# ── Model 2: sigma_hat via conditional variance ───────────────────────────
print(f"\n[Model 2] Training {N_SEEDS} XGBs to predict conditional variance ...")
t0 = time.time()
resid_sq_train = (train['ret_fwd'].values - r_hat_train) ** 2
var_hat_test = np.zeros(len(X_test2))
for seed in range(1, N_SEEDS + 1):
    m = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', random_state=seed, verbosity=0
    )
    m.fit(X_train2, resid_sq_train)
    var_hat_test += m.predict(X_test2)
var_hat_test /= N_SEEDS
sigma_hat_test = np.sqrt(np.maximum(var_hat_test, SIGMA_FLOOR**2))
print(f"  elapsed {time.time()-t0:.1f}s | sigma_hat: "
      f"mean={sigma_hat_test.mean():.3f} median={np.median(sigma_hat_test):.3f} "
      f"p5={np.quantile(sigma_hat_test,0.05):.3f} p95={np.quantile(sigma_hat_test,0.95):.3f}")
print(f"  corr(sigma_hat, trail_sigma) = "
      f"{np.corrcoef(sigma_hat_test, test['trail_sigma'])[0,1]:+.3f}")

test = test.copy()
test['r_hat']     = r_hat_test
test['sigma_hat'] = sigma_hat_test

# ── Portfolio formation ───────────────────────────────────────────────────
def portfolio(df, gamma, fee=TRADING_FEE):
    """Long top decile of r_hat, short bottom decile; within-leg weight ∝ sigma_hat^(-gamma)."""
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['r_hat'].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp['r_hat'] >= hi].copy()
        shorts = grp[grp['r_hat'] <= lo].copy()
        if len(longs) == 0 or len(shorts) == 0:
            continue

        lw_raw = longs['sigma_hat'].values  ** (-gamma)
        sw_raw = shorts['sigma_hat'].values ** (-gamma)
        if lw_raw.sum() == 0 or sw_raw.sum() == 0:
            continue
        longs['w']  = lw_raw  / lw_raw.sum()
        shorts['w'] = sw_raw  / sw_raw.sum()

        lw = dict(zip(longs['permno'],  longs['w']))
        sw = dict(zip(shorts['permno'], shorts['w']))
        tl = sum(abs(lw.get(p,0)-prev_lw.get(p,0)) for p in set(lw)|set(prev_lw)) / 2
        ts = sum(abs(sw.get(p,0)-prev_sw.get(p,0)) for p in set(sw)|set(prev_sw)) / 2

        r_l = (longs['ret_fwd']  * longs['w']).sum()
        r_s = (shorts['ret_fwd'] * shorts['w']).sum()

        # sigma-weighted avg sigma within each leg (for diagnostics)
        sig_l = (longs['trail_sigma']  * longs['w']).sum()
        sig_s = (shorts['trail_sigma'] * shorts['w']).sum()

        monthly.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts),
                        'rL': r_l, 'rS': r_s, 'sigL': sig_l, 'sigS': sig_s})
        prev_lw, prev_sw = lw, sw
    return pd.DataFrame(monthly).set_index('date')

def metrics(r):
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, mdd

# ── Sweep ─────────────────────────────────────────────────────────────────
print("\n=== Portfolio sweep over gamma (within-leg inverse-vol concentration) ===")
print(f"{'gamma':>6} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MDD':>8} "
      f"{'<sigL>':>8} {'<sigS>':>8} {'S/L':>6}")
results = []
for g in GAMMAS:
    pf = portfolio(test, gamma=g)
    ann_ret, ann_vol, sharpe, mdd = metrics(pf['ret'])
    sL, sS = pf['sigL'].mean(), pf['sigS'].mean()
    results.append({'gamma': g, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
                    'sharpe': sharpe, 'mdd': mdd, 'sigma_L': sL, 'sigma_S': sS,
                    'sharpe_gross': pf['ret'].mean()/pf['ret'].std()*np.sqrt(12)})
    print(f"{g:>6.1f} {ann_ret:>+7.1%} {ann_vol:>7.1%} {sharpe:>8.2f} {mdd:>+7.1%} "
          f"{sL:>8.3f} {sS:>8.3f} {sS/sL:>6.2f}")

res_df = pd.DataFrame(results)
res_df.to_csv('results/two_model_crra_test.csv', index=False, float_format='%.4f')
print("\nSaved: results/two_model_crra_test.csv")
