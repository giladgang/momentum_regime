"""
Two-model approach using Denis's EXACT formula as the score.

Model 1: XGB predicts r_fwd              -> r_hat
Model 2: XGB predicts (r_fwd - r_hat)^2  -> sigma_hat (conditional vol)

Score per stock-month (Denis's formula, applied to predictions):
    score_i = sign(r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_hat_i)

Portfolio (matches original CRRA script):
    - Long  top decile of score, value-weighted by me
    - Short bottom decile of score, value-weighted by me
    - 10 bps one-way transaction fee

Sweep gamma = [0, 0.5, 1, 2, 3, 10]. Test run: 5 seeds per model.
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
train['trail_sigma'] = np.sqrt(np.maximum(train['trail_var'].values, SIGMA_FLOOR**2))
test['trail_sigma']  = np.sqrt(np.maximum(test['trail_var'].values,  SIGMA_FLOOR**2))

# Model 2 gets trailing sigma as an extra feature
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
print(f"  elapsed {time.time()-t0:.1f}s | "
      f"corr(r_hat, r_fwd) train={np.corrcoef(r_hat_train, train['ret_fwd'])[0,1]:+.3f} "
      f"test={np.corrcoef(r_hat_test, test['ret_fwd'])[0,1]:+.3f}")

# ── Model 2: sigma_hat from conditional variance ─────────────────────────
print(f"\n[Model 2] Training {N_SEEDS} XGBs to predict squared residual ...")
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
print(f"  elapsed {time.time()-t0:.1f}s | sigma_hat: mean={sigma_hat_test.mean():.3f} "
      f"median={np.median(sigma_hat_test):.3f} "
      f"p5={np.quantile(sigma_hat_test,0.05):.3f} p95={np.quantile(sigma_hat_test,0.95):.3f}")

test = test.copy()
test['r_hat']     = r_hat_test
test['sigma_hat'] = sigma_hat_test

# ── Denis's score + value-weighted L/S portfolio ─────────────────────────
def denis_score(r_hat, sigma_hat, gamma, eps=EPS):
    return np.sign(r_hat) * np.log1p(np.abs(r_hat) / eps) - gamma * np.log(sigma_hat)

def long_short_port(df, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
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
print("\n=== Two-model + Denis's exact formula: sweep over gamma ===")
print("score_i = sign(r_hat)*log(1 + |r_hat|/eps) - gamma*log(sigma_hat)\n")
print(f"{'gamma':>6} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MDD':>8} "
      f"{'<sigL>':>8} {'<sigS>':>8} {'S/L':>6}")

results = []
for g in GAMMAS:
    col = f'denis_g{g}'
    test[col] = denis_score(test['r_hat'].values, test['sigma_hat'].values, g)
    pf = long_short_port(test, col)
    ann_ret, ann_vol, sharpe, mdd = metrics(pf['ret'])
    sL, sS = pf['sigL'].mean(), pf['sigS'].mean()
    results.append({'gamma': g, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
                    'sharpe': sharpe, 'mdd': mdd, 'sigma_L': sL, 'sigma_S': sS})
    print(f"{g:>6.1f} {ann_ret:>+7.1%} {ann_vol:>7.1%} {sharpe:>8.2f} {mdd:>+7.1%} "
          f"{sL:>8.3f} {sS:>8.3f} {sS/sL:>6.2f}")

res_df = pd.DataFrame(results)
res_df.to_csv('results/two_model_denis_test.csv', index=False, float_format='%.4f')
print("\nSaved: results/two_model_denis_test.csv")
