"""
Two-model + dual-utility full-universe selection.

Model 1: XGB predicts r_fwd          -> r_hat
Model 2: XGB predicts sigma_fwd_12m  -> sigma_hat (stock's forward vol)

Per-stock utility under each direction (Denis's CRRA applied to +r if long, -r if short):
    util_long_i  = sign(+r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_hat_i)
    util_short_i = sign(-r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_hat_i)
                 = -sign(r_hat_i) * log(1 + |r_hat_i|/eps) - gamma * log(sigma_hat_i)

Portfolio (full universe, NYSE breakpoints, value-weighted):
    Longs  = top 10% of util_long  (across the full universe)
    Shorts = top 10% of util_short (across the full universe)

Both utilities share the SAME -gamma * log(sigma_hat) term, so both legs prefer low sigma.
Full-universe ranking eliminates the 80/20 sign-split asymmetry of the per-direction version.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import pickle, warnings, time, os
warnings.filterwarnings('ignore')

EPS = 0.01
SIGMA_FLOOR = 0.01
GAMMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
          0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0]
N_SEEDS = 50
CACHE_PATH = 'artefacts/two_model_dual_util_preds.pkl'
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

# ── trailing sigma + forward sigma ───────────────────────────────────────
print("Computing trailing sigma and forward sigma_12m ...")
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw.sort_values(['permno', 'date'])
stocks_raw['trail_std_12'] = (
    stocks_raw.groupby('permno')['ret_adj']
    .transform(lambda x: x.shift(1).rolling(12, min_periods=6).std())
)
rolling_std = stocks_raw.groupby('permno')['ret_adj'].transform(
    lambda x: x.rolling(12, min_periods=6).std())
stocks_raw['sigma_fwd_12m'] = stocks_raw.groupby('permno', group_keys=False).apply(
    lambda g: rolling_std.loc[g.index].shift(-12))

trail_map = stocks_raw.set_index(['permno', 'date'])['trail_std_12']
fwd_map   = stocks_raw.set_index(['permno', 'date'])['sigma_fwd_12m']
train['trail_sigma']   = train.set_index(['permno','date']).index.map(lambda k: trail_map.get(k, np.nan))
train['sigma_fwd_12m'] = train.set_index(['permno','date']).index.map(lambda k: fwd_map.get(k, np.nan))
test['trail_sigma']    = test.set_index(['permno','date']).index.map(lambda k: trail_map.get(k, np.nan))
tv_med = train['trail_sigma'].median()
train['trail_sigma'] = train['trail_sigma'].fillna(tv_med).clip(lower=SIGMA_FLOOR)
test['trail_sigma']  = test['trail_sigma'].fillna(tv_med).clip(lower=SIGMA_FLOOR)

X_train2 = np.column_stack([X_train, train['trail_sigma'].values])
X_test2  = np.column_stack([X_test,  test['trail_sigma'].values])

# ── Predictions (train-or-load cache so we can re-sweep gammas cheaply) ─
if os.path.exists(CACHE_PATH):
    print(f"\nLoading cached predictions from {CACHE_PATH} ...")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    if cache.get('n_seeds') == N_SEEDS and len(cache['r_hat_test']) == len(X_test):
        r_hat_test    = cache['r_hat_test']
        sigma_hat_test = cache['sigma_hat_test']
        print(f"  loaded ({N_SEEDS} seeds). "
              f"corr(r_hat, r_fwd) test = {np.corrcoef(r_hat_test, test['ret_fwd'])[0,1]:+.3f}")
    else:
        print("  cache mismatch, retraining ...")
        cache = None
else:
    cache = None

if cache is None:
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
          f"corr(r_hat, r_fwd) test = {np.corrcoef(r_hat_test, test['ret_fwd'])[0,1]:+.3f}")

    mask_tr = train['sigma_fwd_12m'].notna().values
    print(f"\n[Model 2] Training {N_SEEDS} XGBs on sigma_fwd_12m "
          f"(usable rows {mask_tr.sum():,}/{len(train):,}) ...")
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
          f"mean={sigma_hat_test.mean():.3f} median={np.median(sigma_hat_test):.3f}")

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'n_seeds': N_SEEDS,
                     'r_hat_test': r_hat_test,
                     'sigma_hat_test': sigma_hat_test}, f)
    print(f"  cached predictions -> {CACHE_PATH}")

test = test.copy()
test['r_hat']     = r_hat_test
test['sigma_hat'] = sigma_hat_test
test['util_long']  = (np.sign(r_hat_test) * np.log1p(np.abs(r_hat_test)/EPS)
                      - 0.0 * np.log(sigma_hat_test))   # filled per gamma below
test['util_short'] = (-np.sign(r_hat_test) * np.log1p(np.abs(r_hat_test)/EPS)
                      - 0.0 * np.log(sigma_hat_test))

# ── Portfolio ────────────────────────────────────────────────────────────
def compute_utils(df, gamma):
    ret_term = np.log1p(np.abs(df['r_hat'].values) / EPS)
    sig_pen  = -gamma * np.log(df['sigma_hat'].values)
    df = df.copy()
    df['util_long']  =  np.sign(df['r_hat'].values) * ret_term + sig_pen
    df['util_short'] = -np.sign(df['r_hat'].values) * ret_term + sig_pen
    return df

def portfolio_dual(df, fee=TRADING_FEE):
    """Long = top 10% of util_long; short = top 10% of util_short; full-universe NYSE breakpoints.

    Overlaps (stocks in both top deciles) are KEPT on both sides and net naturally
    in r_L - r_S: a stock with weight w_L in longs and w_S in shorts contributes
    (w_L - w_S) * r to the L/S P&L. No dropping.
    """
    monthly = []
    prev_lw, prev_sw = {}, {}
    overlap_count = 0
    for date, grp in df.groupby('date'):
        nyse_L = grp[grp['exchcd'] == 1]['util_long'].dropna()
        nyse_S = grp[grp['exchcd'] == 1]['util_short'].dropna()
        if len(nyse_L) < 10 or len(nyse_S) < 10:
            continue
        hi_L = nyse_L.quantile(0.90)
        hi_S = nyse_S.quantile(0.90)
        longs  = grp[grp['util_long']  >= hi_L]
        shorts = grp[grp['util_short'] >= hi_S]
        overlap_count += len(set(longs['permno']) & set(shorts['permno']))
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
        ov_month = len(set(longs['permno']) & set(shorts['permno']))
        monthly.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts),
                        'sigL': sig_l, 'sigS': sig_s,
                        'nL': len(longs), 'nS': len(shorts), 'ov': ov_month})
        prev_lw, prev_sw = lw, sw
    pf = pd.DataFrame(monthly).set_index('date')
    return pf, overlap_count

def metrics(r):
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, mdd

print("\n=== Dual-utility full-universe sweep ===")
print(f"{'gamma':>6} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MDD':>8} "
      f"{'<sigL>':>8} {'<sigS>':>8} {'S/L':>6} {'<nL>':>5} {'<nS>':>5} {'ovp':>5}")
results = []
for g in GAMMAS:
    d = compute_utils(test, g)
    pf, overlap_total = portfolio_dual(d)
    ann_ret, ann_vol, sharpe, mdd = metrics(pf['ret'])
    sL, sS = pf['sigL'].mean(), pf['sigS'].mean()
    nL, nS = pf['nL'].mean(), pf['nS'].mean()
    ovp = pf['ov'].mean()
    results.append({'gamma': g, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
                    'sharpe': sharpe, 'mdd': mdd, 'sigma_L': sL, 'sigma_S': sS,
                    'nL': nL, 'nS': nS, 'overlap_per_month': ovp})
    print(f"{g:>6.1f} {ann_ret:>+7.1%} {ann_vol:>7.1%} {sharpe:>8.2f} {mdd:>+7.1%} "
          f"{sL:>8.3f} {sS:>8.3f} {sS/sL:>6.2f} {nL:>5.0f} {nS:>5.0f} {ovp:>5.1f}")

res_df = pd.DataFrame(results)
res_df.to_csv('results/thesis/two_model_dual_util.csv', index=False, float_format='%.4f')
print("\nSaved: results/two_model_dual_util.csv")
