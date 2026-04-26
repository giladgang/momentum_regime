"""
confidence_threshold_test.py
============================
Robustness: replace the decile long/short with a confidence-threshold rule.

Setup
-----
1. Train a small XGB ensemble (3 seeds) to predict r_fwd  ->  r_hat.
2. Calibrate thresholds on the TRAINING set.
3. On the TEST set, build value-weighted long/short portfolios under several
   selection rules and compare to the decile baseline.

Rules compared
--------------
  A. Decile baseline   : long top decile of r_hat,    short bottom decile (NYSE bpts)
  B. Sign              : long r_hat > 0,              short r_hat < 0
  C. Threshold 50 bps  : long r_hat > +0.005,         short r_hat < -0.005
  D. Threshold 100 bps : long r_hat > +0.010,         short r_hat < -0.010
  E. Threshold 200 bps : long r_hat > +0.020,         short r_hat < -0.020
  F. Calibrated        : tau_L, tau_S chosen from training so that stocks above
                         (below) the threshold have mean realized return whose
                         Newey-West t-stat exceeds +2 (is below -2), with tau
                         chosen as the minimum magnitude that clears the bar.

All legs are value-weighted. Transaction cost = 10 bps one-way (matches main
pipeline). Months where a leg is empty under a rule are skipped (not neutral).
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import pickle, warnings, time, os, sys
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE

N_SEEDS = 3

# ── Load artefacts ─────────────────────────────────────────────────────────
print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

train    = artefacts['train'].copy()
test     = artefacts['test'].copy()
X_train  = artefacts['X_train']
X_test   = artefacts['X_test']
FEATURES = artefacts['FEATURES']
print(f"  Train {len(train):,} | Test {len(test):,} | Features {len(FEATURES)}")

# ── XGB ensemble ───────────────────────────────────────────────────────────
print(f"\nTraining {N_SEEDS}-seed XGB ensemble (predicting r_fwd) ...")
t0 = time.time()
r_hat_train = np.zeros(len(X_train))
r_hat_test  = np.zeros(len(X_test))
for seed in range(1, N_SEEDS + 1):
    m = XGBRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        tree_method='hist', random_state=seed, verbosity=0,
    )
    m.fit(X_train, train['ret_fwd'].values)
    r_hat_train += m.predict(X_train)
    r_hat_test  += m.predict(X_test)
r_hat_train /= N_SEEDS
r_hat_test  /= N_SEEDS
print(f"  elapsed {time.time()-t0:.1f}s | "
      f"corr(r_hat, r_fwd)_train={np.corrcoef(r_hat_train, train['ret_fwd'])[0,1]:+.3f} "
      f"| test={np.corrcoef(r_hat_test,  test['ret_fwd'])[0,1]:+.3f}")

train = train.copy()
test  = test.copy()
train['r_hat'] = r_hat_train
test['r_hat']  = r_hat_test

# ── Calibrate thresholds from TRAINING data ────────────────────────────────
print("\nCalibrating thresholds on training data ...")

def _nw_t(x, maxlags=6):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 12:
        return np.nan
    import statsmodels.api as sm
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type='HAC',
                                               cov_kwds={'maxlags': maxlags})
    return float(res.tvalues[0])

# Monthly mean realized return of all training stocks with r_hat >= cutoff
def leg_tstat(df, side, cutoff):
    if side == 'long':
        mask = df['r_hat'] >= cutoff
    else:
        mask = df['r_hat'] <= cutoff
    if mask.sum() == 0:
        return np.nan, 0
    monthly = (df[mask].groupby('date')['ret_fwd'].mean())
    return _nw_t(monthly.values), int(mask.sum())

# Sweep positive cutoffs for long leg; negative cutoffs for short leg.
cut_grid = np.round(np.arange(0.000, 0.0301, 0.0025), 4)
long_scan = [(c, *leg_tstat(train, 'long',  c)) for c in cut_grid]
short_scan = [(c, *leg_tstat(train, 'short', -c)) for c in cut_grid]
print("  Long cutoff  -> NW t-stat (train) / n_obs")
for c, t, n in long_scan:
    print(f"    r_hat >= +{c:.4f}  t={t:+.2f}  n={n:,}")
print("  Short cutoff -> NW t-stat (train) / n_obs")
for c, t, n in short_scan:
    print(f"    r_hat <= -{c:.4f}  t={t:+.2f}  n={n:,}")

# smallest positive cutoff where long t>2 and short t<-2
tau_L = next((c for c, t, _ in long_scan  if t is not None and t > 2), 0.01)
tau_S = next((c for c, t, _ in short_scan if t is not None and t < -2), 0.01)
print(f"  -> calibrated tau_L = +{tau_L:.4f}, tau_S = -{tau_S:.4f}")

# ── Portfolio builders ─────────────────────────────────────────────────────
def _leg_stats(monthly_rows):
    """Turn list of per-month dicts into a tidy DF."""
    return pd.DataFrame(monthly_rows).set_index('date')

def portfolio_decile(df, fee=TRADING_FEE):
    rows, prev_lw, prev_sw = [], {}, {}
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['r_hat'].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp['r_hat'] >= hi]
        shorts = grp[grp['r_hat'] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme, sme = longs['me'].sum(), shorts['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_l = (longs['ret_fwd']  * longs['me']).sum()  / lme
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p,0)-prev_lw.get(p,0)) for p in set(new_lw)|set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p,0)-prev_sw.get(p,0)) for p in set(new_sw)|set(prev_sw)) / 2
        rows.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts),
                     'n_long': len(longs), 'n_short': len(shorts)})
        prev_lw, prev_sw = new_lw, new_sw
    return _leg_stats(rows)

def portfolio_threshold(df, tau_L, tau_S, fee=TRADING_FEE):
    """Long r_hat >= +tau_L, short r_hat <= -tau_S. Skip month if either leg empty."""
    rows, prev_lw, prev_sw = [], {}, {}
    n_skip_L = n_skip_S = n_skip_both = 0
    for date, grp in df.groupby('date'):
        longs  = grp[grp['r_hat'] >= +tau_L]
        shorts = grp[grp['r_hat'] <= -tau_S]
        if len(longs) == 0 and len(shorts) == 0:
            n_skip_both += 1; continue
        if len(longs) == 0 or longs['me'].sum() == 0:
            n_skip_L += 1; continue
        if len(shorts) == 0 or shorts['me'].sum() == 0:
            n_skip_S += 1; continue
        lme, sme = longs['me'].sum(), shorts['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_l = (longs['ret_fwd']  * longs['me']).sum()  / lme
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p,0)-prev_lw.get(p,0)) for p in set(new_lw)|set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p,0)-prev_sw.get(p,0)) for p in set(new_sw)|set(prev_sw)) / 2
        rows.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts),
                     'n_long': len(longs), 'n_short': len(shorts)})
        prev_lw, prev_sw = new_lw, new_sw
    return _leg_stats(rows), dict(skip_L=n_skip_L, skip_S=n_skip_S, skip_both=n_skip_both)

def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return dict(ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan, mdd=np.nan, n_months=len(r))
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12)
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, mdd=mdd, n_months=len(r))

# ── Run all rules ──────────────────────────────────────────────────────────
rules = [
    ('A_decile',          'decile baseline',      dict(kind='decile')),
    ('B_sign',            'r_hat sign',           dict(kind='thr', tau_L=0.000, tau_S=0.000)),
    ('C_thr_50bps',       'r_hat > +/- 50 bps',   dict(kind='thr', tau_L=0.005, tau_S=0.005)),
    ('D_thr_100bps',      'r_hat > +/- 100 bps',  dict(kind='thr', tau_L=0.010, tau_S=0.010)),
    ('E_thr_200bps',      'r_hat > +/- 200 bps',  dict(kind='thr', tau_L=0.020, tau_S=0.020)),
    ('F_calibrated',      f'NW t>|2| calib (tau_L={tau_L:.4f}, tau_S={tau_S:.4f})',
                                                   dict(kind='thr', tau_L=tau_L, tau_S=tau_S)),
]

print("\n=== Rule comparison on TEST set ===")
print(f"{'rule':<15} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MDD':>8} "
      f"{'nL':>5} {'nS':>5} {'nMo':>5} {'skL':>4} {'skS':>4}")
results = []
for key, label, spec in rules:
    if spec['kind'] == 'decile':
        pf = portfolio_decile(test)
        skips = dict(skip_L=0, skip_S=0, skip_both=0)
    else:
        pf, skips = portfolio_threshold(test, spec['tau_L'], spec['tau_S'])
    m = metrics(pf['ret'])
    nL = pf['n_long'].mean()  if len(pf) else np.nan
    nS = pf['n_short'].mean() if len(pf) else np.nan
    results.append(dict(rule=key, label=label, **m, avg_n_long=nL, avg_n_short=nS,
                        **skips))
    print(f"{key:<15} {m['ann_ret']:>+7.1%} {m['ann_vol']:>7.1%} {m['sharpe']:>8.2f} "
          f"{m['mdd']:>+7.1%} {nL:>5.0f} {nS:>5.0f} {m['n_months']:>5d} "
          f"{skips['skip_L']:>4d} {skips['skip_S']:>4d}")

res_df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
res_df.to_csv('results/confidence_threshold_test.csv', index=False, float_format='%.4f')
print("\nSaved: results/confidence_threshold_test.csv")
