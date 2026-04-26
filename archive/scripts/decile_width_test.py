"""
decile_width_test.py
====================
Robustness: sweep the long/short breakpoint width.

Baseline uses NYSE top/bottom 10% (decile). Here we sweep widths from 5% to 30%
to see how Sharpe trades off against book depth.

All other choices match the main pipeline: NYSE breakpoints, value-weighted
within leg, 10 bps one-way transaction costs, r_hat from a 3-seed XGB ensemble.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import pickle, warnings, time, os, sys
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE

N_SEEDS = 3
WIDTHS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # top/bottom X

# ── Load artefacts ─────────────────────────────────────────────────────────
print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)
train    = artefacts['train'].copy()
test     = artefacts['test'].copy()
X_train  = artefacts['X_train']
X_test   = artefacts['X_test']
print(f"  Train {len(train):,} | Test {len(test):,}")

# ── XGB ensemble ───────────────────────────────────────────────────────────
print(f"\nTraining {N_SEEDS}-seed XGB ensemble ...")
t0 = time.time()
r_hat_test = np.zeros(len(X_test))
for seed in range(1, N_SEEDS + 1):
    m = XGBRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        tree_method='hist', random_state=seed, verbosity=0,
    )
    m.fit(X_train, train['ret_fwd'].values)
    r_hat_test += m.predict(X_test)
r_hat_test /= N_SEEDS
print(f"  elapsed {time.time()-t0:.1f}s | "
      f"corr(r_hat, r_fwd)_test={np.corrcoef(r_hat_test, test['ret_fwd'])[0,1]:+.3f}")

test = test.copy()
test['r_hat'] = r_hat_test

# ── Portfolio builder with variable width ──────────────────────────────────
def portfolio(df, width, fee=TRADING_FEE):
    """Long top `width` by r_hat (NYSE bpts), short bottom `width`, value-weighted."""
    q_lo, q_hi = width, 1.0 - width
    rows, prev_lw, prev_sw = [], {}, {}
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['r_hat'].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(q_lo), nyse.quantile(q_hi)
        longs  = grp[grp['r_hat'] >= hi]
        shorts = grp[grp['r_hat'] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme, sme = longs['me'].sum(), shorts['me'].sum()
        new_lw = (longs.set_index('permno')['me']  / lme).to_dict()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_l = (longs['ret_fwd']  * longs['me']).sum()  / lme
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p,0)-prev_lw.get(p,0)) for p in set(new_lw)|set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p,0)-prev_sw.get(p,0)) for p in set(new_sw)|set(prev_sw)) / 2
        rows.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts),
                     'n_long': len(longs), 'n_short': len(shorts),
                     'turnover': tl + ts})
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(rows).set_index('date')

def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12)
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, mdd=mdd)

# ── Sweep ──────────────────────────────────────────────────────────────────
print("\n=== Breakpoint-width sweep on TEST set ===")
print(f"{'width':>6} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MDD':>8} "
      f"{'nL':>5} {'nS':>5} {'turn':>6}")
rows = []
for w in WIDTHS:
    pf = portfolio(test, width=w)
    m = metrics(pf['ret'])
    nL = pf['n_long'].mean()
    nS = pf['n_short'].mean()
    turn = pf['turnover'].mean()
    rows.append(dict(width=w, **m, avg_n_long=nL, avg_n_short=nS, avg_turnover=turn,
                     n_months=len(pf)))
    print(f"{w:>6.2f} {m['ann_ret']:>+7.1%} {m['ann_vol']:>7.1%} {m['sharpe']:>8.2f} "
          f"{m['mdd']:>+7.1%} {nL:>5.0f} {nS:>5.0f} {turn:>6.2f}")

os.makedirs('results', exist_ok=True)
pd.DataFrame(rows).to_csv('results/decile_width_test.csv', index=False, float_format='%.4f')
print("\nSaved: results/decile_width_test.csv")
