"""
signal_weighted_test.py
=======================
Robustness: give the model full flexibility — every stock gets a weight
proportional to its signal, no decile cutoff, no fixed count.

All rules use gross exposure = 2 (100% long + 100% short) to match the decile
baseline. Transaction costs = 10 bps one-way on gross turnover.

Rules
-----
  A. Decile VW baseline        : long top-10% (NYSE bpt), short bottom-10%, VW
  B. Signal-weighted (raw)     : w_i ∝ r_hat_i, then normalized to sum|w|=2
                                  (can be net long/short if r_hat is biased)
  C. Signal-weighted (demean)  : w_i ∝ (r_hat_i - mean_t(r_hat)), dollar-neutral
  D. Rank-weighted (demean)    : w_i ∝ (rank(r_hat_i) - 0.5), dollar-neutral,
                                  robust to prediction-magnitude outliers
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

# ── Portfolio builders ─────────────────────────────────────────────────────
FEE = TRADING_FEE
GROSS = 2.0  # matches decile baseline ($1 long + $1 short per month)

def _run_weighted(df, weight_fn):
    """Generic runner: weight_fn(grp) returns a pd.Series of weights (permno-indexed).
    Normalizes so that sum|w| = GROSS, applies turnover cost on gross diff."""
    rows, prev_w = [], pd.Series(dtype=float)
    for date, grp in df.groupby('date'):
        w = weight_fn(grp)
        if w is None or w.abs().sum() == 0:
            continue
        w = w * (GROSS / w.abs().sum())
        r = (w * grp.set_index('permno').loc[w.index, 'ret_fwd']).sum()
        idx = w.index.union(prev_w.index)
        w_full      = w.reindex(idx).fillna(0)
        prev_w_full = prev_w.reindex(idx).fillna(0)
        turnover = (w_full - prev_w_full).abs().sum() / 2
        rows.append({'date': date, 'ret': r - FEE * turnover,
                     'n_long': int((w > 0).sum()),
                     'n_short': int((w < 0).sum()),
                     'gross': w.abs().sum(),
                     'net': w.sum(),
                     'turnover': turnover})
        prev_w = w
    return pd.DataFrame(rows).set_index('date')

def portfolio_decile(df, fee=FEE):
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
        new_lw = (longs.set_index('permno')['me']  / lme).to_dict()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_l = (longs['ret_fwd']  * longs['me']).sum()  / lme
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p,0)-prev_lw.get(p,0)) for p in set(new_lw)|set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p,0)-prev_sw.get(p,0)) for p in set(new_sw)|set(prev_sw)) / 2
        rows.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts),
                     'n_long': len(longs), 'n_short': len(shorts),
                     'gross': 2.0, 'net': 0.0, 'turnover': tl + ts})
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(rows).set_index('date')

def w_raw(grp):
    """w ∝ r_hat (can be net long/short)."""
    w = grp.set_index('permno')['r_hat'].dropna()
    return w if len(w) > 0 else None

def w_demean(grp):
    """w ∝ r_hat minus cross-sectional mean. Dollar-neutral by construction."""
    s = grp.set_index('permno')['r_hat'].dropna()
    return s - s.mean() if len(s) > 0 else None

def w_rank(grp):
    """w ∝ rank(r_hat) centered at 0.5. Dollar-neutral, outlier-robust."""
    s = grp.set_index('permno')['r_hat'].dropna()
    if len(s) == 0:
        return None
    r = s.rank(pct=True) - 0.5
    return r

# ── Run ─────────────────────────────────────────────────────────────────────
def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12)
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return dict(ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe, mdd=mdd,
                n_months=len(r))

print("\n=== Full-flexibility comparison (TEST set) ===")
print(f"{'rule':<24} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MDD':>8} "
      f"{'nL':>5} {'nS':>5} {'gross':>6} {'net':>6} {'turn':>6}")

rows = []
specs = [
    ('A_decile_VW',          lambda: portfolio_decile(test)),
    ('B_signal_raw',         lambda: _run_weighted(test, w_raw)),
    ('C_signal_demean',      lambda: _run_weighted(test, w_demean)),
    ('D_rank_demean',        lambda: _run_weighted(test, w_rank)),
]
for name, fn in specs:
    pf = fn()
    m = metrics(pf['ret'])
    nL = pf['n_long'].mean()
    nS = pf['n_short'].mean()
    gr = pf['gross'].mean()
    nt = pf['net'].mean()
    turn = pf['turnover'].mean()
    rows.append(dict(rule=name, **m, avg_n_long=nL, avg_n_short=nS,
                     avg_gross=gr, avg_net=nt, avg_turnover=turn))
    print(f"{name:<24} {m['ann_ret']:>+7.1%} {m['ann_vol']:>7.1%} {m['sharpe']:>8.2f} "
          f"{m['mdd']:>+7.1%} {nL:>5.0f} {nS:>5.0f} {gr:>6.2f} {nt:>+6.2f} {turn:>6.2f}")

os.makedirs('results', exist_ok=True)
pd.DataFrame(rows).to_csv('results/signal_weighted_test.csv', index=False, float_format='%.4f')
print("\nSaved: results/signal_weighted_test.csv")
