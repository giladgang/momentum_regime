"""
2026-05-31-weighting-mechanism.py
=================================
EXPLORATORY (experiments/ -- throwaway).

Where does the XGB long-short edge come from -- breadth (many names to sort) or
giving weight to the small-name tail? Hold the SELECTION fixed (top/bottom decile,
NYSE P10/P90) and vary only the WEIGHTING within each leg:
  vw   : value-weighted by me        (baseline; dollars sit in the biggest names)
  cap  : value-weighted, capped at 1% per name, excess redistributed
  ew   : equal-weighted              (each name 1/N; up-weights the small-name tail)

Run on the full universe (breadth) and on Top-500 (no breadth). Compare Sharpe etc.
Also report turnover and the WEIGHTED size percentile of the long leg under each
scheme (how far down the size ladder the capital actually sits).

CAVEAT: 10 bps one-way fee is realistic for value-weighted large-cap turnover but
badly understates equal-weighted small-cap trading costs -- EW numbers are
optimistic (gross of realistic costs / market impact).

Reads only data inputs; writes a summary CSV. Production untouched.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
_s = importlib.util.spec_from_file_location("exp", os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-30-xgb-russell1000.py'))
exp = importlib.util.module_from_spec(_s); _s.loader.exec_module(exp)

from src.utils import metrics
from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)
SEEDS = exp.SEEDS
FEE = 0.001


def leg_weights(leg, scheme, cap=0.01):
    """Series permno->weight for a leg, under vw / ew / capped-vw."""
    me = leg.set_index('permno')['me']
    if scheme == 'ew':
        return pd.Series(1.0 / len(me), index=me.index)
    w = me / me.sum()                       # value weight
    if scheme == 'vw':
        return w
    for _ in range(300):                    # cap at `cap`, redistribute excess
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        pool = w[under].sum()
        if pool <= 0:
            break
        w[under] = w[under] + excess * w[under] / pool
    return w


def ls_port(test, scheme, cap=0.01):
    """Long-short, decile by NYSE P10/P90, weighting = scheme. Returns (ret series, avg turnover, long size pctile)."""
    rows, plw, psw, turns, lps = [], {}, {}, [], []
    for date, grp in test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score'].dropna()
        if len(nyse) < 10:
            continue
        lo_b, hi_b = nyse.quantile(0.10), nyse.quantile(0.90)
        if lo_b >= hi_b:
            continue
        longs = grp[grp['score'] >= hi_b].dropna(subset=['me', 'ret_fwd'])
        shorts = grp[grp['score'] <= lo_b].dropna(subset=['me', 'ret_fwd'])
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lw, sw = leg_weights(longs, scheme, cap), leg_weights(shorts, scheme, cap)
        rl = float((longs.set_index('permno')['ret_fwd'] * lw).sum())
        rs = float((shorts.set_index('permno')['ret_fwd'] * sw).sum())
        tl = sum(abs(lw.get(p, 0) - plw.get(p, 0)) for p in set(lw.index) | set(plw)) / 2
        ts = sum(abs(sw.get(p, 0) - psw.get(p, 0)) for p in set(sw.index) | set(psw)) / 2
        rows.append({'date': date, 'ret': rl - rs - FEE * (tl + ts)})
        turns.append(tl + ts)
        pr = grp.dropna(subset=['me']).assign(_pr=lambda d: d['me'].rank(pct=True)).set_index('permno')['_pr']
        lps.append(float((lw * pr.reindex(lw.index)).sum()))      # weighted size pctile of long leg
        plw, psw = lw.to_dict(), sw.to_dict()
    r = pd.DataFrame(rows).set_index('date')['ret']
    return r, float(np.mean(turns)), float(np.mean(lps))


def scored(df):
    tr = df[df['date'] < TRAIN_END]; te = df[df['date'] >= TRAIN_END].copy()
    Xtr, ytr = tr[exp.FEATURES].values.astype(float), tr['ret_fwd'].values.astype(float)
    Xte = te[exp.FEATURES].values.astype(float)
    pred = np.zeros(len(Xte))
    for seed in SEEDS:
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                         subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr); pred += m.predict(Xte)
    te['score'] = pred / len(SEEDS)
    return te


def table(title, test, pim, schemes):
    print("\n" + "=" * 92)
    print(f"  L/S WEIGHTING MECHANISM -- {title}   (selection fixed; only weighting varies)")
    print("=" * 92)
    print(f"  {'Scheme':<14} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'MDD':>8} {'Calm':>6} {'Panic':>6} "
          f"{'Turnover':>9} {'LongSize%ile':>13}")
    print("  " + "-" * 88)
    out = []
    for name, cap in schemes:
        r, turn, lps = ls_port(test, name, cap)
        rr = r.dropna(); p = pim.reindex(rr.index)
        sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
        ann, vol, sharpe, mdd = metrics(rr)
        lbl = name if cap is None or name != 'cap' else f'cap@{cap:.0%}'
        print(f"  {lbl:<14} {ann:>7.1%} {vol:>6.1%} {sharpe:>7.2f} {mdd:>7.1%} "
              f"{sh(rr[p < 0.5]):>6.2f} {sh(rr[p >= 0.5]):>6.2f} {turn:>8.0%} {lps:>12.0%}")
        out.append({'universe': title, 'scheme': lbl, 'ann': ann, 'vol': vol, 'sharpe': sharpe,
                    'mdd': mdd, 'turnover': turn, 'long_size_pctile': lps})
    print("=" * 92)
    return out


def main():
    base = exp.build_features()
    schemes = [('vw', None), ('cap', 0.01), ('ew', None)]

    print("scoring full universe ...", flush=True)
    te_full = scored(base)
    pim = te_full.drop_duplicates('date').set_index('date')['pi_filter']
    rows = table("FULL UNIVERSE (breadth)", te_full, pim, schemes)

    print("scoring Top-500 ...", flush=True)
    te_500 = scored(exp.apply_size_screen(base, 500))
    pim5 = te_500.drop_duplicates('date').set_index('date')['pi_filter']
    rows += table("S&P 500 / Top-500 (no breadth)", te_500, pim5, [('vw', None), ('ew', None)])

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-weighting-mechanism_summary.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  saved {out}")


if __name__ == '__main__':
    main()
