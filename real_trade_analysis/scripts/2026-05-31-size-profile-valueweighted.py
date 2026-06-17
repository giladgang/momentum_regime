"""
2026-05-31-size-profile-valueweighted.py
========================================
EXPLORATORY (experiments/ -- throwaway).

Corrected size profile of the XGB strategy's holdings, weighting each name by its
ACTUAL portfolio weight (value weight = me / sum me within the leg) -- matching the
value-weighted construction -- shown next to the equal-weighted average for contrast.

Long leg = score >= NYSE P90, Short leg = score <= NYSE P10 (same selection the
L/S and long-only portfolios use). Reported for two universes:
  * Full eligible universe (train+trade)
  * S&P 500 proxy = Top-500 by market cap (train+trade)

pctile = rank of me within that month's universe (1.0 = largest). log_me = log market equity.
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

from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)
SEEDS = exp.SEEDS


def scored_test(df):
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


def profile(test):
    recs = []
    for date, grp in test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score'].dropna()
        if len(nyse) < 10:
            continue
        lo_b, hi_b = nyse.quantile(0.10), nyse.quantile(0.90)
        if lo_b >= hi_b:
            continue
        g = grp.dropna(subset=['me']).copy()
        g['_pr'] = g['me'].rank(pct=True); g['_lm'] = np.log(g['me'])
        legs = {'long': g[g['score'] >= hi_b], 'short': g[g['score'] <= lo_b], 'uni': g}
        for name, leg in legs.items():
            if len(leg) == 0 or leg['me'].sum() == 0:
                continue
            w = leg['me'] / leg['me'].sum()
            recs.append({'date': date, 'leg': name,
                         'ew_lm': leg['_lm'].mean(),  'vw_lm': float((w * leg['_lm']).sum()),
                         'ew_pr': leg['_pr'].mean(),  'vw_pr': float((w * leg['_pr']).sum())})
    return pd.DataFrame(recs).groupby('leg')[['ew_lm', 'vw_lm', 'ew_pr', 'vw_pr']].mean()


def show(title, prof):
    print("\n" + "=" * 78)
    print(f"  SIZE OF HOLDINGS -- EQUAL-WT vs VALUE-WT  ({title})")
    print("=" * 78)
    print(f"  {'Leg':<8} {'EW logme':>9} {'VW logme':>9}   {'EW pctile':>10} {'VW pctile':>10}")
    print("  " + "-" * 56)
    for leg in ['uni', 'long', 'short']:
        r = prof.loc[leg]
        print(f"  {leg:<8} {r['ew_lm']:>9.2f} {r['vw_lm']:>9.2f}   {r['ew_pr']:>9.0%} {r['vw_pr']:>10.0%}")
    print("=" * 78)


def main():
    base = exp.build_features()
    print("scoring full universe ...", flush=True)
    pf_full = profile(scored_test(base))
    print("scoring Top-500 ...", flush=True)
    pf_500 = profile(scored_test(exp.apply_size_screen(base, 500)))

    show("FULL UNIVERSE", pf_full)
    show("S&P 500 (Top-500)", pf_500)

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-size-profile-valueweighted_summary.csv')
    pd.concat({'full': pf_full, 'top500': pf_500}, names=['universe']).to_csv(out)
    print(f"\n  saved {out}")


if __name__ == '__main__':
    main()
