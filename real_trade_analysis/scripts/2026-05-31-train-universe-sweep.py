"""
2026-05-31-train-universe-sweep.py
==================================
EXPLORATORY (experiments/ -- throwaway).

Addresses: "training on the broad universe means learning on stocks I won't trade."
Hold the TRADING universe fixed at Top-500 (the S&P 500 proxy you'd actually trade)
and vary only the TRAINING universe. Find the smallest *tradeable* training universe
(Top-1000 / Top-2000 -- all liquid names, no microcaps) that captures the edge that
full-universe training gives.

  train Top-500  -> trade Top-500   (narrow; the weak baseline, LO ~0.86)
  train Top-1000 -> trade Top-500
  train Top-2000 -> trade Top-500
  train Top-3000 -> trade Top-500
  train Full     -> trade Top-500   (broad; the strong one, LO ~1.07)

Long-only (top-decile NYSE-P90 VW, monthly) and long-short (P10/P90 VW) reported
over full 2011-2024 and the 2015-2024 SPMO overlap. Reference: SPMO replica ~1.00
(full)/1.01 (overlap); real SPMO 1.04 (overlap); cap-wt market ~0.99.

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
_l = importlib.util.spec_from_file_location("lo", os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-31-xgb-longonly-sp500.py'))
lo = importlib.util.module_from_spec(_l); _l.loader.exec_module(lo)
exp = lo.exp

from src.utils import long_short_port, metrics
from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)
SEEDS = exp.SEEDS
TRADE_N = 500
TRAIN_UNIVERSES = [500, 1000, 2000, 3000, None]   # None = full
OVERLAP = (pd.Period('2015-11'), pd.Period('2024-11'))


def st(r, win=None):
    r = r.dropna()
    if win is not None:
        ym = r.index.to_period('M'); r = r[(ym >= win[0]) & (ym <= win[1])]
    if len(r) < 2:
        return (np.nan, np.nan)
    ann, vol, sharpe, mdd = metrics(r)
    return (sharpe, ann)


def main():
    base = exp.build_features()
    # Fixed trading universe: Top-500 test rows.
    test500 = exp.apply_size_screen(base, TRADE_N)
    test500 = test500[test500['date'] >= TRAIN_END].copy()
    print(f"Trading universe fixed = Top-{TRADE_N} ({test500['date'].nunique()} test months)", flush=True)

    Xte = test500[exp.FEATURES].values.astype(float)
    rows = []
    for n in TRAIN_UNIVERSES:
        train_sub = (base if n is None else exp.apply_size_screen(base, n))
        train_sub = train_sub[train_sub['date'] < TRAIN_END]
        Xtr = train_sub[exp.FEATURES].values.astype(float)
        ytr = train_sub['ret_fwd'].values.astype(float)
        pred = np.zeros(len(Xte))
        for seed in SEEDS:
            m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                             subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, tree_method='hist',
                             random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
            m.fit(Xtr, ytr); pred += m.predict(Xte)
        t = test500.copy(); t['score'] = pred / len(SEEDS)
        lo_r = lo.long_only_port(t, 'score')
        ls_r = long_short_port(t, 'score')
        lo_sh_f, lo_ann_f = st(lo_r); lo_sh_o, _ = st(lo_r, OVERLAP)
        ls_sh_f, ls_ann_f = st(ls_r); ls_sh_o, _ = st(ls_r, OVERLAP)
        label = 'Full (~3.3k)' if n is None else f'Top-{n}'
        rows.append((label, len(train_sub), lo_sh_f, lo_ann_f, lo_sh_o, ls_sh_f, ls_ann_f, ls_sh_o))
        print(f"  trained on {label} ({len(train_sub):,} rows) -> LO full Sharpe {lo_sh_f:.2f}", flush=True)

    print("\n" + "=" * 98)
    print(f"  TRAIN-UNIVERSE SWEEP  (trading universe fixed = Top-{TRADE_N})")
    print("=" * 98)
    print(f"  {'Train universe':<14} {'TrainRows':>10} | {'LONG-ONLY':^26} | {'LONG-SHORT':^26}")
    print(f"  {'':<14} {'':>10} | {'Full Sh':>8} {'Full Ann':>9} {'Ovlp Sh':>8} | {'Full Sh':>8} {'Full Ann':>9} {'Ovlp Sh':>8}")
    print("  " + "-" * 94)
    for (label, nrows, lo_f, lo_a, lo_o, ls_f, ls_a, ls_o) in rows:
        print(f"  {label:<14} {nrows:>10,} | {lo_f:>8.2f} {lo_a:>8.1%} {lo_o:>8.2f} | "
              f"{ls_f:>8.2f} {ls_a:>8.1%} {ls_o:>8.2f}")
    print("  " + "-" * 94)
    print(f"  reference: SPMO replica 1.00 (full) / 1.01 (ovlp) ; real SPMO 1.04 (ovlp) ; market ~0.99")
    print("=" * 98)

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-train-universe-sweep_summary.csv')
    pd.DataFrame(rows, columns=['train_universe', 'train_rows', 'lo_sharpe_full', 'lo_ann_full',
                                'lo_sharpe_ovlp', 'ls_sharpe_full', 'ls_ann_full', 'ls_sharpe_ovlp']).to_csv(out, index=False)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
