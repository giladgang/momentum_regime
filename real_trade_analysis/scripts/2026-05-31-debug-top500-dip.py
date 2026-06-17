"""
2026-05-31-debug-top500-dip.py
==============================
DIAGNOSTIC (experiments/). Investigate why XGB long-only Sharpe DIPS at Top-500
(0.86) below Top-250 (0.97) and Top-1000 (0.96) in the universe sweep.

The sweep retrained a SEPARATE model per universe (train+trade on each). This
script removes that confound: train ONE model on the full universe, then trade the
long-only portfolio on each Top-N subset using the SAME predictions. Compare the
fixed-model curve to the retrain-per-universe curve (loaded from the sweep CSV).

  * Dip disappears with a fixed model  -> caused by per-universe retraining (not a bug).
  * Dip persists with a fixed model    -> selection/value-weighting construction effect.

Also reports avg #longs per month (mechanics) for each universe.
Reads only data inputs. Production untouched.
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
UNIVERSES = [100, 250, 500, 1000, 2000, 3000, None]


def avg_longs(sub):
    n = []
    for _, grp in sub.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score'].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        n.append((grp['score'] >= hi).sum())
    return float(np.mean(n)) if n else np.nan


def main():
    base = exp.build_features()
    # ONE model, trained on the full universe.
    tr = base[base['date'] < TRAIN_END]
    te = base[base['date'] >= TRAIN_END].copy()
    Xtr, ytr = tr[exp.FEATURES].values.astype(float), tr['ret_fwd'].values.astype(float)
    Xte = te[exp.FEATURES].values.astype(float)
    pred = np.zeros(len(Xte))
    for i, seed in enumerate(SEEDS, 1):
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                         subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr); pred += m.predict(Xte)
        if i % 10 == 0:
            print(f"  trained {i}/{len(SEEDS)} (full-universe model)", flush=True)
    te['score'] = pred / len(SEEDS)

    # retrain-per-universe numbers from the sweep, for side-by-side
    try:
        sw = pd.read_csv(os.path.join(_ROOT, 'real_trade_analysis', 'results',
                         '2026-05-31-xgb-longonly-universe-sweep_summary.csv')).set_index('strategy')
    except Exception:
        sw = None

    rows = []
    for n in UNIVERSES:
        sub = te if n is None else exp.apply_size_screen(te, n)
        lo_r = lo.long_only_port(sub, 'score')
        ls_r = long_short_port(sub, 'score')
        lo_ann, _, lo_sh, _ = metrics(lo_r)
        ls_ann, _, ls_sh, _ = metrics(ls_r)
        label = 'Full' if n is None else f'Top-{n}'
        swkey = 'Full (~3.3k)' if n is None else f'Top-{n}'
        rt_sh = sw.loc[swkey, 'full_sharpe'] if (sw is not None and swkey in sw.index) else np.nan
        rt_ann = sw.loc[swkey, 'full_ann'] if (sw is not None and swkey in sw.index) else np.nan
        rows.append((label, lo_sh, lo_ann, ls_sh, ls_ann, avg_longs(sub), rt_sh, rt_ann))
        print(f"  done {label}: fixed-model LO Sharpe {lo_sh:.2f}", flush=True)

    print("\n" + "=" * 96)
    print("  TOP-500 DIP DIAGNOSTIC -- FIXED full-universe model traded on each Top-N (full 2011-2024)")
    print("=" * 96)
    print(f"  {'Universe':<10} | {'FIXED MODEL (this run)':^34} | {'RETRAIN-PER-UNIV (sweep)':^24}")
    print(f"  {'':<10} | {'LO Sharpe':>9} {'LO Ann':>8} {'LS Sharpe':>10} {'avg#long':>8} | {'LO Sharpe':>9} {'LO Ann':>8}")
    print("  " + "-" * 92)
    for (label, lo_sh, lo_ann, ls_sh, ls_ann, nlong, rt_sh, rt_ann) in rows:
        print(f"  {label:<10} | {lo_sh:>9.2f} {lo_ann:>7.1%} {ls_sh:>10.2f} {nlong:>8.0f} | "
              f"{rt_sh:>9.2f} {rt_ann:>7.1%}")
    print("=" * 96)

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-debug-top500-dip_summary.csv')
    pd.DataFrame(rows, columns=['universe', 'fixed_lo_sharpe', 'fixed_lo_ann', 'fixed_ls_sharpe',
                                'fixed_ls_ann', 'avg_longs', 'retrain_lo_sharpe', 'retrain_lo_ann']).to_csv(out, index=False)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
