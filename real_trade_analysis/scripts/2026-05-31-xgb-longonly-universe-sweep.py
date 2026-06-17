"""
2026-05-31-xgb-longonly-universe-sweep.py
=========================================
EXPLORATORY (experiments/ -- throwaway).

XGB long-only (the strategy's own construction: top-decile NYSE-P90 breakpoint,
value-weighted, MONTHLY rebalance) swept across increasing universe sizes, vs the
real SPMO ETF benchmark.

For each universe (Top-N by market cap each month; None = full eligible universe):
  train+trade the 50-seed XGB ensemble on that universe, build the long-only
  portfolio, and report performance over TWO windows:
    * Full test     2011-01 .. 2024-11  (167 mo) -- SPMO did not exist; no comparison.
    * SPMO overlap  2015-11 .. 2024-11  (109 mo) -- the only fair window vs SPMO.

SPMO real (Yahoo total return) over the overlap: Ann 16.9% | Vol 16.4% | Sharpe 1.04.

Reads only data inputs; writes only its own CSV. Production untouched.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

# reuse long_only_port + build_features + constants from prior experiments
_ls = importlib.util.spec_from_file_location(
    "lo", os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-31-xgb-longonly-sp500.py'))
lo = importlib.util.module_from_spec(_ls); _ls.loader.exec_module(lo)
exp = lo.exp  # the russell1000 module (build_features, apply_size_screen, FEATURES, SEEDS)

from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)

UNIVERSES = [100, 250, 500, 1000, 2000, 3000, None]   # None = full eligible
SEEDS = exp.SEEDS
OVERLAP = (pd.Period('2015-11'), pd.Period('2024-11'))   # SPMO life


def window_stats(r, lo_p=None, hi_p=None):
    r = r.dropna()
    if lo_p is not None:
        ym = r.index.to_period('M')
        r = r[(ym >= lo_p) & (ym <= hi_p)]
    if len(r) < 2:
        return (len(r), np.nan, np.nan, np.nan)
    sh = r.mean() / r.std() * np.sqrt(12)
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    return (len(r), ann, vol, sh)


def xgb_longonly_returns(sub):
    train = sub[sub['date'] < TRAIN_END]
    test  = sub[sub['date'] >= TRAIN_END].copy()
    Xtr = train[exp.FEATURES].values.astype(float)
    ytr = train['ret_fwd'].values.astype(float)
    Xte = test[exp.FEATURES].values.astype(float)
    preds = np.zeros(len(Xte))
    for seed in SEEDS:
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr)
        preds += m.predict(Xte)
    preds /= len(SEEDS)
    test['score_xgb'] = preds
    return lo.long_only_port(test, 'score_xgb')   # top-decile NYSE-P90 VW, monthly


def _row(label, r):
    """(label, full: mo/ann/vol/sharpe, overlap: mo/ann/vol/sharpe) for a return series."""
    nf, af, vf, sf = window_stats(r)
    no, ao, vo, so = window_stats(r, *OVERLAP)
    return (label, nf, af, vf, sf, no, ao, vo, so)


def main():
    base = exp.build_features()

    # Benchmarks: real SPMO (2015-2024 only), our validated SPMO replica and the
    # cap-weighted market (both full 2011-2024) from the replica run.
    spmo = pd.read_csv(os.path.join(_ROOT, 'real_trade_analysis', 'data', '_spmo_real_monthly.csv'),
                       index_col=0, parse_dates=True).iloc[:, 0]
    repl = pd.read_csv(os.path.join(_ROOT, 'real_trade_analysis', 'results',
                       '2026-05-31-spmo-replica-vs-xgb_returns.csv'),
                       index_col=0, parse_dates=True)

    rows = []
    for n in UNIVERSES:
        sub = base if n is None else exp.apply_size_screen(base, n)
        label = 'Full (~3.3k)' if n is None else f'Top-{n}'
        rows.append(_row(label, xgb_longonly_returns(sub)))
        print(f"  done {label}: full Sharpe {rows[-1][4]:.2f} | overlap Sharpe {rows[-1][8]:.2f}", flush=True)

    bench = [
        _row('SPMO replica',  repl['SPMO replica (risk-adj)']),
        _row('S&P500 (cap-wt)', repl['S&P500 proxy (cap-wt)']),
        _row('SPMO (real ETF)', spmo),   # full cols will be the 2015-2024 life (no pre-2015 data)
    ]

    print("\n" + "=" * 96)
    print("  XGB LONG-ONLY by universe size  vs  SPMO   (long-only, top-decile VW, monthly)")
    print("  Primary window: FULL TEST 2011-2024.  Overlap col = 2015-2024 (only window real SPMO exists).")
    print("=" * 96)
    print(f"  {'Strategy':<16} | {'FULL TEST 2011-2024':^26} | {'2015-2024':^9}")
    print(f"  {'':<16} | {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} | {'Sharpe':>9}")
    print("  " + "-" * 60)
    for (label, nf, af, vf, sf, no, ao, vo, so) in rows:
        print(f"  {label:<16} | {nf:>4} {af:>7.1%} {vf:>6.1%} {sf:>7.2f} | {so:>9.2f}")
    print("  " + "-" * 60)
    for (label, nf, af, vf, sf, no, ao, vo, so) in bench:
        # real SPMO has no pre-2015 data, so its "full" cols are its 2015-2024 life
        print(f"  {label:<16} | {nf:>4} {af:>7.1%} {vf:>6.1%} {sf:>7.2f} | {so:>9.2f}")
    print("=" * 96)
    print("  Note: 'SPMO (real ETF)' full-test columns cover only 2015-2024 (fund launched Nov 2015);")
    print("        use the 'SPMO replica' row for the true 2011-2024 comparison.")

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-xgb-longonly-universe-sweep_summary.csv')
    cols = ['strategy', 'full_mo', 'full_ann', 'full_vol', 'full_sharpe',
            'ov_mo', 'ov_ann', 'ov_vol', 'ov_sharpe']
    pd.DataFrame(rows + bench, columns=cols).to_csv(out, index=False)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
