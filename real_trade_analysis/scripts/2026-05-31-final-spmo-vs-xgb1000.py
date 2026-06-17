"""
2026-05-31-final-spmo-vs-xgb1000.py
===================================
EXPLORATORY (experiments/ -- throwaway).

Clean head-to-head over the test period:
  SPMO replica (S&P 500 Momentum rules, full 2011-2024)
  SPMO real ETF (2015-2024, overlap only)
  S&P 500 cap-weighted market (reference)
  XGB trained on Top-1000, traded on S&P 500 (Top-500):  long-only AND long-short

Full metrics (Ann/Vol/Sharpe/MDD + calm/panic) over FULL 2011-2024 and the
2015-2024 SPMO-overlap window. SPMO replica / market / real-SPMO returns are
loaded from prior runs; only the XGB(train1000/trade500) lines are computed here.

Reads only data inputs; production untouched.
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
OVERLAP = (pd.Period('2015-11'), pd.Period('2024-11'))


def stats(r, pim_p, window=None):
    r = r.dropna()
    if window is not None:
        ym = r.index.to_period('M'); r = r[(ym >= window[0]) & (ym <= window[1])]
    if len(r) < 2:
        return dict(mo=len(r), ann=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan, calm=np.nan, panic=np.nan)
    p = pim_p.reindex(r.index.to_period('M')).values
    sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    ann, vol, sharpe, mdd = metrics(r)
    return dict(mo=len(r), ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, calm=sh(r[p < 0.5]), panic=sh(r[p >= 0.5]))


def main():
    base = exp.build_features()
    # XGB: train Top-1000, trade Top-500
    train = exp.apply_size_screen(base, 1000); train = train[train['date'] < TRAIN_END]
    test = exp.apply_size_screen(base, 500); test = test[test['date'] >= TRAIN_END].copy()
    Xtr, ytr = train[exp.FEATURES].values.astype(float), train['ret_fwd'].values.astype(float)
    Xte = test[exp.FEATURES].values.astype(float)
    pred = np.zeros(len(Xte))
    for i, seed in enumerate(SEEDS, 1):
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                         subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr); pred += m.predict(Xte)
        if i % 10 == 0:
            print(f"  xgb seed {i}/{len(SEEDS)}", flush=True)
    test['score'] = pred / len(SEEDS)
    xgb_lo = lo.long_only_port(test, 'score')
    xgb_ls = long_short_port(test, 'score')

    pim = test.drop_duplicates('date').set_index('date')['pi_filter']
    pim_p = pim.copy(); pim_p.index = pim.index.to_period('M')

    repl = pd.read_csv(os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-spmo-replica-vs-xgb_returns.csv'),
                       index_col=0, parse_dates=True)
    real = pd.read_csv(os.path.join(_ROOT, 'real_trade_analysis', 'data', '_spmo_real_monthly.csv'),
                       index_col=0, parse_dates=True).iloc[:, 0]

    series = [
        ('SPMO replica',            repl['SPMO replica (risk-adj)']),
        ('SPMO (real ETF)',         real),
        ('S&P500 (cap-wt)',         repl['S&P500 proxy (cap-wt)']),
        ('XGB LO (tr1000/S&P500)',  xgb_lo),
        ('XGB L/S (tr1000/S&P500)', xgb_ls),
    ]

    def render(title, win):
        print("\n" + "=" * 86)
        print(f"  {title}")
        print("=" * 86)
        print(f"  {'Strategy':<24} {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'MDD':>8} {'Calm':>6} {'Panic':>6}")
        print("  " + "-" * 78)
        out = []
        for lbl, r in series:
            s = stats(r, pim_p, win)
            out.append({'strategy': lbl, 'window': title, **s})
            if s['mo'] >= 2:
                print(f"  {lbl:<24} {s['mo']:>4} {s['ann']:>7.1%} {s['vol']:>6.1%} {s['sharpe']:>7.2f} "
                      f"{s['mdd']:>7.1%} {s['calm']:>6.2f} {s['panic']:>6.2f}")
            else:
                print(f"  {lbl:<24} {'(no data in window)':>50}")
        print("=" * 86)
        return out

    rows = render("FULL TEST 2011-2024", None) + render("OVERLAP 2015-2024 (vs real SPMO)", OVERLAP)
    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-final-spmo-vs-xgb1000_summary.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame({'xgb_lo': xgb_lo, 'xgb_ls': xgb_ls}).to_csv(
        os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-final-spmo-vs-xgb1000_returns.csv'))
    print(f"\n  saved {out}")


if __name__ == '__main__':
    main()
