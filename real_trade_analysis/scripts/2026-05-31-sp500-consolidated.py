"""
2026-05-31-sp500-consolidated.py
================================
EXPLORATORY (experiments/ -- throwaway).

Consolidated comparison on ONE universe: S&P 500 proxy = Top-500 by market cap
each month. All strategies built from a SINGLE 50-seed XGB training (so numbers
are mutually consistent), reported over two windows:
  * FULL    2011-2024  (167 mo)
  * OVERLAP 2015-2024  (109 mo, the only window the real SPMO ETF exists)

Strategies (all on Top-500):
  S&P500 proxy (cap-wt)      -- market benchmark
  SPMO replica (risk-adj)    -- S&P 500 Momentum rules (top quintile, FMC x score, semi-annual)
  SPMO (real ETF)            -- actual fund, overlap only
  XGB long-short             -- P10/P90 value-weighted (the thesis construction)
  XGB long-only (monthly)    -- top-decile NYSE-P90 value-weighted
  XGB SPMO-style             -- XGB signal through the S&P 500 Momentum machinery

Reuses build_longonly/add_spmo_inputs (spmo module), long_only_port (lo module),
long_short_port/metrics (src). Reads only data inputs; writes a summary CSV.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

def _load(name, path):
    s = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, 'real_trade_analysis', 'scripts', path))
    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod

sp = _load("spmo", "2026-05-31-spmo-replica-vs-xgb.py")
lo = _load("lo",   "2026-05-31-xgb-longonly-sp500.py")
exp = sp.exp

from src.utils import long_short_port, metrics
from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)

SEEDS   = exp.SEEDS
OVERLAP = (pd.Period('2015-11'), pd.Period('2024-11'))


def stats(r, pim_p, window=None):
    r = r.dropna()
    if window is not None:
        ym = r.index.to_period('M')
        r = r[(ym >= window[0]) & (ym <= window[1])]
    if len(r) < 2:
        return dict(mo=len(r), ann=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan, calm=np.nan, panic=np.nan)
    p = pim_p.reindex(r.index.to_period('M')).values
    sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    ann, vol, sharpe, mdd = metrics(r)
    return dict(mo=len(r), ann=ann, vol=vol, sharpe=sharpe, mdd=mdd,
                calm=sh(r[p < 0.5]), panic=sh(r[p >= 0.5]))


def main():
    df = exp.apply_size_screen(sp.add_spmo_inputs(exp.build_features()), 500)
    train = df[df['date'] < TRAIN_END]
    test  = df[df['date'] >= TRAIN_END].copy()
    print(f"Top-500 panel: {len(df):,} rows | test months {test['date'].nunique()}", flush=True)

    Xtr, ytr = train[exp.FEATURES].values.astype(float), train['ret_fwd'].values.astype(float)
    Xte = test[exp.FEATURES].values.astype(float)
    preds = np.zeros(len(Xte))
    for i, seed in enumerate(SEEDS, 1):
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr)
        preds += m.predict(Xte)
        if i % 10 == 0:
            print(f"  xgb seed {i}/{len(SEEDS)}", flush=True)
    test['score'] = preds / len(SEEDS)
    test['spmo_signal'] = test['mom_value'] / test['sigma_m']

    pim = test.drop_duplicates('date').set_index('date')['pi_filter']
    pim_p = pim.copy(); pim_p.index = pim.index.to_period('M')

    real = pd.read_csv(os.path.join(_ROOT, 'real_trade_analysis', 'data', '_spmo_real_monthly.csv'),
                       index_col=0, parse_dates=True).iloc[:, 0]

    series = [
        ('S&P500 proxy (cap-wt)',   sp.build_longonly(test, 'me',          'cap',   1.00, None)),
        ('SPMO replica (risk-adj)', sp.build_longonly(test, 'spmo_signal', 'score', 0.20, 0.09)),
        ('SPMO (real ETF)',         real),
        ('XGB long-short',          long_short_port(test, 'score')),
        ('XGB long-only (monthly)', lo.long_only_port(test, 'score')),
        ('XGB SPMO-style',          sp.build_longonly(test, 'score',       'score', 0.20, 0.09)),
    ]

    def render(title, window):
        print("\n" + "=" * 86)
        print(f"  S&P 500 UNIVERSE -- {title}")
        print("=" * 86)
        print(f"  {'Strategy':<26} {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'MDD':>8} {'Calm':>6} {'Panic':>6}")
        print("  " + "-" * 80)
        rows = []
        for lbl, r in series:
            s = stats(r, pim_p, window)
            rows.append({'strategy': lbl, 'window': title, **s})
            line = (f"  {lbl:<26} {s['mo']:>4} {s['ann']:>7.1%} {s['vol']:>6.1%} "
                    f"{s['sharpe']:>7.2f} {s['mdd']:>7.1%} {s['calm']:>6.2f} {s['panic']:>6.2f}"
                    if s['mo'] >= 2 else f"  {lbl:<26} {'(no data in window)':>40}")
            print(line)
        print("=" * 86)
        return rows

    allrows = render("FULL TEST 2011-2024", None) + render("OVERLAP 2015-2024 (vs real SPMO)", OVERLAP)
    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-sp500-consolidated_summary.csv')
    pd.DataFrame(allrows).to_csv(out, index=False)
    print(f"\n  saved {out}")

    # ── Size profile of holdings WITHIN the S&P 500 (Top-500) universe ──────────
    recs = []
    for date, grp in test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score'].dropna()
        if len(nyse) < 10:
            continue
        lo_b, hi_b = nyse.quantile(0.10), nyse.quantile(0.90)
        if lo_b >= hi_b:
            continue
        g = grp.dropna(subset=['me']).copy()
        g['_pr'] = g['me'].rank(pct=True)                 # 1.0 = largest within Top-500 that month
        longs, shorts = g[g['score'] >= hi_b], g[g['score'] <= lo_b]
        recs.append({'date': date,
                     'uni_logme': np.log(g['me']).mean(),
                     'long_logme': np.log(longs['me']).mean(),
                     'short_logme': np.log(shorts['me']).mean(),
                     'long_pctile': longs['_pr'].mean(),
                     'short_pctile': shorts['_pr'].mean()})
    szp = pd.DataFrame(recs).set_index('date'); szp['year'] = szp.index.year
    ann = szp.groupby('year')[['uni_logme', 'long_logme', 'short_logme', 'long_pctile', 'short_pctile']].mean()
    print("\n" + "=" * 84)
    print("  SIZE OF HOLDINGS *WITHIN* THE S&P 500 (Top-500) -- pctile is rank inside Top-500")
    print("=" * 84)
    print(f"  {'Year':<6} {'Uni logme':>10} {'Long logme':>11} {'Short logme':>12} {'Long pctile':>12} {'Short pctile':>13}")
    print("  " + "-" * 76)
    for y, row in ann.iterrows():
        print(f"  {y:<6} {row['uni_logme']:>10.2f} {row['long_logme']:>11.2f} {row['short_logme']:>12.2f} "
              f"{row['long_pctile']:>11.0%} {row['short_pctile']:>12.0%}")
    ov = szp[['uni_logme', 'long_logme', 'short_logme', 'long_pctile', 'short_pctile']].mean()
    print("  " + "-" * 76)
    print(f"  {'ALL':<6} {ov['uni_logme']:>10.2f} {ov['long_logme']:>11.2f} {ov['short_logme']:>12.2f} "
          f"{ov['long_pctile']:>11.0%} {ov['short_pctile']:>12.0%}")
    print("=" * 84)
    ann.to_csv(os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-sp500-size_by_year.csv'))
    print("  saved 2026-05-31-sp500-size_by_year.csv")


if __name__ == '__main__':
    main()
