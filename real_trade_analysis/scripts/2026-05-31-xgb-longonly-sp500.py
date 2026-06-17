"""
2026-05-31-xgb-longonly-sp500.py
================================
EXPLORATORY (experiments/ -- throwaway).

Long-only version of the THESIS strategy (XGB regime-momentum signal) on the
S&P 500 universe (Top-500 by market cap proxy), using the strategy's OWN
long-only construction -- NOT S&P's momentum-score weighting:
  * Signal     : XGB ensemble (mom_1..12 + pi_filter), train+trade on Top-500, 50 seeds.
  * Long leg   : top decile by score via NYSE P90 breakpoint, value-weighted by me
                 (identical to scripts/cross_sectional_model.py long_only_port).
  * Rebalance  : monthly (native) and quarterly.
  * Fees       : 10 bps one-way on turnover. Long-only.

Reference points already established this session (Top-500 universe, 167 mo):
  market (Top-500 cap-wt) Sharpe ~0.99 ; XGB long-short Top-500 Sharpe 0.28 ;
  XGB SPMO-style long-only Sharpe 0.81.

Reads only data inputs; writes only its own CSVs. Production untouched.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

_spec = importlib.util.spec_from_file_location(
    "xgb_russell1000", os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-30-xgb-russell1000.py'))
exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp)

from src.utils import metrics
from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)

TOP_N = 500
FEE   = 0.001
SEEDS = exp.SEEDS
QUARTERLY = {3, 6, 9, 12}


def long_only_port(df_test, score_col, fee=FEE, rebal_months=None):
    """Top-decile (NYSE P90) long-only, value-weighted. Mirrors production."""
    if rebal_months is None:
        rebal_months = set(range(1, 13))
    monthly, prev = [], {}
    for date, grp in df_test.groupby('date'):
        if date.month in rebal_months:
            nyse = grp[grp['exchcd'] == 1][score_col].dropna()
            if len(nyse) < 10:
                continue
            hi = nyse.quantile(0.90)
            longs = grp[grp[score_col] >= hi].dropna(subset=['me'])
            if longs['me'].sum() == 0:
                continue
            tot = longs['me'].sum()
            new = (longs.set_index('permno')['me'] / tot).to_dict()
            turn = sum(abs(new.get(p, 0) - prev.get(p, 0)) for p in set(new) | set(prev)) / 2
            r = (longs['ret_fwd'] * longs['me']).sum() / tot
            monthly.append({'date': date, 'ret': r - fee * turn})
            prev = new
        else:
            held = grp[grp['permno'].isin(prev)]
            if held.empty or held['me'].sum() == 0:
                continue
            tot = held['me'].sum()
            r = (held['ret_fwd'] * held['me']).sum() / tot
            prev = (held.set_index('permno')['me'] / tot).to_dict()
            monthly.append({'date': date, 'ret': r})
    return pd.DataFrame(monthly).set_index('date')['ret']


def market_port(df_test):
    """Cap-weighted Top-500 (S&P 500 proxy), all names, monthly."""
    rows = []
    for date, grp in df_test.groupby('date'):
        g = grp.dropna(subset=['me', 'ret_fwd'])
        if g['me'].sum() == 0:
            continue
        rows.append({'date': date, 'ret': (g['ret_fwd'] * g['me']).sum() / g['me'].sum()})
    return pd.DataFrame(rows).set_index('date')['ret']


def regime_row(r, pim, label):
    r = r.dropna()
    p = pim.reindex(r.index)
    calm, panic = r[p < 0.5], r[p >= 0.5]
    sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    ann, vol, sharpe, mdd = metrics(r)
    return {'label': label, 'months': len(r), 'ann_ret': ann, 'vol': vol,
            'sharpe': sharpe, 'mdd': mdd, 'calm': sh(calm), 'panic': sh(panic)}


def main():
    df = exp.apply_size_screen(exp.build_features(), TOP_N)
    train = df[df['date'] < TRAIN_END]
    test  = df[df['date'] >= TRAIN_END].copy()
    print(f"Top-{TOP_N} panel: {len(df):,} rows | test months {test['date'].nunique()}", flush=True)

    Xtr = train[exp.FEATURES].values.astype(float)
    ytr = train['ret_fwd'].values.astype(float)
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
    preds /= len(SEEDS)
    test['score_xgb'] = preds

    pim = test.drop_duplicates('date').set_index('date')['pi_filter']
    strategies = [
        ('S&P500 proxy (cap-wt)',     market_port(test)),
        ('XGB long-only (monthly)',   long_only_port(test, 'score_xgb')),
        ('XGB long-only (quarterly)', long_only_port(test, 'score_xgb', rebal_months=QUARTERLY)),
    ]
    rows = [regime_row(r, pim, lbl) for lbl, r in strategies]

    print("\n" + "=" * 82)
    print("  THESIS XGB -- LONG-ONLY on S&P 500 (Top-500) universe, top-decile value-weighted")
    print("=" * 82)
    print(f"  {'Strategy':<28} {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} "
          f"{'MDD':>8} {'CalmSh':>7} {'PanicSh':>8}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['label']:<28} {r['months']:>4} {r['ann_ret']:>7.1%} {r['vol']:>6.1%} "
              f"{r['sharpe']:>7.2f} {r['mdd']:>7.1%} {r['calm']:>7.2f} {r['panic']:>8.2f}")
    print("=" * 82)

    out = os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-xgb-longonly-sp500_returns.csv')
    pd.DataFrame({lbl: r for (lbl, r) in strategies}).to_csv(out)
    print(f"  saved {out}")


if __name__ == '__main__':
    main()
