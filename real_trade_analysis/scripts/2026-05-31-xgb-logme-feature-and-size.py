"""
2026-05-31-xgb-logme-feature-and-size.py
========================================
EXPLORATORY (experiments/ -- throwaway). Full eligible universe, test 2011-2024.

Two questions:
 (A) Does adding log market equity (log_me, lagged 1 month -- the market-cap
     feature) to the XGB inputs improve results?
       baseline : mom_1..12 + pi_filter            (13 features)
       +size    : mom_1..12 + pi_filter + log_me    (14 features)
     Both built into the strategy's L/S (P10/P90 VW) and long-only (P90 VW,
     monthly) portfolios; Sharpe/Ann/Vol/MDD + calm/panic reported.
 (B) Size profile of the CURRENT strategy (baseline) holdings: average log_me and
     within-universe size percentile of the long leg vs the short leg, per year and
     overall. (The long-only long leg == the L/S long leg: both are score >= NYSE P90.)

Same sample for both models (rows with a valid lagged log_me) so the comparison
isolates the feature, not the universe. Reads only data inputs; writes only its
own CSVs. Production untouched.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

_ls = importlib.util.spec_from_file_location(
    "lo", os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-31-xgb-longonly-sp500.py'))
lo = importlib.util.module_from_spec(_ls); _ls.loader.exec_module(lo)
exp = lo.exp

from src.utils import long_short_port, metrics
from config import (TRAIN_END, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)

SEEDS = exp.SEEDS
BASE  = exp.FEATURES                 # mom_1..12 + pi_filter
SIZE  = exp.FEATURES + ['log_me']    # + market cap
_CRSP = os.path.join(_ROOT, 'data', 'crsp_msf_raw.parquet')


def add_log_me(df):
    """Merge lagged log market equity (log of prior-month me) -- no look-ahead."""
    raw = pd.read_parquet(_CRSP, columns=['permno', 'date', 'me', 'shrcd', 'exchcd', 'prc'])
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.sort_values(['permno', 'date'])
    raw = raw[raw['shrcd'].isin([10, 11]) & raw['exchcd'].isin([1, 2, 3]) & (raw['prc'].abs() > 1.0)]
    raw['log_me'] = np.log(raw.groupby('permno')['me'].shift(1).replace(0, np.nan))
    return df.merge(raw[['permno', 'date', 'log_me']], on=['permno', 'date'], how='left')


def train_predict(df, feats):
    tr = df[df['date'] < TRAIN_END]
    te = df[df['date'] >= TRAIN_END].copy()
    Xtr, ytr = tr[feats].values.astype(float), tr['ret_fwd'].values.astype(float)
    Xte = te[feats].values.astype(float)
    pred = np.zeros(len(Xte))
    for seed in SEEDS:
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(Xtr, ytr)
        pred += m.predict(Xte)
    te['score'] = pred / len(SEEDS)
    return te


def perf_row(r, pim, label):
    r = r.dropna(); p = pim.reindex(r.index)
    sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    ann, vol, sharpe, mdd = metrics(r)
    return {'strategy': label, 'mo': len(r), 'ann': ann, 'vol': vol, 'sharpe': sharpe,
            'mdd': mdd, 'calm': sh(r[p < 0.5]), 'panic': sh(r[p >= 0.5])}


def size_profile(test, score_col):
    """Per-month mean log_me + size percentile of long (>=P90) and short (<=P10) legs."""
    recs = []
    for date, grp in test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo_b, hi_b = nyse.quantile(0.10), nyse.quantile(0.90)
        if lo_b >= hi_b:
            continue
        g = grp.dropna(subset=['me']).copy()
        g['_pr'] = g['me'].rank(pct=True)                     # 1.0 = largest in universe
        longs = g[g[score_col] >= hi_b]
        shorts = g[g[score_col] <= lo_b]
        recs.append({'date': date,
                     'uni_logme':  np.log(g['me']).mean(),
                     'long_logme': np.log(longs['me']).mean(),
                     'short_logme': np.log(shorts['me']).mean(),
                     'long_pctile':  longs['_pr'].mean(),
                     'short_pctile': shorts['_pr'].mean(),
                     'uni_med_me':  g['me'].median(),
                     'long_med_me': longs['me'].median(),
                     'short_med_me': shorts['me'].median()})
    return pd.DataFrame(recs).set_index('date')


def main():
    df = add_log_me(exp.build_features())
    df = df.dropna(subset=['log_me']).reset_index(drop=True)   # same sample for both models
    print(f"Full universe (valid log_me): {len(df):,} rows | "
          f"test months {df[df['date'] >= TRAIN_END]['date'].nunique()}", flush=True)

    te_base = train_predict(df, BASE); print("  trained baseline", flush=True)
    te_size = train_predict(df, SIZE); print("  trained +log_me", flush=True)
    pim = te_base.drop_duplicates('date').set_index('date')['pi_filter']

    perf = [
        perf_row(long_short_port(te_base, 'score'),          pim, 'L/S  baseline (mom+pi)'),
        perf_row(long_short_port(te_size, 'score'),          pim, 'L/S  +log_me'),
        perf_row(lo.long_only_port(te_base, 'score'),        pim, 'LongOnly baseline (mom+pi)'),
        perf_row(lo.long_only_port(te_size, 'score'),        pim, 'LongOnly +log_me'),
    ]
    print("\n" + "=" * 84)
    print("  (A) DOES log_me HELP?  full universe, test 2011-2024")
    print("=" * 84)
    print(f"  {'Strategy':<28} {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'MDD':>8} {'Calm':>6} {'Panic':>6}")
    print("  " + "-" * 80)
    for r in perf:
        print(f"  {r['strategy']:<28} {r['mo']:>4} {r['ann']:>7.1%} {r['vol']:>6.1%} "
              f"{r['sharpe']:>7.2f} {r['mdd']:>7.1%} {r['calm']:>6.2f} {r['panic']:>6.2f}")
    print("=" * 84)

    # (B) Size profile of current (baseline) holdings
    sp = size_profile(te_base, 'score')
    sp['year'] = sp.index.year
    annual = sp.groupby('year')[['uni_logme', 'long_logme', 'short_logme',
                                 'long_pctile', 'short_pctile']].mean()
    print("\n" + "=" * 84)
    print("  (B) SIZE OF HOLDINGS (baseline strategy) -- avg log_me & size percentile per year")
    print("      pctile: 1.00 = largest in that month's universe. Longs = LO & L/S long leg.")
    print("=" * 84)
    print(f"  {'Year':<6} {'Uni logme':>10} {'Long logme':>11} {'Short logme':>12} "
          f"{'Long pctile':>12} {'Short pctile':>13}")
    print("  " + "-" * 76)
    for y, row in annual.iterrows():
        print(f"  {y:<6} {row['uni_logme']:>10.2f} {row['long_logme']:>11.2f} {row['short_logme']:>12.2f} "
              f"{row['long_pctile']:>11.0%} {row['short_pctile']:>12.0%}")
    ov = sp[['uni_logme', 'long_logme', 'short_logme', 'long_pctile', 'short_pctile']].mean()
    print("  " + "-" * 76)
    print(f"  {'ALL':<6} {ov['uni_logme']:>10.2f} {ov['long_logme']:>11.2f} {ov['short_logme']:>12.2f} "
          f"{ov['long_pctile']:>11.0%} {ov['short_pctile']:>12.0%}")
    print("=" * 84)

    pd.DataFrame(perf).to_csv(os.path.join(_ROOT, 'real_trade_analysis', 'results',
                              '2026-05-31-xgb-logme-feature_perf.csv'), index=False)
    annual.to_csv(os.path.join(_ROOT, 'real_trade_analysis', 'results', '2026-05-31-xgb-logme-size_by_year.csv'))
    print("  saved _perf.csv and _size_by_year.csv")


if __name__ == '__main__':
    main()
