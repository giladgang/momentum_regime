"""
2026-05-30-xgb-russell1000.py
=============================
EXPLORATORY (experiments/ -- throwaway, no reproducibility guarantee).

Question: how does the headline XGB long-short momentum strategy behave if we
restrict it to "big companies" only -- a Russell-1000 proxy (the largest
TOP_N stocks by market equity each month)?

Design (approved 2026-05-30):
  * Define "big" = top TOP_N=1000 by market equity (me) each month. This is a
    Russell-1000 PROXY reconstructed from CRSP market caps -- the real Russell
    1000 reconstitutes annually in June; here we re-rank monthly, point-in-time,
    using only month-t caps (no look-ahead).
  * Filter scope = TRAIN + TRADE on big only (the XGB learns from and trades
    only large caps).
  * Isolation: this script imports only the SAFE pure helpers from src.utils
    and constants from config. It NEVER imports scripts/cross_sectional_model.py
    (that module executes on import and overwrites production artefacts). It
    writes only its own returns CSVs next to this file. Nothing in results/,
    tables/, artefacts/, or plots/ is touched.

For an apples-to-apples comparison we run BOTH universes under identical code:
  1. Full universe   -> should reproduce the logged production XGB (Sharpe ~1.11)
  2. Top-1000 (big)  -> the experiment

Production "before" (RESULTS_LOG.md, M2 XGB mom+pi, full universe, L/S, 167mo):
  Ann 21.9% | Vol 19.7% | Sharpe 1.11 | MDD -24.8% | Calm 0.82 | Panic 1.56
"""

import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Project root on path so config / src import cleanly regardless of cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from config import (TRAIN_END, XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH,
                    LEARNING_RATE, SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)
from src.utils import long_short_port, metrics

# ── Experiment knobs ──────────────────────────────────────────────────────────
TOP_N   = 1000              # Russell-1000 proxy: largest TOP_N stocks by me/month
N_SEEDS = len(XGB_SEEDS)    # 50, matching production
SEEDS   = XGB_SEEDS

MOM_LBS      = list(range(1, 13))
MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
FEATURES     = MOM_FEATURES + ['pi_filter']   # fundamentals off, == production baseline

_CRSP_PATH   = os.path.join(_ROOT, 'data', 'crsp_msf_raw.parquet')
_REGIME_PATH = os.path.join(_ROOT, 'data', 'panel_with_regimes.parquet')


# ── Data prep (mirrors scripts/cross_sectional_model.py, no look-ahead) ───────
def build_features():
    """Load CRSP, apply eligibility, compute momentum + ret_fwd, merge pi_filter.

    Returns a stock-month frame with FEATURES + ['ret_fwd','me','exchcd',
    'permno','date'], already dropna'd on the model's core columns. No size
    screen applied here -- that is done per-universe downstream.
    """
    stocks = pd.read_parquet(_CRSP_PATH)
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

    # Eligibility: ordinary common shares, major exchanges, share price > $1.
    stocks = stocks[stocks['shrcd'].isin([10, 11])]
    stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
    stocks = stocks[stocks['prc'].abs() > 1.0]
    stocks = stocks.reset_index(drop=True)

    # Regime signal: merge pi_filter on date, then group-wise ffill by permno
    # (prevents one permno's late pi_filter bleeding into the next permno).
    regimes = pd.read_parquet(_REGIME_PATH)[['date', 'pi_filter']]
    regimes['date'] = pd.to_datetime(regimes['date'])
    stocks = stocks.merge(regimes, on='date', how='left')
    stocks['pi_filter'] = stocks.groupby('permno')['pi_filter'].ffill()

    # Momentum at 1..12 month lookbacks via shifted cumulative log returns.
    stocks['_lr']    = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
    stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)   # shift -> no look-ahead
    for lb in MOM_LBS:
        roll = (stocks.groupby('permno', sort=False)['_lr_s1']
                .rolling(lb, min_periods=lb).sum()
                .reset_index(level='permno', drop=True).sort_index())
        stocks[f'mom_{lb}'] = np.expm1(roll)
    stocks.drop(columns=['_lr', '_lr_s1'], inplace=True)

    # Next-month return target.
    stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

    # Drop rows missing target or core features (NOT on me -- the portfolio
    # builder handles NaN me, matching production).
    df = stocks.dropna(subset=['ret_fwd'] + FEATURES).reset_index(drop=True)
    return df


def apply_size_screen(df, top_n):
    """Keep the largest top_n stocks by me each month. None -> full universe."""
    if top_n is None:
        return df
    sized = df[df['me'].notna()].copy()
    rank  = sized.groupby('date')['me'].rank(ascending=False, method='first')
    return sized[rank <= top_n].reset_index(drop=True)


# ── Model + evaluation ────────────────────────────────────────────────────────
def run_xgb_ls(df, label):
    """Train the 50-seed XGB ensemble, build the L/S portfolio, return metrics."""
    train = df[df['date'] < TRAIN_END].copy()
    test  = df[df['date'] >= TRAIN_END].copy()

    X_train = train[FEATURES].values.astype(float)
    y_train = train['ret_fwd'].values.astype(float)
    X_test  = test[FEATURES].values.astype(float)

    print(f"\n[{label}] train rows={len(train):,}  test rows={len(test):,}  "
          f"avg names/mo (test)={len(test) / test['date'].nunique():.0f}")

    preds = np.zeros(len(X_test))
    for i, seed in enumerate(SEEDS, 1):
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
        m.fit(X_train, y_train)
        preds += m.predict(X_test)
        if i % 10 == 0:
            print(f"    [{label}] seed {i}/{N_SEEDS}")
    preds /= N_SEEDS

    test = test.copy()
    test['score_xgb'] = preds
    r = long_short_port(test, 'score_xgb')

    # Regime-conditional Sharpe (pi >= 0.5 == panic), pi taken per month.
    pi_m  = test.drop_duplicates('date').set_index('date')['pi_filter']
    pim   = pi_m.reindex(r.index)
    calm  = r[pim < 0.5]
    panic = r[pim >= 0.5]
    sh = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan

    ann, vol, sharpe, mdd = metrics(r)
    return {
        'label': label, 'returns': r,
        'months': len(r), 'n_calm': int((pim < 0.5).sum()), 'n_panic': int((pim >= 0.5).sum()),
        'ann_ret': ann, 'vol': vol, 'sharpe': sharpe, 'mdd': mdd,
        'calm_sharpe': sh(calm), 'panic_sharpe': sh(panic),
    }


def main():
    df = build_features()
    print(f"Eligible stock-months after feature build: {len(df):,}  "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    universes = [
        ('Full universe',            None),
        (f'Top-{TOP_N} (big only)',  TOP_N),
    ]
    results = []
    for label, top_n in universes:
        sub = apply_size_screen(df, top_n)
        res = run_xgb_ls(sub, label)
        results.append(res)
        out = os.path.join(_ROOT, 'real_trade_analysis', 'results',
                           f"2026-05-30-xgb-russell1000_returns_{'full' if top_n is None else f'top{top_n}'}.csv")
        res['returns'].to_csv(out, header=['ret'])
        print(f"    saved {out}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  XGB LONG-SHORT (mom + pi):  FULL UNIVERSE  vs  TOP-1000 (BIG ONLY)")
    print("=" * 78)
    print(f"  {'Universe':<24} {'Mo':>4} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} "
          f"{'MDD':>8} {'CalmSh':>7} {'PanicSh':>8}")
    print("  " + "-" * 74)
    print(f"  {'PRODUCTION (logged)':<24} {167:>4} {0.219:>7.1%} {0.197:>6.1%} "
          f"{1.11:>7.2f} {-0.248:>7.1%} {0.82:>7.2f} {1.56:>8.2f}")
    for r in results:
        print(f"  {r['label']:<24} {r['months']:>4} {r['ann_ret']:>7.1%} {r['vol']:>6.1%} "
              f"{r['sharpe']:>7.2f} {r['mdd']:>7.1%} {r['calm_sharpe']:>7.2f} {r['panic_sharpe']:>8.2f}")
    print("=" * 78)
    print(f"  (calm n={results[0]['n_calm']}, panic n={results[0]['n_panic']} for full;  "
          f"big: calm n={results[1]['n_calm']}, panic n={results[1]['n_panic']})")


if __name__ == '__main__':
    main()
