"""
expanding_window_backtest_parallel.py
=====================================
Production script for the 30-year expanding-window OOS backtest (Section 5.4.3).
Parallel-HMM variant of expanding_window_backtest.py. Same outputs, same
file format, same resume logic — only difference is the HMM seed loop runs
across multiple CPU cores via multiprocessing.Pool instead of one Python
serial loop. ~6x faster on an 8-core machine; bit-identical to serial
(verified by tests/test_parallel_hmm_determinism.py).

This is the script that produced results/expanding_returns_prod.csv,
results/expanding_pi_filter_prod.csv, and results/expanding_summary_prod.csv,
which feed Section 5.4.3 (Historical Stress: Out-of-Sample Evidence) and
Appendix app:expanding_window.

NOT TO BE CONFUSED WITH scripts/expanding_window.py, which is the older
tri-annual HMM re-estimation analysis (test period fixed at 2011-2025)
feeding app:expanding in the appendix. Different test, different outputs.

DETERMINISM CHECK FIRST. Before relying on this for a production run that
mixes serial and parallel results in the same output file, run:
    python -m pytest tests/test_parallel_hmm_determinism.py
to confirm that parallel `fit_hmm` returns the same pi_filter as the serial
version within numerical tolerance.

Usage:
    python scripts/expanding_window_backtest_parallel.py \\
        --first-retrain-year 2010 --last-retrain-year 2024 \\
        --hmm-seeds 200 --xgb-seeds 50 --tag prod \\
        --workers 8

The --tag is used identically to the serial script. Resume support: if a
year already has a row in expanding_summary_<tag>.csv it is skipped, so
this can be invoked with the same --tag as a stopped serial run to
continue from where it left off.
"""

import argparse
import numpy as np
import pandas as pd
import warnings
import time
import sys
import os
from functools import partial
from multiprocessing import Pool, cpu_count

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.chdir(_ROOT)

from xgboost import XGBRegressor

from config import (HMM_FEATURES, HMM_ITERATIONS, HMM_BURNIN, N_ESTIMATORS,
                    MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE, MOM_FEATURES,
                    RESULTS_DIR, RESULTS_THESIS_DIR)

from historical_oos_production import fit_hmm  # reuse tested code
from config import TRADING_FEE as FEE

def build_lo(df_test, score_col):
    """Long-only top-decile (NYSE P90), value-weighted; mirrors build_ls long leg (no short)."""
    prev_lw = {}; monthly = []
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi].dropna(subset=['me'])
        if longs['me'].sum() == 0: continue
        lw = (longs.set_index('permno')['me'] / longs['me'].sum()).to_dict()
        tl = sum(abs(lw.get(p,0)-prev_lw.get(p,0)) for p in set(lw)|set(prev_lw))/2
        r_l = (longs['ret_fwd']*longs['me']).sum()/longs['me'].sum()
        monthly.append({'date': date, 'ret': r_l - FEE*tl}); prev_lw = lw
    return pd.DataFrame(monthly).set_index('date')['ret'] if monthly else pd.Series(dtype=float)

warnings.filterwarnings('ignore')

FEE = 0.001


def crisis_mask_for_training(train_dates):
    """Same windows as the serial version."""
    windows = [
        ('1990-07-01', '1991-03-31'),
        ('1998-07-01', '1998-10-31'),
        ('2000-03-01', '2002-10-31'),
        ('2007-10-01', '2009-06-30'),
        ('2020-02-01', '2020-04-30'),
    ]
    td = pd.to_datetime(train_dates).values
    mask = np.zeros(len(td), dtype=bool)
    for s, e in windows:
        mask |= ((td >= np.datetime64(s)) & (td <= np.datetime64(e)))
    return mask


def _fit_one_seed(seed, Z_train, Z_full, crisis_mask):
    """Top-level worker function (must be importable for spawn-based pools)."""
    return fit_hmm(Z_train, Z_full, seed=seed, crisis_mask=crisis_mask)


def fit_hmm_parallel(Z_train, Z_full, seeds, crisis_mask, n_workers):
    """Run fit_hmm across `seeds` using a Pool of `n_workers`. Returns the
    averaged pi_filter (same shape as a single fit_hmm call's output)."""
    f = partial(_fit_one_seed, Z_train=Z_train, Z_full=Z_full,
                crisis_mask=crisis_mask)
    with Pool(processes=n_workers) as pool:
        pi_per_seed = pool.map(f, seeds)
    return np.mean(pi_per_seed, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first-retrain-year', type=int, default=1995)
    ap.add_argument('--last-retrain-year', type=int, default=2024)
    ap.add_argument('--hmm-seeds', type=int, default=200)
    ap.add_argument('--xgb-seeds', type=int, default=50)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--workers', type=int, default=max(cpu_count() - 1, 1),
                    help='Number of HMM worker processes (default: cpu_count - 1)')
    args = ap.parse_args()

    HMM_SEEDS = list(range(1, args.hmm_seeds + 1))
    XGB_SEEDS = list(range(1, args.xgb_seeds + 1))
    TAG = args.tag
    FEATURES = MOM_FEATURES + ['pi_filter']
    N_WORKERS = max(1, min(args.workers, cpu_count()))

    os.makedirs(RESULTS_THESIS_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  EXPANDING-WINDOW BACKTEST (PARALLEL HMM): tag={TAG}")
    print(f"  First retrain year: {args.first_retrain_year}")
    print(f"  Last retrain year:  {args.last_retrain_year}")
    print(f"  HMM seeds: {len(HMM_SEEDS)}, XGB seeds: {len(XGB_SEEDS)}")
    print(f"  HMM workers: {N_WORKERS} (of {cpu_count()} cores)")
    print("=" * 70)

    # Load data
    print("\n[setup] Loading data ...")
    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    sub = (panel[['date'] + HMM_FEATURES]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date'))
    sub = sub[sub['date'] >= '1990-01-01'].reset_index(drop=True)

    stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks = stocks[
        stocks['shrcd'].isin([10, 11])
        & stocks['exchcd'].isin([1, 2, 3])
        & (stocks['prc'].abs() > 1)
    ].sort_values(['permno', 'date']).reset_index(drop=True)
    stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
    stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)
    for lb in range(1, 13):
        rs = (stocks.groupby('permno', sort=False)['_lr_s1']
              .rolling(lb, min_periods=lb)
              .sum()
              .reset_index(level='permno', drop=True)
              .sort_index())
        stocks[f'mom_{lb}'] = np.expm1(rs)
    stocks.drop(columns=['_lr', '_lr_s1'], inplace=True)
    stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

    Z_full = sub[HMM_FEATURES].values.astype(float)
    print(f"  Feature panel: {len(sub)} months, stocks: {len(stocks):,} rows")

    # Output paths (thesis-canonical layout post-restructure)
    ret_path = os.path.join(RESULTS_THESIS_DIR, f'expanding_returns_{TAG}.csv')
    pi_path = os.path.join(RESULTS_THESIS_DIR, f'expanding_pi_filter_{TAG}.csv')
    log_path = os.path.join(RESULTS_THESIS_DIR, f'expanding_summary_{TAG}.csv')

    # Resume support
    completed_years = set()
    if os.path.exists(log_path):
        try:
            existing_log = pd.read_csv(log_path)
            completed_years = set(int(y) for y in existing_log['predict_year'].values)
        except Exception as e:
            print(f"  Could not parse existing log ({e}); starting fresh.")
            completed_years = set()
    if completed_years:
        print(f"  Resume: {len(completed_years)} year(s) already complete: "
              f"{sorted(completed_years)}")
        if os.path.exists(ret_path):
            df_r = pd.read_csv(ret_path, parse_dates=['date'])
            df_r = df_r[df_r['date'].dt.year.isin(completed_years)]
            df_r.to_csv(ret_path, index=False)
        if os.path.exists(pi_path):
            df_p = pd.read_csv(pi_path, parse_dates=['date'])
            df_p = df_p[df_p['date'].dt.year.isin(completed_years)]
            df_p.to_csv(pi_path, index=False)

    t_total = time.time()

    for predict_year in range(args.first_retrain_year, args.last_retrain_year + 1):
        if predict_year in completed_years:
            print(f"\n[{predict_year}] SKIP (already in summary log)")
            continue

        t0 = time.time()
        train_end = pd.Timestamp(f'{predict_year}-01-01')
        test_end = pd.Timestamp(f'{predict_year + 1}-01-01')

        train_date_mask = sub['date'] < train_end
        Z_train = sub.loc[train_date_mask, HMM_FEATURES].values.astype(float)
        if len(Z_train) < 24:
            print(f"\n  SKIP {predict_year}: only {len(Z_train)} training months")
            continue

        train_dates = sub.loc[train_date_mask, 'date']
        crisis_mask = crisis_mask_for_training(train_dates.values)

        print(f"\n[{predict_year}] Train: {len(Z_train)} months "
              f"(crisis mask: {crisis_mask.sum()}) | "
              f"HMM seeds = {len(HMM_SEEDS)} | workers = {N_WORKERS}")

        # ── Parallel HMM: fan out seeds across worker processes ──
        pi_all = fit_hmm_parallel(Z_train, Z_full, HMM_SEEDS,
                                  crisis_mask, N_WORKERS)

        # ── XGB ensemble (kept serial; xgboost itself is multi-threaded) ──
        pi_df = pd.DataFrame({'date': sub['date'], 'pi_filter': pi_all})
        stocks_w = stocks.merge(pi_df, on='date', how='left')
        stocks_w['pi_filter'] = stocks_w['pi_filter'].ffill()

        df = stocks_w.dropna(subset=['ret_fwd'] + FEATURES).copy()
        train_xgb = df[df['date'] < train_end]
        test_xgb = df[(df['date'] >= train_end) & (df['date'] < test_end)].copy()

        if len(train_xgb) == 0 or len(test_xgb) == 0:
            print(f"  SKIP {predict_year}: empty train or test")
            continue

        X_tr = train_xgb[FEATURES].values.astype(float)
        X_te = test_xgb[FEATURES].values.astype(float)
        y_tr = train_xgb['ret_fwd'].values.astype(float)

        mask_tr = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        mask_te = ~np.isnan(X_te).any(axis=1)
        X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
        X_te = X_te[mask_te]
        test_xgb = test_xgb[mask_te].copy()

        preds = np.zeros(len(X_te))
        for xs in XGB_SEEDS:
            xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                               learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                               colsample_bytree=COLSAMPLE, tree_method='hist',
                               random_state=xs, verbosity=0)
            xgb.fit(X_tr, y_tr)
            preds += xgb.predict(X_te)
        preds /= len(XGB_SEEDS)
        test_xgb['score'] = preds

        r_year = build_lo(test_xgb, 'score')
        if len(r_year) == 0:
            print(f"  {predict_year}: no portfolio months produced")
            continue

        year_ret_df = pd.DataFrame([{'date': d, 'ret': v} for d, v in r_year.items()])
        pred_mask = (sub['date'] >= train_end) & (sub['date'] < test_end)
        year_pi_df = pd.DataFrame({
            'date': sub.loc[pred_mask, 'date'].values,
            'pi_filter': pi_all[pred_mask.values],
        })

        elapsed = time.time() - t0
        log_row = {
            'predict_year': predict_year,
            'train_months': len(Z_train),
            'crisis_months_in_train': int(crisis_mask.sum()),
            'xgb_train_rows': len(X_tr),
            'xgb_test_rows': len(X_te),
            'n_portfolio_months': len(r_year),
            'mean_pi_predicted_year': float(pi_all[pred_mask.values].mean()),
            'ann_sharpe_year': (r_year.mean() / r_year.std() * np.sqrt(12))
                                if len(r_year) > 1 and r_year.std() > 0 else np.nan,
            'elapsed_s': elapsed,
        }

        # Incremental save: returns -> pi -> log
        ret_needs_header = not os.path.exists(ret_path) or os.path.getsize(ret_path) == 0
        year_ret_df.to_csv(ret_path, mode='a', header=ret_needs_header, index=False)

        pi_needs_header = not os.path.exists(pi_path) or os.path.getsize(pi_path) == 0
        year_pi_df.to_csv(pi_path, mode='a', header=pi_needs_header, index=False)

        log_needs_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        pd.DataFrame([log_row]).to_csv(log_path, mode='a',
                                       header=log_needs_header, index=False)

        total_elapsed = (time.time() - t_total) / 60
        years_left = args.last_retrain_year - predict_year
        print(f"  {predict_year} done in {elapsed:.0f}s | "
              f"Sharpe {log_row['ann_sharpe_year']:+.2f} | "
              f"total {total_elapsed:.1f} min | saved")

    print("\n" + "=" * 70)
    if os.path.exists(ret_path):
        out_ret = pd.read_csv(ret_path, parse_dates=['date']).set_index('date').sort_index()
        if len(out_ret) > 0:
            print(f"  COMPLETE: {len(out_ret)} OOS months from "
                  f"{out_ret.index.min().date()} to {out_ret.index.max().date()}")
            if len(out_ret) > 1 and out_ret['ret'].std() > 0:
                full_sharpe = out_ret['ret'].mean() / out_ret['ret'].std() * np.sqrt(12)
                full_cum = (1 + out_ret['ret']).prod() - 1
                cum_ser = (1 + out_ret['ret']).cumprod()
                mdd = ((cum_ser - cum_ser.cummax()) / cum_ser.cummax()).min()
                print(f"  Full OOS Sharpe: {full_sharpe:.2f}")
                print(f"  Cumulative:      {full_cum * 100:+.1f}%")
                print(f"  Max DD:          {mdd * 100:+.1f}%")
    print(f"  Total time this session: {(time.time() - t_total) / 60:.1f} min")
    print("=" * 70)
    print(f"\nSaved:\n  {ret_path}\n  {pi_path}\n  {log_path}")


if __name__ == '__main__':
    main()
