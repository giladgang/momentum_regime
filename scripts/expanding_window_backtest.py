"""
expanding_window_backtest.py
============================
30-year expanding-window out-of-sample backtest (Section 5.4.3 historical
stress evidence). Retrain the full HMM + XGBoost pipeline at the start of
each year using only data available through the prior year-end, then use
the resulting model to predict the 12 months of the new year. Stitch the
resulting monthly OOS L/S returns into a single long time series spanning
1995-2024.

NOT TO BE CONFUSED WITH scripts/expanding_window.py, which is a different
analysis (tri-annual HMM re-estimation with fixed 2011-2025 test period
for the appendix robustness check app:expanding). This script produces a
true 30-year out-of-sample backtest including dot-com and GFC.

For production use the parallel variant: scripts/expanding_window_backtest_parallel.py
gives bit-identical output with ~6x speedup. See tests/test_parallel_hmm_determinism.py
for the bit-exactness check.

Every prediction is strictly causal: the model used for month t was fitted only
on data through the year preceding t. No look-ahead, no training-test overlap.

Usage (smoke test, ~90 min at reduced seeds):
    python scripts/expanding_window_backtest.py \
        --first-retrain-year 1995 --last-retrain-year 2024 \
        --hmm-seeds 20 --xgb-seeds 10 --tag smoke

Usage (production, ~17 hrs):
    python scripts/expanding_window_backtest.py \
        --first-retrain-year 1995 --last-retrain-year 2024 \
        --hmm-seeds 200 --xgb-seeds 50 --tag prod

Outputs (per tag):
    results/thesis/expanding_returns_<tag>.csv       -- monthly OOS L/S returns
    results/thesis/expanding_pi_filter_<tag>.csv     -- monthly averaged pi_filter
    results/thesis/expanding_summary_<tag>.csv       -- per-retraining diagnostics
"""

import argparse
import numpy as np
import pandas as pd
import warnings
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor

from config import (HMM_FEATURES, HMM_ITERATIONS, HMM_BURNIN, N_ESTIMATORS,
                    MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE, MOM_FEATURES,
                    RESULTS_DIR, RESULTS_THESIS_DIR)

from historical_oos_production import fit_hmm, build_ls  # reuse tested code

warnings.filterwarnings('ignore')

FEE = 0.001


def crisis_mask_for_training(train_dates):
    """Return a boolean mask over train_dates flagging known crisis windows
    that fall inside the training range. Used by fit_hmm for sign correction
    (though fit_hmm now falls back to theoretical signs when the mask is small).
    """
    windows = [
        ('1990-07-01', '1991-03-31'),   # early 90s recession
        ('1998-07-01', '1998-10-31'),   # LTCM/Russia
        ('2000-03-01', '2002-10-31'),   # dot-com bust
        ('2007-10-01', '2009-06-30'),   # GFC
        ('2020-02-01', '2020-04-30'),   # COVID crash
    ]
    td = pd.to_datetime(train_dates).values
    mask = np.zeros(len(td), dtype=bool)
    for s, e in windows:
        mask |= ((td >= np.datetime64(s)) & (td <= np.datetime64(e)))
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first-retrain-year', type=int, default=1995,
                    help='First year to predict; model fitted on data up to its start')
    ap.add_argument('--last-retrain-year', type=int, default=2024,
                    help='Last year to predict')
    ap.add_argument('--hmm-seeds', type=int, default=200)
    ap.add_argument('--xgb-seeds', type=int, default=50)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    HMM_SEEDS = list(range(1, args.hmm_seeds + 1))
    XGB_SEEDS = list(range(1, args.xgb_seeds + 1))
    TAG = args.tag
    FEATURES = MOM_FEATURES + ['pi_filter']

    os.makedirs(RESULTS_THESIS_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  EXPANDING-WINDOW BACKTEST: tag={TAG}")
    print(f"  First retrain year: {args.first_retrain_year}")
    print(f"  Last retrain year:  {args.last_retrain_year}")
    print(f"  HMM seeds: {len(HMM_SEEDS)}, XGB seeds: {len(XGB_SEEDS)}")
    print("=" * 70)

    # Load data ONCE (shared across all retrainings)
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

    # Output paths (per tag) — thesis-canonical layout post-restructure
    ret_path = os.path.join(RESULTS_THESIS_DIR, f'expanding_returns_{TAG}.csv')
    pi_path = os.path.join(RESULTS_THESIS_DIR, f'expanding_pi_filter_{TAG}.csv')
    log_path = os.path.join(RESULTS_THESIS_DIR, f'expanding_summary_{TAG}.csv')

    # Resume support: a year is considered complete only if it has a row in
    # the summary log. On restart we skip those years and append fresh rows
    # for the rest. We also trim the returns and pi CSVs to only the
    # completed years to avoid duplicate rows from partial prior writes.
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

        # Trim returns CSV to only completed years
        if os.path.exists(ret_path):
            df_r = pd.read_csv(ret_path, parse_dates=['date'])
            df_r = df_r[df_r['date'].dt.year.isin(completed_years)]
            df_r.to_csv(ret_path, index=False)
        # Trim pi CSV similarly
        if os.path.exists(pi_path):
            df_p = pd.read_csv(pi_path, parse_dates=['date'])
            df_p = df_p[df_p['date'].dt.year.isin(completed_years)]
            df_p.to_csv(pi_path, index=False)

    t_total = time.time()
    per_year_log_in_memory = []  # for final aggregate summary print

    for predict_year in range(args.first_retrain_year, args.last_retrain_year + 1):
        if predict_year in completed_years:
            print(f"\n[{predict_year}] SKIP (already in summary log)")
            continue

        t0 = time.time()
        train_end = pd.Timestamp(f'{predict_year}-01-01')
        test_end = pd.Timestamp(f'{predict_year + 1}-01-01')

        # ── HMM ──
        train_date_mask = sub['date'] < train_end
        Z_train = sub.loc[train_date_mask, HMM_FEATURES].values.astype(float)
        if len(Z_train) < 24:
            print(f"\n  SKIP {predict_year}: only {len(Z_train)} training months")
            continue

        train_dates = sub.loc[train_date_mask, 'date']
        crisis_mask = crisis_mask_for_training(train_dates.values)

        print(f"\n[{predict_year}] Train: {len(Z_train)} months "
              f"(crisis mask: {crisis_mask.sum()}) | HMM seeds = {len(HMM_SEEDS)}")

        pi_all = np.zeros(len(Z_full))
        for i, hs in enumerate(HMM_SEEDS, 1):
            pi_all += fit_hmm(Z_train, Z_full, seed=hs, crisis_mask=crisis_mask)
        pi_all /= len(HMM_SEEDS)

        # ── Build pi-augmented stock panel ──
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

        # ── L/S portfolio for this year ──
        r_year = build_ls(test_xgb, 'score')
        if len(r_year) == 0:
            print(f"  {predict_year}: no portfolio months produced")
            continue

        # Build per-year output dataframes
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
        per_year_log_in_memory.append(log_row)

        # Incremental save: returns -> pi -> log (log last; its presence marks
        # the year as complete for the resume logic).
        ret_needs_header = not os.path.exists(ret_path) or os.path.getsize(ret_path) == 0
        year_ret_df.to_csv(ret_path, mode='a', header=ret_needs_header, index=False)

        pi_needs_header = not os.path.exists(pi_path) or os.path.getsize(pi_path) == 0
        year_pi_df.to_csv(pi_path, mode='a', header=pi_needs_header, index=False)

        log_needs_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        pd.DataFrame([log_row]).to_csv(log_path, mode='a',
                                       header=log_needs_header, index=False)

        total_elapsed = (time.time() - t_total) / 60
        years_done_this_session = (predict_year - args.first_retrain_year + 1) \
                                  - len(completed_years.intersection(
                                      range(args.first_retrain_year, predict_year + 1)))
        years_left = args.last_retrain_year - predict_year
        eta_min = (total_elapsed / max(years_done_this_session, 1)) * years_left \
                   if years_done_this_session > 0 else 0
        print(f"  {predict_year} done in {elapsed:.0f}s | "
              f"Sharpe {log_row['ann_sharpe_year']:+.2f} | "
              f"total {total_elapsed:.1f} min | ETA {eta_min:.1f} min | saved")

    # ── Final summary: read back the on-disk returns (covers both resumed
    # and newly-computed years) and report aggregate stats.
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
