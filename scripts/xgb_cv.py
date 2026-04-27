"""
xgb_cv.py
=========
Expanding-window cross-validation of XGBoost hyperparameters on the
training period (1990--2010). Selects tree depth, learning rate, and
number of estimators on held-out *validation* folds before any test-period
(2011--2025) data is touched.

Purpose
-------
Replaces the thesis's current "hyperparameters selected via cross-validation
on the training set" claim (which was previously not actually implemented)
with a real CV procedure. The winning configuration is then the one used
for the main-results pipeline; the 2011--2025 Sharpe is computed *once* on
that selection.

Folds
-----
5 expanding-window folds; 2007--2010 held as buffer; 2011--2025 untouched.

  Fold 1:  train 1990-1996  |  validate 1997-1998
  Fold 2:  train 1990-1998  |  validate 1999-2000
  Fold 3:  train 1990-2000  |  validate 2001-2002
  Fold 4:  train 1990-2002  |  validate 2003-2004
  Fold 5:  train 1990-2004  |  validate 2005-2006

Grid
----
max_depth       : 3, 4, 5, 6
learning_rate   : 0.01, 0.05, 0.10
n_estimators    : 200, 500, 1000

36 hyperparameter combinations, 5 folds, 5 seeds  =>  900 XGB fits.

Metric
------
Net-of-cost long-short Sharpe on each fold's validation window, computed
with the production trading fee (config.TRADING_FEE, 10 bps). Using the
production fee rather than fee=0 matters for hyperparameter ranking
because deeper trees / higher learning rates produce more volatile
predictions and therefore more turnover; a net-Sharpe ranking penalises
that correctly.

Assumptions
-----------
- pi_filter in the input data is the production HMM's filter, estimated
  on the full 1990-2010 training period. For the XGB hyperparameter
  selection we treat the regime signal as fixed; HMM feature selection
  is validated separately in scripts/hmm_cv.py with per-fold HMM refits.
- Per-fold refitting of the HMM for XGB CV would confound "is depth=4
  the right XGB hyperparameter?" with "does this HMM fit have enough
  data?". Fixing pi_filter isolates the XGB question.

Checkpointing
-------------
Appends one row per (combo, fold, seed) cell to results/xgb_cv_results.csv
immediately on completion. On restart, existing rows are skipped. The
final summary (winner + per-fold breakdown) is emitted in three formats:
  - results/xgb_cv_results.csv         (raw per-cell table)
  - results/xgb_cv_winner.json         (winner config for downstream use)
  - tables/table_xgb_cv.tex            (thesis-ready summary table)

Usage
-----
    python scripts/xgb_cv.py --smoke          # ~1 min smoke test
    python scripts/xgb_cv.py                  # full run (~2-5 h)
    python scripts/xgb_cv.py --seeds 3        # override seed count
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (CS_FEATURES, MOM_FEATURES, SUBSAMPLE, COLSAMPLE,
                    TRADING_FEE, USE_FUNDAMENTALS)
from src.utils import long_short_port, compute_sharpe

# Minimum validation months required for a fold to produce a meaningful
# Sharpe. 12 months = 1 year; Sharpe on <12 months is too noisy to rank.
MIN_VAL_MONTHS = 12


# ═══════════════════════════════════════════════════════════════════════════════
# Folds
# ═══════════════════════════════════════════════════════════════════════════════

FOLDS = [
    # (fold_id, train_end,       val_start,      val_end)
    (1, '1997-01-01', '1997-01-01', '1999-01-01'),
    (2, '1999-01-01', '1999-01-01', '2001-01-01'),
    (3, '2001-01-01', '2001-01-01', '2003-01-01'),
    (4, '2003-01-01', '2003-01-01', '2005-01-01'),
    (5, '2005-01-01', '2005-01-01', '2007-01-01'),
]

GRID_FULL = {
    'max_depth':     [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.10],
    'n_estimators':  [200, 500, 1000],
}
GRID_SMOKE = {
    'max_depth':     [3, 4],
    'learning_rate': [0.05],
    'n_estimators':  [200],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (mirrors cross_sectional_model.py feature build)
# ═══════════════════════════════════════════════════════════════════════════════

def load_panel():
    """Load + filter stock panel and merge pi_filter. Returns df with:
    date, permno, exchcd, me, ret_fwd, <CS_FEATURES>, ..."""
    stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

    stocks = stocks[stocks['shrcd'].isin([10, 11])]
    stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
    stocks = stocks[stocks['prc'].abs() > 1.0]
    stocks = stocks.reset_index(drop=True)

    regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'pi_filter']]
    regimes['date'] = pd.to_datetime(regimes['date'])
    stocks = stocks.merge(regimes, on='date', how='left')
    stocks['pi_filter'] = stocks['pi_filter'].ffill()

    # Momentum features via shifted log-returns
    stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
    stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)
    for lb in range(1, 13):
        roll = (stocks.groupby('permno', sort=False)['_log_ret_s1']
                      .rolling(lb, min_periods=lb).sum()
                      .reset_index(level='permno', drop=True).sort_index())
        stocks[f'mom_{lb}'] = np.expm1(roll)
    stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

    # log_me lagged
    stocks['log_me'] = np.log(
        stocks.groupby('permno')['me'].transform(lambda x: x.shift(1))
              .replace(0, np.nan)
    )

    # Forward return target
    stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(
        lambda x: x.shift(-1)
    )

    core = MOM_FEATURES + ['pi_filter'] + (['log_me'] if USE_FUNDAMENTALS else [])
    df = stocks.dropna(subset=['ret_fwd'] + core).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CV cell: one (combo, fold) evaluation using a seed ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
#
# This mirrors production (scripts/cross_sectional_model.py §8): train
# N_SEEDS XGBRegressors with different random_state, average their
# per-stock predictions into a single score, build ONE long-short
# portfolio from that score, and compute ONE Sharpe on validation. Taking
# the Sharpe of the ensemble portfolio (as production does) is NOT the
# same as averaging per-seed Sharpes — the ensemble portfolio has lower
# per-month ranking noise, giving a tighter metric.

def run_cell(train_df, val_df, *, max_depth, learning_rate, n_estimators,
             seeds, fee, xgb_n_jobs=None):
    """Train a SEEDED ENSEMBLE of XGB regressors on train_df, average
    their predictions, form one long-short portfolio, return
    (net-of-fee Sharpe, n_val_months).

    Parameters
    ----------
    seeds : list[int]
        XGB random_state values. Production uses range(1, 51); CV uses
        a shorter list to cut runtime.
    fee : float
        Transaction fee passed to long_short_port. Production value is
        config.TRADING_FEE (10 bps).
    xgb_n_jobs : int or None
        Pass 1 when called from a worker process to avoid nested-thread
        oversubscription; None lets XGBoost use all cores (default).
    """
    X_train = train_df[CS_FEATURES].values.astype(float)
    y_train = train_df['ret_fwd'].values.astype(float)
    X_val = val_df[CS_FEATURES].values.astype(float)

    xgb_kwargs = dict(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE, tree_method='hist', verbosity=0,
    )
    if xgb_n_jobs is not None:
        xgb_kwargs['n_jobs'] = xgb_n_jobs

    preds = np.zeros(len(X_val))
    for seed in seeds:
        model = XGBRegressor(random_state=seed, **xgb_kwargs)
        model.fit(X_train, y_train)
        preds += model.predict(X_val)
    preds /= len(seeds)

    val_df = val_df.copy()
    val_df['score'] = preds
    r_ls = long_short_port(val_df, 'score', fee=fee)
    return compute_sharpe(r_ls), len(r_ls)


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop with incremental checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_PATH = 'results/cv/xgb_cv_results.csv'
SMOKE_RESULTS_PATH = 'results/cv/xgb_cv_smoke.csv'
WINNER_PATH = 'results/cv/xgb_cv_winner.json'
TABLE_PATH = 'tables/table_xgb_cv.tex'

# One row per (depth, lr, n_estimators, fold). `n_seeds` records how many
# XGB seeds went into the ensemble; `fee` records the transaction cost
# used. These make the CSV self-describing.
COLS = ['max_depth', 'learning_rate', 'n_estimators', 'fold',
        'n_seeds', 'fee', 'val_sharpe', 'n_val_months', 'elapsed_sec']


def completed_cells(path):
    """Load existing CSV if any; return a set of (depth, lr, n, fold)
    tuples already computed."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(
        (int(r.max_depth), float(r.learning_rate), int(r.n_estimators),
         int(r.fold))
        for r in df.itertuples(index=False)
    )


def append_row(path, row):
    """Append a single row to the CSV, creating header if needed."""
    write_header = not os.path.exists(path)
    pd.DataFrame([row], columns=COLS).to_csv(
        path, mode='a', header=write_header, index=False
    )


def emit_latex_table(summary, out_path, n_seeds, fee):
    """Write a thesis-ready LaTeX table of the top-10 configurations."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    top = summary.head(10)
    lines = [
        r'% Auto-generated by scripts/xgb_cv.py. Do not edit by hand.',
        r'\begin{table}[H]',
        r'\centering',
        r'\begin{tabular}{r r r r r r}',
        r'\toprule',
        (r'depth & $\eta$ & $M$ & '
         r'mean val Sharpe & fold-std & \# folds \\'),
        r'\midrule',
    ]
    for (d, lr, n), row in top.iterrows():
        lines.append(
            f'{int(d)} & {lr:.2f} & {int(n)} & '
            f'{row["mean"]:+.3f} & {row["std"]:.3f} & {int(row["count"])} \\\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        (r'\caption{XGBoost hyperparameter selection via '
         + f'{len(summary.index.get_level_values("max_depth").unique())}-'
         + r'depth $\times$ 3-$\eta$ $\times$ 3-$M$ grid, evaluated over '
           r'5 expanding-window folds on the training period (1990--2010). '
         + f'Each cell is the net-of-fee ({fee*10000:.0f} bps) long-short '
           r'Sharpe from an ensemble of '
         + f'{n_seeds} XGBoost seeds, matching the production pipeline '
           r'with fewer seeds for CV tractability. The selected configuration '
           r'is the top row.}'),
        r'\label{tab:xgb_cv}',
        r'\end{table}',
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def save_winner_json(summary, out_path, per_fold_for_winner, fee, n_seeds):
    """Persist the winning config as JSON for downstream consumers."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    depth, lr, n_est = summary.index[0]
    winner = {
        'max_depth': int(depth),
        'learning_rate': float(lr),
        'n_estimators': int(n_est),
        'mean_val_sharpe': float(summary.iloc[0]['mean']),
        'fold_std_val_sharpe': float(summary.iloc[0]['std']),
        'n_folds': int(summary.iloc[0]['count']),
        'n_seeds_per_cell': n_seeds,
        'fee_decimal': fee,
        'per_fold_val_sharpe': per_fold_for_winner,
    }
    with open(out_path, 'w') as f:
        json.dump(winner, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Worker-process plumbing for `--workers > 1`
# ═══════════════════════════════════════════════════════════════════════════════
# Same template as hmm_cv.py: each worker loads the panel once at startup
# (initializer), then processes (depth, lr, n_estimators, fold) cells in
# parallel via Pool.imap_unordered. We pin threads to 1 inside workers so 6
# workers x XGB calls don't oversubscribe the 8 cores.

_PANEL_DF = None


def _init_worker_xgb():
    import os as _os
    _os.environ['OMP_NUM_THREADS'] = '1'
    _os.environ['MKL_NUM_THREADS'] = '1'
    _os.environ['OPENBLAS_NUM_THREADS'] = '1'
    global _PANEL_DF
    _PANEL_DF = load_panel()


def _execute_cell_xgb(df, payload, xgb_n_jobs=None):
    """Run one (depth, lr, n_estimators, fold) cell.

    Returns (status, row, msg) where status is 'ok'/'short_val'.
    """
    (max_depth, learning_rate, n_estimators, fold_id,
     train_end, val_start, val_end, seeds_list, fee, n_seeds) = payload

    train = df[df['date'] < train_end]
    val = df[(df['date'] >= val_start) & (df['date'] < val_end)]
    n_val_unique = val['date'].nunique()
    if n_val_unique < MIN_VAL_MONTHS:
        return ('short_val', None,
                f'only {n_val_unique} val months, need {MIN_VAL_MONTHS}')

    t0 = time.time()
    sharpe, n_months = run_cell(
        train, val,
        max_depth=max_depth, learning_rate=learning_rate,
        n_estimators=n_estimators, seeds=seeds_list, fee=fee,
        xgb_n_jobs=xgb_n_jobs,
    )
    dt = time.time() - t0

    row = [max_depth, learning_rate, n_estimators, fold_id,
           n_seeds, fee, sharpe, n_months, dt]
    return ('ok', row, None)


def _run_cell_xgb_worker(payload):
    global _PANEL_DF
    status, row, msg = _execute_cell_xgb(_PANEL_DF, payload, xgb_n_jobs=1)
    return (payload, status, row, msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny grid (2 depths × 2 folds × 1 seed) for end-to-end test')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of XGB seeds per ensemble cell (default 5; '
                             'production uses 50)')
    parser.add_argument('--fee', type=float, default=TRADING_FEE,
                        help='Transaction fee for L/S Sharpe (default from config: '
                             f'{TRADING_FEE})')
    parser.add_argument('--output', default=None,
                        help='Results CSV path (append, supports resume). '
                             'Defaults: results/xgb_cv_results.csv for full '
                             'runs, results/xgb_cv_smoke.csv for --smoke. '
                             'Smoke NEVER overwrites full results.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes (default 1 = serial; '
                             'recommended 6 on an 8-core machine).')
    args = parser.parse_args()

    # Safety: smoke runs must not overwrite full-run results.
    if args.output is None:
        args.output = SMOKE_RESULTS_PATH if args.smoke else RESULTS_PATH

    grid = GRID_SMOKE if args.smoke else GRID_FULL
    n_seeds = 1 if args.smoke else args.seeds
    folds = FOLDS[:2] if args.smoke else FOLDS
    seeds_list = list(range(n_seeds))

    print('=' * 70)
    print(f'  XGB cross-validation ({"SMOKE" if args.smoke else "FULL"})')
    print('=' * 70)
    print(f'  Grid:             {grid}')
    print(f'  Folds:            {len(folds)}')
    print(f'  Seeds / ensemble: {n_seeds}  (production: 50)')
    print(f'  Fee:              {args.fee}  ({args.fee*10000:.0f} bps)')
    total_cells = (len(grid['max_depth']) * len(grid['learning_rate'])
                   * len(grid['n_estimators']) * len(folds))
    total_fits = total_cells * n_seeds
    print(f'  Cells:            {total_cells}  (= {total_fits} XGB fits total)')
    print(f'  Output:           {args.output}')
    print('=' * 70)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    done = completed_cells(args.output)
    print(f'  Resuming:   {len(done)} cells already done, '
          f'{total_cells - len(done)} remaining\n')

    use_parallel = args.workers > 1
    if use_parallel:
        print(f'  Workers:          {args.workers}  (parallel)')
    else:
        print(f'  Workers:          1  (serial)')

    if not use_parallel:
        print('  Loading panel ...')
        df = load_panel()
        print(f'  Panel: {len(df):,} rows, {df["date"].min().date()} '
              f'→ {df["date"].max().date()}\n')

    # Build flat list of work units (cells), filtering against `done`.
    work_units = []
    for max_depth in grid['max_depth']:
        for learning_rate in grid['learning_rate']:
            for n_estimators in grid['n_estimators']:
                for fold_id, train_end, val_start, val_end in folds:
                    key = (max_depth, learning_rate, n_estimators, fold_id)
                    if key in done:
                        continue
                    work_units.append((
                        max_depth, learning_rate, n_estimators, fold_id,
                        train_end, val_start, val_end,
                        seeds_list, args.fee, n_seeds,
                    ))

    if not work_units:
        print('  All cells already complete; nothing to run.\n')

    t_start = time.time()
    n_done = 0
    n_short_val = 0

    def _record(payload, status, row, msg):
        nonlocal n_done, n_short_val
        max_depth, learning_rate, n_estimators, fold_id = payload[:4]
        if status == 'short_val':
            n_short_val += 1
            print(f'  [!] d={max_depth} lr={learning_rate} '
                  f'n={n_estimators} fold={fold_id}  SKIPPED ({msg})')
            return
        if row is not None:
            append_row(args.output, row)
        if status == 'ok':
            n_done += 1
            sharpe = row[6]
            dt = row[8]
            elapsed = time.time() - t_start
            total_progress = n_done + n_short_val
            print(f'  [{total_progress:4d}/{len(work_units):4d}] '
                  f'd={max_depth} lr={learning_rate} n={n_estimators} '
                  f'fold={fold_id}  Sharpe={sharpe:+.3f}  '
                  f'({dt:.1f}s, {elapsed/60:.1f}m total)')

    if use_parallel:
        from multiprocessing import get_context
        ctx = get_context('spawn')
        with ctx.Pool(processes=args.workers,
                      initializer=_init_worker_xgb) as pool:
            for payload, status, row, msg in pool.imap_unordered(
                    _run_cell_xgb_worker, work_units, chunksize=1):
                _record(payload, status, row, msg)
    else:
        for payload in work_units:
            status, row, msg = _execute_cell_xgb(df, payload, xgb_n_jobs=None)
            _record(payload, status, row, msg)

    print('\n' + '=' * 70)
    print('  DONE')
    print('=' * 70)
    print(f'  ok:           {n_done}')
    print(f'  short_val:    {n_short_val}')

    # ── Summary: aggregate across folds per hyperparameter combo ─────────────
    if not os.path.exists(args.output):
        print('\n  No rows written; skipping summary.')
        return
    results = pd.read_csv(args.output)
    if results.empty or 'val_sharpe' not in results.columns:
        print('\n  Empty results CSV; skipping summary.')
        return
    summary = (results.groupby(['max_depth', 'learning_rate', 'n_estimators'])
                      ['val_sharpe'].agg(['mean', 'std', 'count'])
                      .sort_values('mean', ascending=False))

    print('\n  Top 10 by mean validation Sharpe (net of fees):')
    print(summary.head(10).to_string())

    winner = summary.index[0]
    print(f'\n  Winner: depth={winner[0]}, lr={winner[1]}, n_estimators={winner[2]}')
    print(f'          mean Sharpe = {summary.iloc[0]["mean"]:+.3f} '
          f'(fold-std {summary.iloc[0]["std"]:.3f})')

    # Per-fold breakdown for the winner — useful for a defence Q&A.
    winner_rows = results[(results['max_depth'] == winner[0])
                          & (results['learning_rate'] == winner[1])
                          & (results['n_estimators'] == winner[2])]
    per_fold = (winner_rows.set_index('fold')['val_sharpe']
                           .round(3).to_dict())
    per_fold_int = {int(k): float(v) for k, v in per_fold.items()}
    print('\n  Winner per-fold Sharpes:')
    for fold, sh in sorted(per_fold_int.items()):
        print(f'    fold {fold}: {sh:+.3f}')

    # ── Thesis-ready outputs ─────────────────────────────────────────────────
    if not args.smoke:
        emit_latex_table(summary, TABLE_PATH, n_seeds=n_seeds, fee=args.fee)
        save_winner_json(summary, WINNER_PATH, per_fold_int,
                         fee=args.fee, n_seeds=n_seeds)
        print(f'\n  Wrote: {TABLE_PATH}')
        print(f'  Wrote: {WINNER_PATH}')


if __name__ == '__main__':
    main()
