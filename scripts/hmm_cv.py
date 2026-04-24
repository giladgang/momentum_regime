"""
hmm_cv.py
=========
Expanding-window cross-validation of HMM feature-set selection on the
training period (1990--2010). Evaluates all DD-inclusive feature
combinations of size 1--4 on held-out *validation* folds before any
test-period (2011--2025) data is touched.

Purpose
-------
Replaces the thesis caveat in data_section.tex: "the HMM feature set was
selected with reference to test-period performance" with a proper CV
selection so the selected combination truly has no test-set information
in it.

Procedure (per fold × combination × HMM seed)
---------------------------------------------
  1. Build raw feature panel, slice into train and validation partitions.
  2. Z-score each candidate feature using TRAIN-PARTITION mean and std
     (no leakage from validation or test).
  3. Fit HMM (K=2) on training Z via existing fit_hmm routine.
  4. Forward-filter pi_filter on full panel (causal).
  5. Train a SEEDED ENSEMBLE of XGB regressors on training stocks +
     training pi_filter, mirroring production's §9 ensemble logic.
  6. Average ensemble predictions, form ONE long-short portfolio, compute
     ONE net-of-fee Sharpe on the validation window.
  7. Append one row per (combo, fold, hmm_seed) to the CSV.

Metric
------
Net-of-fee (config.TRADING_FEE, 10 bps) long-short Sharpe from an
ensemble-averaged XGB prediction, matching the production pipeline
(scripts/cross_sectional_model.py §9) but with fewer seeds per ensemble
for CV tractability.

Folds
-----
Same 5 expanding-window folds as scripts/xgb_cv.py.

Scope
-----
K=2 is held fixed. K selection CV (K=2 vs K=3,4,5) is out of scope for
this script; the K choice has an in-sample / interpretability defence in
the appendix and does not depend on held-out performance.

Feature candidates (9)
----------------------
DD, VOL, CS, LVIX, GDP_g, DISP, REL_N, TERM, SKEW.
DD is required in every combination (economically primary per the
thesis's existing justification).

  DD alone         : C(8,0) = 1
  DD + 1 of 8      : C(8,1) = 8
  DD + 2 of 8      : C(8,2) = 28
  DD + 3 of 8      : C(8,3) = 56
  Total            : 93 combinations

Checkpointing
-------------
Appends one row per (combo, fold, seed) cell to
results/hmm_cv_features.csv immediately on completion. On restart,
completed cells are skipped.

Usage
-----
    python scripts/hmm_cv.py --smoke          # ~5-10 min smoke test
    python scripts/hmm_cv.py                  # full run (~15-30 h)
    python scripts/hmm_cv.py --hmm-seeds 1    # override HMM seeds
    python scripts/hmm_cv.py --xgb-seeds 5    # override XGB seeds
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (MOM_FEATURES, SUBSAMPLE, COLSAMPLE, USE_FUNDAMENTALS,
                    TRADING_FEE, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE)
from src.utils import long_short_port, compute_sharpe
from historical_oos_production import fit_hmm  # K=2 Bayesian HMM

# Minimum validation months required to compute a meaningful Sharpe.
MIN_VAL_MONTHS = 12


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CANDIDATES = ['DD', 'VOL', 'CS', 'LVIX', 'GDP_g', 'DISP', 'REL_N', 'TERM', 'SKEW']
REQUIRED = 'DD'
MAX_FEATURES = 4

FOLDS = [
    # (fold_id, train_end,       val_start,      val_end)
    (1, '1997-01-01', '1997-01-01', '1999-01-01'),
    (2, '1999-01-01', '1999-01-01', '2001-01-01'),
    (3, '2001-01-01', '2001-01-01', '2003-01-01'),
    (4, '2003-01-01', '2003-01-01', '2005-01-01'),
    (5, '2005-01-01', '2005-01-01', '2007-01-01'),
]

# Crisis windows for HMM sign-correction (used by fit_hmm when enough
# months are present in the training partition)
CRISIS_WINDOWS = [
    ('1990-07-01', '1991-03-31'),  # early-90s recession
    ('1998-07-01', '1998-10-31'),  # LTCM
    ('2000-03-01', '2002-10-31'),  # dot-com
    ('2007-10-01', '2009-06-30'),  # GFC
    ('2020-02-01', '2020-04-30'),  # COVID
]


# ═══════════════════════════════════════════════════════════════════════════════
# Combinations
# ═══════════════════════════════════════════════════════════════════════════════

def enumerate_combinations():
    """Return list of tuples of feature names. DD is required."""
    others = [c for c in CANDIDATES if c != REQUIRED]
    combos = []
    for k in range(0, MAX_FEATURES):  # DD + 0, 1, 2, or 3 others
        for subset in itertools.combinations(others, k):
            combos.append((REQUIRED,) + subset)
    return combos


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_hmm_panel():
    """Load raw HMM feature panel (one row per month)."""
    panel = pd.read_parquet('data/panel_with_regimes.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    keep = ['date'] + CANDIDATES
    panel = panel[keep].dropna().sort_values('date').reset_index(drop=True)
    return panel


def load_stock_panel():
    """Load stock panel with momentum features and ret_fwd.
    Same construction as cross_sectional_model.py, but pi_filter is left
    empty — it will be filled per-fold from each fold's HMM fit."""
    stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

    stocks = stocks[stocks['shrcd'].isin([10, 11])]
    stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
    stocks = stocks[stocks['prc'].abs() > 1.0]
    stocks = stocks.reset_index(drop=True)

    stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
    stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)
    for lb in range(1, 13):
        roll = (stocks.groupby('permno', sort=False)['_log_ret_s1']
                      .rolling(lb, min_periods=lb).sum()
                      .reset_index(level='permno', drop=True).sort_index())
        stocks[f'mom_{lb}'] = np.expm1(roll)
    stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

    stocks['log_me'] = np.log(
        stocks.groupby('permno')['me'].transform(lambda x: x.shift(1))
              .replace(0, np.nan)
    )
    stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(
        lambda x: x.shift(-1)
    )
    core = MOM_FEATURES + (['log_me'] if USE_FUNDAMENTALS else [])
    stocks = stocks.dropna(subset=['ret_fwd'] + core).reset_index(drop=True)
    return stocks


# ═══════════════════════════════════════════════════════════════════════════════
# HMM fit + XGB eval for one CV cell
# ═══════════════════════════════════════════════════════════════════════════════

def zscore_train(panel, features, train_end):
    """Z-score `features` using statistics from the train partition only.
    Returns a DataFrame with <feature>_z columns alongside date."""
    train_mask = panel['date'] < pd.Timestamp(train_end)
    out = panel[['date']].copy()
    for f in features:
        mu = panel.loc[train_mask, f].mean()
        sd = panel.loc[train_mask, f].std()
        if sd == 0 or not np.isfinite(sd):
            sd = 1.0
        out[f + '_z'] = (panel[f] - mu) / sd
    return out


def crisis_mask_for_training(train_dates):
    """Boolean mask marking crisis months inside a training-date array."""
    dates = pd.to_datetime(train_dates).values
    mask = np.zeros(len(dates), dtype=bool)
    for start, end in CRISIS_WINDOWS:
        mask |= ((dates >= np.datetime64(start)) & (dates <= np.datetime64(end)))
    return mask


def fit_pi_filter(panel_z, features_z, train_end, seed, n_iter, n_burnin):
    """Fit HMM on training partition, forward-filter on full series.
    Returns DataFrame of (date, pi_filter)."""
    Z_full = panel_z[features_z].values.astype(float)
    train_mask = (panel_z['date'] < pd.Timestamp(train_end)).values
    Z_train = Z_full[train_mask]
    train_dates = panel_z.loc[train_mask, 'date'].values
    cmask = crisis_mask_for_training(train_dates)

    pi = fit_hmm(Z_train, Z_full, seed=seed, crisis_mask=cmask,
                 n_iter=n_iter, n_burnin=n_burnin)
    return pd.DataFrame({'date': panel_z['date'].values, 'pi_filter': pi})


def eval_xgb_ensemble(stocks, pi_df, train_end, val_start, val_end,
                      xgb_seeds, fee, n_estimators=None):
    """Merge pi_filter into stocks, train an ENSEMBLE of XGB regressors
    on training months, average their predictions, form ONE long-short
    portfolio, return (net-of-fee Sharpe, n_val_months).

    Mirrors production (scripts/cross_sectional_model.py §9): predictions
    are averaged BEFORE ranking stocks into deciles; one Sharpe per
    (combo, fold, hmm_seed) cell.
    """
    merged = stocks.merge(pi_df, on='date', how='left')
    merged['pi_filter'] = merged['pi_filter'].ffill()
    merged = merged.dropna(subset=['pi_filter']).reset_index(drop=True)

    feats = MOM_FEATURES + ['pi_filter']
    if USE_FUNDAMENTALS:
        from config import FUND_FEATURES
        feats = feats + FUND_FEATURES

    train = merged[merged['date'] < train_end]
    val = merged[(merged['date'] >= val_start) & (merged['date'] < val_end)]
    if len(train) == 0 or len(val) == 0:
        return np.nan, 0

    X_train = train[feats].values.astype(float)
    y_train = train['ret_fwd'].values.astype(float)
    X_val = val[feats].values.astype(float)

    preds = np.zeros(len(X_val))
    for seed in xgb_seeds:
        model = XGBRegressor(
            n_estimators=n_estimators or N_ESTIMATORS,
            max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
            subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE,
            tree_method='hist', random_state=seed, verbosity=0,
        )
        model.fit(X_train, y_train)
        preds += model.predict(X_val)
    preds /= len(xgb_seeds)

    val = val.copy()
    val['score'] = preds
    r_ls = long_short_port(val, 'score', fee=fee)
    return compute_sharpe(r_ls), len(r_ls)


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_PATH = 'results/hmm_cv_features.csv'
SMOKE_RESULTS_PATH = 'results/hmm_cv_smoke.csv'
WINNER_PATH = 'results/hmm_cv_winner.json'
TABLE_PATH = 'tables/table_hmm_cv.tex'

# One row per (combo, fold, hmm_seed). All XGB seeds are baked into the
# ensemble score recorded in that row, matching production semantics.
COLS = ['combo', 'n_features', 'fold', 'hmm_seed',
        'n_xgb_seeds', 'fee', 'val_sharpe', 'n_val_months',
        'hmm_elapsed_sec', 'xgb_elapsed_sec']


def combo_key(combo):
    return '+'.join(combo)


def completed_cells(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set((r.combo, int(r.fold), int(r.hmm_seed))
               for r in df.itertuples(index=False))


def append_row(path, row):
    write_header = not os.path.exists(path)
    pd.DataFrame([row], columns=COLS).to_csv(
        path, mode='a', header=write_header, index=False
    )


def emit_latex_table(summary, out_path, hmm_seeds, xgb_seeds, fee):
    """Write a thesis-ready LaTeX table of the top-15 feature combinations."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    top = summary.head(15)
    lines = [
        r'% Auto-generated by scripts/hmm_cv.py. Do not edit by hand.',
        r'\begin{table}[H]',
        r'\centering',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        (r'HMM feature set & \# features & mean val Sharpe & '
         r'fold-std & \# folds \\'),
        r'\midrule',
    ]
    for combo, row in top.iterrows():
        pretty = combo.replace('_', r'\_')
        lines.append(
            f'{pretty} & {int(row["n_features"])} & '
            f'{row["mean"]:+.3f} & {row["std"]:.3f} & {int(row["count"])} \\\\'
        )
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        (r'\caption{HMM feature-set selection via 5-fold expanding-window '
           r'cross-validation on the training period (1990--2010). Each '
           r'combination is the mean net-of-fee ('
         + f'{fee*10000:.0f} bps) long-short Sharpe across folds; each '
           r'fold refits the HMM on the fold-train partition with per-fold '
           r'z-scoring (no leakage), then forms portfolios via an ensemble '
           r'of '
         + f'{hmm_seeds} HMM seeds $\\times$ {xgb_seeds} XGBoost seeds, '
           r'matching the production pipeline at reduced seed budget for '
           r'CV tractability.}'),
        r'\label{tab:hmm_cv}',
        r'\end{table}',
    ]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def save_winner_json(summary, out_path, per_fold_for_winner, hmm_seeds,
                     xgb_seeds, fee):
    winner_combo = summary.index[0]
    winner = {
        'combo': winner_combo,
        'features': winner_combo.split('+'),
        'n_features': int(summary.iloc[0]['n_features']),
        'mean_val_sharpe': float(summary.iloc[0]['mean']),
        'fold_std_val_sharpe': float(summary.iloc[0]['std']),
        'n_folds': int(summary.iloc[0]['count']),
        'n_hmm_seeds': hmm_seeds,
        'n_xgb_seeds': xgb_seeds,
        'fee_decimal': fee,
        'per_fold_val_sharpe': per_fold_for_winner,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(winner, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny: 3 combos × 2 folds × 1 HMM seed × 1 XGB seed, '
                             'reduced HMM iterations')
    parser.add_argument('--hmm-seeds', type=int, default=3,
                        help='HMM Gibbs seeds per cell (default 3; production: 200)')
    parser.add_argument('--xgb-seeds', type=int, default=5,
                        help='XGB seeds per ensemble (default 5; production: 50)')
    parser.add_argument('--fee', type=float, default=TRADING_FEE,
                        help=f'Transaction fee (default from config: {TRADING_FEE})')
    parser.add_argument('--hmm-iter', type=int, default=2000)
    parser.add_argument('--hmm-burnin', type=int, default=500)
    parser.add_argument('--n-estimators', type=int, default=None,
                        help='Override XGB n_estimators (default: use config)')
    parser.add_argument('--output', default=None,
                        help='Results CSV path (append, supports resume). '
                             'Defaults: results/hmm_cv_features.csv for full '
                             'runs, results/hmm_cv_smoke.csv for --smoke. '
                             'Smoke NEVER overwrites full results.')
    args = parser.parse_args()

    # Safety: smoke runs must not overwrite full-run results.
    if args.output is None:
        args.output = SMOKE_RESULTS_PATH if args.smoke else RESULTS_PATH

    combos = enumerate_combinations()
    folds = FOLDS

    if args.smoke:
        combos = [('DD',), ('DD', 'CS'), ('DD', 'DISP', 'REL_N', 'CS')]
        folds = FOLDS[:2]
        args.hmm_seeds = 1
        args.xgb_seeds = 1
        args.hmm_iter = 500
        args.hmm_burnin = 100
        args.n_estimators = 200

    xgb_seed_list = list(range(args.xgb_seeds))

    total_cells = len(combos) * len(folds) * args.hmm_seeds
    total_hmm_fits = total_cells
    total_xgb_fits = total_cells * args.xgb_seeds
    print('=' * 70)
    print(f'  HMM feature-selection CV ({"SMOKE" if args.smoke else "FULL"})')
    print('=' * 70)
    print(f'  Combos:            {len(combos)}')
    print(f'  Folds:             {len(folds)}')
    print(f'  HMM seeds / cell:  {args.hmm_seeds}  (production: 200)')
    print(f'  XGB seeds / ens.:  {args.xgb_seeds}  (production: 50)')
    print(f'  Fee:               {args.fee}  ({args.fee*10000:.0f} bps)')
    print(f'  HMM iter / burnin: {args.hmm_iter} / {args.hmm_burnin}')
    print(f'  Cells:             {total_cells}  '
          f'(= {total_hmm_fits} HMM + {total_xgb_fits} XGB fits)')
    print(f'  Output:            {args.output}')
    print('=' * 70)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    done = completed_cells(args.output)
    print(f'  Resuming:   {len(done)} cells already done, '
          f'{total_cells - len(done)} remaining\n')

    print('  Loading HMM panel ...')
    panel = load_hmm_panel()
    print(f'  HMM panel: {len(panel)} months  '
          f'{panel["date"].min().date()} → {panel["date"].max().date()}')

    print('  Loading stock panel ...')
    stocks = load_stock_panel()
    print(f'  Stocks: {len(stocks):,} rows, {stocks["permno"].nunique():,} permnos\n')

    t_start = time.time()
    cell_idx = 0
    for combo in combos:
        feats_z = [f + '_z' for f in combo]

        for fold_id, train_end, val_start, val_end in folds:
            panel_z = zscore_train(panel, list(combo), train_end)

            # Validation-window size guard (same dates used by eval_xgb)
            val_months = (
                (panel_z['date'] >= pd.Timestamp(val_start))
                & (panel_z['date'] < pd.Timestamp(val_end))
            ).sum()
            if val_months < MIN_VAL_MONTHS:
                print(f'  [!] {combo_key(combo):<30s} fold={fold_id}  '
                      f'SKIPPED (only {val_months} val months)')
                continue

            for hmm_seed in range(args.hmm_seeds):
                cell_idx += 1
                key = (combo_key(combo), fold_id, hmm_seed)
                if key in done:
                    continue

                # Fit HMM once per (combo, fold, hmm_seed) — expensive
                t_hmm = time.time()
                try:
                    pi_df = fit_pi_filter(panel_z, feats_z, train_end,
                                          seed=hmm_seed,
                                          n_iter=args.hmm_iter,
                                          n_burnin=args.hmm_burnin)
                except Exception as e:
                    print(f'  [!] HMM fit failed for {combo_key(combo)} '
                          f'fold={fold_id} hmm_seed={hmm_seed}: {e}')
                    append_row(args.output, [
                        combo_key(combo), len(combo), fold_id, hmm_seed,
                        args.xgb_seeds, args.fee,
                        np.nan, 0, time.time() - t_hmm, 0.0,
                    ])
                    continue
                hmm_dt = time.time() - t_hmm

                # Ensemble XGB evaluation (one Sharpe, matches production)
                t_xgb = time.time()
                sharpe, n_months = eval_xgb_ensemble(
                    stocks, pi_df, train_end, val_start, val_end,
                    xgb_seeds=xgb_seed_list, fee=args.fee,
                    n_estimators=args.n_estimators,
                )
                xgb_dt = time.time() - t_xgb

                append_row(args.output, [
                    combo_key(combo), len(combo), fold_id, hmm_seed,
                    args.xgb_seeds, args.fee,
                    sharpe, n_months, hmm_dt, xgb_dt,
                ])

                total_elapsed = time.time() - t_start
                print(f'  [{cell_idx:5d}/{total_cells:5d}] '
                      f'{combo_key(combo):<30s} '
                      f'fold={fold_id} hmm={hmm_seed}  '
                      f'Sharpe={sharpe:+.3f}  '
                      f'(HMM {hmm_dt:.0f}s + XGB-ens {xgb_dt:.1f}s, '
                      f'{total_elapsed/60:.1f}m total)')

    print('\n' + '=' * 70)
    print('  DONE')
    print('=' * 70)

    # ── Summary: average across HMM seeds within each fold, then across
    # folds per combination. This matches what production reports: the
    # ensemble portfolio's Sharpe, stable across regime-signal sampling noise.
    results = pd.read_csv(args.output)
    per_fold = (results.groupby(['combo', 'fold'])['val_sharpe']
                       .mean().reset_index())
    summary = (per_fold.groupby('combo')['val_sharpe']
                       .agg(['mean', 'std', 'count'])
                       .sort_values('mean', ascending=False))
    summary['n_features'] = [c.count('+') + 1 for c in summary.index]

    print('\n  Top 15 combinations by mean validation Sharpe across folds:')
    print(summary.head(15).to_string())

    winner = summary.index[0]
    print(f'\n  Winner by mean Sharpe:              {winner}  '
          f'(mean={summary.iloc[0]["mean"]:+.3f}, '
          f'fold-std={summary.iloc[0]["std"]:.3f})')
    stable = summary.sort_values('std')
    print(f'  Winner by stability (low fold-std): {stable.index[0]}  '
          f'(mean={stable.iloc[0]["mean"]:+.3f}, '
          f'fold-std={stable.iloc[0]["std"]:.3f})')

    # Per-fold breakdown for the top combo
    winner_per_fold = (per_fold[per_fold['combo'] == winner]
                       .set_index('fold')['val_sharpe']
                       .round(3).to_dict())
    winner_per_fold_int = {int(k): float(v) for k, v in winner_per_fold.items()}
    print('\n  Winner per-fold Sharpes:')
    for fold, sh in sorted(winner_per_fold_int.items()):
        print(f'    fold {fold}: {sh:+.3f}')

    # ── Thesis-ready outputs ─────────────────────────────────────────────────
    if not args.smoke:
        emit_latex_table(summary, TABLE_PATH,
                         hmm_seeds=args.hmm_seeds, xgb_seeds=args.xgb_seeds,
                         fee=args.fee)
        save_winner_json(summary, WINNER_PATH, winner_per_fold_int,
                         hmm_seeds=args.hmm_seeds, xgb_seeds=args.xgb_seeds,
                         fee=args.fee)
        print(f'\n  Wrote: {TABLE_PATH}')
        print(f'  Wrote: {WINNER_PATH}')


if __name__ == '__main__':
    main()
