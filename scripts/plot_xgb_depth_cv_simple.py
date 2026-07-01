"""
Cross-validation validation Sharpe by tree depth, depths 1 to 6, at the
production hyperparameters (lr=0.05, n=500). Produces the data the appendix
depth figure consumes (the figure itself is rendered by
scripts/plot_xgb_depth_cv.py).

Depths 3-6 are read from the existing CV run (results/cv/xgb_cv_results.csv,
the lr=0.05, n=500 configuration). Depths 1-2 were not in the original CV grid,
so they are computed here on the same five expanding-window folds and 5-seed
ensemble, using scripts/xgb_cv.py machinery.

Writes:
    results/cv/xgb_depth_cv_1to6.csv
"""
import os
import sys

import pandas as pd

ROOT = '/Users/giladgang/momentum_regime'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import LEARNING_RATE, N_ESTIMATORS, TRADING_FEE
from scripts.xgb_cv import load_panel, run_cell, FOLDS

LR, N = LEARNING_RATE, N_ESTIMATORS      # production: 0.05, 500
SEEDS = list(range(5))                    # match CV's 5-seed ensemble
N_JOBS = 6                                # leave 2 cores free

rows = {}  # depth -> (mean, std) of per-fold validation Sharpe

# ── depths 3-6 from the existing CV run, at the production (lr, n) config ──
cv = pd.read_csv('results/cv/xgb_cv_results.csv')
fixed = cv[(cv['learning_rate'] == LR) & (cv['n_estimators'] == N)]
for d in [3, 4, 5, 6]:
    s = fixed[fixed['max_depth'] == d]['val_sharpe']
    rows[d] = (s.mean(), s.std())
    print(f'  depth {d} (from CSV): {s.mean():+.3f} +/- {s.std():.3f}  (n_folds={len(s)})')

# ── depths 1-2: run now on the same folds ──
print('Loading panel for depths 1-2 ...', flush=True)
df = load_panel()
print(f'  panel: {len(df):,} rows', flush=True)
for d in [1, 2]:
    fold_sharpes = []
    for (fid, train_end, val_start, val_end) in FOLDS:
        train = df[df['date'] < train_end]
        val = df[(df['date'] >= val_start) & (df['date'] < val_end)]
        sh, nm = run_cell(train, val, max_depth=d, learning_rate=LR,
                          n_estimators=N, seeds=SEEDS, fee=TRADING_FEE,
                          xgb_n_jobs=N_JOBS)
        fold_sharpes.append(sh)
        print(f'  depth {d} fold {fid}: Sharpe={sh:+.3f}', flush=True)
    s = pd.Series(fold_sharpes)
    rows[d] = (s.mean(), s.std())
    print(f'  depth {d} (computed): {s.mean():+.3f} +/- {s.std():.3f}', flush=True)

# ── persist ──
depths = [1, 2, 3, 4, 5, 6]
means = [rows[d][0] for d in depths]
stds = [rows[d][1] for d in depths]
out = pd.DataFrame({'depth': depths, 'cv_val_sharpe': means, 'fold_std': stds})
out.to_csv('results/cv/xgb_depth_cv_1to6.csv', index=False)
print('Saved: results/cv/xgb_depth_cv_1to6.csv')

print('\nDepth 1-6 CV Sharpe:')
for d, m, s in zip(depths, means, stds):
    print(f'  depth {d}: {m:+.3f} +/- {s:.3f}')
