"""
2026-05-31-reconcile-retrain.py
===============================
DIAGNOSTIC (experiments/). Confirm the 1.08 vs 1.107 gap is XGB scoring
non-determinism, not a pipeline error. The sample is already proven identical
(reconcile-production.py). Here we retrain the 50-seed ensemble on the EXACT
production arrays saved in the pickle and rebuild the L/S portfolio.

  result ~= 1.107  -> gap was purely my feature/row-ordering pipeline (production reproduces).
  result  < 1.107  -> irreducible XGB/library non-determinism (parallel FP / row order).

READ-ONLY on production artefacts; writes nothing to production.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from src.utils import long_short_port, metrics
from config import (XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_N_JOBS)

with open(os.path.join(_ROOT, 'artefacts', 'cs_artefacts_data.pkl'), 'rb') as f:
    art = pickle.load(f)

Xtr, ytr, Xte = art['X_train'], art['y_train'], art['X_test']
print(f"production arrays: X_train {Xtr.shape}, X_test {Xte.shape}, seeds {len(XGB_SEEDS)}", flush=True)

preds = np.zeros(len(Xte))
for i, seed in enumerate(XGB_SEEDS, 1):
    m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
                     subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE, tree_method='hist',
                     random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS)
    m.fit(Xtr, ytr)
    preds += m.predict(Xte)
    if i % 10 == 0:
        print(f"  seed {i}/{len(XGB_SEEDS)}", flush=True)
preds /= len(XGB_SEEDS)

t = art['test'].copy()
prod_score = t['score_xgb'].values
t['score_new'] = preds

print("\n=== RECONCILE ===")
print(f"  production score_xgb -> L/S Sharpe: {metrics(long_short_port(t, 'score_xgb'))[2]:.3f}")
print(f"  retrained on prod arrays -> L/S Sharpe: {metrics(long_short_port(t, 'score_new'))[2]:.3f}")
print(f"  pred correlation (retrain vs production): {np.corrcoef(prod_score, preds)[0, 1]:.5f}")
print(f"  mean |pred diff|: {np.abs(prod_score - preds).mean():.6e}")
