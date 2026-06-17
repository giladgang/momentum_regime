"""
2026-05-31-reconcile-production.py
==================================
DIAGNOSTIC (experiments/). Reconcile the 1.08 (my harness) vs 1.11 (logged
production) full-universe L/S Sharpe gap. READ-ONLY: loads the existing
production artefact pickle (artefacts/cs_artefacts_data.pkl, from the May-7
production run) and compares it to my experiment harness sample. Writes nothing
to production.

Steps:
  1. Recompute long_short_port on the production scored test set -> production Sharpe.
  2. Cross-check vs the saved production strategy returns.
  3. Compare production test SAMPLE (date range, rows, permno-date pairs) to my
     exp.build_features() full-universe test -> is the gap a SAMPLE difference?
"""
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
_s = importlib.util.spec_from_file_location("exp", os.path.join(_ROOT, 'real_trade_analysis', 'scripts', '2026-05-30-xgb-russell1000.py'))
exp = importlib.util.module_from_spec(_s); _s.loader.exec_module(exp)

from src.utils import long_short_port, metrics
from config import TRAIN_END

print("loading production artefact pickle (912MB) ...", flush=True)
with open(os.path.join(_ROOT, 'artefacts', 'cs_artefacts_data.pkl'), 'rb') as f:
    art = pickle.load(f)
print("keys:", list(art.keys()), flush=True)

ptest = art['test']
print("\n=== PRODUCTION test set ===")
print("rows:", len(ptest), "| cols incl:", [c for c in ['permno', 'date', 'score_xgb', 'me', 'ret_fwd', 'exchcd'] if c in ptest.columns])
print("date range:", ptest['date'].min().date(), "->", ptest['date'].max().date(), "| months:", ptest['date'].nunique())

# 1. Recompute L/S from production scores
r_prod = long_short_port(ptest, 'score_xgb')
print(f"\nL/S Sharpe recomputed from production score_xgb: {metrics(r_prod)[2]:.3f}  (logged 1.11)")
print(f"  ann {metrics(r_prod)[0]:.1%} vol {metrics(r_prod)[1]:.1%} months {len(r_prod)}")

# 2. Saved production strategy returns (cross-check)
if 'strategies_lo' in art:
    for k, v in art['strategies_lo'].items():
        if 'XGB' in k:
            print(f"  saved strategies['{k}'] Sharpe: {metrics(v)[2]:.3f}")

# 3. Compare sample vs my experiment harness
print("\n=== MY experiment harness (exp.build_features, full universe) ===")
mine = exp.build_features()
mtest = mine[mine['date'] >= TRAIN_END]
print("rows:", len(mtest), "| date range:", mtest['date'].min().date(), "->", mtest['date'].max().date(), "| months:", mtest['date'].nunique())

pk = set(map(tuple, ptest[['permno', 'date']].values))
mk = set(map(tuple, mtest[['permno', 'date']].values))
print("\n=== SAMPLE OVERLAP (permno,date pairs) ===")
print(f"  in both:       {len(pk & mk):,}")
print(f"  prod only:     {len(pk - mk):,}")
print(f"  mine only:     {len(mk - pk):,}")
print(f"  prod total:    {len(pk):,}  | mine total: {len(mk):,}")

# month-level row counts compare
pm = ptest.groupby(ptest['date'].dt.to_period('M')).size()
mm = mtest.groupby(mtest['date'].dt.to_period('M')).size()
diff = (pm - mm).dropna()
print("\n  months where row counts differ:", int((diff != 0).sum()), "of", len(diff))
if (diff != 0).any():
    print("  sample diffs (prod-mine) head:\n", diff[diff != 0].head(10).to_string())
