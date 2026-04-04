"""
cs_plots.py
===========
Fast re-plotting script — loads pre-saved artefacts from cross_sectional_model.py.
Run this instead of re-running the full model when you only want to add/change plots.

Usage:
    python cs_plots.py
"""

import matplotlib
matplotlib.use('Agg')

import pickle, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import re
from collections import defaultdict, Counter

# ── Load artefacts ────────────────────────────────────────────────────────────

print("Loading artefacts ...")
xgb = joblib.load('cs_artefacts_xgb.pkl')
lr  = joblib.load('cs_artefacts_lr.pkl')

with open('cs_artefacts_data.pkl', 'rb') as f:
    d = pickle.load(f)

test         = d['test']
train        = d['train']
X_test       = d['X_test']
X_train      = d['X_train']
X_te_s       = d['X_te_s']
X_tr_s       = d['X_tr_s']
y_train      = d['y_train']
FEATURES     = d['FEATURES']
strategies_lo= d['strategies_lo']
r_mkt        = d['r_mkt']
shap_values  = d['shap_values']
imp_calm     = d['imp_calm']
imp_panic    = d['imp_panic']
pi_grid      = d['pi_grid']
pdp_scores   = d['pdp_scores']
all_trees    = d['all_trees']
avg_tree     = d['avg_tree']

print("Artefacts loaded. Add your plot code below.")

# ── Add new plots here ────────────────────────────────────────────────────────

# Example: LR weights
fig, ax = plt.subplots(figsize=(8, 6))
coef = pd.Series(lr.coef_[0], index=FEATURES).sort_values()
colors_lr = ['crimson' if c < 0 else 'steelblue' for c in coef]
coef.plot(kind='barh', ax=ax, color=colors_lr, alpha=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Learned weight (standardized)', fontsize=9)
ax.set_title('Logistic Regression weights\n(positive = predicts above-median return)', fontsize=10)
plt.tight_layout()
fig.savefig('cs_lr_weights.png', dpi=150)
plt.close(fig)
print("Saved: cs_lr_weights.png")
