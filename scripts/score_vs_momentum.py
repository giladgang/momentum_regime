"""
How XGBoost scores stocks as a function of their momentum, split by regime.
Shows the relationship between a stock's trailing momentum and the score
the model assigns it -- and how that relationship flips between calm and panic.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MOM_FEATURES, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_SEEDS)

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

REDUCED = MOM_FEATURES + ['pi_filter']
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

from xgboost import XGBRegressor

X_tr = train[REDUCED].values.astype(float)
X_te = test[REDUCED].values.astype(float)
y_tr = train['ret_fwd'].values.astype(float)

print("Training XGB ensemble ...")
preds = np.zeros(len(X_te))
for xs in XGB_SEEDS[:5]:
    xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                       learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                       colsample_bytree=COLSAMPLE, tree_method='hist',
                       random_state=xs, verbosity=0)
    xgb.fit(X_tr, y_tr)
    preds += xgb.predict(X_te)
preds /= 5
test['score'] = preds

test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['pi_month'] = test_pi['pi_month'].fillna(0.5)
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

# ── Bin stocks by 12-month momentum, compute average score per bin ──
test_pi['mom12_pct'] = test_pi['mom_12'] * 100  # to %

# Use percentile bins to get equal-sized groups
n_bins = 20
for regime in ['Calm', 'Panic']:
    mask = test_pi['regime'] == regime
    test_pi.loc[mask, 'mom12_bin'] = pd.qcut(
        test_pi.loc[mask, 'mom12_pct'], n_bins, labels=False, duplicates='drop')

# Compute mean score and mean momentum per bin per regime
binned = test_pi.dropna(subset=['mom12_bin']).groupby(['regime', 'mom12_bin']).agg(
    mean_mom12=('mom12_pct', 'mean'),
    mean_score=('score', 'mean'),
    count=('score', 'size')
).reset_index()

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 6))

for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned[binned['regime'] == regime].sort_values('mean_mom12')
    ax.plot(d['mean_mom12'], d['mean_score'], f'{marker}-', color=color,
            linewidth=2.5, markersize=7, label=f'{regime}', alpha=0.85)

ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)

ax.set_xlabel('12-month trailing momentum (%)', fontsize=12)
ax.set_ylabel('Average XGBoost predicted score', fontsize=12)
ax.set_title('How XGBoost scores stocks based on their momentum\nby market regime',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig('plots/score_vs_momentum.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: score_vs_momentum.png")

# Also do 1-month momentum
test_pi['mom1_pct'] = test_pi['mom_1'] * 100
for regime in ['Calm', 'Panic']:
    mask = test_pi['regime'] == regime
    test_pi.loc[mask, 'mom1_bin'] = pd.qcut(
        test_pi.loc[mask, 'mom1_pct'], n_bins, labels=False, duplicates='drop')

binned1 = test_pi.dropna(subset=['mom1_bin']).groupby(['regime', 'mom1_bin']).agg(
    mean_mom1=('mom1_pct', 'mean'),
    mean_score=('score', 'mean'),
).reset_index()

# 2-panel: 12-mo and 1-mo side by side
fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel 1: 12-month
ax = axes[0]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned[binned['regime'] == regime].sort_values('mean_mom12')
    ax.plot(d['mean_mom12'], d['mean_score'], f'{marker}-', color=color,
            linewidth=2.5, markersize=6, label=f'{regime}', alpha=0.85)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlabel('12-month trailing momentum (%)', fontsize=11)
ax.set_ylabel('Average XGBoost predicted score', fontsize=11)
ax.set_title('12-month momentum', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: 1-month
ax = axes[1]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned1[binned1['regime'] == regime].sort_values('mean_mom1')
    ax.plot(d['mean_mom1'], d['mean_score'], f'{marker}-', color=color,
            linewidth=2.5, markersize=6, label=f'{regime}', alpha=0.85)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlabel('1-month trailing momentum (%)', fontsize=11)
ax.set_title('1-month momentum', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('How XGBoost scores stocks based on momentum: calm vs panic',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig('plots/score_vs_momentum_2panel.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: score_vs_momentum_2panel.png")

from PIL import Image
for name in ['score_vs_momentum', 'score_vs_momentum_2panel']:
    img = Image.open(f'{name}.png')
    img.save(f'{name}.pdf', 'PDF', resolution=150)
print("Saved PDFs")
