"""
Two charts for the thesis:
- Figure for 5.2.1/5.2.2: mean signed SHAP contribution of each momentum horizon,
  calm vs panic, separate panels for long and short legs.

Uses production scores (score_xgb from artefacts).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, sys, os
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MOM_FEATURES, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, XGB_SEEDS)

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
shap_values = art['shap_values']
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

# Use production scores
test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['pi_month'] = test_pi['pi_month'].fillna(0.5)
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

# Assign legs using production scores
test_pi['leg'] = 'middle'
for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    long_mask = test_pi.loc[grp.index, 'score_xgb'] >= hi
    short_mask = test_pi.loc[grp.index, 'score_xgb'] <= lo
    test_pi.loc[grp.index[long_mask], 'leg'] = 'long'
    test_pi.loc[grp.index[short_mask], 'leg'] = 'short'

# Get SHAP values for momentum horizons
mom_indices = [FEATURES.index(f'mom_{h}') for h in range(1, 13)]
regime_arr = test_pi['regime'].values
leg_arr = test_pi['leg'].values

# Compute mean signed SHAP
results = {}
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (regime_arr == regime) & (leg_arr == leg)
        mean_shap = shap_values[mask][:, mom_indices].mean(axis=0)
        results[(regime, leg)] = mean_shap

# Print
print(f"{'Horizon':>8s} {'Calm Long':>10s} {'Calm Short':>11s} {'Panic Long':>11s} {'Panic Short':>12s}")
for i, h in enumerate(range(1, 13)):
    vals = [results[('Calm','long')][i], results[('Calm','short')][i],
            results[('Panic','long')][i], results[('Panic','short')][i]]
    print(f"  mom_{h:<3d} {vals[0]:>+10.5f} {vals[1]:>+11.5f} {vals[2]:>+11.5f} {vals[3]:>+12.5f}")

# ── Two separate figures ──
x = np.arange(1, 13)

# Figure 1: Long leg
fig1, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - 0.18, results[('Calm', 'long')] * 10000, 0.35,
       label='Calm', color='steelblue', alpha=0.85, edgecolor='white')
ax.bar(x + 0.18, results[('Panic', 'long')] * 10000, 0.35,
       label='Panic', color='#E53935', alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=10)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
ax.set_ylabel('Mean SHAP contribution (bps)', fontsize=11)
ax.set_title('Long leg: how each momentum horizon pushes stocks into the long portfolio',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig1.savefig('plots/shap_horizon_long.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("\nSaved: shap_horizon_long.png")

# Figure 2: Short leg
fig2, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - 0.18, results[('Calm', 'short')] * 10000, 0.35,
       label='Calm', color='steelblue', alpha=0.85, edgecolor='white')
ax.bar(x + 0.18, results[('Panic', 'short')] * 10000, 0.35,
       label='Panic', color='#E53935', alpha=0.85, edgecolor='white')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=10)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
ax.set_ylabel('Mean SHAP contribution (bps)', fontsize=11)
ax.set_title('Short leg: how each momentum horizon pushes stocks into the short portfolio',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig2.savefig('plots/shap_horizon_short.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: shap_horizon_short.png")

from PIL import Image
for name in ['shap_horizon_long', 'shap_horizon_short']:
    img = Image.open(f'{name}.png')
    img.save(f'{name}.pdf', 'PDF', resolution=150)
print("Saved PDFs")
