"""
SHAP dependence plot for mom_12: how the model uses 12-month momentum
differently in calm vs panic. Binned for clarity.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, sys, os

with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
shap_values = art['shap_values']
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['pi_month'] = test_pi['pi_month'].fillna(0.5)
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

mom12_idx = FEATURES.index('mom_12')
test_pi['mom12_val'] = test_pi['mom_12'].values * 100
test_pi['mom12_shap'] = shap_values[:, mom12_idx] * 10000  # bps

# Bin by mom_12 value within each regime (20 bins)
n_bins = 20

binned_data = []
for regime in ['Calm', 'Panic']:
    sub = test_pi[test_pi['regime'] == regime].copy()
    sub['bin'] = pd.qcut(sub['mom12_val'].rank(method='first'), n_bins,
                         labels=False, duplicates='drop')
    grouped = sub.groupby('bin').agg(
        mean_mom12=('mom12_val', 'mean'),
        mean_shap=('mom12_shap', 'mean'),
    ).reset_index()
    grouped['regime'] = regime
    binned_data.append(grouped)

binned = pd.concat(binned_data)

# ── Single panel ──
fig, ax = plt.subplots(figsize=(10, 6))

for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned[binned['regime'] == regime].sort_values('mean_mom12')
    ax.plot(d['mean_mom12'], d['mean_shap'], f'{marker}-', color=color,
            linewidth=2.5, markersize=7, label=regime, alpha=0.85)

ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)

ax.set_xlabel('12-month trailing momentum (%)', fontsize=12)
ax.set_ylabel('SHAP contribution of mom_12 (bps)', fontsize=12)
ax.set_title('How XGBoost uses 12-month momentum: calm vs panic',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig('shap_dependence_mom12.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: shap_dependence_mom12.png")

# ── Also do mom_1 ──
mom1_idx = FEATURES.index('mom_1')
test_pi['mom1_val'] = test_pi['mom_1'].values * 100
test_pi['mom1_shap'] = shap_values[:, mom1_idx] * 10000

binned1 = []
for regime in ['Calm', 'Panic']:
    sub = test_pi[test_pi['regime'] == regime].copy()
    sub['bin'] = pd.qcut(sub['mom1_val'].rank(method='first'), n_bins,
                         labels=False, duplicates='drop')
    grouped = sub.groupby('bin').agg(
        mean_mom1=('mom1_val', 'mean'),
        mean_shap=('mom1_shap', 'mean'),
    ).reset_index()
    grouped['regime'] = regime
    binned1.append(grouped)

binned1 = pd.concat(binned1)

# 2-panel
fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned[binned['regime'] == regime].sort_values('mean_mom12')
    ax.plot(d['mean_mom12'], d['mean_shap'], f'{marker}-', color=color,
            linewidth=2.5, markersize=6, label=regime, alpha=0.85)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_xlabel('12-month momentum (%)', fontsize=11)
ax.set_ylabel('SHAP contribution (bps)', fontsize=11)
ax.set_title('12-month momentum', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned1[binned1['regime'] == regime].sort_values('mean_mom1')
    ax.plot(d['mean_mom1'], d['mean_shap'], f'{marker}-', color=color,
            linewidth=2.5, markersize=6, label=regime, alpha=0.85)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)
ax.set_xlabel('1-month momentum (%)', fontsize=11)
ax.set_title('1-month momentum', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('How XGBoost uses momentum differently by regime',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig('shap_dependence_2panel.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: shap_dependence_2panel.png")

from PIL import Image
for name in ['shap_dependence_mom12', 'shap_dependence_2panel']:
    img = Image.open(f'{name}.png')
    img.save(f'{name}.pdf', 'PDF', resolution=150)
print("Saved PDFs")
