"""
Percentage of model attention (SHAP importance) on each momentum horizon,
split by calm vs panic. Shows which horizons the model leans on in each regime.
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
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

REDUCED = MOM_FEATURES + ['pi_filter']
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

from xgboost import XGBRegressor

X_tr = train[REDUCED].values.astype(float)
X_te = test[REDUCED].values.astype(float)
y_tr = train['ret_fwd'].values.astype(float)

print("Training XGB ...")
xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                   learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                   colsample_bytree=COLSAMPLE, tree_method='hist',
                   random_state=42, verbosity=0)
xgb.fit(X_tr, y_tr)

print("Computing SHAP values ...")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_te)

# Regime labels per row
test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['pi_month'] = test_pi['pi_month'].fillna(0.5)
regime = np.where(test_pi['pi_month'].values >= 0.5, 'Panic', 'Calm')

# Mean |SHAP| per feature, split by regime
mom_indices = [REDUCED.index(f'mom_{h}') for h in range(1, 13)]
pi_idx = REDUCED.index('pi_filter')

shares = {}
for r in ['Calm', 'Panic']:
    mask = regime == r
    abs_shap = np.abs(shap_values[mask])

    # Mean |SHAP| for each momentum horizon
    mom_importance = abs_shap[:, mom_indices].mean(axis=0)
    pi_importance = abs_shap[:, pi_idx].mean()
    total = mom_importance.sum() + pi_importance

    # As percentage of total
    mom_pct = mom_importance / total * 100
    pi_pct = pi_importance / total * 100

    shares[r] = {'mom': mom_pct, 'pi': pi_pct}

# Print
print(f"\n{'Horizon':>8s} {'Calm %':>8s} {'Panic %':>9s} {'Shift':>8s}")
for i, h in enumerate(range(1, 13)):
    c = shares['Calm']['mom'][i]
    p = shares['Panic']['mom'][i]
    print(f"  mom_{h:<3d} {c:>7.1f}% {p:>8.1f}% {p-c:>+7.1f}%")
print(f"  pi_flt {shares['Calm']['pi']:>7.1f}% {shares['Panic']['pi']:>8.1f}% {shares['Panic']['pi']-shares['Calm']['pi']:>+7.1f}%")

# ── Plot: grouped bar chart, calm vs panic side by side for each horizon ──
fig, ax = plt.subplots(figsize=(12, 5.5))

x = np.arange(13)  # 12 horizons + pi_filter
w = 0.35

calm_vals = np.append(shares['Calm']['mom'], shares['Calm']['pi'])
panic_vals = np.append(shares['Panic']['mom'], shares['Panic']['pi'])

labels = [f'{h}' for h in range(1, 13)] + [r'$\pi^{filter}$']

bars_c = ax.bar(x - w/2, calm_vals, w, label='Calm', color='steelblue', alpha=0.85, edgecolor='white')
bars_p = ax.bar(x + w/2, panic_vals, w, label='Panic', color='#E53935', alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel('Feature', fontsize=11)
ax.set_ylabel('Share of total importance (%)', fontsize=11)
ax.set_title('How XGB allocates attention across features: calm vs panic',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add a vertical separator before pi_filter
ax.axvline(11.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig('plots/horizon_weight_share.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: horizon_weight_share.png")

from PIL import Image
img = Image.open('horizon_weight_share.png')
img.save('horizon_weight_share.pdf', 'PDF', resolution=150)
print("Saved: horizon_weight_share.pdf")
