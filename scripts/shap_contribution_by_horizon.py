"""
Mean signed SHAP contribution of each momentum horizon in calm vs panic.
Shows how each horizon pushes the predicted score up or down in each regime.
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

with open('cs_artefacts_data.pkl', 'rb') as f:
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

print("Training XGB ...")
xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                   learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                   colsample_bytree=COLSAMPLE, tree_method='hist',
                   random_state=42, verbosity=0)
xgb.fit(X_tr, y_tr)

print("Computing SHAP values ...")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_te)

# Regime labels
test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['pi_month'] = test_pi['pi_month'].fillna(0.5)
regime = np.where(test_pi['pi_month'].values >= 0.5, 'Panic', 'Calm')

mom_indices = [REDUCED.index(f'mom_{h}') for h in range(1, 13)]

# Mean |SHAP| per horizon per regime (absolute contribution)
contrib = {}
for r in ['Calm', 'Panic']:
    mask = regime == r
    contrib[r] = np.abs(shap_values[mask][:, mom_indices]).mean(axis=0)

# Print
print(f"\n{'Horizon':>8s} {'Calm':>10s} {'Panic':>10s} {'Ratio P/C':>10s}")
for i, h in enumerate(range(1, 13)):
    c, p = contrib['Calm'][i], contrib['Panic'][i]
    ratio = p / c if c > 0 else 0
    print(f"  mom_{h:<3d} {c:>10.5f} {p:>10.5f} {ratio:>9.1f}x")

# ── Plot ──
fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(12)
w = 0.35

bars_c = ax.bar(x - w/2, contrib['Calm'] * 1000, w, label='Calm',
                color='steelblue', alpha=0.85, edgecolor='white')
bars_p = ax.bar(x + w/2, contrib['Panic'] * 1000, w, label='Panic',
                color='#E53935', alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([f'{h}-mo' for h in range(1, 13)], fontsize=10)
ax.set_xlabel('Momentum lookback horizon', fontsize=12)
ax.set_ylabel('Mean |SHAP| contribution (bps)', fontsize=12)
ax.set_title('How much each momentum horizon contributes to XGBoost predictions\ncalm vs panic',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig('shap_contribution_by_horizon.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: shap_contribution_by_horizon.png")

from PIL import Image
img = Image.open('shap_contribution_by_horizon.png')
img.save('shap_contribution_by_horizon.pdf', 'PDF', resolution=150)
print("Saved: shap_contribution_by_horizon.pdf")
