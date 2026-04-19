"""
Share of each momentum horizon as % of total momentum SHAP importance,
split by calm vs panic. Excludes pi_filter -- shows only how the model
distributes attention across the 12 momentum horizons.
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

# Momentum-only indices
mom_indices = [REDUCED.index(f'mom_{h}') for h in range(1, 13)]

shares = {}
for r in ['Calm', 'Panic']:
    mask = regime == r
    abs_shap = np.abs(shap_values[mask][:, mom_indices]).mean(axis=0)
    total_mom = abs_shap.sum()
    shares[r] = abs_shap / total_mom * 100

# Print
print(f"\n{'Horizon':>8s} {'Calm %':>8s} {'Panic %':>9s} {'Shift':>8s}")
for i, h in enumerate(range(1, 13)):
    c, p = shares['Calm'][i], shares['Panic'][i]
    print(f"  mom_{h:<3d} {c:>7.1f}% {p:>8.1f}% {p-c:>+7.1f}%")
print(f"  {'Total':>7s} {shares['Calm'].sum():>7.1f}% {shares['Panic'].sum():>8.1f}%")

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(12)
w = 0.35

bars_c = ax.bar(x - w/2, shares['Calm'], w, label='Calm', color='steelblue', alpha=0.85, edgecolor='white')
bars_p = ax.bar(x + w/2, shares['Panic'], w, label='Panic', color='#E53935', alpha=0.85, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([f'{h}-mo' for h in range(1, 13)], fontsize=9)
ax.set_xlabel('Momentum lookback horizon', fontsize=11)
ax.set_ylabel('Share of total momentum importance (%)', fontsize=11)
ax.set_title('Distribution of model attention across momentum horizons: calm vs panic',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig('plots/momentum_horizon_share.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: momentum_horizon_share.png")

from PIL import Image
img = Image.open('momentum_horizon_share.png')
img.save('momentum_horizon_share.pdf', 'PDF', resolution=150)
print("Saved: momentum_horizon_share.pdf")
