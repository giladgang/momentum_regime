"""
SHAP importance of each momentum horizon, split by leg (long/short) and regime (calm/panic).
Shows which horizons XGBoost relies on for each portfolio decision.
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

# ── Load data ──
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
FEATURES = art['FEATURES']
X_test = art['X_test']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

REDUCED = MOM_FEATURES + ['pi_filter']
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

# Train XGB and get SHAP values
from xgboost import XGBRegressor

X_tr = train[REDUCED].values.astype(float)
X_te = test[REDUCED].values.astype(float)
y_tr = train['ret_fwd'].values.astype(float)

print("Training XGB ensemble (5 seeds) ...")
# Use one model for SHAP (consistent with main script)
xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                   learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                   colsample_bytree=COLSAMPLE, tree_method='hist',
                   random_state=42, verbosity=0)
xgb.fit(X_tr, y_tr)

# Score stocks
preds = np.zeros(len(X_te))
for xs in XGB_SEEDS[:5]:
    xgb_i = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=xs, verbosity=0)
    xgb_i.fit(X_tr, y_tr)
    preds += xgb_i.predict(X_te)
preds /= 5
test['score'] = preds

print("Computing SHAP values ...")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_te)  # (N, 13)

# Add regime and leg labels
test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['pi_month'] = test_pi['pi_month'].fillna(0.5)
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

test_pi['leg'] = 'middle'
for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    long_mask = test_pi.loc[grp.index, 'score'] >= hi
    short_mask = test_pi.loc[grp.index, 'score'] <= lo
    test_pi.loc[grp.index[long_mask], 'leg'] = 'long'
    test_pi.loc[grp.index[short_mask], 'leg'] = 'short'

# ── Compute mean SHAP by regime x leg for each momentum horizon ──
mom_indices = [REDUCED.index(f'mom_{h}') for h in range(1, 13)]
mom_labels = [f'{h}' for h in range(1, 13)]

results = {}
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (test_pi['regime'].values == regime) & (test_pi['leg'].values == leg)
        # Mean SHAP (signed, not absolute) -- shows direction of contribution
        mean_shap = shap_values[mask][:, mom_indices].mean(axis=0)
        results[(regime, leg)] = mean_shap

# Print
print(f"\n{'Horizon':>8s} {'Calm Long':>10s} {'Calm Short':>11s} {'Panic Long':>11s} {'Panic Short':>12s}")
for i, h in enumerate(range(1, 13)):
    vals = [results[('Calm','long')][i], results[('Calm','short')][i],
            results[('Panic','long')][i], results[('Panic','short')][i]]
    print(f"  mom_{h:<3d} {vals[0]:>+10.5f} {vals[1]:>+11.5f} {vals[2]:>+11.5f} {vals[3]:>+12.5f}")

# Also show pi_filter
pi_idx = REDUCED.index('pi_filter')
print(f"\n  pi_flt ", end='')
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (test_pi['regime'].values == regime) & (test_pi['leg'].values == leg)
        v = shap_values[mask][:, pi_idx].mean()
        print(f" {v:>+10.5f}", end='')
print()

# ── Plot: 2x2 grid ──
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)

x = np.arange(12)
titles = {
    ('Calm', 'long'): 'Calm -- Long side',
    ('Calm', 'short'): 'Calm -- Short side',
    ('Panic', 'long'): 'Panic -- Long side',
    ('Panic', 'short'): 'Panic -- Short side',
}
panel_colors = {
    ('Calm', 'long'): '#2196F3',
    ('Calm', 'short'): '#E53935',
    ('Panic', 'long'): '#2196F3',
    ('Panic', 'short'): '#E53935',
}

for idx, ((regime, leg), ax) in enumerate(zip(
        [('Calm','long'), ('Calm','short'), ('Panic','long'), ('Panic','short')], axes.flat)):
    vals = results[(regime, leg)]
    colors = ['#2196F3' if v >= 0 else '#E53935' for v in vals]
    ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=8)
    ax.set_xlabel('Momentum horizon (months)', fontsize=9)
    ax.set_title(titles[(regime, leg)], fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    if idx % 2 == 0:
        ax.set_ylabel('Mean SHAP value', fontsize=9)

plt.suptitle('How XGBoost uses each momentum horizon\n(mean SHAP contribution by regime and portfolio leg)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig('plots/shap_by_leg_regime.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: shap_by_leg_regime.png")

# ── Also make a simpler 2-panel version (Calm | Panic), long vs short overlaid ──
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
w = 0.35

for i, regime in enumerate(['Calm', 'Panic']):
    ax = axes2[i]
    long_vals = results[(regime, 'long')]
    short_vals = results[(regime, 'short')]

    ax.bar(x - w/2, long_vals, w, label='Long side', color='#2196F3', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, short_vals, w, label='Short side', color='#E53935', alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=9)
    ax.set_xlabel('Momentum horizon (months)', fontsize=10)
    n = 108 if regime == 'Calm' else 59
    ax.set_title(f'{regime} months ({n} months)', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    if i == 0:
        ax.set_ylabel('Mean SHAP value\n(positive = pushes score up)', fontsize=10)
        ax.legend(fontsize=9)

plt.suptitle('How XGBoost weights each momentum horizon by regime',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig('plots/shap_by_leg_regime_2panel.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: shap_by_leg_regime_2panel.png")
