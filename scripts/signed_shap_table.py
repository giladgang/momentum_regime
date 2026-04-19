"""
signed_shap_table.py
====================
Compute signed (not absolute) SHAP by horizon, regime, and leg.
"""

import numpy as np
import pandas as pd
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)

test = art['test'].copy()
shap_values = art['shap_values']
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test['regime'] = np.where(test['pi_month'] >= 0.5, 'Panic', 'Calm')

# Assign legs
test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

horizons = list(range(1, 13))
mom_indices = [FEATURES.index(f'mom_{h}') for h in horizons]

# Also get pi_filter index
pi_idx = FEATURES.index('pi_filter')

print(f"\n{'':>8s} | {'Calm Long':>10s} {'Calm Short':>11s} | {'Panic Long':>11s} {'Panic Short':>12s}")
print("-" * 65)

for i, h in enumerate(horizons):
    idx = mom_indices[i]
    vals = []
    for regime in ['Calm', 'Panic']:
        for leg in ['long', 'short']:
            mask = (test['regime'] == regime) & (test['leg'] == leg)
            v = shap_values[mask.values, idx].mean()
            vals.append(v)
    print(f"mom_{h:>2d}   | {vals[0]:>+10.5f} {vals[1]:>+11.5f} | {vals[2]:>+11.5f} {vals[3]:>+12.5f}")

# pi_filter
print("-" * 65)
vals = []
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (test['regime'] == regime) & (test['leg'] == leg)
        v = shap_values[mask.values, pi_idx].mean()
        vals.append(v)
print(f"pi_filt  | {vals[0]:>+10.5f} {vals[1]:>+11.5f} | {vals[2]:>+11.5f} {vals[3]:>+12.5f}")

# Also show |SHAP| for comparison
print(f"\n\n{'':>8s} | {'Calm Long':>10s} {'Calm Short':>11s} | {'Panic Long':>11s} {'Panic Short':>12s}")
print("-" * 65)
print("ABSOLUTE |SHAP| (for comparison):")
print("-" * 65)

for i, h in enumerate(horizons):
    idx = mom_indices[i]
    vals = []
    for regime in ['Calm', 'Panic']:
        for leg in ['long', 'short']:
            mask = (test['regime'] == regime) & (test['leg'] == leg)
            v = np.abs(shap_values[mask.values, idx]).mean()
            vals.append(v)
    print(f"mom_{h:>2d}   | {vals[0]:>10.5f} {vals[1]:>11.5f} | {vals[2]:>11.5f} {vals[3]:>12.5f}")

vals = []
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (test['regime'] == regime) & (test['leg'] == leg)
        v = np.abs(shap_values[mask.values, pi_idx]).mean()
        vals.append(v)
print("-" * 65)
print(f"pi_filt  | {vals[0]:>10.5f} {vals[1]:>11.5f} | {vals[2]:>11.5f} {vals[3]:>12.5f}")

print("\nDone.")
