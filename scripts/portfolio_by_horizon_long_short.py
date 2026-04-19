"""
Two panels (Long leg | Short leg): median momentum at each lookback horizon (1-12 months),
calm vs panic.
X-axis = lookback horizon in months.
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

portfolio = test_pi[test_pi['leg'].isin(['long', 'short'])].copy()

# Compute medians
horizons = list(range(1, 13))
mom_cols = [f'mom_{h}' for h in horizons]

profiles = {}
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (portfolio['regime'] == regime) & (portfolio['leg'] == leg)
        profiles[(regime, leg)] = portfolio.loc[mask, mom_cols].median().values * 100

# ── Plot: 2 panels ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
x = np.arange(1, 13)

# Panel 1: Long leg
ax = axes[0]
ax.plot(x, profiles[('Calm', 'long')], 'o-', color='steelblue',
        linewidth=2.5, markersize=7, label='Calm', alpha=0.85)
ax.plot(x, profiles[('Panic', 'long')], 's-', color='#E53935',
        linewidth=2.5, markersize=7, label='Panic', alpha=0.85)
ax.axhline(0, color='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in horizons], fontsize=10)
ax.set_xlabel('Lookback horizon (months)', fontsize=11)
ax.set_ylabel('Median trailing momentum (%)', fontsize=11)
ax.set_title('Long leg: what M2 buys', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: Short leg
ax = axes[1]
ax.plot(x, profiles[('Calm', 'short')], 'o-', color='steelblue',
        linewidth=2.5, markersize=7, label='Calm', alpha=0.85)
ax.plot(x, profiles[('Panic', 'short')], 's-', color='#E53935',
        linewidth=2.5, markersize=7, label='Panic', alpha=0.85)
ax.axhline(0, color='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in horizons], fontsize=10)
ax.set_xlabel('Lookback horizon (months)', fontsize=11)
ax.set_title('Short leg: what M2 sells', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('Momentum profile of M2 portfolio by regime',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/portfolio_by_horizon_long_short.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: portfolio_by_horizon_long_short.png")

from PIL import Image
img = Image.open('portfolio_by_horizon_long_short.png')
img.save('portfolio_by_horizon_long_short.pdf', 'PDF', resolution=150)
print("Saved: portfolio_by_horizon_long_short.pdf")
