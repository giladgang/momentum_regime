"""
Portfolio momentum profile: median momentum at each of the 12 horizons
for long/short legs in calm/panic regimes.
Shows what the portfolio actually holds across the full term structure.
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

# ── Compute median momentum at each horizon ──
horizons = list(range(1, 13))
mom_cols = [f'mom_{h}' for h in horizons]

profiles = {}
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (portfolio['regime'] == regime) & (portfolio['leg'] == leg)
        medians = portfolio.loc[mask, mom_cols].median() * 100  # to %
        profiles[(regime, leg)] = medians.values

# Print table
print(f"\n{'Horizon':>8s} {'Calm Long':>10s} {'Calm Short':>11s} {'Panic Long':>11s} {'Panic Short':>12s}")
for i, h in enumerate(horizons):
    vals = [profiles[('Calm','long')][i], profiles[('Calm','short')][i],
            profiles[('Panic','long')][i], profiles[('Panic','short')][i]]
    print(f"  mom_{h:<3d} {vals[0]:>+9.1f}% {vals[1]:>+10.1f}% {vals[2]:>+10.1f}% {vals[3]:>+11.1f}%")

# ── Plot: 2-panel (Calm | Panic), lines for long and short ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

x = np.arange(1, 13)

for i, regime in enumerate(['Calm', 'Panic']):
    ax = axes[i]

    long_vals = profiles[(regime, 'long')]
    short_vals = profiles[(regime, 'short')]

    ax.plot(x, long_vals, 'o-', color='#2196F3', linewidth=2.5, markersize=7,
            label='Long side', zorder=3)
    ax.plot(x, short_vals, 's-', color='#E53935', linewidth=2.5, markersize=7,
            label='Short side', zorder=3)

    # Shade the spread
    ax.fill_between(x, long_vals, short_vals, alpha=0.10,
                    color='#2196F3' if regime == 'Calm' else '#E53935')

    ax.axhline(0, color='black', linewidth=0.6, linestyle='-')
    ax.set_xticks(x)
    ax.set_xlabel('Momentum lookback horizon (months)', fontsize=10)
    n = 108 if regime == 'Calm' else 59
    ax.set_title(f'{regime} months ({n} months)', fontsize=12, fontweight='bold')
    ax.grid(axis='both', alpha=0.3)

    if i == 0:
        ax.set_ylabel('Median trailing momentum (%)', fontsize=11)
        ax.legend(fontsize=10, loc='upper left')

plt.suptitle('What XGB holds: momentum profile across all 12 horizons',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/portfolio_momentum_profile.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: portfolio_momentum_profile.png")

# Convert to PDF
from PIL import Image
img = Image.open('portfolio_momentum_profile.png')
img.save('portfolio_momentum_profile.pdf', 'PDF', resolution=150)
print("Saved: portfolio_momentum_profile.pdf")
