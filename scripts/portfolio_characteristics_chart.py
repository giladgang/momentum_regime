"""
portfolio_characteristics_chart.py
==================================
Creates a chart showing what M2 holds in calm vs panic regimes.
Visualises the inversion: trend-riders in calm, beaten-down stocks in panic.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRADING_FEE, MOM_FEATURES, N_ESTIMATORS,
                    MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE, XGB_SEEDS)

# ── Load data ──
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

REDUCED = MOM_FEATURES + ['pi_filter']

pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

# Train XGB ensemble (5 seeds for speed)
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

# Classify months and legs
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

# ── Compute characteristics by regime x leg ──
chars = {
    'mom_1': '1-mo momentum',
    'mom_6': '6-mo momentum',
    'mom_12': '12-mo momentum',
}

# Also compute median market cap in $M
results = {}
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (portfolio['regime'] == regime) & (portfolio['leg'] == leg)
        subset = portfolio[mask]
        key = (regime, leg)
        results[key] = {}
        for col in chars:
            if col in subset.columns:
                results[key][col] = subset[col].median() * 100  # convert to %
        if 'me' in subset.columns:
            results[key]['me'] = subset['me'].median()  # CRSP me is in $M
        if 'bm' in subset.columns:
            results[key]['bm'] = subset['me'].median() / 1e6  # we'll use B/M directly
            results[key]['bm'] = subset['bm'].median()

# Print summary
print("\n  Portfolio characteristics (medians):")
print(f"  {'':20s} {'Calm Long':>12s} {'Calm Short':>12s} {'Panic Long':>12s} {'Panic Short':>12s}")
for col, label in chars.items():
    vals = [results[(r, l)].get(col, float('nan')) for r, l in
            [('Calm', 'long'), ('Calm', 'short'), ('Panic', 'long'), ('Panic', 'short')]]
    print(f"  {label:20s} {vals[0]:>+11.1f}% {vals[1]:>+11.1f}% {vals[2]:>+11.1f}% {vals[3]:>+11.1f}%")
me_vals = [results[(r, l)].get('me', float('nan')) for r, l in
           [('Calm', 'long'), ('Calm', 'short'), ('Panic', 'long'), ('Panic', 'short')]]
print(f"  {'Market cap ($M)':20s} {me_vals[0]:>11.0f}  {me_vals[1]:>11.0f}  {me_vals[2]:>11.0f}  {me_vals[3]:>11.0f}")

# ── Create figure ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

bar_width = 0.35
regimes = ['Calm', 'Panic']
legs = ['long', 'short']
colors_leg = {'long': '#2196F3', 'short': '#E53935'}
leg_labels = {'long': 'Long side', 'short': 'Short side'}

for i, (col, label) in enumerate(chars.items()):
    ax = axes[i]
    x = np.arange(len(regimes))

    for j, leg in enumerate(legs):
        vals = [results[(r, leg)].get(col, 0) for r in regimes]
        offset = -bar_width/2 + j * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      label=leg_labels[leg], color=colors_leg[leg], alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        # Add value labels on bars
        for bar, v in zip(bars, vals):
            va = 'bottom' if v >= 0 else 'top'
            offset_y = 0.5 if v >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2, v + offset_y,
                    f'{v:+.1f}%', ha='center', va=va, fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Calm\n(108 months)', f'Panic\n(59 months)'], fontsize=9)
    ax.set_ylabel('Median momentum (%)', fontsize=9)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.grid(axis='y', alpha=0.3)

    if i == 0:
        ax.legend(fontsize=9, loc='upper left')

plt.suptitle("What M2 holds: portfolio characteristics by regime",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/portfolio_characteristics_by_regime.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: portfolio_characteristics_by_regime.png")
