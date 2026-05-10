"""
Generate Option A and Option C portfolio characteristics charts.
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

# ── Load and prep data ──
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

# ── Compute medians ──
horizons = {'mom_1': '1-month', 'mom_6': '6-month', 'mom_12': '12-month'}
data = {}
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (portfolio['regime'] == regime) & (portfolio['leg'] == leg)
        subset = portfolio[mask]
        for col in horizons:
            data[(regime, leg, col)] = subset[col].median() * 100

# Print summary
print(f"\n{'':18s} {'Calm Long':>10s} {'Calm Short':>11s} {'Panic Long':>11s} {'Panic Short':>12s}")
for col, label in horizons.items():
    vals = [data[('Calm','long',col)], data[('Calm','short',col)],
            data[('Panic','long',col)], data[('Panic','short',col)]]
    print(f"  {label:16s} {vals[0]:>+9.1f}% {vals[1]:>+10.1f}% {vals[2]:>+10.1f}% {vals[3]:>+11.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION A: Single bar chart, 12-month momentum only
# ═══════════════════════════════════════════════════════════════════════════════

fig_a, ax = plt.subplots(figsize=(7, 5))

x = np.arange(2)
w = 0.32
colors = {'long': '#2196F3', 'short': '#E53935'}

long_vals = [data[('Calm', 'long', 'mom_12')], data[('Panic', 'long', 'mom_12')]]
short_vals = [data[('Calm', 'short', 'mom_12')], data[('Panic', 'short', 'mom_12')]]

bars_l = ax.bar(x - w/2, long_vals, w, label='Long side', color=colors['long'], alpha=0.85, edgecolor='white')
bars_s = ax.bar(x + w/2, short_vals, w, label='Short side', color=colors['short'], alpha=0.85, edgecolor='white')

for bars in [bars_l, bars_s]:
    for bar in bars:
        v = bar.get_height()
        va = 'bottom' if v >= 0 else 'top'
        off = 1.0 if v >= 0 else -1.0
        ax.text(bar.get_x() + bar.get_width()/2, v + off,
                f'{v:+.1f}%', ha='center', va=va, fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['Calm\n(108 months)', 'Panic\n(59 months)'], fontsize=11)
ax.set_ylabel('Median 12-month momentum (%)', fontsize=11)
ax.set_title('What XGB holds: 12-month trailing momentum by regime', fontsize=12, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.6)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(min(short_vals + long_vals) - 8, max(long_vals + short_vals) + 8)

plt.tight_layout()
fig_a.savefig('plots/portfolio_chars_option_a.png', dpi=150, bbox_inches='tight')
plt.close(fig_a)
print("\nSaved: portfolio_chars_option_a.png")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION C: 2-panel (Calm | Panic), 3 horizons per panel
# ═══════════════════════════════════════════════════════════════════════════════

fig_c, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

horizon_labels = ['1-mo', '6-mo', '12-mo']
horizon_keys = ['mom_1', 'mom_6', 'mom_12']

for i, (regime, ax) in enumerate(zip(['Calm', 'Panic'], axes)):
    x = np.arange(len(horizon_keys))
    w = 0.32

    long_vals = [data[(regime, 'long', h)] for h in horizon_keys]
    short_vals = [data[(regime, 'short', h)] for h in horizon_keys]

    bars_l = ax.bar(x - w/2, long_vals, w, label='Long side', color=colors['long'], alpha=0.85, edgecolor='white')
    bars_s = ax.bar(x + w/2, short_vals, w, label='Short side', color=colors['short'], alpha=0.85, edgecolor='white')

    for bars in [bars_l, bars_s]:
        for bar in bars:
            v = bar.get_height()
            va = 'bottom' if v >= 0 else 'top'
            off = 1.2 if v >= 0 else -1.2
            ax.text(bar.get_x() + bar.get_width()/2, v + off,
                    f'{v:+.1f}%', ha='center', va=va, fontsize=8.5, fontweight='bold')

    n_months = 108 if regime == 'Calm' else 59
    ax.set_title(f'{regime} months ({n_months} months)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels, fontsize=10)
    ax.set_xlabel('Momentum horizon', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.grid(axis='y', alpha=0.3)

    if i == 0:
        ax.set_ylabel('Median momentum (%)', fontsize=11)
        ax.legend(fontsize=9, loc='upper left')

plt.suptitle('What XGB holds: portfolio momentum characteristics by regime',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig_c.savefig('plots/portfolio_chars_option_c.png', dpi=150, bbox_inches='tight')
plt.close(fig_c)
print("Saved: portfolio_chars_option_c.png")
