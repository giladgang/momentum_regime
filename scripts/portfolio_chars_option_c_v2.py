"""
Option C v2: clearer axis labels, larger panels.
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
horizons_to_show = [1, 3, 6, 9, 12]
horizon_labels = ['1\nmonth', '3\nmonths', '6\nmonths', '9\nmonths', '12\nmonths']
mom_cols = [f'mom_{h}' for h in horizons_to_show]

results = {}
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (portfolio['regime'] == regime) & (portfolio['leg'] == leg)
        medians = portfolio.loc[mask, mom_cols].median() * 100
        results[(regime, leg)] = medians.values

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

colors = {'long': '#2196F3', 'short': '#E53935'}
w = 0.32

for i, regime in enumerate(['Calm', 'Panic']):
    ax = axes[i]
    x = np.arange(len(horizons_to_show))

    long_vals = results[(regime, 'long')]
    short_vals = results[(regime, 'short')]

    bars_l = ax.bar(x - w/2, long_vals, w, label='Long side',
                    color=colors['long'], alpha=0.85, edgecolor='white')
    bars_s = ax.bar(x + w/2, short_vals, w, label='Short side',
                    color=colors['short'], alpha=0.85, edgecolor='white')

    for bars in [bars_l, bars_s]:
        for bar in bars:
            v = bar.get_height()
            va = 'bottom' if v >= 0 else 'top'
            off = 1.5 if v >= 0 else -1.5
            ax.text(bar.get_x() + bar.get_width()/2, v + off,
                    f'{v:+.0f}%', ha='center', va=va, fontsize=10, fontweight='bold')

    n = 108 if regime == 'Calm' else 59
    ax.set_title(f'{regime} months ({n} months)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels, fontsize=11)
    ax.set_xlabel('Lookback horizon', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.grid(axis='y', alpha=0.3)

    if i == 0:
        ax.set_ylabel('Median trailing momentum (%)', fontsize=12)
        ax.legend(fontsize=11, loc='upper left')

plt.suptitle('Portfolio composition by regime: what M2 holds',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/portfolio_chars_option_c_v2.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: portfolio_chars_option_c_v2.png")

from PIL import Image
img = Image.open('portfolio_chars_option_c_v2.png')
img.save('portfolio_chars_option_c_v2.pdf', 'PDF', resolution=150)
print("Saved: portfolio_chars_option_c_v2.pdf")
