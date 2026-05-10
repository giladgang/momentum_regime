"""
Two panels: long leg selection rate and short leg selection rate
by momentum decile, calm vs panic.
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

n_bins = 10
records = []

for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    regime = grp['regime'].iloc[0]
    grp = grp.copy()
    grp['mom12_decile'] = pd.qcut(grp['mom_12'].rank(method='first'), n_bins,
                                   labels=list(range(1, n_bins + 1)))
    for _, row in grp.iterrows():
        if pd.isna(row['mom12_decile']):
            continue
        records.append({
            'regime': regime,
            'mom12_decile': int(row['mom12_decile']),
            'in_long': 1 if row['score'] >= hi else 0,
            'in_short': 1 if row['score'] <= lo else 0,
        })

df = pd.DataFrame(records)
rates = df.groupby(['regime', 'mom12_decile']).agg(
    long_rate=('in_long', 'mean'),
    short_rate=('in_short', 'mean'),
).reset_index()

# ── Plot: 2 panels, long and short ──
x_labels = ['Biggest\nlosers', '2', '3', '4', '5', '6', '7', '8', '9', 'Biggest\nwinners']
x = np.arange(n_bins)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

# Panel 1: Long leg
ax = axes[0]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = rates[rates['regime'] == regime].sort_values('mom12_decile')
    ax.plot(x, d['long_rate'].values * 100, f'{marker}-', color=color,
            linewidth=2.5, markersize=8, label=regime, alpha=0.85)
ax.axhline(10, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_xlabel('Stocks ranked by 12-month momentum', fontsize=10)
ax.set_ylabel('% of stocks selected', fontsize=11)
ax.set_title('Long leg: which stocks does XGB buy?', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Panel 2: Short leg
ax = axes[1]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = rates[rates['regime'] == regime].sort_values('mom12_decile')
    ax.plot(x, d['short_rate'].values * 100, f'{marker}-', color=color,
            linewidth=2.5, markersize=8, label=regime, alpha=0.85)
ax.axhline(10, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_xlabel('Stocks ranked by 12-month momentum', fontsize=10)
ax.set_title('Short leg: which stocks does XGB sell?', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('XGB stock selection by momentum decile: calm vs panic',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/selection_long_short_separate.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: selection_long_short_separate.png")

from PIL import Image
img = Image.open('selection_long_short_separate.png')
img.save('selection_long_short_separate.pdf', 'PDF', resolution=150)
print("Saved: selection_long_short_separate.pdf")
