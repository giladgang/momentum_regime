"""
Score vs momentum with clean quintile x-axis.
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

# ── Decile bins computed within each month (cross-sectional rank) ──
n_bins = 10
bin_labels = list(range(1, n_bins + 1))
x_labels = ['Biggest\nlosers', '2', '3', '4', '5', '6', '7', '8', '9', 'Biggest\nwinners']

def assign_decile(group, col, new_col):
    group[new_col] = pd.qcut(group[col].rank(method='first'), n_bins, labels=bin_labels)
    return group

# 12-month momentum deciles within each month
test_pi = test_pi.groupby('date', group_keys=False).apply(
    assign_decile, col='mom_12', new_col='mom12_decile')

# 1-month momentum deciles within each month
test_pi = test_pi.groupby('date', group_keys=False).apply(
    assign_decile, col='mom_1', new_col='mom1_decile')

# ── Compute mean score per decile per regime ──
def get_binned(df, decile_col):
    return df.groupby(['regime', decile_col]).agg(
        mean_score=('score', 'mean'),
    ).reset_index()

binned12 = get_binned(test_pi.dropna(subset=['mom12_decile']), 'mom12_decile')
binned1 = get_binned(test_pi.dropna(subset=['mom1_decile']), 'mom1_decile')

# ── Single panel: 12-month ──
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(n_bins)

for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned12[binned12['regime'] == regime].sort_values('mom12_decile')
    ax.plot(x, d['mean_score'].values, f'{marker}-', color=color,
            linewidth=2.5, markersize=8, label=regime, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_xlabel('Stocks ranked by 12-month trailing momentum (cross-sectional decile)', fontsize=11)
ax.set_ylabel('Average XGBoost predicted score', fontsize=11)
ax.set_title('How XGBoost scores stocks based on their past returns\nby market regime',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Annotate the key insight
ax.annotate('In panic: model\nfavors losers', xy=(0.5, 0.065), fontsize=9,
            color='#E53935', fontweight='bold', ha='center')
ax.annotate('In calm: model\nfavors winners', xy=(8.5, 0.038), fontsize=9,
            color='steelblue', fontweight='bold', ha='center')

plt.tight_layout()
fig.savefig('plots/score_vs_momentum_v2.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: score_vs_momentum_v2.png")

# ── 2-panel: 12-mo and 1-mo ──
fig2, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

# Panel 1: 12-month
ax = axes[0]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned12[binned12['regime'] == regime].sort_values('mom12_decile')
    ax.plot(x, d['mean_score'].values, f'{marker}-', color=color,
            linewidth=2.5, markersize=7, label=regime, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_xlabel('Stocks ranked by 12-month momentum', fontsize=10)
ax.set_ylabel('Average XGBoost predicted score', fontsize=11)
ax.set_title('12-month momentum', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Panel 2: 1-month
ax = axes[1]
for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = binned1[binned1['regime'] == regime].sort_values('mom1_decile')
    ax.plot(x, d['mean_score'].values, f'{marker}-', color=color,
            linewidth=2.5, markersize=7, label=regime, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_xlabel('Stocks ranked by 1-month momentum', fontsize=10)
ax.set_title('1-month momentum', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('How XGBoost scores stocks based on past returns: calm vs panic',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig2.savefig('plots/score_vs_momentum_v2_2panel.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: score_vs_momentum_v2_2panel.png")

from PIL import Image
for name in ['score_vs_momentum_v2', 'score_vs_momentum_v2_2panel']:
    img = Image.open(f'{name}.png')
    img.save(f'{name}.pdf', 'PDF', resolution=150)
print("Saved PDFs")
