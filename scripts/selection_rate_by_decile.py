"""
For each cross-sectional momentum decile, what fraction of stocks
does M2 select for its long leg? Calm vs panic.

This directly shows the inversion: in calm, winners go long;
in panic, losers go long.
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

with open('cs_artefacts_data.pkl', 'rb') as f:
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

# Assign long/short legs and momentum deciles within each month
n_bins = 10
records = []

for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score'].dropna()
    if len(nyse) < 10:
        continue

    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    regime = grp['regime'].iloc[0]

    # Assign momentum decile within this month
    grp = grp.copy()
    grp['mom12_decile'] = pd.qcut(grp['mom_12'].rank(method='first'), n_bins,
                                   labels=list(range(1, n_bins + 1)))

    for _, row in grp.iterrows():
        if pd.isna(row['mom12_decile']):
            continue
        in_long = 1 if row['score'] >= hi else 0
        in_short = 1 if row['score'] <= lo else 0
        records.append({
            'regime': regime,
            'mom12_decile': int(row['mom12_decile']),
            'in_long': in_long,
            'in_short': in_short,
        })

df = pd.DataFrame(records)

# Compute: for each (regime, decile), what % of stocks are selected for long?
rates = df.groupby(['regime', 'mom12_decile']).agg(
    long_rate=('in_long', 'mean'),
    short_rate=('in_short', 'mean'),
    count=('in_long', 'size'),
).reset_index()

# Print
print(f"\n{'Decile':>7s} {'Calm Long%':>10s} {'Calm Short%':>12s} {'Panic Long%':>12s} {'Panic Short%':>13s}")
for d in range(1, 11):
    c = rates[(rates['regime'] == 'Calm') & (rates['mom12_decile'] == d)]
    p = rates[(rates['regime'] == 'Panic') & (rates['mom12_decile'] == d)]
    cl = c['long_rate'].values[0] * 100 if len(c) > 0 else 0
    cs = c['short_rate'].values[0] * 100 if len(c) > 0 else 0
    pl = p['long_rate'].values[0] * 100 if len(p) > 0 else 0
    ps = p['short_rate'].values[0] * 100 if len(p) > 0 else 0
    print(f"  {d:>5d} {cl:>9.1f}% {cs:>11.1f}% {pl:>11.1f}% {ps:>12.1f}%")

# ── Plot: long selection rate by decile, calm vs panic ──
x_labels = ['Biggest\nlosers', '2', '3', '4', '5', '6', '7', '8', '9', 'Biggest\nwinners']
x = np.arange(n_bins)

fig, ax = plt.subplots(figsize=(10, 6))

for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = rates[rates['regime'] == regime].sort_values('mom12_decile')
    ax.plot(x, d['long_rate'].values * 100, f'{marker}-', color=color,
            linewidth=2.5, markersize=8, label=f'{regime}', alpha=0.85)

ax.axhline(10, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax.text(9.3, 10.5, '10% baseline\n(random selection)', fontsize=8, color='grey', ha='right')

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_xlabel('Stocks ranked by 12-month trailing momentum (within-month decile)', fontsize=11)
ax.set_ylabel('% of stocks selected for the long leg', fontsize=12)
ax.set_title('Which stocks does M2 buy? Selection rate by momentum decile',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig('selection_rate_by_decile.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: selection_rate_by_decile.png")

# ── Also: net selection (long rate minus short rate) ──
fig2, ax = plt.subplots(figsize=(10, 6))

for regime, color, marker in [('Calm', 'steelblue', 'o'), ('Panic', '#E53935', 's')]:
    d = rates[rates['regime'] == regime].sort_values('mom12_decile')
    net = (d['long_rate'].values - d['short_rate'].values) * 100
    ax.plot(x, net, f'{marker}-', color=color,
            linewidth=2.5, markersize=8, label=f'{regime}', alpha=0.85)

ax.axhline(0, color='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_xlabel('Stocks ranked by 12-month trailing momentum (within-month decile)', fontsize=11)
ax.set_ylabel('Net selection rate (% long minus % short)', fontsize=12)
ax.set_title('M2 net position by momentum decile: calm vs panic',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig2.savefig('net_selection_by_decile.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved: net_selection_by_decile.png")

from PIL import Image
for name in ['selection_rate_by_decile', 'net_selection_by_decile']:
    img = Image.open(f'{name}.png')
    img.save(f'{name}.pdf', 'PDF', resolution=150)
print("Saved PDFs")
