"""
Long leg momentum profile across all 12 horizons, calm vs panic. Bar chart.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

test_pi['leg'] = 'middle'
for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10: continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test_pi.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'

long_stocks = test_pi[test_pi['leg'] == 'long']
mom_cols = [f'mom_{h}' for h in range(1, 13)]

calm_vals = long_stocks[long_stocks['regime'] == 'Calm'].groupby('date')[mom_cols].median().mean().values * 100
panic_vals = long_stocks[long_stocks['regime'] == 'Panic'].groupby('date')[mom_cols].median().mean().values * 100

x = np.arange(1, 13)
w = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - w/2, calm_vals, w, label='Calm (108 months)', color='steelblue', alpha=0.85, edgecolor='white')
ax.bar(x + w/2, panic_vals, w, label='Panic (59 months)', color='#E53935', alpha=0.85, edgecolor='white')

ax.axhline(0, color='black', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=10)
ax.set_xlabel('Lookback horizon (months)', fontsize=12)
ax.set_ylabel('Median trailing momentum of long leg (%)', fontsize=12)
ax.set_title('What M2 buys: momentum profile of the long leg by regime',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig('plots/long_leg_all_horizons_bar.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: long_leg_all_horizons_bar.png")

from PIL import Image
img = Image.open('long_leg_all_horizons_bar.png')
img.save('long_leg_all_horizons_bar.pdf', 'PDF', resolution=150)
print("Saved: long_leg_all_horizons_bar.pdf")
