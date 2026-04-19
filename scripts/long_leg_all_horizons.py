"""
Long leg momentum profile across all 12 horizons, calm vs panic.
Single panel, production scores.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, sys, os

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
    test_pi.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

long_stocks = test_pi[test_pi['leg'] == 'long']

# Monthly median at each horizon, then average across months
x = np.arange(1, 13)
mom_cols = [f'mom_{h}' for h in range(1, 13)]

calm_vals = long_stocks[long_stocks['regime'] == 'Calm'].groupby('date')[mom_cols].median().mean().values * 100
panic_vals = long_stocks[long_stocks['regime'] == 'Panic'].groupby('date')[mom_cols].median().mean().values * 100

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, calm_vals, 'o-', color='steelblue', linewidth=2.5, markersize=8,
        label='Calm (108 months)', alpha=0.85)
ax.plot(x, panic_vals, 's-', color='#E53935', linewidth=2.5, markersize=8,
        label='Panic (59 months)', alpha=0.85)

ax.fill_between(x, calm_vals, panic_vals, alpha=0.08, color='grey')
ax.axhline(0, color='black', linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels([f'{h}' for h in range(1, 13)], fontsize=10)
ax.set_xlabel('Lookback horizon (months)', fontsize=12)
ax.set_ylabel('Median trailing momentum of long leg (%)', fontsize=12)
ax.set_title('What M2 buys: momentum profile of the long leg by regime',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig('plots/long_leg_all_horizons.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: long_leg_all_horizons.png")

from PIL import Image
img = Image.open('long_leg_all_horizons.png')
img.save('long_leg_all_horizons.pdf', 'PDF', resolution=150)
print("Saved: long_leg_all_horizons.pdf")

# Print values
print(f"\n{'Horizon':>8s} {'Calm Long':>10s} {'Panic Long':>11s}")
for i, h in enumerate(range(1, 13)):
    print(f"  mom_{h:<3d} {calm_vals[i]:>+9.1f}% {panic_vals[i]:>+10.1f}%")
