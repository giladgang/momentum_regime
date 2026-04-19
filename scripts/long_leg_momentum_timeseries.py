"""
Time series of average 12-month momentum of stocks in M2's long leg,
with panic regime shading. Shows the flip: positive in calm, negative in panic.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MOM_FEATURES

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

# Use production scores
test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

# Monthly median 12-month momentum of long leg
long_stocks = test[test['leg'] == 'long']
monthly_mom = long_stocks.groupby('date')['mom_12'].median() * 100

# Get pi_filter for shading
pi = pi_monthly.reindex(monthly_mom.index)['pi_filter']

# ── Plot ──
fig, ax = plt.subplots(figsize=(13, 5.5))

ax.plot(monthly_mom.index, monthly_mom.values, color='steelblue', linewidth=1.5, alpha=0.9)
ax.axhline(0, color='black', linewidth=0.6)

# Shade panic periods
panic_months = pi[pi >= 0.5].index
for i, date in enumerate(panic_months):
    # Find contiguous blocks
    if i == 0 or (date - panic_months[i-1]).days > 45:
        start = date
    if i == len(panic_months) - 1 or (panic_months[i+1] - date).days > 45:
        ax.axvspan(start, date, alpha=0.15, color='#E53935', zorder=0)

# Add regime labels
ax.text(pd.Timestamp('2014-06-01'), monthly_mom.max() * 0.85,
        'Calm: buys winners\n(positive momentum)',
        fontsize=10, color='steelblue', fontweight='bold', ha='center')
ax.text(pd.Timestamp('2020-04-01'), monthly_mom.min() * 0.7,
        'Panic: buys losers\n(negative momentum)',
        fontsize=10, color='#E53935', fontweight='bold', ha='center')

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Median 12-month momentum of long leg (%)', fontsize=11)
ax.set_title('What M2 buys over time: momentum of the long leg flips with market regime',
             fontsize=12, fontweight='bold')

# Add legend for shading
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#E53935', alpha=0.15, label='Panic regime')],
          fontsize=10, loc='lower left')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig('plots/long_leg_momentum_ts.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: long_leg_momentum_ts.png")

from PIL import Image
img = Image.open('long_leg_momentum_ts.png')
img.save('long_leg_momentum_ts.pdf', 'PDF', resolution=150)
print("Saved: long_leg_momentum_ts.pdf")
