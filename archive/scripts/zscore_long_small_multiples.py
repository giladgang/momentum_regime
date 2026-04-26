"""
zscore_long_small_multiples.py
==============================
Small-multiple line plots: one panel per momentum horizon, z-score over time,
with panic months shaded.

Input  : results/zscore_long_by_month.csv
         data/panel_with_regimes.parquet (for pi_filter)
Output : plots/zscore_long_small_multiples.png
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('results/zscore_long_by_month.csv', parse_dates=['date'])
panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date')
df = df.merge(pi_monthly, on='date', how='left').sort_values('date').reset_index(drop=True)

horizons = list(range(1, 13))

vmax = float(np.nanmax(np.abs(df[[f'mom_{h}' for h in horizons]].values)))
ymin, ymax = -vmax * 1.05, vmax * 1.05

fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
axes = axes.flatten()

panic_mask = (df['pi_filter'] >= 0.5).values
# Identify contiguous panic spans for axvspan
spans = []
start = None
for i, p in enumerate(panic_mask):
    if p and start is None:
        start = i
    elif not p and start is not None:
        spans.append((start, i - 1))
        start = None
if start is not None:
    spans.append((start, len(panic_mask) - 1))

for idx, h in enumerate(horizons):
    ax = axes[idx]
    for s, e in spans:
        ax.axvspan(df['date'].iloc[s], df['date'].iloc[e],
                   color='#E53935', alpha=0.12, linewidth=0)
    ax.plot(df['date'], df[f'mom_{h}'], color='#2196F3', linewidth=1.4)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    ax.set_title(f'mom_{h}', fontsize=11, fontweight='bold')
    ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)

# Only bottom row gets x labels; only left column gets y labels
for ax in axes[-4:]:
    ax.tick_params(axis='x', rotation=0, labelsize=8)
for ax in axes[::4]:
    ax.set_ylabel('Z-score', fontsize=10)

fig.suptitle('Long-leg z-score over time, by momentum horizon (panic months shaded)',
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig('plots/zscore_long_small_multiples.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/zscore_long_small_multiples.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: plots/zscore_long_small_multiples.png')
print('Saved: plots/zscore_long_small_multiples.pdf')
