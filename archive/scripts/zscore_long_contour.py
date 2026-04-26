"""
zscore_long_contour.py
======================
Filled contour plot of long-leg z-score over (time, momentum horizon).

Input : results/zscore_long_by_month.csv
Output: plots/zscore_long_contour.png
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('results/zscore_long_by_month.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

horizons = np.arange(1, 13)
z = df[[f'mom_{h}' for h in horizons]].values  # (T, 12)
t_idx = np.arange(len(df))

H, T = np.meshgrid(horizons, t_idx)
vmax = float(np.nanmax(np.abs(z)))
levels = np.linspace(-vmax, vmax, 21)

fig, ax = plt.subplots(figsize=(10, 14))
cf = ax.contourf(H, T, z, levels=levels, cmap='RdBu_r', extend='both')
cl = ax.contour(H, T, z, levels=levels[::4], colors='black', linewidths=0.4, alpha=0.5)
ax.clabel(cl, fmt='%.1f', fontsize=7, inline=True)

ax.set_xticks(horizons)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)

year_ticks = [i for i, d in enumerate(df['date']) if d.month == 1]
year_labels = [df['date'].iloc[i].strftime('%Y') for i in year_ticks]
ax.set_yticks(year_ticks)
ax.set_yticklabels(year_labels, fontsize=9)
ax.set_ylabel('Date', fontsize=11)
ax.set_title('Long-leg cross-sectional z-score: filled contour',
             fontsize=13, fontweight='bold')

cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('z-score', fontsize=10)

plt.tight_layout()
fig.savefig('plots/zscore_long_contour.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/zscore_long_contour.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: plots/zscore_long_contour.png')
print('Saved: plots/zscore_long_contour.pdf')
