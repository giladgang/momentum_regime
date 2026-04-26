"""
zscore_long_3d.py
=================
3D surface plot of the long-leg z-score over (time, momentum horizon).

Input : results/zscore_long_by_month.csv
Output: plots/zscore_long_3d.png
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

df = pd.read_csv('results/zscore_long_by_month.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

horizons = np.arange(1, 13)
z = df[[f'mom_{h}' for h in horizons]].values  # shape (T, 12)
t_idx = np.arange(len(df))

H, T = np.meshgrid(horizons, t_idx)

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

vmax = np.nanmax(np.abs(z))
surf = ax.plot_surface(
    T, H, z,
    cmap=cm.RdBu_r,
    vmin=-vmax, vmax=vmax,
    linewidth=0, antialiased=True, rstride=1, cstride=1, alpha=0.95,
)

# Year ticks
years = df['date'].dt.year.values
year_ticks = []
year_labels = []
for y in range(years.min(), years.max() + 1, 2):
    idxs = np.where(years == y)[0]
    if len(idxs):
        year_ticks.append(idxs[0])
        year_labels.append(str(y))
ax.set_xticks(year_ticks)
ax.set_xticklabels(year_labels, fontsize=9)

ax.set_yticks(horizons)
ax.set_yticklabels([str(h) for h in horizons], fontsize=9)

ax.set_xlabel('Date', fontsize=11, labelpad=12)
ax.set_ylabel('Momentum lookback (months)', fontsize=11, labelpad=10)
ax.set_zlabel('Long-leg z-score', fontsize=11, labelpad=6)

ax.set_title('Long-leg cross-sectional z-score: time x momentum horizon',
             fontsize=13, fontweight='bold', pad=14)

ax.view_init(elev=28, azim=-62)

cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
cbar.set_label('z-score', fontsize=10)

plt.tight_layout()
out = 'plots/zscore_long_3d.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
