"""
PNG preview of the time-smoothed 3D surface, so it can be shown inline.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline

df = pd.read_csv('results/zscore_long_by_month.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

horizons = np.arange(1, 13)
z_raw = df[[f'mom_{h}' for h in horizons]].values
t_raw = np.arange(len(df))

t_fine = np.linspace(t_raw[0], t_raw[-1], len(df) * 6)
spline = RectBivariateSpline(t_raw, horizons, z_raw, kx=3, ky=3)
z = spline(t_fine, horizons)

H, T = np.meshgrid(horizons, t_fine)

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

vmax = float(np.nanmax(np.abs(z)))
surf = ax.plot_surface(
    T, H, z,
    cmap=cm.RdBu_r, vmin=-vmax, vmax=vmax,
    linewidth=0, antialiased=True, rstride=1, cstride=1,
)

year_ticks, year_labels = [], []
for y in range(df['date'].dt.year.min(), df['date'].dt.year.max() + 1, 2):
    idxs = np.where(df['date'].dt.year.values == y)[0]
    if len(idxs):
        year_ticks.append(idxs[0])
        year_labels.append(str(y))

ax.set_xticks(year_ticks)
ax.set_xticklabels(year_labels, fontsize=9)
ax.set_yticks(horizons)
ax.set_yticklabels([str(h) for h in horizons], fontsize=9)
ax.set_xlabel('Date', labelpad=12)
ax.set_ylabel('Momentum lookback (months)', labelpad=10)
ax.set_zlabel('Long-leg z-score', labelpad=6)
ax.set_title('Long-leg z-score (time smoothed)', fontweight='bold', pad=14)
ax.view_init(elev=28, azim=-62)

fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08).set_label('z-score')
fig.savefig('plots/zscore_long_3d_smooth.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: plots/zscore_long_3d_smooth.png')
