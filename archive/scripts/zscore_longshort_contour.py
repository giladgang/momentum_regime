"""
zscore_longshort_contour.py
===========================
Filled contour plot of the long-minus-short z-score over (time, horizon),
with a pi_filter line panel on the left. Same style as
zscore_long_contour.py but for the L-S spread.

Input  : results/zscore_longshort_by_month.csv
         data/panel_with_regimes.parquet
Outputs: plots/zscore_longshort_contour.png / .pdf
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('results/zscore_longshort_by_month.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date')
df = df.merge(pi_monthly, on='date', how='left')

horizons = np.arange(1, 13)
z = df[[f'mom_{h}' for h in horizons]].values
t_idx = np.arange(len(df))
dates = df['date']
pi = df['pi_filter'].values

# Cap at 95th percentile magnitude so the colour scale isn't dominated by outliers.
vmax = float(np.nanpercentile(np.abs(z), 95))
z_plot = np.clip(z, -vmax, vmax)

H, T = np.meshgrid(horizons, t_idx)
levels = np.linspace(-vmax, vmax, 21)

fig, (ax_pi, ax) = plt.subplots(
    1, 2, figsize=(12, 14),
    gridspec_kw={'width_ratios': [0.22, 1], 'wspace': 0.04},
    sharey=True,
)

# π panel
ax_pi.plot(pi, t_idx + 0.5, color='#333333', linewidth=1.2)
ax_pi.fill_betweenx(t_idx + 0.5, 0, pi, color='#E53935', alpha=0.25)
ax_pi.axvline(0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
ax_pi.set_xlim(0, 1)
ax_pi.set_xticks([0, 0.5, 1])
ax_pi.set_xlabel(r'$\pi^{\mathrm{filter}}$', fontsize=11)
ax_pi.set_title('Panic prob.', fontsize=10, fontweight='bold')
ax_pi.grid(axis='x', alpha=0.3)

# Contour
cf = ax.contourf(H, T, z_plot, levels=levels, cmap='RdBu_r', extend='both')
cl = ax.contour(H, T, z_plot, levels=levels[::4], colors='black',
                linewidths=0.4, alpha=0.5)
ax.clabel(cl, fmt='%.1f', fontsize=7, inline=True)

ax.set_xticks(horizons)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)

year_ticks = [i for i, d in enumerate(dates) if d.month == 1]
year_labels = [dates.iloc[i].strftime('%Y') for i in year_ticks]
ax_pi.set_yticks(year_ticks)
ax_pi.set_yticklabels(year_labels, fontsize=9)
ax_pi.set_ylabel('Date', fontsize=11)
ax_pi.set_ylim(0, len(df))

fig.suptitle('Long MINUS short cross-sectional z-score: filled contour',
             fontsize=13, fontweight='bold', y=0.995)

cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, extend='both')
cbar.set_label('z-score spread (long − short)', fontsize=10)

plt.tight_layout()
fig.savefig('plots/zscore_longshort_contour.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/zscore_longshort_contour.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: plots/zscore_longshort_contour.png')
print('Saved: plots/zscore_longshort_contour.pdf')
