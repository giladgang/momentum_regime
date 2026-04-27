"""
zscore_long_short_heatmap.py
============================
Side-by-side heatmaps of long-leg and short-leg z-scores with a shared
pi_filter line panel on the left. No spread panel.

Outputs:
- plots/zscore_long_short_heatmap.png / .pdf
- plots/zscore_long_short_heatmap.html
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

long_df = pd.read_csv('results/thesis/zscore_long_by_month.csv', parse_dates=['date'])
short_df = pd.read_csv('results/thesis/zscore_short_by_month.csv', parse_dates=['date'])

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date')
pi_df = long_df[['date']].merge(pi_monthly, on='date', how='left')
pi = pi_df['pi_filter'].values

horizons = list(range(1, 13))
z_long = long_df[[f'mom_{h}' for h in horizons]].values
z_short = short_df[[f'mom_{h}' for h in horizons]].values
dates = long_df['date']

vmax = float(max(np.nanmax(np.abs(z_long)), np.nanmax(np.abs(z_short))))

# ── Static PNG / PDF ──
y_idx = np.arange(len(long_df)) + 0.5

fig, axes = plt.subplots(
    1, 3, figsize=(16, 14),
    gridspec_kw={'width_ratios': [0.22, 1, 1], 'wspace': 0.06},
    sharey=True,
)
ax_pi, ax_long, ax_short = axes

ax_pi.plot(pi, y_idx, color='#333333', linewidth=1.2)
ax_pi.fill_betweenx(y_idx, 0, pi, color='#E53935', alpha=0.25)
ax_pi.axvline(0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
ax_pi.set_xlim(0, 1)
ax_pi.set_xticks([0, 0.5, 1])
ax_pi.set_xlabel(r'$\pi^{\mathrm{filter}}$', fontsize=11)
ax_pi.set_title('Panic prob.', fontsize=11, fontweight='bold')
ax_pi.grid(axis='x', alpha=0.3)

def _draw(ax, z_mat, title):
    im = ax.imshow(
        z_mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        origin='lower', extent=[0.5, 12.5, 0, len(long_df)],
    )
    ax.set_xticks(horizons)
    ax.set_xlabel('Momentum lookback (months)', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    return im

_draw(ax_long, z_long, 'Long leg')
im = _draw(ax_short, z_short, 'Short leg')

year_ticks = [i for i, d in enumerate(dates) if d.month == 1]
year_labels = [dates.iloc[i].strftime('%Y') for i in year_ticks]
ax_pi.set_yticks(year_ticks)
ax_pi.set_yticklabels(year_labels, fontsize=9)
ax_pi.set_ylabel('Date', fontsize=11)
ax_pi.set_ylim(0, len(long_df))

fig.suptitle('Cross-sectional z-score of selected stocks: long and short legs',
             fontsize=14, fontweight='bold', y=0.995)

cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
cbar.set_label('z-score', fontsize=10)

fig.savefig('plots/zscore_long_short_heatmap.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/zscore_long_short_heatmap.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: plots/zscore_long_short_heatmap.png')
print('Saved: plots/zscore_long_short_heatmap.pdf')

# ── Interactive HTML ──
date_labels = dates.dt.strftime('%Y-%m').tolist()
fig2 = make_subplots(
    rows=1, cols=3,
    column_widths=[0.16, 0.42, 0.42],
    shared_yaxes=True,
    horizontal_spacing=0.015,
    subplot_titles=('Panic prob. π', 'Long leg z', 'Short leg z'),
)

fig2.add_trace(
    go.Scatter(
        x=pi, y=date_labels, mode='lines',
        line=dict(color='#333333', width=1.4),
        fill='tozerox',
        fillcolor='rgba(229,57,53,0.25)',
        hovertemplate='Date: %{y}<br>π: %{x:.3f}<extra></extra>',
        showlegend=False,
    ),
    row=1, col=1,
)
fig2.add_vline(x=0.5, line_dash='dash', line_color='grey',
               line_width=1, row=1, col=1)

for col_idx, (z_mat, label) in enumerate(
    [(z_long, 'Long z'), (z_short, 'Short z')], start=2
):
    fig2.add_trace(
        go.Heatmap(
            z=z_mat, x=horizons, y=date_labels,
            colorscale='RdBu_r', zmin=-vmax, zmax=vmax,
            colorbar=dict(title='z-score') if col_idx == 3 else None,
            showscale=(col_idx == 3),
            hovertemplate=f'Date: %{{y}}<br>Horizon: %{{x}} mo<br>{label}: %{{z:.3f}}<extra></extra>',
        ),
        row=1, col=col_idx,
    )

fig2.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], title_text='π', row=1, col=1)
for c in (2, 3):
    fig2.update_xaxes(title_text='Momentum lookback (months)',
                      tickmode='array', tickvals=horizons, row=1, col=c)
fig2.update_yaxes(autorange='reversed', title_text='Date', row=1, col=1)
for c in (2, 3):
    fig2.update_yaxes(autorange='reversed', row=1, col=c)

fig2.update_layout(
    title='Cross-sectional z-score by month × horizon: long and short legs',
    width=1300, height=1200,
    margin=dict(l=80, r=40, t=80, b=60),
)
fig2.write_html('plots/zscore_long_short_heatmap.html', include_plotlyjs='cdn')
print('Saved: plots/zscore_long_short_heatmap.html')
