"""
zscore_longshort_heatmap.py
===========================
Build a monthly long-minus-short z-score table and render a heatmap
with the same pi_filter line panel on the left.

Outputs:
- results/zscore_longshort_by_month.csv
- plots/zscore_longshort_heatmap.png / .pdf
- plots/zscore_longshort_heatmap.html
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load artefacts and assign legs ──
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])

test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

# ── Monthly long-minus-short z-score per horizon ──
horizons = list(range(1, 13))
rows = []
for date, grp in test.groupby('date'):
    row = {'date': date}
    long_mask = grp['leg'] == 'long'
    short_mask = grp['leg'] == 'short'
    for h in horizons:
        col = f'mom_{h}'
        m = grp[col].mean()
        s = grp[col].std()
        if s > 0 and long_mask.any() and short_mask.any():
            z_long = ((grp.loc[long_mask, col] - m) / s).mean()
            z_short = ((grp.loc[short_mask, col] - m) / s).mean()
            row[f'mom_{h}'] = z_long - z_short
        else:
            row[f'mom_{h}'] = np.nan
    rows.append(row)

df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
df.to_csv('results/zscore_longshort_by_month.csv', index=False, float_format='%.4f')
print(f"Saved: results/zscore_longshort_by_month.csv  ({len(df)} rows)")

# ── Merge pi_filter for the left panel ──
panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date')
df = df.merge(pi_monthly, on='date', how='left')

z = df[[f'mom_{h}' for h in horizons]].values
dates = df['date']
pi = df['pi_filter'].values
vmax = float(np.nanmax(np.abs(z)))

# ── Static PNG / PDF ──
y_idx = np.arange(len(df)) + 0.5

fig, (ax_pi, ax) = plt.subplots(
    1, 2, figsize=(12, 14),
    gridspec_kw={'width_ratios': [0.22, 1], 'wspace': 0.04},
    sharey=True,
)

ax_pi.plot(pi, y_idx, color='#333333', linewidth=1.2)
ax_pi.fill_betweenx(y_idx, 0, pi, color='#E53935', alpha=0.25)
ax_pi.axvline(0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
ax_pi.set_xlim(0, 1)
ax_pi.set_xticks([0, 0.5, 1])
ax_pi.set_xlabel(r'$\pi^{\mathrm{filter}}$', fontsize=11)
ax_pi.set_title('Panic prob.', fontsize=10, fontweight='bold')
ax_pi.grid(axis='x', alpha=0.3)

im = ax.imshow(
    z, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
    origin='lower', extent=[0.5, 12.5, 0, len(df)],
)
ax.set_xticks(horizons)
ax.set_xlabel('Momentum lookback horizon (months)', fontsize=11)

year_ticks = [i for i, d in enumerate(dates) if d.month == 1]
year_labels = [dates.iloc[i].strftime('%Y') for i in year_ticks]
ax_pi.set_yticks(year_ticks)
ax_pi.set_yticklabels(year_labels, fontsize=9)
ax_pi.set_ylabel('Date', fontsize=11)
ax_pi.set_ylim(0, len(df))

fig.suptitle('Long MINUS short cross-sectional z-score\nby month and momentum horizon',
             fontsize=13, fontweight='bold', y=0.995)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('z-score spread (long − short)', fontsize=10)

plt.tight_layout()
fig.savefig('plots/zscore_longshort_heatmap.png', dpi=150, bbox_inches='tight')
fig.savefig('plots/zscore_longshort_heatmap.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved: plots/zscore_longshort_heatmap.png')
print('Saved: plots/zscore_longshort_heatmap.pdf')

# ── Interactive HTML ──
date_labels = dates.dt.strftime('%Y-%m').tolist()
fig2 = make_subplots(
    rows=1, cols=2,
    column_widths=[0.18, 0.82],
    shared_yaxes=True,
    horizontal_spacing=0.015,
    subplot_titles=('Panic probability π', 'Long − short z-score'),
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
fig2.add_trace(
    go.Heatmap(
        z=z, x=horizons, y=date_labels,
        colorscale='RdBu_r', zmin=-vmax, zmax=vmax,
        colorbar=dict(title='L−S z'),
        hovertemplate='Date: %{y}<br>Horizon: %{x} mo<br>L−S z: %{z:.3f}<extra></extra>',
    ),
    row=1, col=2,
)
fig2.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], title_text='π', row=1, col=1)
fig2.update_xaxes(title_text='Momentum lookback (months)',
                  tickmode='array', tickvals=horizons, row=1, col=2)
fig2.update_yaxes(autorange='reversed', title_text='Date', row=1, col=1)
fig2.update_yaxes(autorange='reversed', row=1, col=2)
fig2.update_layout(
    title='Long − short z-score by month × momentum horizon',
    width=1100, height=1200,
    margin=dict(l=80, r=40, t=80, b=60),
)
fig2.write_html('plots/zscore_longshort_heatmap.html', include_plotlyjs='cdn')
print('Saved: plots/zscore_longshort_heatmap.html')
