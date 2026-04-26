"""
zscore_long_3d_html.py
======================
Interactive 3D surface of long-leg z-score over (time, momentum horizon).

Input : results/zscore_long_by_month.csv
Output: plots/zscore_long_3d.html
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('results/zscore_long_by_month.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

horizons = np.arange(1, 13)
z = df[[f'mom_{h}' for h in horizons]].values  # (T, 12), raw
t_raw = np.arange(len(df))
dates = df['date'].dt.strftime('%Y-%m').tolist()

vmax = float(np.nanmax(np.abs(z)))

fig = go.Figure(
    data=[
        go.Surface(
            x=horizons,
            y=t_raw,
            z=z,
            colorscale='RdBu_r',
            cmin=-vmax,
            cmax=vmax,
            colorbar=dict(title='z-score'),
            hovertemplate=(
                'Date: %{customdata}<br>'
                'Horizon: %{x} mo<br>'
                'Z-score: %{z:.3f}<extra></extra>'
            ),
            customdata=np.tile(np.array(dates)[:, None], (1, len(horizons))),
        )
    ]
)

# Y-axis tick every January
tick_idx = [i for i, d in enumerate(df['date']) if d.month == 1]
tick_labels = [df['date'].iloc[i].strftime('%Y') for i in tick_idx]

fig.update_layout(
    title='Long-leg cross-sectional z-score: time × momentum horizon',
    scene=dict(
        xaxis=dict(title='Momentum lookback (months)', tickmode='array',
                   tickvals=list(horizons)),
        yaxis=dict(title='Date', tickmode='array',
                   tickvals=tick_idx, ticktext=tick_labels),
        zaxis=dict(title='Z-score'),
        camera=dict(eye=dict(x=1.6, y=-1.8, z=1.0)),
    ),
    width=1100,
    height=750,
    margin=dict(l=0, r=0, t=60, b=0),
)

out = 'plots/zscore_long_3d.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f"Saved: {out}")
