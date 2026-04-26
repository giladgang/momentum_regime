"""
zscore_long_animation.py
========================
Animated monthly profile: z-score vs horizon, one frame per month.
Panic-regime frames switch the line colour to red.

Input  : results/zscore_long_by_month.csv
         data/panel_with_regimes.parquet
Outputs: plots/zscore_long_animation.html   (interactive, scrubber)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('results/zscore_long_by_month.csv', parse_dates=['date'])
panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date')
df = df.merge(pi_monthly, on='date', how='left').sort_values('date').reset_index(drop=True)

horizons = list(range(1, 13))
z = df[[f'mom_{h}' for h in horizons]].values
vmax = float(np.nanmax(np.abs(z))) * 1.05

frames = []
for i in range(len(df)):
    is_panic = df['pi_filter'].iloc[i] >= 0.5
    color = '#E53935' if is_panic else '#2196F3'
    label = 'Panic' if is_panic else 'Calm'
    date_str = df['date'].iloc[i].strftime('%Y-%m')
    frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=horizons,
                    y=z[i],
                    mode='lines+markers',
                    line=dict(color=color, width=3),
                    marker=dict(size=9, color=color),
                    name=date_str,
                )
            ],
            name=date_str,
            layout=go.Layout(
                title=f'Long-leg z-score by horizon — {date_str} ({label}, '
                      f'π={df["pi_filter"].iloc[i]:.2f})'
            ),
        )
    )

fig = go.Figure(
    data=[
        go.Scatter(
            x=horizons, y=z[0],
            mode='lines+markers',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=9),
        )
    ],
    frames=frames,
)

# Horizontal zero line
fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.4)

slider_steps = [
    dict(
        method='animate',
        args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                             mode='immediate', transition=dict(duration=0))],
        label=f.name,
    )
    for f in frames
]

fig.update_layout(
    title=f'Long-leg z-score by horizon — {df["date"].iloc[0].strftime("%Y-%m")} '
          f'({"Panic" if df["pi_filter"].iloc[0] >= 0.5 else "Calm"}, '
          f'π={df["pi_filter"].iloc[0]:.2f})',
    xaxis=dict(title='Momentum lookback (months)', tickmode='array', tickvals=horizons),
    yaxis=dict(title='Z-score', range=[-vmax, vmax]),
    width=950, height=600,
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        y=1.12, x=1.0, xanchor='right', yanchor='top',
        buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=250, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0))]),
        ],
    )],
    sliders=[dict(
        steps=slider_steps,
        transition=dict(duration=0),
        x=0, y=0, currentvalue=dict(prefix='Month: ', font=dict(size=14)),
        len=1.0,
    )],
)

out = 'plots/zscore_long_animation.html'
fig.write_html(out, include_plotlyjs='cdn')
print(f'Saved: {out}')
