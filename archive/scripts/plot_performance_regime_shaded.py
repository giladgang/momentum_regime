"""
Generate cumulative wealth plot with panic regime shading.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load artefacts
with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

strategies_lo = artefacts['strategies_lo']
r_mkt = artefacts['r_mkt']
test = artefacts['test'].copy()

# Get pi_filter monthly
pi_monthly = test[['date', 'pi_filter']].drop_duplicates('date').sort_values('date')
pi_monthly = pi_monthly.set_index('date')

# Identify panic months (pi >= 0.5)
panic_months = pi_monthly[pi_monthly['pi_filter'] >= 0.5].index

# Strategies to plot
strats = {
    'Market':              r_mkt,
    'Fixed 12-mo mom':     strategies_lo['Fixed 12-mo mom'],
    'M0: Formula':         strategies_lo['Method 0: Formula'],
    'M1: LR':              strategies_lo['Method 1: LR'],
    'M2: XGB':             strategies_lo['Method 2: XGB'],
}

colors = {
    'Market':          ('black',      '--', 1.3),
    'Fixed 12-mo mom': ('steelblue',  '--', 1.0),
    'M0: Formula':     ('grey',       '-',  1.0),
    'M1: LR':          ('crimson',    '-',  1.6),
    'M2: XGB':         ('darkorange', '-',  1.6),
}

fig, ax = plt.subplots(figsize=(12, 6.5))

# Shade panic periods
for i, dt in enumerate(panic_months):
    # Shade from this month to ~1 month later
    ax.axvspan(dt, dt + pd.DateOffset(months=1),
               color='#FFE0E0', alpha=0.6,
               label='Panic regime' if i == 0 else None)

# Plot cumulative wealth
for name, r in strats.items():
    c, ls, lw = colors[name]
    r = r.dropna().sort_index()
    cum = (1 + r).cumprod()
    terminal = cum.iloc[-1]
    ax.plot(r.index, cum, color=c, linestyle=ls, linewidth=lw,
            label=f'{name} (${terminal:.1f})')

ax.set_yscale('log')
ax.set_ylabel('Cumulative wealth (log scale, \\$1 invested)', fontsize=11)
ax.set_xlabel('')
ax.set_title('Out-of-sample portfolio performance, January 2011 -- November 2025',
             fontsize=12, fontweight='bold')

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.0f}' if y >= 1 else f'${y:.1f}'))
ax.set_ylim(bottom=0.7)

# Legend
handles, labels = ax.get_legend_handles_labels()
# Move "Panic regime" to the end
panic_idx = labels.index('Panic regime')
handles.append(handles.pop(panic_idx))
labels.append(labels.pop(panic_idx))
ax.legend(handles, labels, fontsize=9, loc='upper left', framealpha=0.9)

ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig('cs_performance_regime_shaded.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved: cs_performance_regime_shaded.png")
