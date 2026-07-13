"""Exploratory: the four canonical k=4 momentum term-structure clusters
(thesis objects: 12-d cross-sectional z-curves of the production long-leg
picks, 2011-01..2024-11), annotated with the APPLIED long-only strategy's
per-cluster performance (pi rule_r, gross, from paper/results).

Output: plots/cluster_k4_curves_applied.png (exploratory, not thesis).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

curves = pd.read_csv('results/thesis/zscore_long_by_month.csv',
                     parse_dates=['date'])
labels = pd.read_csv('results/thesis/cluster_k4_member_dates.csv',
                     parse_dates=['date'])
df = curves.merge(labels, on='date', how='inner')
moms = [f'mom_{h}' for h in range(1, 13)]
H = np.arange(1, 13)

# applied per-cluster stats (pi rule_r, computed 2026-07-13 from
# paper/results/walk_returns.csv + canonical member dates)
STATS = {0: ('C1 · calm',           53, 1.17, 0.16, 0.15),
         1: ('C2 · mild stress',    62, 0.33, -0.16, 0.32),
         2: ('C3 · stress/recovery', 31, 1.55, 0.64, 0.27),
         3: ('C4 · deep panic',     21, 0.66, 0.91, 0.78)}
COLORS = {0: '#2a78d6', 1: '#3b9c52', 2: '#9469d0', 3: '#e05c4b'}

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
ymin = df[moms].min().min()
ymax = df[moms].max().max()
for c, ax in zip(range(4), axes.ravel()):
    g = df[df['cluster'] == c]
    for _, row in g.iterrows():
        ax.plot(H, row[moms].values.astype(float), color='#999999',
                lw=0.6, alpha=0.35, zorder=1)
    centroid = g[moms].mean().values.astype(float)
    ax.plot(H, centroid, color=COLORS[c], lw=2.6, zorder=3)
    ax.axhline(0, color='#666666', lw=0.7, zorder=2)
    name, n, sh, ir, pi = STATS[c]
    ax.set_title(f'{name}  (n={n})', fontsize=11, color='#222222',
                 loc='left', fontweight='bold')
    ax.text(0.02, 0.04,
            f'applied long-only (pi rule_r):  Sharpe {sh:.2f} · '
            f'IR {ir:+.2f} · mean pi {pi:.2f}',
            transform=ax.transAxes, fontsize=8.5, color='#555555')
    ax.grid(True, color='#e8e8e8', lw=0.6, zorder=0)
    ax.set_xticks([1, 3, 6, 9, 12])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel('momentum horizon (months)', fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel('cross-sectional z-score\nof long-leg picks', fontsize=10)
fig.suptitle('Momentum term-structure clusters (canonical k=4, 2011-01..2024-11)\n'
             'gray = member months · bold = cluster mean curve',
             fontsize=12, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = 'plots/cluster_k4_curves_applied.png'
fig.savefig(out, dpi=160)
fig.savefig(out.replace('.png', '.pdf'))
print('saved', out, '+ .pdf | months plotted:', len(df),
      '| y-range', round(ymin, 2), '..', round(ymax, 2))
