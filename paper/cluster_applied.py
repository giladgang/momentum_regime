"""Applied-window k=4 clustering of the strategy's own long-leg z-curves
(2011-01..2025-11), replacing the thesis-window clustering for the paper.
Method mirrors scripts/cluster_zscore_l2.py: plain KMeans, L2, no
preprocessing, on the 12-d monthly curves; k=4; stability checked across
the thesis seed list (ARI).

Outputs: paper/results/tables/applied_cluster_k4_labels.csv,
         applied_cluster_k4_stats.csv,
         plots/applied_cluster_k4_curves.{png,pdf}
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import walk_report as W                              # noqa: E402
from paper.src import metrics as M                              # noqa: E402

MOMS = [f'mom_{h}' for h in range(1, 13)]
SEEDS = [42, 123, 456, 789, 1011, 1213]      # thesis convention

zc = pd.read_csv('paper/results/tables/applied_zscore_long_by_month.csv',
                 parse_dates=['date']).set_index('date')
X = zc[MOMS].values.astype(float)
labs_by_seed = {}
for s in SEEDS:
    labs_by_seed[s] = KMeans(n_clusters=4, n_init=20,
                             random_state=s).fit_predict(X)
base = labs_by_seed[42]
aris = [adjusted_rand_score(base, labs_by_seed[s]) for s in SEEDS[1:]]
print(f'[cluster] stability: min ARI vs seed-42 = {min(aris):.3f}')

# order clusters by mean curve level: C1 = most winner-tilted ... C4 = deepest loser
order = np.argsort([-X[base == k].mean() for k in range(4)])
remap = {old: new for new, old in enumerate(order)}
labels = np.array([remap[l] for l in base])
out = pd.DataFrame({'date': zc.index, 'cluster': labels})
out.to_csv('paper/results/tables/applied_cluster_k4_labels.csv', index=False)

rets = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
sel = pd.read_csv(C.SELECTIONS_CSV)
_, _, _, bench, pi_m = W._comparator_series(sel)
rr = rets[rets.rule == 'rule_r'].set_index('date')['strat_ret']
lab_s = out.set_index('date')['cluster'].reindex(rr.index)

stats, names = [], {}
for k in range(4):
    m = (lab_s == k).values
    v, b = rr[m], bench.reindex(rr.index)[m]
    a = v - b
    pi_k = float(pi_m.reindex(rr.index)[m].mean())
    stats.append({'cluster': f'C{k+1}', 'n': int(m.sum()),
                  'mean_pi': pi_k, 'mean_z': float(X[labels == k].mean()),
                  'sharpe': M.sharpe(v),
                  'ir': float(a.mean() / a.std() * np.sqrt(12)),
                  'n_2025': int((lab_s[m].index.year == 2025).sum())})
st = pd.DataFrame(stats)
st.to_csv('paper/results/tables/applied_cluster_k4_stats.csv', index=False)
print(st.round(3).to_string(index=False))
y25 = out[out['date'].dt.year >= 2025]
print('\n2025 month assignments:')
print(y25.assign(ym=y25['date'].dt.to_period('M').astype(str))
      [['ym', 'cluster']].to_string(index=False))

COLORS = {0: '#2a78d6', 1: '#3b9c52', 2: '#9469d0', 3: '#e05c4b'}
NAME = {0: 'C1 · winner book', 1: 'C2 · neutral book',
        2: 'C3 · loser tilt', 3: 'C4 · deep-loser book'}
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
H = np.arange(1, 13)
for k, ax in zip(range(4), axes.ravel()):
    mem = X[labels == k]
    for row in mem:
        ax.plot(H, row, color='#999999', lw=0.6, alpha=0.35, zorder=1)
    ax.plot(H, mem.mean(0), color=COLORS[k], lw=2.6, zorder=3)
    ax.axhline(0, color='#666666', lw=0.7, zorder=2)
    r = st.iloc[k]
    ax.set_title(f"{NAME[k]}  (n={int(r['n'])}, {int(r['n_2025'])} in 2025)",
                 fontsize=11, loc='left', fontweight='bold')
    ax.text(0.02, 0.04, f"Sharpe {r['sharpe']:.2f} · IR {r['ir']:+.2f} · "
            f"mean pi {r['mean_pi']:.2f}", transform=ax.transAxes,
            fontsize=8.5, color='#555555')
    ax.grid(True, color='#e8e8e8', lw=0.6, zorder=0)
    ax.set_xticks([1, 3, 6, 9, 12])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel('momentum horizon (months)', fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel('cross-sectional z-score\nof long-leg picks', fontsize=10)
fig.suptitle('Applied strategy: momentum term-structure clusters '
             '(k=4, own picks, 2011-01..2025-11)\n'
             'gray = member months · bold = cluster mean curve',
             fontsize=12, x=0.02, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig('plots/applied_cluster_k4_curves.png', dpi=160)
fig.savefig('plots/applied_cluster_k4_curves.pdf')
print('\nsaved plots/applied_cluster_k4_curves.{png,pdf}')
