"""
landscape_k2_dispersion_plot.py
================================
Within-cluster dispersion view of the k=2 input-landscape clusters,
formatted to match the existing thesis dispersion plot
(plots/thesis/zscore_subregime_dispersion.pdf).

For each input-defined cluster: plots every member month's long-leg
z-profile as a thin line, the cluster centroid as a heavy line, and the
all-calm-months centroid as a dashed reference. Title includes cluster
name, n, and conditional Sharpe.

Output:
- plots/thesis/landscape_k2_dispersion.png
- plots/thesis/landscape_k2_dispersion.pdf
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'
RETURNS_PATH = 'results/thesis/fundamentals_returns.pkl'
RNG_SEED = 42
N_INIT = 20
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])

agg_mean = test.groupby('date')[MOM_COLS].mean()
agg_std = test.groupby('date')[MOM_COLS].std()
pi = test.groupby('date')['pi_filter'].first()

long_z = (pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                      parse_dates=['date'])
          .set_index('date'))

with open(RETURNS_PATH, 'rb') as f:
    rets = pickle.load(f)
m2_ret = rets['baseline_mom_pi']['returns']
m2_ret.index = pd.to_datetime(m2_ret.index)

calm_dates = pi.index[pi <= 0.5]
panic_dates = pi.index[pi > 0.5]

# ----------------------------------------------------------------------
# Cluster (k=2) on means+std features
# ----------------------------------------------------------------------
def cluster_k2(dates):
    feats = pd.concat([
        agg_mean.loc[dates].rename(columns=lambda c: f'{c}_mean'),
        agg_std.loc[dates].rename(columns=lambda c: f'{c}_std'),
    ], axis=1)
    X = StandardScaler().fit_transform(feats.values)
    km = KMeans(n_clusters=2, random_state=RNG_SEED, n_init=N_INIT).fit(X)
    return pd.Series(km.labels_, index=dates, name='cluster')


def relabel(labels, dates, names):
    df = agg_mean.loc[dates].copy()
    df['cluster'] = labels.values
    depth = df.groupby('cluster').mean().mean(axis=1).sort_values()
    order = depth.index.tolist()
    return labels.map({order[i]: names[i] for i in range(len(names))})


calm_lbl = relabel(cluster_k2(calm_dates), calm_dates,
                   ['typical_calm', 'strong_continuation'])
panic_lbl = relabel(cluster_k2(panic_dates), panic_dates,
                    ['widespread_loss', 'humped_panic'])

all_labels = pd.concat([calm_lbl, panic_lbl])

# ----------------------------------------------------------------------
# Conditional Sharpe per cluster
# ----------------------------------------------------------------------
def ann_sharpe(x):
    x = x.dropna()
    if len(x) == 0 or x.std() == 0:
        return np.nan
    return (x.mean() / x.std()) * np.sqrt(12)


def cluster_stats(dates, labels):
    df = pd.DataFrame({'date': dates, 'cluster': labels.values})
    df = df.merge(m2_ret.rename('m2_ret'),
                  left_on='date', right_index=True, how='left')
    return (df.groupby('cluster')
              .agg(n=('date', 'count'),
                   sharpe=('m2_ret', ann_sharpe)))


calm_stats = cluster_stats(calm_dates, calm_lbl)
panic_stats = cluster_stats(panic_dates, panic_lbl)
stats = pd.concat([calm_stats, panic_stats])

# ----------------------------------------------------------------------
# Calm reference centroid (all calm months, long-leg z-profile)
# ----------------------------------------------------------------------
calm_ref = long_z.loc[calm_dates, MOM_COLS].mean().values

# ----------------------------------------------------------------------
# Plot — 1x4 panels, dispersion view
# ----------------------------------------------------------------------
PANEL_ORDER = ['typical_calm', 'strong_continuation',
               'humped_panic', 'widespread_loss']
PRETTY = {
    'widespread_loss':     'Widespread loss',
    'typical_calm':        'Typical calm',
    'humped_panic':        'Humped panic',
    'strong_continuation': 'Strong continuation',
}
REGIME = {
    'typical_calm':        'calm',
    'strong_continuation': 'calm',
    'humped_panic':        'panic',
    'widespread_loss':     'panic',
}
COLORS = {
    'widespread_loss':     '#d62728',  # deep red
    'typical_calm':        '#1f77b4',  # blue
    'humped_panic':        '#2ca02c',  # green (mirrors mild panic in ref plot)
    'strong_continuation': '#6a3d9a',  # purple
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)

YLIM = (-1.5, 1.0)

for ax, name in zip(axes, PANEL_ORDER):
    color = COLORS[name]
    member_dates = all_labels.index[all_labels == name]
    Z = long_z.loc[member_dates, MOM_COLS].values  # n_members x 12
    centroid = Z.mean(axis=0)
    n = stats.loc[name, 'n']
    sharpe = stats.loc[name, 'sharpe']

    # Member-month profiles (thin)
    for row in Z:
        ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.35)

    # Cluster centroid (heavy)
    ax.plot(HORIZONS, centroid, color=color, lw=2.5, marker='o', ms=6,
            label=f'{PRETTY[name]} centroid')

    # Calm reference (dashed black)
    ax.plot(HORIZONS, calm_ref, color='#333', lw=1.5, ls='--',
            label='Calm centroid (reference)')

    ax.axhline(0, color='#888', lw=0.6)
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(*YLIM)
    ax.set_xlabel('Momentum lookback (months)')
    ax.set_title(f'{PRETTY[name]} — {REGIME[name]} '
                 f'(n={int(n)}, Sharpe={sharpe:.2f})')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, frameon=True)

axes[0].set_ylabel('Long-leg cross-sectional z-score')

fig.tight_layout()

png_path = f'{PLOT_DIR}/landscape_k2_dispersion.png'
pdf_path = f'{PLOT_DIR}/landscape_k2_dispersion.pdf'
fig.savefig(png_path, dpi=200, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'Saved: {png_path}')
print(f'Saved: {pdf_path}')

print('\nCluster sizes and conditional Sharpes:')
print(stats.to_string())
