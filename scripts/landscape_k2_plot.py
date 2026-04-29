"""
landscape_k2_plot.py
=====================
Produces the k=2 cluster visualisation for both regimes. For each regime
we cluster months on the input landscape (means+std, standardised, k=2),
then plot:

  - the input landscape centroid per cluster (typical-stock raw mom_h)
  - the output selection centroid per cluster (long-leg cross-sectional
    z-score profile averaged over months in that input cluster)

This shows: "given a market state of type X, what selection shape does
the model produce on average?"

2x2 figure (calm input, calm output, panic input, panic output) saved to
plots/thesis/landscape_k2_clusters.{png,pdf}.
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
RNG_SEED = 42
N_INIT = 20
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Load data
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

calm_dates = pi.index[pi <= 0.5]
panic_dates = pi.index[pi > 0.5]

# ----------------------------------------------------------------------
# Cluster (k=2) per regime on means+std features
# ----------------------------------------------------------------------
def cluster_k2(dates):
    feats = pd.concat([
        agg_mean.loc[dates].rename(columns=lambda c: f'{c}_mean'),
        agg_std.loc[dates].rename(columns=lambda c: f'{c}_std'),
    ], axis=1)
    X = StandardScaler().fit_transform(feats.values)
    km = KMeans(n_clusters=2, random_state=RNG_SEED, n_init=N_INIT).fit(X)
    return pd.Series(km.labels_, index=dates, name='cluster')


calm_lbl = cluster_k2(calm_dates)
panic_lbl = cluster_k2(panic_dates)


def relabel_by_depth(labels, dates, naming):
    """Order clusters by mean landscape depth across the 12 horizons; map
    the most-negative cluster to naming[0] and most-positive to naming[-1]."""
    df = agg_mean.loc[dates].copy()
    df['cluster'] = labels.values
    depth = df.groupby('cluster').mean().mean(axis=1).sort_values()
    order = depth.index.tolist()
    return labels.map({order[i]: naming[i] for i in range(len(naming))})


calm_lbl = relabel_by_depth(calm_lbl, calm_dates,
                            ['typical_calm', 'strong_continuation'])
panic_lbl = relabel_by_depth(panic_lbl, panic_dates,
                             ['widespread_loss', 'humped_panic'])

# ----------------------------------------------------------------------
# Compute centroids in raw / z-score units
# ----------------------------------------------------------------------
def centroids(dates, labels, source_df):
    """Mean of source_df rows grouped by labels."""
    df = source_df.loc[dates].copy()
    df['cluster'] = labels.values
    return df.groupby('cluster')[MOM_COLS].mean()


calm_input = centroids(calm_dates, calm_lbl, agg_mean)
calm_output = centroids(calm_dates, calm_lbl, long_z)
panic_input = centroids(panic_dates, panic_lbl, agg_mean)
panic_output = centroids(panic_dates, panic_lbl, long_z)

calm_sizes = calm_lbl.value_counts()
panic_sizes = panic_lbl.value_counts()

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)

CALM_COLORS = {'strong_continuation': '#1f77b4', 'typical_calm': '#9ecae1'}
PANIC_COLORS = {'humped_panic': '#fdae6b', 'widespread_loss': '#a63603'}

CALM_DISPLAY = ['strong_continuation', 'typical_calm']
PANIC_DISPLAY = ['humped_panic', 'widespread_loss']


def plot_panel(ax, df, sizes, name_order, color_map, ylabel, title,
               zero_line=False, ypct=False):
    for name in name_order:
        if name not in df.index:
            continue
        y = df.loc[name].values
        if ypct:
            y = 100 * y
        ax.plot(HORIZONS, y, marker='o', color=color_map[name],
                lw=2.0, ms=5,
                label=f'{name} (n={int(sizes[name])})')
    if zero_line:
        ax.axhline(0, color='#888', lw=0.8, ls='--')
    ax.set_xticks(HORIZONS)
    ax.set_xlabel('Momentum horizon (months)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', frameon=True)


# Top row: calm
plot_panel(axes[0, 0], calm_input, calm_sizes, CALM_DISPLAY, CALM_COLORS,
           ylabel='Typical-stock return (%)',
           title='Calm: input landscape centroid (k=2)',
           zero_line=True, ypct=True)
plot_panel(axes[0, 1], calm_output, calm_sizes, CALM_DISPLAY, CALM_COLORS,
           ylabel='Long-leg cross-sectional z',
           title='Calm: model selection shape',
           zero_line=True)

# Bottom row: panic
plot_panel(axes[1, 0], panic_input, panic_sizes, PANIC_DISPLAY, PANIC_COLORS,
           ylabel='Typical-stock return (%)',
           title='Panic: input landscape centroid (k=2)',
           zero_line=True, ypct=True)
plot_panel(axes[1, 1], panic_output, panic_sizes, PANIC_DISPLAY, PANIC_COLORS,
           ylabel='Long-leg cross-sectional z',
           title='Panic: model selection shape',
           zero_line=True)

fig.suptitle('Input landscape vs model selection — k=2 clusters per regime',
             fontsize=13, y=1.00)
fig.tight_layout()

png_path = f'{PLOT_DIR}/landscape_k2_clusters.png'
pdf_path = f'{PLOT_DIR}/landscape_k2_clusters.pdf'
fig.savefig(png_path, dpi=200, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'Saved: {png_path}')
print(f'Saved: {pdf_path}')

# ----------------------------------------------------------------------
# Console summary
# ----------------------------------------------------------------------
print('\nCalm cluster sizes:')
print(calm_sizes.to_string())
print('\nPanic cluster sizes:')
print(panic_sizes.to_string())

print('\nCalm input centroids (raw mom_h, %):')
print((100 * calm_input).round(2).to_string())
print('\nPanic input centroids (raw mom_h, %):')
print((100 * panic_input).round(2).to_string())
print('\nCalm output centroids (long-leg z):')
print(calm_output.round(3).to_string())
print('\nPanic output centroids (long-leg z):')
print(panic_output.round(3).to_string())
