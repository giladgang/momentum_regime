"""
landscape_unified_dispersion_plot.py
=====================================
Unified clustering across all 167 OOS months (no regime split). Features:

  12 cross-sectional means of mom_h
  12 cross-sectional stds  of mom_h
   1 pi_filter
  ----
  25 standardised features

Step 1: sweep k = 2..6, report silhouette and cluster sizes per k.
Step 2: plot the dispersion view at whichever k has the highest silhouette.

Outputs:
- results/thesis/landscape_unified_k_sweep.csv
- plots/thesis/landscape_unified_dispersion.{png,pdf}
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'
RETURNS_PATH = 'results/thesis/fundamentals_returns.pkl'
RNG_SEED = 42
N_INIT = 20
K_RANGE = list(range(2, 7))  # 2..6 inclusive
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

all_dates = agg_mean.index
assert (agg_std.index == all_dates).all()
assert (pi.index == all_dates).all()
assert (long_z.index == all_dates).all()

# ----------------------------------------------------------------------
# Build feature matrix (4-d: short/mid/long mom tertile means + pi)
#
# This is the parsimonious config — the full 25-feature version
# (12 means + 12 stds + pi) was dominated by an 11-month outlier in the
# coordinated mom-mean dimensions. Grouping into 3 tertiles removes the
# collinearity that gave that outlier a 12x weight boost, and dropping
# stds removes the second outlier-amplifying channel. See
# landscape_feature_reduction_sweep.py for the comparison.
# ----------------------------------------------------------------------
SHORT = [1, 2, 3, 4]
MID   = [5, 6, 7, 8]
LONG  = [9, 10, 11, 12]


def tertile(df, group):
    return df[[f'mom_{h}' for h in group]].mean(axis=1)


feats = pd.DataFrame({
    'mean_short': tertile(agg_mean, SHORT),
    'mean_mid':   tertile(agg_mean, MID),
    'mean_long':  tertile(agg_mean, LONG),
    'pi_filter':  pi,
})
assert feats.shape[1] == 4
print(f'Feature matrix: {feats.shape[0]} months x {feats.shape[1]} features')

X = StandardScaler().fit_transform(feats.values)

# ----------------------------------------------------------------------
# Step 1: k-sweep
# ----------------------------------------------------------------------
sweep_rows = []
fits = {}  # cache cluster labels per k
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=N_INIT).fit(X)
    sil = float(silhouette_score(X, km.labels_))
    sizes = tuple(sorted(np.bincount(km.labels_, minlength=k).tolist()))
    fits[k] = pd.Series(km.labels_, index=all_dates, name='cluster')
    sweep_rows.append({
        'k': k,
        'silhouette': round(sil, 3),
        'min_cluster_size': sizes[0],
        'max_cluster_size': sizes[-1],
        'cluster_sizes': str(sizes),
    })
sweep_df = pd.DataFrame(sweep_rows)
print(f'\nk-sweep on unified {feats.shape[1]}-d features '
      f'(mom tertile means + pi_filter):')
print(sweep_df.to_string(index=False))
sweep_path = f'{RES_DIR}/landscape_unified_k_sweep.csv'
sweep_df.to_csv(sweep_path, index=False)
print(f'Saved: {sweep_path}')

# Pick best k by silhouette
best_k = int(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'k'])
best_sil = float(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'silhouette'])
print(f'\nBest k by silhouette: k={best_k} (silhouette={best_sil:.3f})')

labels = fits[best_k]

# ----------------------------------------------------------------------
# Step 2: characterise clusters and assign generic depth-ordered names
# ----------------------------------------------------------------------
char = pd.DataFrame({
    'cluster': labels.values,
    'pi': pi.values,
    'land_mean': agg_mean.mean(axis=1).values,  # avg of 12 horizon means
}).groupby('cluster').agg(
    n=('pi', 'count'),
    mean_pi=('pi', 'mean'),
    mean_land=('land_mean', 'mean'),
)
print('\nRaw cluster characteristics:')
print(char.round(3).to_string())

# Generic depth ordering — most-negative landscape -> state_1, etc.
order = char.sort_values('mean_land').index.tolist()
relabel = {order[i]: f'state_{i+1}' for i in range(best_k)}
labels_named = labels.map(relabel)

# Resort `char` by depth and add the new name
char = char.loc[order].copy()
char['name'] = [f'state_{i+1}' for i in range(best_k)]

# ----------------------------------------------------------------------
# Conditional Sharpe per cluster
# ----------------------------------------------------------------------
def ann_sharpe(x):
    x = x.dropna()
    if len(x) == 0 or x.std() == 0:
        return np.nan
    return (x.mean() / x.std()) * np.sqrt(12)


df_ret = pd.DataFrame({
    'cluster': labels_named.values,
    'm2_ret': m2_ret.reindex(all_dates).values,
}, index=all_dates)
sharpe_by_cluster = df_ret.groupby('cluster')['m2_ret'].apply(ann_sharpe)
char['sharpe'] = char['name'].map(sharpe_by_cluster).round(2)
print('\nDepth-ordered clusters with names + Sharpe:')
print(char.round(3).to_string())

# ----------------------------------------------------------------------
# Calm reference (all months with pi <= 0.5)
# ----------------------------------------------------------------------
calm_mask = pi <= 0.5
calm_ref = long_z.loc[calm_mask, MOM_COLS].mean().values

# ----------------------------------------------------------------------
# Plot dispersion view at best k
# ----------------------------------------------------------------------
PANEL_ORDER = [f'state_{i+1}' for i in range(best_k)]

# Color cycle: span red -> blue -> purple as depth increases
CMAP = plt.cm.viridis(np.linspace(0.15, 0.85, best_k))

# Layout: single row up to k=4, otherwise 2 rows
if best_k <= 4:
    n_rows, n_cols = 1, best_k
else:
    n_rows, n_cols = 2, int(np.ceil(best_k / 2))

fig, axes_2d = plt.subplots(n_rows, n_cols,
                            figsize=(5.0 * n_cols, 5.5 * n_rows),
                            sharey=True, squeeze=False)
axes = axes_2d.flatten()
YLIM = (-1.5, 1.0)

for i, name in enumerate(PANEL_ORDER):
    ax = axes[i]
    color = CMAP[i]
    member_dates = labels_named.index[labels_named == name]
    Z = long_z.loc[member_dates, MOM_COLS].values
    centroid = Z.mean(axis=0)
    n = int(char.loc[char['name'] == name, 'n'].iloc[0])
    mp = float(char.loc[char['name'] == name, 'mean_pi'].iloc[0])
    sh = float(char.loc[char['name'] == name, 'sharpe'].iloc[0])
    regime_tag = 'panic' if mp > 0.5 else 'calm'

    for row in Z:
        ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.35)
    ax.plot(HORIZONS, centroid, color=color, lw=2.5, marker='o', ms=6,
            label=f'{name} centroid')
    ax.plot(HORIZONS, calm_ref, color='#333', lw=1.5, ls='--',
            label='Calm centroid (reference)')

    ax.axhline(0, color='#888', lw=0.6)
    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(*YLIM)
    ax.set_xlabel('Momentum lookback (months)')
    ax.set_title(f'{name} — {regime_tag} '
                 f'(n={n}, $\\bar\\pi$={mp:.2f}, Sharpe={sh:.2f})')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, frameon=True)

# Hide any extra subplots if best_k doesn't fill the grid
for j in range(best_k, len(axes)):
    axes[j].set_visible(False)

# Y-label on first column only
for r in range(n_rows):
    axes_2d[r, 0].set_ylabel('Long-leg cross-sectional z-score')

fig.suptitle(f'Unified k={best_k} clustering on '
             f'short/mid/long mom tertile means + '
             f'$\\pi_t^{{filter}}$ (silhouette = {best_sil:.3f})',
             y=1.00 if n_rows == 1 else 1.01)
fig.tight_layout()

png_path = f'{PLOT_DIR}/landscape_unified_dispersion.png'
pdf_path = f'{PLOT_DIR}/landscape_unified_dispersion.pdf'
fig.savefig(png_path, dpi=200, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'\nSaved: {png_path}')
print(f'Saved: {pdf_path}')
