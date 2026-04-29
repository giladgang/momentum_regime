"""
output_first_clustering.py
===========================
Inverts the direction of the unified clustering: cluster all 167 OOS
months by the *model's selection shape* (long-leg z-profile, 12-d), then
characterise each cluster by the INPUT features (pi_filter, mom tertile
means, mom tertile stds) and the realised performance.

Where landscape_unified_dispersion_plot.py asks "given a market state,
what does the model do?", this script asks "given that the model
produced this shape, what was the market doing?".

Steps:
1. Build 12-d output feature (long-leg z-profile, no scaling — z already
   on a comparable scale across horizons).
2. Sweep k=2..6, pick best by silhouette.
3. Per cluster, report n, regime composition, input landscape signature,
   and realised return/Sharpe.
4. Plot 1xK dispersion view of output centroids with member months and
   input signature in the title.

Outputs:
- results/thesis/output_first_k_sweep.csv
- results/thesis/output_first_cluster_characteristics.csv
- plots/thesis/output_first_dispersion.{png,pdf}
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

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'
RETURNS_PATH = 'results/thesis/fundamentals_returns.pkl'
RNG_SEED = 42
N_INIT = 20
K_RANGE = list(range(2, 7))
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

SHORT = [1, 2, 3, 4]
MID = [5, 6, 7, 8]
LONG = [9, 10, 11, 12]

os.makedirs(PLOT_DIR, exist_ok=True)


def tertile_mean(df, group):
    return df[[f'mom_{h}' for h in group]].mean(axis=1)


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
assert (long_z.index == all_dates).all()

# Output feature matrix: 12-d long-leg z-profile per month
Y = long_z.loc[all_dates, MOM_COLS].values
print(f'Output feature matrix: {Y.shape[0]} months x {Y.shape[1]} horizons')

# ----------------------------------------------------------------------
# Sweep k on output
# ----------------------------------------------------------------------
sweep_rows, fits = [], {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=N_INIT).fit(Y)
    sil = float(silhouette_score(Y, km.labels_))
    sizes = tuple(sorted(np.bincount(km.labels_, minlength=k).tolist()))
    fits[k] = pd.Series(km.labels_, index=all_dates)
    sweep_rows.append({
        'k': k,
        'silhouette': round(sil, 3),
        'cluster_sizes': str(sizes),
    })
sweep_df = pd.DataFrame(sweep_rows)
print('\nk-sweep on output (long-leg z-profile, 12-d):')
print(sweep_df.to_string(index=False))
sweep_df.to_csv(f'{RES_DIR}/output_first_k_sweep.csv', index=False)

best_k = int(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'k'])
best_sil = float(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'silhouette'])
print(f'\nBest k by silhouette: k={best_k} (silhouette={best_sil:.3f})')

labels = fits[best_k]

# ----------------------------------------------------------------------
# Per-cluster input/output characterisation
# ----------------------------------------------------------------------
def ann_sharpe(x):
    x = x.dropna()
    if len(x) == 0 or x.std() == 0:
        return np.nan
    return (x.mean() / x.std()) * np.sqrt(12)


short_mean = tertile_mean(agg_mean, SHORT)
mid_mean = tertile_mean(agg_mean, MID)
long_mean = tertile_mean(agg_mean, LONG)
short_std = tertile_mean(agg_std, SHORT)
mid_std = tertile_mean(agg_std, MID)
long_std = tertile_mean(agg_std, LONG)
ret_aligned = m2_ret.reindex(all_dates)

per_cluster = []
for cl in sorted(labels.unique()):
    mask = labels == cl
    member_dates = all_dates[mask]
    n = mask.sum()
    n_panic = int((pi.loc[member_dates] > 0.5).sum())
    sharpe = ann_sharpe(ret_aligned.loc[member_dates])
    per_cluster.append({
        'cluster': int(cl),
        'n': int(n),
        'n_panic': n_panic,
        'panic_share': round(n_panic / n, 2),
        'mean_pi': round(pi.loc[member_dates].mean(), 3),
        'mean_short': round(short_mean.loc[member_dates].mean(), 3),
        'mean_mid':   round(mid_mean.loc[member_dates].mean(), 3),
        'mean_long':  round(long_mean.loc[member_dates].mean(), 3),
        'std_short':  round(short_std.loc[member_dates].mean(), 3),
        'std_mid':    round(mid_std.loc[member_dates].mean(), 3),
        'std_long':   round(long_std.loc[member_dates].mean(), 3),
        'mean_ret_pct': round(100 * ret_aligned.loc[member_dates].mean(), 3),
        'sharpe': round(sharpe, 2),
        'hit_rate': round((ret_aligned.loc[member_dates] > 0).mean(), 3),
    })
char_df = pd.DataFrame(per_cluster)

# Order clusters by mean_long (output centroid's long-end value, computed
# as the mean of the long-leg z-profile member months at horizons 9..12).
# Most-negative -> cluster_1 (uniformly-loser shape), most-positive ->
# cluster_K (strong continuation shape).
# The simplest proxy for output depth is the mean of the centroid across
# all 12 horizons; use that.
centroid_depth = {}
for cl in sorted(labels.unique()):
    mask = labels == cl
    centroid_depth[cl] = Y[mask.values].mean()
order = sorted(centroid_depth, key=centroid_depth.get)
relabel = {old: f'cluster_{new+1}' for new, old in enumerate(order)}
char_df['name'] = char_df['cluster'].map(relabel)
char_df = char_df.set_index('name').loc[[f'cluster_{i+1}' for i in range(best_k)]]
labels_named = labels.map(relabel)

print('\nPer-cluster input signature, output ordering, and performance:')
print(char_df.to_string())
char_df.to_csv(f'{RES_DIR}/output_first_cluster_characteristics.csv')

# ----------------------------------------------------------------------
# Calm reference (all months with pi <= 0.5)
# ----------------------------------------------------------------------
calm_mask = pi <= 0.5
calm_ref = long_z.loc[calm_mask, MOM_COLS].mean().values

# ----------------------------------------------------------------------
# Plot — 1xK dispersion view
# ----------------------------------------------------------------------
PANEL_ORDER = [f'cluster_{i+1}' for i in range(best_k)]
CMAP = plt.cm.viridis(np.linspace(0.15, 0.85, best_k))

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
    info = char_df.loc[name]

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
    # Title shows INPUT signature: panic share, pi, tertile mean returns
    ax.set_title(
        f'{name} (n={int(info["n"])}, '
        f'panic={int(info["n_panic"])}/{int(info["n"])}, '
        f'$\\bar\\pi$={info["mean_pi"]:.2f})\n'
        f'mom S/M/L = {100*info["mean_short"]:.1f}/{100*info["mean_mid"]:.1f}/'
        f'{100*info["mean_long"]:.1f}% '
        f'| Sharpe={info["sharpe"]:.2f}',
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, frameon=True)

for j in range(best_k, len(axes)):
    axes[j].set_visible(False)
for r in range(n_rows):
    axes_2d[r, 0].set_ylabel('Long-leg cross-sectional z-score')

fig.suptitle(f'Output-first clustering: k={best_k} on long-leg z-profile '
             f'(silhouette = {best_sil:.3f}); titles show INPUT signature',
             y=1.00 if n_rows == 1 else 1.01)
fig.tight_layout()

png_path = f'{PLOT_DIR}/output_first_dispersion.png'
pdf_path = f'{PLOT_DIR}/output_first_dispersion.pdf'
fig.savefig(png_path, dpi=200, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'\nSaved: {png_path}')
print(f'Saved: {pdf_path}')
