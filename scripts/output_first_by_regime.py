"""
output_first_by_regime.py
==========================
Output-first clustering, split by HMM regime. For each regime separately:

  1. Cluster months on the long-leg z-profile (12-d output).
  2. Sweep k=2..6, pick best by silhouette.
  3. Per cluster, characterise the INPUT signature (pi_filter, mom
     tertile means/stds) and realised performance.
  4. Plot the dispersion view at the best k.

This is the by-regime version of output_first_clustering.py. Use this
when you want to see the model's selection rules as they vary within
calm and within panic separately.

Outputs:
- results/thesis/output_first_{calm,panic}_k_sweep.csv
- results/thesis/output_first_{calm,panic}_cluster_characteristics.csv
- plots/thesis/output_first_{calm,panic}_dispersion.{png,pdf}
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

SHORT, MID, LONG = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]

os.makedirs(PLOT_DIR, exist_ok=True)


def tertile_mean(df, group):
    return df[[f'mom_{h}' for h in group]].mean(axis=1)


def ann_sharpe(x):
    x = x.dropna()
    if len(x) == 0 or x.std() == 0:
        return np.nan
    return (x.mean() / x.std()) * np.sqrt(12)


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

# Pre-compute tertile aggregates
short_mean = tertile_mean(agg_mean, SHORT)
mid_mean = tertile_mean(agg_mean, MID)
long_mean = tertile_mean(agg_mean, LONG)
short_std = tertile_mean(agg_std, SHORT)
mid_std = tertile_mean(agg_std, MID)
long_std = tertile_mean(agg_std, LONG)

# Calm reference (all months with pi <= 0.5) — same in both panels
calm_mask = pi <= 0.5
calm_ref = long_z.loc[calm_mask, MOM_COLS].mean().values

# ----------------------------------------------------------------------
# Per-regime evaluation
# ----------------------------------------------------------------------
def run_regime(regime_name, dates):
    print(f'\n========= {regime_name.upper()} (n={len(dates)}) =========')
    Y = long_z.loc[dates, MOM_COLS].values

    # k-sweep
    sweep_rows, fits = [], {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=N_INIT).fit(Y)
        sil = float(silhouette_score(Y, km.labels_))
        sizes = tuple(sorted(np.bincount(km.labels_, minlength=k).tolist()))
        fits[k] = pd.Series(km.labels_, index=dates)
        sweep_rows.append({
            'k': k,
            'silhouette': round(sil, 3),
            'cluster_sizes': str(sizes),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    print('k-sweep on long-leg z-profile:')
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(f'{RES_DIR}/output_first_{regime_name}_k_sweep.csv',
                    index=False)

    best_k = int(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'k'])
    best_sil = float(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'silhouette'])
    print(f'Best k by silhouette: k={best_k} (silhouette={best_sil:.3f})')

    labels = fits[best_k]

    # Per-cluster characterisation
    rows = []
    for cl in sorted(labels.unique()):
        member = dates[labels == cl]
        n = len(member)
        sharpe = ann_sharpe(m2_ret.reindex(member))
        rows.append({
            'cluster': int(cl),
            'n': int(n),
            'mean_pi': round(pi.loc[member].mean(), 3),
            'mean_short': round(short_mean.loc[member].mean(), 3),
            'mean_mid':   round(mid_mean.loc[member].mean(), 3),
            'mean_long':  round(long_mean.loc[member].mean(), 3),
            'std_short':  round(short_std.loc[member].mean(), 3),
            'std_mid':    round(mid_std.loc[member].mean(), 3),
            'std_long':   round(long_std.loc[member].mean(), 3),
            'mean_ret_pct': round(100 * m2_ret.reindex(member).mean(), 3),
            'sharpe': round(sharpe, 2),
            'hit_rate': round((m2_ret.reindex(member) > 0).mean(), 3),
        })
    char_df = pd.DataFrame(rows)

    # Order clusters by output centroid depth (mean of centroid across 12 horizons)
    centroid_depth = {}
    for cl in sorted(labels.unique()):
        idx = (labels == cl).values
        centroid_depth[cl] = Y[idx].mean()
    order = sorted(centroid_depth, key=centroid_depth.get)
    relabel = {old: f'cluster_{new+1}' for new, old in enumerate(order)}
    char_df['name'] = char_df['cluster'].map(relabel)
    char_df = char_df.set_index('name').loc[
        [f'cluster_{i+1}' for i in range(best_k)]
    ]
    labels_named = labels.map(relabel)

    print('\nPer-cluster signature (ordered by output depth):')
    print(char_df.to_string())
    char_df.to_csv(
        f'{RES_DIR}/output_first_{regime_name}_cluster_characteristics.csv'
    )

    # ----- Plot -----
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
        member = dates[labels_named == name]
        Z = long_z.loc[member, MOM_COLS].values
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
        ax.set_title(
            f'{name} (n={int(info["n"])}, $\\bar\\pi$={info["mean_pi"]:.2f})\n'
            f'mom S/M/L = {100*info["mean_short"]:.1f}/'
            f'{100*info["mean_mid"]:.1f}/{100*info["mean_long"]:.1f}% '
            f'| Sharpe={info["sharpe"]:.2f}',
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, frameon=True)

    for j in range(best_k, len(axes)):
        axes[j].set_visible(False)
    for r in range(n_rows):
        axes_2d[r, 0].set_ylabel('Long-leg cross-sectional z-score')

    fig.suptitle(
        f'Output-first clustering — {regime_name.upper()} '
        f'(k={best_k}, silhouette={best_sil:.3f}); titles show INPUT signature',
        y=1.00 if n_rows == 1 else 1.01,
    )
    fig.tight_layout()

    png = f'{PLOT_DIR}/output_first_{regime_name}_dispersion.png'
    pdf = f'{PLOT_DIR}/output_first_{regime_name}_dispersion.pdf'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f'Saved: {png}')
    print(f'Saved: {pdf}')


calm_dates = pi.index[pi <= 0.5]
panic_dates = pi.index[pi > 0.5]
run_regime('calm', calm_dates)
run_regime('panic', panic_dates)
