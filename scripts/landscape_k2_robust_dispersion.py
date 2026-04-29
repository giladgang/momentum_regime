"""
landscape_k2_robust_dispersion.py
==================================
Robust version of output_first_by_regime.py dispersion plots, using the
modal cluster partition across the 6-seed stability check (rather than
the seed-42 partition that turned out to be a single-seed dissenter for
panic).

Stability-check results from cluster_robustness.py:
  calm:  ARI = 1.000 across all 6 seeds, all give (42, 66) — seed 42 is fine
  panic: ARI = 0.977 across 6 seeds. Seed 42 alone gives (21, 38);
         seeds 123, 456, 789, 1011, 1213 all give the modal (22, 37)
         and are pairwise identical (ARI = 1.0 between them). Use one
         of those (seed 123) for the panic plot.

Outputs:
- plots/thesis/output_first_calm_dispersion_robust.{png,pdf}
- plots/thesis/output_first_panic_dispersion_robust.{png,pdf}
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from bootstrap_helpers import block_bootstrap_sharpe

RES_DIR = 'results/thesis'
PLOT_DIR = 'plots/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'
RETURNS_PATH = 'results/thesis/fundamentals_returns.pkl'

K = 2
N_INIT = 20
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

# Modal seeds chosen so panel reflects the (22, 37) panic partition
# agreed upon by 5 of 6 stability seeds. Calm seed 42 is fine since all
# 6 stability seeds give identical labels for calm.
SEEDS = {'calm': 42, 'panic': 123}

SHORT, MID, LONG = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]

os.makedirs(PLOT_DIR, exist_ok=True)


def tertile_mean(df, group):
    return df[[f'mom_{h}' for h in group]].mean(axis=1)


def ann_sharpe(x):
    x = x.dropna()
    if len(x) == 0 or x.std(ddof=1) == 0:
        return np.nan
    return (x.mean() / x.std(ddof=1)) * np.sqrt(12)


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

# Calm reference: long-leg z-profile averaged across all calm months
calm_mask = pi <= 0.5
calm_ref = long_z.loc[calm_mask, MOM_COLS].mean().values

short_mean = tertile_mean(agg_mean, SHORT)
mid_mean = tertile_mean(agg_mean, MID)
long_mean = tertile_mean(agg_mean, LONG)
short_std = tertile_mean(agg_std, SHORT)
mid_std = tertile_mean(agg_std, MID)
long_std = tertile_mean(agg_std, LONG)


def cluster_and_plot(regime_name, dates, seed, ari_value):
    print(f'\n===== {regime_name.upper()} (n={len(dates)}, seed {seed}) =====')
    Y = long_z.loc[dates, MOM_COLS].values
    km = KMeans(n_clusters=K, random_state=seed, n_init=N_INIT).fit(Y)
    sil = float(silhouette_score(Y, km.labels_))
    sizes = sorted(np.bincount(km.labels_, minlength=K).tolist())
    print(f'  silhouette = {sil:.3f}, sizes = {tuple(sizes)}')

    labels = pd.Series(km.labels_, index=dates)

    # Order by output centroid depth: most-negative -> cluster_1
    depths = {cl: Y[(labels == cl).values].mean() for cl in [0, 1]}
    order = sorted(depths, key=depths.get)
    relabel = {old: f'cluster_{i+1}' for i, old in enumerate(order)}
    labels_named = labels.map(relabel)

    # Per-cluster info incl bootstrap CI
    info = {}
    for name in ['cluster_1', 'cluster_2']:
        member = dates[labels_named == name]
        n = len(member)
        rs = m2_ret.reindex(member).dropna().values
        ci = block_bootstrap_sharpe(rs, block_size=6, n_reps=5000, seed=42)
        depth = Y[(labels_named == name).values].mean()
        info[name] = {
            'n': n,
            'depth': depth,
            'mean_pi': pi.loc[member].mean(),
            'mean_short': short_mean.loc[member].mean(),
            'mean_mid':   mid_mean.loc[member].mean(),
            'mean_long':  long_mean.loc[member].mean(),
            'sharpe_point': ci['sharpe_point'],
            'sharpe_lo95':  ci['sharpe_lo95'],
            'sharpe_hi95':  ci['sharpe_hi95'],
        }
        print(f'  {name}: n={n}, depth={depth:+.3f}, '
              f'Sharpe={ci["sharpe_point"]:.2f} '
              f'[{ci["sharpe_lo95"]:.2f}, {ci["sharpe_hi95"]:.2f}]')

    # ----- Plot -----
    PANEL_ORDER = ['cluster_1', 'cluster_2']
    CMAP = plt.cm.viridis(np.linspace(0.15, 0.85, K))

    fig, axes = plt.subplots(1, K, figsize=(11.5, 5.5), sharey=True)
    YLIM = (-1.5, 1.0)

    for i, name in enumerate(PANEL_ORDER):
        ax = axes[i]
        color = CMAP[i]
        member = dates[labels_named == name]
        Z = long_z.loc[member, MOM_COLS].values
        centroid = Z.mean(axis=0)
        d = info[name]

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
            f'{name} (n={d["n"]}, $\\bar\\pi$={d["mean_pi"]:.2f})\n'
            f'mom S/M/L = {100*d["mean_short"]:.1f}/'
            f'{100*d["mean_mid"]:.1f}/{100*d["mean_long"]:.1f}% | '
            f'Sharpe = {d["sharpe_point"]:.2f} '
            f'[{d["sharpe_lo95"]:.2f}, {d["sharpe_hi95"]:.2f}]',
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, frameon=True)

    axes[0].set_ylabel('Long-leg cross-sectional z-score')
    fig.suptitle(
        f'{regime_name.upper()} — k=2 (silhouette={sil:.3f}, '
        f'seed-stability ARI={ari_value:.3f}, modal seed {seed}); '
        f'titles show INPUT signature + bootstrap 95% CI',
        y=1.00,
    )
    fig.tight_layout()

    png = f'{PLOT_DIR}/output_first_{regime_name}_dispersion_robust.png'
    pdf = f'{PLOT_DIR}/output_first_{regime_name}_dispersion_robust.pdf'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f'  Saved: {png}')
    print(f'  Saved: {pdf}')


calm_dates = pi.index[pi <= 0.5]
panic_dates = pi.index[pi > 0.5]

cluster_and_plot('calm', calm_dates, seed=SEEDS['calm'], ari_value=1.000)
cluster_and_plot('panic', panic_dates, seed=SEEDS['panic'], ari_value=0.977)
