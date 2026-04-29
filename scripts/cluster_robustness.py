"""
cluster_robustness.py
======================
Robustness checks for the output-first per-regime k=2 clusterings
produced by output_first_by_regime.py.

For each regime (calm, panic):
  1. Seed stability — refit KMeans(k=2) at 6 seeds; report sorted cluster
     sizes per seed and mean pairwise ARI between seed-label vectors.
     Mirrors thesis 5.2.3 stability check.
  2. Block-bootstrap Sharpe CI — for each of the two clusters, compute
     a 95% CI on annualised Sharpe via moving-block bootstrap (block size
     6 months, 5000 reps).

Uses verified primitives from bootstrap_helpers.py (tested in
test_bootstrap_helpers.py).

Outputs:
- results/thesis/cluster_robustness_seed_stability.csv
- results/thesis/cluster_robustness_sharpe_ci.csv
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bootstrap_helpers import block_bootstrap_sharpe, seed_stability
from sklearn.cluster import KMeans

RES_DIR = 'results/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'
RETURNS_PATH = 'results/thesis/fundamentals_returns.pkl'

K = 2
N_INIT = 20
SEEDS = [42, 123, 456, 789, 1011, 1213]
BLOCK_SIZE = 6
N_BOOT = 5000
BOOT_SEED = 42

HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
pi = test.groupby('date')['pi_filter'].first()

long_z = (pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                      parse_dates=['date'])
          .set_index('date'))

with open(RETURNS_PATH, 'rb') as f:
    rets = pickle.load(f)
m2_ret = rets['baseline_mom_pi']['returns']
m2_ret.index = pd.to_datetime(m2_ret.index)

# ----------------------------------------------------------------------
# Per-regime robustness
# ----------------------------------------------------------------------
def run_regime(regime_name, dates):
    print(f'\n========== {regime_name.upper()} (n={len(dates)}) ==========')
    Y = long_z.loc[dates, MOM_COLS].values

    # --- seed stability ---
    stab = seed_stability(Y, k=K, seeds=SEEDS, n_init=N_INIT)
    print(f'\nSeed stability across {len(SEEDS)} seeds (k={K}):')
    print('  sorted cluster sizes per seed:')
    for s, sizes in zip(SEEDS, stab['sorted_sizes_per_seed']):
        print(f'    seed {s}: {sizes}')
    print(f'  mean pairwise ARI: {stab["mean_ari"]:.4f}  '
          f'(1.0 = identical labels across seeds)')
    print(f'  individual pairwise ARIs: '
          f'{[round(a, 3) for a in stab["pairwise_ari"]]}')

    # --- bootstrap Sharpe CI per cluster (using seed-42 labels) ---
    km = KMeans(n_clusters=K, random_state=42, n_init=N_INIT).fit(Y)
    labels = pd.Series(km.labels_, index=dates)

    # Order clusters by output centroid depth (most-negative -> cluster_1)
    depths = [Y[(labels == cl).values].mean() for cl in [0, 1]]
    order = sorted([0, 1], key=lambda cl: depths[cl])
    relabel = {old: f'cluster_{i+1}' for i, old in enumerate(order)}
    labels_named = labels.map(relabel)

    print(f'\nBlock-bootstrap Sharpe 95% CI '
          f'(block_size={BLOCK_SIZE}, n_reps={N_BOOT}):')
    sharpe_rows = []
    for name in ['cluster_1', 'cluster_2']:
        member = dates[labels_named == name]
        n = len(member)
        rs = m2_ret.reindex(member).dropna().values
        ci = block_bootstrap_sharpe(rs, block_size=BLOCK_SIZE,
                                    n_reps=N_BOOT, seed=BOOT_SEED)
        depth = Y[(labels_named == name).values].mean()
        print(f'  {name} (n={n}, output_depth={depth:+.3f}): '
              f'Sharpe={ci["sharpe_point"]:.2f}  '
              f'95% CI [{ci["sharpe_lo95"]:.2f}, {ci["sharpe_hi95"]:.2f}]'
              f'  excludes_zero={(ci["sharpe_lo95"] > 0)}')
        sharpe_rows.append({
            'regime': regime_name,
            'cluster': name,
            'n': n,
            'output_depth': round(depth, 4),
            'sharpe_point': round(ci['sharpe_point'], 4),
            'sharpe_lo95': round(ci['sharpe_lo95'], 4),
            'sharpe_hi95': round(ci['sharpe_hi95'], 4),
            'ci_excludes_zero': bool(ci['sharpe_lo95'] > 0),
            'n_reps_valid': ci['n_reps_valid'],
        })

    # Pairwise: are the two clusters' Sharpes statistically distinguishable?
    # (Quick check: do their CIs overlap?)
    s1, s2 = sharpe_rows[0], sharpe_rows[1]
    cis_overlap = not (s1['sharpe_hi95'] < s2['sharpe_lo95']
                       or s2['sharpe_hi95'] < s1['sharpe_lo95'])
    print(f'\nCIs overlap between cluster_1 and cluster_2: {cis_overlap}')
    print(f'  (non-overlapping CIs => Sharpe difference is robust)')

    return stab, sharpe_rows, cis_overlap


calm_dates = pi.index[pi <= 0.5]
panic_dates = pi.index[pi > 0.5]

stab_calm, sh_calm, overlap_calm = run_regime('calm', calm_dates)
stab_panic, sh_panic, overlap_panic = run_regime('panic', panic_dates)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
seed_rows = []
for regime, stab in [('calm', stab_calm), ('panic', stab_panic)]:
    for s, sizes in zip(SEEDS, stab['sorted_sizes_per_seed']):
        seed_rows.append({
            'regime': regime,
            'seed': s,
            'sorted_sizes': str(sizes),
        })
    seed_rows.append({
        'regime': regime,
        'seed': 'mean_pairwise_ari',
        'sorted_sizes': round(stab['mean_ari'], 4),
    })
seed_df = pd.DataFrame(seed_rows)
seed_df.to_csv(f'{RES_DIR}/cluster_robustness_seed_stability.csv', index=False)
print(f'\nSaved: {RES_DIR}/cluster_robustness_seed_stability.csv')

sh_df = pd.DataFrame(sh_calm + sh_panic)
sh_df.to_csv(f'{RES_DIR}/cluster_robustness_sharpe_ci.csv', index=False)
print(f'Saved: {RES_DIR}/cluster_robustness_sharpe_ci.csv')

print('\n========== HEADLINE ==========')
print(f'CALM seed stability ARI: {stab_calm["mean_ari"]:.4f}  '
      f'(>= 0.95 => stable cluster assignments)')
print(f'PANIC seed stability ARI: {stab_panic["mean_ari"]:.4f}')
print(f'CALM CIs overlap (sharpe difference inconclusive): {overlap_calm}')
print(f'PANIC CIs overlap: {overlap_panic}')
