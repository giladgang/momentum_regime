"""
landscape_subregime_analysis.py
================================
Input-side counterpart to zscore_subregime_analysis.py. Clusters market
states by the *cross-sectional momentum landscape* — the mean of raw
mom_1..mom_12 across ALL stocks each month — rather than by the model's
output selection. This decouples the cluster definition from anything the
model produced and lets us ask: does the model's selection shape line up
with the input landscape?

The OOS sample is split by HMM regime first, then clustered separately:

  - calm  months (pi_filter <= 0.5, n=108): k=3 on landscape
  - panic months (pi_filter >  0.5, n= 59): k=3 on landscape (parallel to
                                            the thesis output-side
                                            decomposition)

Both clusterings standardise each landscape column (across the relevant
month set) before KMeans, so all 12 horizons contribute equally to the
Euclidean distance metric. Without standardisation, the longer horizons
(mom_12 std is ~4.2x mom_1 std across months) would dominate.

For each regime we also recompute the output-side k=3 clustering on the
same date set (long-leg z-profile) and report Adjusted Rand Index between
the two label vectors. ARI ~ 1 means the input landscape determines the
model's selection shape; ARI ~ 0 means the model is using something the
landscape mean doesn't capture.

Outputs:
- results/thesis/landscape_calm_subregime_summary.csv
- results/thesis/landscape_panic_subregime_summary.csv
- results/thesis/landscape_subregime_robustness.csv

Usage:
    python scripts/landscape_subregime_analysis.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
RES_DIR = 'results/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'
RETURNS_PATH = 'results/thesis/fundamentals_returns.pkl'

RNG_SEED = 42
N_INIT = 20
K = 3
HORIZONS = list(range(1, 13))
COLS = [f'mom_{h}' for h in HORIZONS]
SEEDS_FOR_STABILITY = [42, 123, 456, 789, 1011, 1213]

# Depth-ordered names (most-negative landscape -> most-positive)
CALM_NAMES = ['weak_continuation', 'moderate_continuation', 'strong_continuation']
PANIC_NAMES = ['deep_crisis', 'transition', 'mild_panic']

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])

# Landscape vector: mean of RAW mom_h across ALL stocks each month.
landscape = (test
             .groupby('date')[COLS]
             .mean()
             .reset_index())

pi = (test.groupby('date')['pi_filter']
          .first()
          .reset_index())
landscape = landscape.merge(pi, on='date').sort_values('date').reset_index(drop=True)
assert landscape['pi_filter'].notna().all()
print(f'Landscape vectors: {len(landscape)} months x {len(COLS)} horizons')

long_z = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv', parse_dates=['date'])
short_z = pd.read_csv(f'{RES_DIR}/zscore_short_by_month.csv', parse_dates=['date'])
long_z = long_z.rename(columns={c: f'long_{c}' for c in long_z.columns if c != 'date'})
short_z = short_z.rename(columns={c: f'short_{c}' for c in short_z.columns if c != 'date'})

with open(RETURNS_PATH, 'rb') as f:
    rets = pickle.load(f)
m2_ret = rets['baseline_mom_pi']['returns']
m2_ret.index = pd.to_datetime(m2_ret.index)
ret_df = pd.DataFrame({'date': m2_ret.index, 'm2_ret': m2_ret.values})

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def ann_sharpe(x):
    x = x.dropna()
    if len(x) == 0 or x.std() == 0:
        return np.nan
    return (x.mean() / x.std()) * np.sqrt(12)


def cluster_landscape(land_df, k, seed=RNG_SEED):
    """Standardise the 12 landscape columns across the rows of land_df,
    fit KMeans(k), and return labels + silhouette."""
    X = StandardScaler().fit_transform(land_df[COLS].values)
    km = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT).fit(X)
    sil = float(silhouette_score(X, km.labels_)) if k >= 2 else np.nan
    return km.labels_, sil


def cluster_output(long_z_df, k, seed=RNG_SEED):
    """KMeans on the long-leg z-profile (12-d) — already z-scored, so no
    re-standardisation. Mirrors zscore_subregime_analysis.py."""
    cols = [f'long_mom_{h}' for h in HORIZONS]
    X = long_z_df[cols].values
    km = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT).fit(X)
    return km.labels_


def relabel_by_depth(land_df, labels, k, naming):
    """Order clusters by mean landscape depth across the 12 horizons in
    RAW mom_h units. Most-negative -> naming[0], most-positive -> naming[-1]."""
    tmp = land_df.copy()
    tmp['_lbl'] = labels
    mean_depth = (tmp.groupby('_lbl')[COLS]
                     .mean()
                     .mean(axis=1)
                     .sort_values())
    order = mean_depth.index.tolist()
    return {order[i]: naming[i] for i in range(k)}


def per_state_summary(land_df, state_col, naming_order):
    """Per-cluster summary merged with realised returns and selection shape."""
    merged = (land_df
              .merge(long_z, on='date', how='left')
              .merge(short_z, on='date', how='left')
              .merge(ret_df, on='date', how='left'))
    rows = []
    for state in naming_order:
        sub = merged[merged[state_col] == state]
        n = len(sub)
        if n == 0:
            continue
        row = {
            'state': state,
            'n_months': n,
            'mean_pi': round(sub['pi_filter'].mean(), 3),
            'mean_monthly_ret_pct': round(100 * sub['m2_ret'].mean(), 3),
            'ann_sharpe': round(ann_sharpe(sub['m2_ret']), 2),
            'hit_rate': round((sub['m2_ret'] > 0).mean(), 3),
        }
        for h in HORIZONS:
            row[f'land_mom_{h}'] = round(sub[f'mom_{h}'].mean(), 4)
        for h in HORIZONS:
            row[f'long_z_mom_{h}'] = round(sub[f'long_mom_{h}'].mean(), 3)
        for h in HORIZONS:
            row[f'short_z_mom_{h}'] = round(sub[f'short_mom_{h}'].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def stability_sizes(land_df, k, seeds):
    out = {}
    X = StandardScaler().fit_transform(land_df[COLS].values)
    for s in seeds:
        km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(X)
        out[s] = tuple(sorted(np.bincount(km.labels_, minlength=k).tolist()))
    return out


def run_regime(regime_name, regime_df, names, display_order, ari_label):
    """Run input-landscape and output-selection clustering for one regime,
    compute ARI, and return summary + diagnostics."""
    print(f'\n=== {regime_name} clustering (k={K}) ===')
    print(f'n_months = {len(regime_df)}')

    # Input-landscape clustering
    labels_in, sil = cluster_landscape(regime_df, k=K)
    regime_df = regime_df.copy()
    regime_df['cluster_in'] = labels_in
    relabel = relabel_by_depth(regime_df, labels_in, K, names)
    regime_df['state'] = regime_df['cluster_in'].map(relabel)

    # Output-selection clustering on the same dates
    out_z = (long_z[long_z['date'].isin(regime_df['date'])]
             .sort_values('date')
             .reset_index(drop=True))
    labels_out = cluster_output(out_z, k=K)
    out_z['out_cluster'] = labels_out

    # ARI
    aligned = (regime_df[['date', 'cluster_in', 'state']]
               .merge(out_z[['date', 'out_cluster']], on='date', how='inner'))
    ari = float(adjusted_rand_score(aligned['cluster_in'],
                                    aligned['out_cluster']))

    # Summary
    summary = per_state_summary(regime_df, 'state', display_order)
    disp_cols = ['state', 'n_months', 'mean_pi', 'mean_monthly_ret_pct',
                 'ann_sharpe', 'hit_rate']
    print(summary[disp_cols].to_string(index=False))
    print(f'silhouette = {sil:.3f}')
    print(f'ARI ({ari_label}): {ari:.3f}')

    ct = pd.crosstab(aligned['state'], aligned['out_cluster'],
                     rownames=['landscape state'],
                     colnames=['output cluster'])
    print('Cross-tab: landscape state x output cluster')
    print(ct)

    return summary, sil, ari, regime_df


# ----------------------------------------------------------------------
# Calm and panic clusterings
# ----------------------------------------------------------------------
calm = landscape[landscape['pi_filter'] <= 0.5].reset_index(drop=True)
panic = landscape[landscape['pi_filter'] > 0.5].reset_index(drop=True)

summary_calm, sil_calm, ari_calm, _ = run_regime(
    'CALM', calm, CALM_NAMES,
    display_order=CALM_NAMES[::-1],   # strong -> weak in display
    ari_label='calm landscape vs calm output',
)
out_calm = f'{RES_DIR}/landscape_calm_subregime_summary.csv'
summary_calm.to_csv(out_calm, index=False)
print(f'Saved: {out_calm}')

summary_panic, sil_panic, ari_panic, _ = run_regime(
    'PANIC', panic, PANIC_NAMES,
    display_order=['mild_panic', 'transition', 'deep_crisis'],
    ari_label='panic landscape vs panic output',
)
out_panic = f'{RES_DIR}/landscape_panic_subregime_summary.csv'
summary_panic.to_csv(out_panic, index=False)
print(f'Saved: {out_panic}')

# ----------------------------------------------------------------------
# Robustness sidecar
# ----------------------------------------------------------------------
stab_calm = stability_sizes(calm, k=K, seeds=SEEDS_FOR_STABILITY)
stab_panic = stability_sizes(panic, k=K, seeds=SEEDS_FOR_STABILITY)

robust_rows = [
    {'metric': 'k', 'value': K},
    {'metric': 'n_calm_months', 'value': len(calm)},
    {'metric': 'n_panic_months', 'value': len(panic)},
    {'metric': 'silhouette_calm_seed42', 'value': round(sil_calm, 4)},
    {'metric': 'silhouette_panic_seed42', 'value': round(sil_panic, 4)},
    {'metric': 'ARI_calm_landscape_vs_calm_output', 'value': round(ari_calm, 4)},
    {'metric': 'ARI_panic_landscape_vs_panic_output', 'value': round(ari_panic, 4)},
]
for s, sizes in stab_calm.items():
    robust_rows.append({'metric': f'calm_cluster_sizes_seed{s}',
                        'value': str(sizes)})
for s, sizes in stab_panic.items():
    robust_rows.append({'metric': f'panic_cluster_sizes_seed{s}',
                        'value': str(sizes)})

robust = pd.DataFrame(robust_rows)
robust_path = f'{RES_DIR}/landscape_subregime_robustness.csv'
robust.to_csv(robust_path, index=False)
print(f'\nSaved: {robust_path}')
