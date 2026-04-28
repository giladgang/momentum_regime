"""
zscore_subregime_xgb_seed_robustness.py
========================================
Tests whether the three-cluster panic decomposition documented in
zscore_subregime_analysis.py is sensitive to the specific 50 XGBoost
seeds used in the production ensemble.

Procedure:
  1. Load the existing production cs_artefacts (X_train, X_test, y_train).
  2. Train a second 50-seed XGB ensemble using seeds 101..150 (production
     uses 1..50). All other hyperparameters identical.
  3. Recompute score_xgb_v2, reassign top/bottom-decile legs.
  4. Recompute long-leg cross-sectional z-profile per month.
  5. Re-cluster panic months with k=3.
  6. Compare to the production clustering: cluster sizes, Adjusted Rand
     Index (ARI), centroid pairwise distances, Sharpe per cluster.

Output:
  results/thesis/zscore_subregime_xgb_seed_robustness.csv
  logs/zscore_subregime_xgb_seed_robustness.log (when run via pipeline)

This is a robustness check; it does NOT overwrite production artefacts.
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ARTEFACTS_PATH, RESULTS_THESIS_DIR,
    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE,
    XGB_N_JOBS, PANEL_WITH_REGIMES_PATH,
)

os.makedirs(RESULTS_THESIS_DIR, exist_ok=True)

PROD_SEEDS = list(range(1, 51))      # match config.XGB_SEEDS
ROBUST_SEEDS = list(range(101, 151))  # disjoint set of 50 seeds
HORIZONS = list(range(1, 13))
COLS = [f'mom_{h}' for h in HORIZONS]
RNG_SEED = 42  # for k-means

print('=' * 72)
print('XGB seed robustness check for panic sub-regime clustering')
print(f'Production seeds: {PROD_SEEDS[0]}..{PROD_SEEDS[-1]}  (n={len(PROD_SEEDS)})')
print(f'Robustness seeds: {ROBUST_SEEDS[0]}..{ROBUST_SEEDS[-1]} (n={len(ROBUST_SEEDS)})')
print('=' * 72)

t0 = time.time()
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
X_train = art['X_train']
y_train = art['y_train']
X_test  = art['X_test']
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])
print(f'Loaded artefacts in {time.time()-t0:.1f}s. '
      f'X_train={X_train.shape}, X_test={X_test.shape}')

# Train 50-seed XGB ensemble with disjoint seeds
print(f'\nTraining {len(ROBUST_SEEDS)} XGB seeds with n_jobs={XGB_N_JOBS}...')
t0 = time.time()
preds_v2 = np.zeros(len(X_test))
for i, seed in enumerate(ROBUST_SEEDS, 1):
    m = XGBRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE, tree_method='hist',
        random_state=seed, verbosity=0, n_jobs=XGB_N_JOBS,
    )
    m.fit(X_train, y_train)
    preds_v2 += m.predict(X_test)
    if i % 10 == 0:
        elapsed = time.time() - t0
        eta = elapsed / i * (len(ROBUST_SEEDS) - i)
        print(f'  seed {i:2d}/{len(ROBUST_SEEDS)}  '
              f'elapsed={elapsed:5.0f}s  eta={eta:5.0f}s')
preds_v2 /= len(ROBUST_SEEDS)
print(f'Total training: {time.time()-t0:.0f}s')

# Reassign legs using new score
test['score_xgb_v2'] = preds_v2
test['leg_v2'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb_v2'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb_v2'] >= hi].index, 'leg_v2'] = 'long'
    test.loc[grp[grp['score_xgb_v2'] <= lo].index, 'leg_v2'] = 'short'

# Compute long-leg z-by-month under new ensemble
rows_v2 = []
for date, grp in test.groupby('date'):
    long_mask = grp['leg_v2'] == 'long'
    row = {'date': date}
    if long_mask.sum() == 0:
        for h in HORIZONS:
            row[f'mom_{h}'] = np.nan
    else:
        for h in HORIZONS:
            col = f'mom_{h}'
            mean = grp[col].mean()
            std = grp[col].std()
            if std > 0:
                zs = (grp.loc[long_mask, col] - mean) / std
                row[f'mom_{h}'] = zs.mean()
            else:
                row[f'mom_{h}'] = np.nan
    rows_v2.append(row)
long_v2 = pd.DataFrame(rows_v2).sort_values('date').reset_index(drop=True)

# Production long-by-month z (already computed)
long_prod = pd.read_csv(f'{RESULTS_THESIS_DIR}/zscore_long_by_month.csv',
                        parse_dates=['date'])

# Attach pi_filter
panel = pd.read_parquet(PANEL_WITH_REGIMES_PATH)
panel['date'] = pd.to_datetime(panel['date'])
pi = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').sort_values('date')

L_prod = long_prod.merge(pi, on='date').dropna(subset=['pi_filter']).reset_index(drop=True)
L_v2   = long_v2  .merge(pi, on='date').dropna(subset=['pi_filter']).reset_index(drop=True)
assert (L_prod['date'].values == L_v2['date'].values).all()

# Cluster panic months under both ensembles
panic = L_prod['pi_filter'] > 0.5
X_prod = L_prod.loc[panic, COLS].values
X_v2   = L_v2  .loc[panic, COLS].values

km_prod = KMeans(n_clusters=3, random_state=RNG_SEED, n_init=20).fit(X_prod)
km_v2   = KMeans(n_clusters=3, random_state=RNG_SEED, n_init=20).fit(X_v2)

# Re-label both clusterings by depth-of-mean (so labels match across runs)
def relabel_by_depth(km, X):
    labels = km.labels_.copy()
    means = pd.DataFrame(X).groupby(labels).mean().mean(axis=1)
    order = means.sort_values().index.tolist()
    new = {old: new_i for new_i, old in enumerate(order)}
    return np.array([new[l] for l in labels])

lab_prod = relabel_by_depth(km_prod, X_prod)
lab_v2   = relabel_by_depth(km_v2,   X_v2)

# Compare
sizes_prod = tuple(sorted(np.bincount(lab_prod, minlength=3).tolist()))
sizes_v2   = tuple(sorted(np.bincount(lab_v2,   minlength=3).tolist()))
ari        = adjusted_rand_score(lab_prod, lab_v2)
agree      = (lab_prod == lab_v2).mean()
sil_prod   = silhouette_score(X_prod, km_prod.labels_)
sil_v2     = silhouette_score(X_v2,   km_v2.labels_)

# Centroid pairwise distances after depth-relabel (depth 0 = deep_crisis)
def centroids(km, X, labels_relabeled):
    return np.vstack([X[labels_relabeled == k].mean(axis=0) for k in range(3)])

C_prod = centroids(km_prod, X_prod, lab_prod)
C_v2   = centroids(km_v2,   X_v2,   lab_v2)
cluster_centroid_shifts = np.linalg.norm(C_prod - C_v2, axis=1)  # one per cluster

# Cross-tab of cluster membership
cross = pd.crosstab(pd.Series(lab_prod, name='production'),
                    pd.Series(lab_v2,   name='robustness'))

# Save all outputs to one csv
out_rows = []
out_rows.append({'metric': 'n_panic_months', 'value': int(panic.sum())})
out_rows.append({'metric': 'cluster_sizes_production', 'value': '|'.join(map(str, sizes_prod))})
out_rows.append({'metric': 'cluster_sizes_robustness', 'value': '|'.join(map(str, sizes_v2))})
out_rows.append({'metric': 'ARI_prod_vs_robust', 'value': round(float(ari), 4)})
out_rows.append({'metric': 'pct_months_same_cluster', 'value': round(float(agree)*100, 2)})
out_rows.append({'metric': 'silhouette_production', 'value': round(float(sil_prod), 4)})
out_rows.append({'metric': 'silhouette_robustness', 'value': round(float(sil_v2), 4)})
for k, names in enumerate(['deep_crisis', 'transition', 'mild_panic']):
    out_rows.append({'metric': f'centroid_shift_L2_{names}',
                     'value': round(float(cluster_centroid_shifts[k]), 4)})

out_path = f'{RESULTS_THESIS_DIR}/zscore_subregime_xgb_seed_robustness.csv'
pd.DataFrame(out_rows).to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')

# Console report
print('\n' + '=' * 72)
print('XGB SEED ROBUSTNESS RESULTS')
print('=' * 72)
print(f'  Cluster sizes (sorted):')
print(f'    production : {sizes_prod}')
print(f'    robustness : {sizes_v2}')
print(f'  Adjusted Rand Index (1.0 = identical clustering): {ari:.4f}')
print(f'  Months in same cluster across runs: {agree*100:.1f}%')
print(f'  Silhouette score:')
print(f'    production : {sil_prod:.4f}')
print(f'    robustness : {sil_v2:.4f}')
print(f'  Cluster centroid L2 shifts (in 12-d z-space):')
for k, name in enumerate(['deep_crisis', 'transition', 'mild_panic']):
    print(f'    {name:12s}: {cluster_centroid_shifts[k]:.4f}')
print(f'\n  Cross-tab (production rows vs robustness cols):')
print('  ' + cross.to_string().replace('\n', '\n  '))
print('\nDone.')
