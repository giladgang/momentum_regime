# scripts/picked_stock_cluster_search.py
"""Picked-stock cluster + feature search.

Stage 1 (this task): cluster months on the 15-d picked-stock fingerprint;
sweep K=2..5 with 6-seed stability and silhouette; pick smallest stable K
with sane size distribution; save labels + centroids + sweep CSV.

Stages 2-3 (later tasks) extend this script with predictor features,
ANOVA, multinomial logistic, decision tree, and per-cluster summary.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_feature_search_helpers import picked_stock_fingerprint

RES_DIR = 'results/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]
K_GRID = [2, 3, 4, 5]
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
N_INIT = 20
ARI_THRESHOLD = 0.90
MIN_CLUSTER_SIZE_FRAC = 0.05  # smallest cluster must be >=5% of months


def cluster_sweep(X_std):
    """Run KMeans for each K in K_GRID, with 6-seed stability + silhouette."""
    rows = []
    labels_by_k = {}
    for k in K_GRID:
        fits = []
        sils = []
        for s in STABILITY_SEEDS:
            km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(X_std)
            fits.append(km.labels_)
            sils.append(silhouette_score(X_std, km.labels_))
        pair_aris = []
        for i in range(len(STABILITY_SEEDS)):
            for j in range(i + 1, len(STABILITY_SEEDS)):
                pair_aris.append(adjusted_rand_score(fits[i], fits[j]))
        sizes = sorted(np.bincount(fits[0], minlength=k).tolist())
        rows.append({
            'k': k,
            'mean_silhouette': round(float(np.mean(sils)), 4),
            'mean_ari': round(float(np.mean(pair_aris)), 4),
            'min_size_frac': round(min(sizes) / len(X_std), 4),
            'sizes_seed42': str(tuple(sizes)),
            'stable': bool(np.mean(pair_aris) >= ARI_THRESHOLD),
        })
        labels_by_k[k] = fits[0]
        print(f'  k={k}: silhouette={rows[-1]["mean_silhouette"]:.3f}, '
              f'ari={rows[-1]["mean_ari"]:.3f}, sizes={sizes}, '
              f'stable={rows[-1]["stable"]}', flush=True)
    return pd.DataFrame(rows), labels_by_k


def pick_k(sweep_df):
    """Smallest K that is stable AND whose smallest cluster is >=5%."""
    eligible = sweep_df[
        sweep_df['stable']
        & (sweep_df['min_size_frac'] >= MIN_CLUSTER_SIZE_FRAC)
    ].sort_values('k')
    if len(eligible) == 0:
        # Fall back to most stable K
        return int(sweep_df.sort_values('mean_ari', ascending=False).iloc[0]['k'])
    return int(eligible.iloc[0]['k'])


def main():
    print('Loading data...', flush=True)
    picks = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv',
                        parse_dates=['date'])[['date', 'permno']]
    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    panel = art['test'].copy()
    panel['date'] = pd.to_datetime(panel['date'])

    print('Computing fingerprint...', flush=True)
    fp = picked_stock_fingerprint(picks, panel, MOM_COLS)
    fp = fp.dropna()
    print(f'  fingerprint shape: {fp.shape}', flush=True)

    X = fp.values.astype(np.float64)
    X_std = StandardScaler().fit_transform(X)

    print('\n=== Cluster sweep K=2..5 ===', flush=True)
    sweep_df, labels_by_k = cluster_sweep(X_std)
    sweep_df.to_csv(f'{RES_DIR}/picked_stock_cluster_sweep.csv', index=False)
    print(f'\nSaved: {RES_DIR}/picked_stock_cluster_sweep.csv', flush=True)
    print(sweep_df.to_string(index=False))

    chosen_k = pick_k(sweep_df)
    print(f'\n>>> CHOSEN K = {chosen_k}', flush=True)

    labels = labels_by_k[chosen_k]
    label_df = pd.DataFrame({
        'date': fp.index,
        'cluster': labels,
    })
    label_df.to_csv(f'{RES_DIR}/picked_stock_cluster_labels.csv', index=False)
    print(f'Saved: {RES_DIR}/picked_stock_cluster_labels.csv', flush=True)

    # Centroids on raw (un-standardised) fingerprint
    centroids = fp.assign(cluster=labels).groupby('cluster').mean()
    centroids['n_months'] = pd.Series(labels).value_counts().sort_index().values
    centroids.to_csv(f'{RES_DIR}/picked_stock_cluster_centroids.csv')
    print(f'Saved: {RES_DIR}/picked_stock_cluster_centroids.csv', flush=True)
    print('\nCentroids (raw mom) per cluster:')
    print(centroids.round(3).to_string())


if __name__ == '__main__':
    main()
