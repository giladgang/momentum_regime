"""Per-regime leaf clustering, memory-safe via shared PCA.

Strategy: load full LV once, do PCA -> 50 dims (small), then subset
calm/panic in the 50-d PCA space and cluster each subset. The 50-d
matrix is ~17 MB which fits trivially.

  - calm subset: KMeans k=2 (matches earlier output-side k=2)
  - panic subset: KMeans k=3 (look for within-panic substructure)

Outputs:
  results/thesis/rule_path_per_regime_v3_sweep.csv
  results/thesis/rule_path_per_regime_v3_labels.csv
  artefacts/rule_path_per_regime_v3_pca.npz   (50-d PCA matrix)
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_clustering_helpers import silhouette_sparse_subsample

LEAF_PATH = 'artefacts/leaf_signatures.npz'
PCA_PATH = 'artefacts/rule_path_per_regime_v3_pca.npz'
RES_DIR = 'results/thesis'
RNG_SEED = 42
N_INIT = 20
PCA_DIM = 50
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
ARI_THRESHOLD = 0.90


def cluster_subset_pca(Z_subset, regime_name, k):
    """KMeans on already-PCA-reduced subset."""
    print(f'\n{regime_name.upper()} (PCA-reduced): {Z_subset.shape}, k={k}',
          flush=True)
    fits = []
    sils = []
    for s in STABILITY_SEEDS:
        km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(Z_subset)
        sil = silhouette_sparse_subsample(Z_subset, km.labels_,
                                          n_samples=5000, seed=s)
        fits.append(km.labels_)
        sils.append(sil)
    pair_aris = []
    for i in range(len(STABILITY_SEEDS)):
        for j in range(i + 1, len(STABILITY_SEEDS)):
            pair_aris.append(float(adjusted_rand_score(fits[i], fits[j])))
    mean_sil = float(np.mean(sils))
    mean_ari = float(np.mean(pair_aris))
    sizes = sorted(np.bincount(fits[0], minlength=k).tolist())
    print(f'  k={k}: mean_sil={mean_sil:.3f}, mean_ari={mean_ari:.3f}, '
          f'sizes(seed42)={tuple(sizes)}, '
          f'stable={"YES" if mean_ari >= ARI_THRESHOLD else "NO"}', flush=True)
    return fits[0], mean_sil, mean_ari, sizes


def get_pca_or_compute():
    """Load cached PCA representation or compute and save."""
    if os.path.exists(PCA_PATH):
        print(f'Loading cached PCA from {PCA_PATH}...', flush=True)
        d = np.load(PCA_PATH, allow_pickle=False)
        return d['Z'], d['date'], d['permno']

    print(f'Computing PCA (will be saved to {PCA_PATH})...', flush=True)
    d = np.load(LEAF_PATH, allow_pickle=False)
    LV = d['leaf_values'].astype(np.float32)
    date = d['date']
    permno = d['permno']
    d.close()
    print(f'  LV shape: {LV.shape}', flush=True)

    col_std = LV.std(axis=0).astype(np.float32)
    keep = col_std > 0
    print(f'  zero-variance trees: {(~keep).sum()}', flush=True)
    if (~keep).sum() > 0:
        LV = np.ascontiguousarray(LV[:, keep], dtype=np.float32)
        col_std = col_std[keep]
    col_mean = LV.mean(axis=0).astype(np.float32)
    print(f'  standardising in-place...', flush=True)
    LV -= col_mean[np.newaxis, :]
    LV /= col_std[np.newaxis, :]

    print(f'  TruncatedSVD to {PCA_DIM} components...', flush=True)
    svd = TruncatedSVD(n_components=PCA_DIM, random_state=RNG_SEED)
    Z = svd.fit_transform(LV)
    del LV
    print(f'  PCA explained variance ratio sum: '
          f'{svd.explained_variance_ratio_.sum():.3f}', flush=True)

    np.savez_compressed(PCA_PATH, Z=Z, date=date, permno=permno)
    print(f'  saved PCA to {PCA_PATH} ({Z.nbytes / 1e6:.1f} MB)', flush=True)
    return Z, date, permno


def main():
    Z, date, permno = get_pca_or_compute()

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    pi = test.groupby('date')['pi_filter'].first()

    label_dates = pd.to_datetime(date)
    regime = pd.Series(label_dates).map(
        lambda x: 'panic' if pi.get(x, 0) > 0.5 else 'calm'
    ).values
    is_calm = regime == 'calm'
    is_panic = regime == 'panic'
    print(f'\ncalm rows: {is_calm.sum()}, panic rows: {is_panic.sum()}',
          flush=True)

    sweep_rows = []

    # Calm k=2
    Z_calm = Z[is_calm]
    print(f'  calm Z shape: {Z_calm.shape} '
          f'({Z_calm.nbytes / 1e6:.1f} MB)', flush=True)
    calm_labels, calm_sil, calm_ari, calm_sizes = cluster_subset_pca(
        Z_calm, 'calm', k=2
    )
    sweep_rows.append({
        'regime': 'calm', 'k': 2,
        'mean_silhouette': round(calm_sil, 4),
        'mean_ari': round(calm_ari, 4),
        'stable_at_threshold': calm_ari >= ARI_THRESHOLD,
        'sizes_seed42': str(tuple(calm_sizes)),
    })

    # Panic k=3
    Z_panic = Z[is_panic]
    print(f'  panic Z shape: {Z_panic.shape} '
          f'({Z_panic.nbytes / 1e6:.1f} MB)', flush=True)
    panic_labels, panic_sil, panic_ari, panic_sizes = cluster_subset_pca(
        Z_panic, 'panic', k=3
    )
    sweep_rows.append({
        'regime': 'panic', 'k': 3,
        'mean_silhouette': round(panic_sil, 4),
        'mean_ari': round(panic_ari, 4),
        'stable_at_threshold': panic_ari >= ARI_THRESHOLD,
        'sizes_seed42': str(tuple(panic_sizes)),
    })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f'{RES_DIR}/rule_path_per_regime_v3_sweep.csv', index=False)
    print(f'\nSaved: {RES_DIR}/rule_path_per_regime_v3_sweep.csv', flush=True)
    print(sweep_df.to_string(index=False))

    # Combine: calm 0..1, panic 2..4
    full_labels = np.zeros(len(date), dtype=np.int8)
    full_labels[is_calm] = calm_labels
    full_labels[is_panic] = panic_labels + 2

    final_df = pd.DataFrame({
        'date': pd.to_datetime(date),
        'permno': permno,
        'rule_id': full_labels,
    })
    final_df.to_csv(f'{RES_DIR}/rule_path_per_regime_v3_labels.csv',
                    index=False)
    print(f'Saved: {RES_DIR}/rule_path_per_regime_v3_labels.csv', flush=True)


if __name__ == '__main__':
    main()
