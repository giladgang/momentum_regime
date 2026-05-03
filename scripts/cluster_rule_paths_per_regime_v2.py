"""Phase 2 supplementary v2: per-regime leaf clustering with regime-specific k.

  - calm subset: k=2
  - panic subset: k=3

Compare to unified k=3 split: does the within-panic substructure replicate
when we cluster panic in isolation (instead of pooled with calm)?

Memory-aware loading: load only the subset's rows from the npz, never
hold the full LV matrix and the subset matrix simultaneously.

Outputs:
  results/thesis/rule_path_per_regime_v2_sweep.csv
  results/thesis/rule_path_per_regime_v2_labels.csv
  artefacts/rule_path_per_regime_v2_labels.npz
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
LABELS_NPZ = 'artefacts/rule_path_per_regime_v2_labels.npz'
RES_DIR = 'results/thesis'
RNG_SEED = 42
N_INIT = 20
PCA_DIM = 50
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
ARI_THRESHOLD = 0.90


def cluster_subset(LV_subset, regime_name, k):
    """Standardise + PCA + KMeans with 6-seed stability."""
    print(f'\n{regime_name.upper()} subset: {LV_subset.shape}, k={k}',
          flush=True)
    col_std = LV_subset.std(axis=0).astype(np.float32)
    keep = col_std > 0
    n_zero = int((~keep).sum())
    print(f'  zero-variance trees in subset: {n_zero}', flush=True)
    if n_zero > 0:
        LVk = np.ascontiguousarray(LV_subset[:, keep], dtype=np.float32)
        col_std_keep = col_std[keep]
    else:
        LVk = LV_subset
        col_std_keep = col_std
    col_mean = LVk.mean(axis=0).astype(np.float32)
    print(f'  standardising in-place...', flush=True)
    LVk -= col_mean[np.newaxis, :]
    LVk /= col_std_keep[np.newaxis, :]

    print(f'  TruncatedSVD to {PCA_DIM} components...', flush=True)
    svd = TruncatedSVD(n_components=min(PCA_DIM, LVk.shape[1] - 1),
                       random_state=RNG_SEED)
    Z = svd.fit_transform(LVk)
    print(f'  PCA explained variance ratio sum: '
          f'{svd.explained_variance_ratio_.sum():.3f}', flush=True)

    fits = []
    sils = []
    for s in STABILITY_SEEDS:
        km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(Z)
        sil = silhouette_sparse_subsample(Z, km.labels_, n_samples=5000, seed=s)
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


def main():
    print('Loading metadata + regime labels...', flush=True)
    d = np.load(LEAF_PATH, allow_pickle=False)
    date = d['date']
    permno = d['permno']
    d.close()

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
    print(f'  calm rows: {is_calm.sum()}, panic rows: {is_panic.sum()}',
          flush=True)

    sweep_rows = []

    # Calm — k=2
    print('\nLoading leaf_values (calm rows only)...', flush=True)
    d2 = np.load(LEAF_PATH, allow_pickle=False)
    LV_calm = np.ascontiguousarray(d2['leaf_values'][is_calm], dtype=np.float32)
    d2.close()
    print(f'  calm subset memory: {LV_calm.nbytes / 1e9:.1f} GB', flush=True)
    calm_labels, calm_sil, calm_ari, calm_sizes = cluster_subset(
        LV_calm, 'calm', k=2
    )
    sweep_rows.append({
        'regime': 'calm', 'k': 2,
        'mean_silhouette': round(calm_sil, 4),
        'mean_ari': round(calm_ari, 4),
        'stable_at_threshold': calm_ari >= ARI_THRESHOLD,
        'sizes_seed42': str(tuple(calm_sizes)),
    })
    del LV_calm

    # Panic — k=3
    print('\nLoading leaf_values (panic rows only)...', flush=True)
    d3 = np.load(LEAF_PATH, allow_pickle=False)
    LV_panic = np.ascontiguousarray(d3['leaf_values'][is_panic], dtype=np.float32)
    d3.close()
    print(f'  panic subset memory: {LV_panic.nbytes / 1e9:.1f} GB', flush=True)
    panic_labels, panic_sil, panic_ari, panic_sizes = cluster_subset(
        LV_panic, 'panic', k=3
    )
    sweep_rows.append({
        'regime': 'panic', 'k': 3,
        'mean_silhouette': round(panic_sil, 4),
        'mean_ari': round(panic_ari, 4),
        'stable_at_threshold': panic_ari >= ARI_THRESHOLD,
        'sizes_seed42': str(tuple(panic_sizes)),
    })
    del LV_panic

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f'{RES_DIR}/rule_path_per_regime_v2_sweep.csv', index=False)
    print(f'\nSaved: {RES_DIR}/rule_path_per_regime_v2_sweep.csv', flush=True)
    print(sweep_df.to_string(index=False))

    # Combine into a single label vector with rule_id encoded:
    #   0..1   = calm sub-rules
    #   2..4   = panic sub-rules
    full_labels = np.zeros(len(date), dtype=np.int8)
    full_labels[is_calm] = calm_labels       # 0 or 1
    full_labels[is_panic] = panic_labels + 2  # 2, 3, or 4

    np.savez_compressed(
        LABELS_NPZ,
        labels=full_labels,
        date=date,
        permno=permno,
    )
    final_df = pd.DataFrame({
        'date': pd.to_datetime(date),
        'permno': permno,
        'rule_id': full_labels,
    })
    final_df.to_csv(f'{RES_DIR}/rule_path_per_regime_v2_labels.csv',
                    index=False)
    print(f'Saved: {RES_DIR}/rule_path_per_regime_v2_labels.csv '
          f'({len(final_df)} rows)', flush=True)


if __name__ == '__main__':
    main()
