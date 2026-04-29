"""Phase 2 supplementary: per-regime leaf-level clustering.

Mirrors the methodology of output_first_by_regime.py but at the leaf
level: split long-leg stock-months by HMM regime first, then run KMeans
k=2 with 6-seed stability gate within each subset.

Produces 4 total rule labels (2 per regime). For comparison with the
unified k=2 (panic vs calm) and to recover within-regime substructure
that the unified clustering pools.

Outputs:
  results/thesis/rule_path_per_regime_sweep.csv
  results/thesis/rule_path_per_regime_labels.csv
  artefacts/rule_path_per_regime_labels.npz
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
LABELS_NPZ = 'artefacts/rule_path_per_regime_labels.npz'
RES_DIR = 'results/thesis'
RNG_SEED = 42
N_INIT = 20
PCA_DIM = 50
K = 2
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
ARI_THRESHOLD = 0.90


def cluster_subset(LV_subset, regime_name):
    """Standardise + PCA + KMeans k=2 with 6-seed stability gating."""
    print(f'\n{regime_name.upper()} subset: {LV_subset.shape}', flush=True)
    col_std = LV_subset.std(axis=0).astype(np.float32)
    keep = col_std > 0
    print(f'  zero-variance trees in subset: {(~keep).sum()}', flush=True)
    LVk = LV_subset if (~keep).sum() == 0 else np.ascontiguousarray(
        LV_subset[:, keep], dtype=np.float32
    )
    col_std_keep = col_std[keep] if (~keep).sum() > 0 else col_std
    col_mean = LVk.mean(axis=0).astype(np.float32)
    LVk -= col_mean[np.newaxis, :]
    LVk /= col_std_keep[np.newaxis, :]
    print(f'  standardised. TruncatedSVD to {PCA_DIM} components...',
          flush=True)
    svd = TruncatedSVD(n_components=min(PCA_DIM, LVk.shape[1] - 1),
                       random_state=RNG_SEED)
    Z = svd.fit_transform(LVk)
    print(f'  PCA explained variance ratio sum: '
          f'{svd.explained_variance_ratio_.sum():.3f}', flush=True)

    fits = []
    sils = []
    for s in STABILITY_SEEDS:
        km = KMeans(n_clusters=K, random_state=s, n_init=N_INIT).fit(Z)
        sil = silhouette_sparse_subsample(Z, km.labels_, n_samples=5000, seed=s)
        fits.append(km.labels_)
        sils.append(sil)
    pair_aris = []
    for i in range(len(STABILITY_SEEDS)):
        for j in range(i + 1, len(STABILITY_SEEDS)):
            pair_aris.append(float(adjusted_rand_score(fits[i], fits[j])))
    mean_sil = float(np.mean(sils))
    mean_ari = float(np.mean(pair_aris))
    sizes = sorted(np.bincount(fits[0], minlength=K).tolist())
    stable = mean_ari >= ARI_THRESHOLD
    print(f'  k={K}: mean_sil={mean_sil:.3f}, mean_ari={mean_ari:.3f}, '
          f'sizes(seed42)={tuple(sizes)}, stable={"YES" if stable else "NO"}',
          flush=True)
    return fits[0], mean_sil, mean_ari, sizes, stable


def main():
    print('Loading metadata + regime labels...', flush=True)
    d = np.load(LEAF_PATH, allow_pickle=False)
    date = d['date']
    permno = d['permno']

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    pi = test.groupby('date')['pi_filter'].first()

    # Map each long-leg row to its regime
    label_dates = pd.to_datetime(date)
    regime = pd.Series(label_dates).map(
        lambda d: 'panic' if pi.get(d, 0) > 0.5 else 'calm'
    ).values
    is_calm = regime == 'calm'
    is_panic = regime == 'panic'
    print(f'  calm rows: {is_calm.sum()}, panic rows: {is_panic.sum()}',
          flush=True)

    sweep_rows = []

    # Calm — load only what we need, immediately subset, then free
    print('\nLoading leaf_values (calm rows only)...', flush=True)
    LV_full = d['leaf_values']
    LV_calm = np.ascontiguousarray(LV_full[is_calm], dtype=np.float32)
    del LV_full   # free the full matrix
    d.close()
    print(f'  calm subset memory: {LV_calm.nbytes / 1e9:.1f} GB', flush=True)

    calm_labels, calm_sil, calm_ari, calm_sizes, calm_stable = cluster_subset(
        LV_calm, 'calm'
    )
    del LV_calm   # free before loading panic
    sweep_rows.append({
        'regime': 'calm',
        'k': K,
        'mean_silhouette': round(calm_sil, 4),
        'mean_ari': round(calm_ari, 4),
        'stable_at_threshold': calm_stable,
        'sizes_seed42': str(tuple(calm_sizes)),
    })

    # Panic — reload, subset, free
    print('\nLoading leaf_values (panic rows only)...', flush=True)
    d2 = np.load(LEAF_PATH, allow_pickle=False)
    LV_panic = np.ascontiguousarray(d2['leaf_values'][is_panic], dtype=np.float32)
    d2.close()
    print(f'  panic subset memory: {LV_panic.nbytes / 1e9:.1f} GB', flush=True)

    panic_labels, panic_sil, panic_ari, panic_sizes, panic_stable = cluster_subset(
        LV_panic, 'panic'
    )
    del LV_panic
    sweep_rows.append({
        'regime': 'panic',
        'k': K,
        'mean_silhouette': round(panic_sil, 4),
        'mean_ari': round(panic_ari, 4),
        'stable_at_threshold': panic_stable,
        'sizes_seed42': str(tuple(panic_sizes)),
    })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f'{RES_DIR}/rule_path_per_regime_sweep.csv', index=False)
    print(f'\nSaved: {RES_DIR}/rule_path_per_regime_sweep.csv', flush=True)
    print(sweep_df.to_string(index=False))

    # Combine into a single label vector with rule_id in {calm0, calm1, panic0, panic1}
    # encoded as integers: 0=calm0, 1=calm1, 2=panic0, 3=panic1
    full_labels = np.zeros(LV.shape[0], dtype=np.int8)
    full_labels[is_calm] = calm_labels  # 0 or 1
    full_labels[is_panic] = panic_labels + 2  # 2 or 3

    np.savez_compressed(
        LABELS_NPZ,
        labels=full_labels,
        date=date,
        permno=permno,
    )
    print(f'\nSaved: {LABELS_NPZ}', flush=True)

    final_df = pd.DataFrame({
        'date': pd.to_datetime(date),
        'permno': permno,
        'rule_id': full_labels,
    })
    final_df.to_csv(f'{RES_DIR}/rule_path_per_regime_labels.csv', index=False)
    print(f'Saved: {RES_DIR}/rule_path_per_regime_labels.csv '
          f'({len(final_df)} rows)', flush=True)


if __name__ == '__main__':
    main()
