"""Phase 2: cluster long-leg stock-months on standardised leaf-value matrix.

Each row of the leaf-value matrix is a stock-month's per-tree prediction
contribution. Standardising columns automatically weights trees by their
variance — high-impact early trees dominate Euclidean distance.

Pipeline:
  1. Load leaf_values matrix (~55,200 x 25,000) from Phase 1 artefact.
  2. Drop zero-variance columns; StandardScaler the rest per column.
  3. TruncatedSVD to 50 components.
  4. KMeans for k in [2, 3] x 6 stability seeds.
  5. Pick best k by joint silhouette + ARI >= 0.90 stability gate.

Outputs:
  results/thesis/rule_path_clustering_sweep.csv
  results/thesis/rule_path_labels.csv
  artefacts/rule_path_labels.npz
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_clustering_helpers import silhouette_sparse_subsample

LEAF_PATH = 'artefacts/leaf_signatures.npz'
LABELS_PATH = 'artefacts/rule_path_labels.npz'
RES_DIR = 'results/thesis'
RNG_SEED = 42
N_INIT = 20
PCA_DIM = 50
K_RANGE = [2, 3]
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
ARI_THRESHOLD = 0.90


def main():
    print('Loading leaf-value signatures...')
    d = np.load(LEAF_PATH, allow_pickle=False)
    LV = d['leaf_values'].astype(np.float32)
    date = d['date']
    permno = d['permno']
    n = LV.shape[0]
    print(f'  leaf-value matrix: {LV.shape}')

    # ---- Compute per-column std; drop zero-variance trees if any ----
    col_std = LV.std(axis=0).astype(np.float32)
    keep = col_std > 0
    n_zero = int((~keep).sum())
    print(f'  zero-variance trees: {n_zero} (all kept if 0)', flush=True)

    # If there are zero-variance trees, drop them. Otherwise reuse LV
    # in-place to avoid an 8.5 GB copy (which paged on this machine).
    if n_zero > 0:
        print(f'  copying to drop {n_zero} zero-variance columns...',
              flush=True)
        LVk = np.ascontiguousarray(LV[:, keep], dtype=np.float32)
        del LV
        col_std_keep = col_std[keep]
    else:
        LVk = LV   # reuse — no copy
        col_std_keep = col_std

    # ---- Standardise in-place (vectorised broadcast, float32) ----
    print(f'  standardising {LVk.shape[1]} columns in-place (vectorised)...',
          flush=True)
    col_mean = LVk.mean(axis=0).astype(np.float32)
    LVk -= col_mean[np.newaxis, :]
    LVk /= col_std_keep[np.newaxis, :]

    # ---- PCA reduction ----
    print(f'  TruncatedSVD to {PCA_DIM} components...', flush=True)
    svd = TruncatedSVD(n_components=PCA_DIM, random_state=RNG_SEED)
    Z = svd.fit_transform(LVk)
    del LVk  # free the standardised matrix; Z is much smaller (n x 50)
    print(f'  PCA explained variance ratio sum: '
          f'{svd.explained_variance_ratio_.sum():.3f}', flush=True)

    # ---- Sweep k with stability gating ----
    sweep_rows = []
    fits_by_seed = {}

    for k in K_RANGE:
        fits_by_seed[k] = []
        sils = []
        for s in STABILITY_SEEDS:
            km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(Z)
            sil = silhouette_sparse_subsample(Z, km.labels_,
                                              n_samples=5000, seed=s)
            fits_by_seed[k].append(km.labels_)
            sils.append(sil)

        # Pairwise ARI across seeds
        pair_aris = []
        for i in range(len(STABILITY_SEEDS)):
            for j in range(i + 1, len(STABILITY_SEEDS)):
                pair_aris.append(float(adjusted_rand_score(
                    fits_by_seed[k][i], fits_by_seed[k][j]
                )))

        mean_sil = float(np.mean(sils))
        mean_ari = float(np.mean(pair_aris))
        sizes_seed42 = sorted(np.bincount(fits_by_seed[k][0],
                                          minlength=k).tolist())
        sweep_rows.append({
            'k': k,
            'mean_silhouette': round(mean_sil, 4),
            'mean_ari': round(mean_ari, 4),
            'stable_at_threshold': bool(mean_ari >= ARI_THRESHOLD),
            'sizes_seed42': str(tuple(sizes_seed42)),
        })
        print(f'  k={k}: mean_sil={mean_sil:.3f}, mean_ari={mean_ari:.3f}, '
              f'sizes(seed42)={tuple(sizes_seed42)}, '
              f'stable={"YES" if mean_ari >= ARI_THRESHOLD else "NO"}')

    sweep_df = pd.DataFrame(sweep_rows)
    os.makedirs(RES_DIR, exist_ok=True)
    sweep_df.to_csv(f'{RES_DIR}/rule_path_clustering_sweep.csv', index=False)
    print(f'\nSaved: {RES_DIR}/rule_path_clustering_sweep.csv')

    # ---- Pick winner ----
    eligible = sweep_df[sweep_df['stable_at_threshold']]
    if len(eligible) > 0:
        best_row = eligible.loc[eligible['mean_silhouette'].idxmax()]
        print(f'\nStability gate satisfied. Best stable k={int(best_row["k"])}, '
              f'mean_sil={best_row["mean_silhouette"]:.3f}, '
              f'mean_ari={best_row["mean_ari"]:.3f}')
    else:
        best_row = sweep_df.loc[sweep_df['mean_silhouette'].idxmax()]
        print(f'\n!!! WARNING: no k passed stability gate '
              f'(ARI >= {ARI_THRESHOLD}). Falling back to '
              f'highest-silhouette k={int(best_row["k"])}, '
              f'mean_ari={best_row["mean_ari"]:.3f}.')

    best_k = int(best_row['k'])
    best_sil = float(best_row['mean_silhouette'])
    best_ari = float(best_row['mean_ari'])
    labels_final = fits_by_seed[best_k][0]   # seed-42 canonical assignment

    # ---- Save artefacts ----
    np.savez_compressed(
        LABELS_PATH,
        labels=labels_final,
        best_k=best_k,
        best_sil=best_sil,
        best_ari=best_ari,
        cols_kept=keep,
        date=date,
        permno=permno,
    )
    print(f'Saved: {LABELS_PATH}')

    final_df = pd.DataFrame({
        'date': pd.to_datetime(date),
        'permno': permno,
        'rule_id': labels_final,
    })
    final_df.to_csv(f'{RES_DIR}/rule_path_labels.csv', index=False)
    print(f'Saved: {RES_DIR}/rule_path_labels.csv ({len(final_df)} rows)')


if __name__ == '__main__':
    main()
