"""Phase 2 supplementary: unified k=3 leaf-level clustering.

The main cluster_rule_paths.py picked k=2 as winner. k=3 also passed the
stability gate (mean ARI 0.9999) but had slightly lower silhouette.
This script saves the k=3 labels for downstream characterisation —
we want to see the within-panic substructure (k=3 splits the panic
cluster of size 33836 into two sub-clusters of size 13772 and 20064).

Outputs:
  results/thesis/rule_path_k3_labels.csv
  artefacts/rule_path_k3_labels.npz
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEAF_PATH = 'artefacts/leaf_signatures.npz'
LABELS_NPZ = 'artefacts/rule_path_k3_labels.npz'
RES_DIR = 'results/thesis'
RNG_SEED = 42
N_INIT = 20
PCA_DIM = 50
K = 3


def main():
    print('Loading leaf-value signatures...', flush=True)
    d = np.load(LEAF_PATH, allow_pickle=False)
    LV = d['leaf_values'].astype(np.float32)
    date = d['date']
    permno = d['permno']
    print(f'  shape: {LV.shape}', flush=True)

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

    print(f'  KMeans k={K}...', flush=True)
    km = KMeans(n_clusters=K, random_state=RNG_SEED, n_init=N_INIT).fit(Z)
    sizes = sorted(np.bincount(km.labels_, minlength=K).tolist())
    print(f'  sizes: {tuple(sizes)}', flush=True)

    np.savez_compressed(
        LABELS_NPZ,
        labels=km.labels_,
        date=date,
        permno=permno,
    )
    print(f'Saved: {LABELS_NPZ}', flush=True)

    final_df = pd.DataFrame({
        'date': pd.to_datetime(date),
        'permno': permno,
        'rule_id': km.labels_,
    })
    final_df.to_csv(f'{RES_DIR}/rule_path_k3_labels.csv', index=False)
    print(f'Saved: {RES_DIR}/rule_path_k3_labels.csv ({len(final_df)} rows)',
          flush=True)


if __name__ == '__main__':
    main()
