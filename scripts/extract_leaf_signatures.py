"""Phase 1: extract leaf-VALUE matrix for long-leg stock-months across the
50-seed x 500-tree ensemble.

Each cell leaf_values[i, t] is the prediction contribution of tree t to
stock-month i — i.e. the real-valued leaf score at the leaf where stock i
landed in tree t. The sum across t (x learning_rate) recovers the stock's
score.

Standardising the columns of leaf_values automatically weights trees by
the variance of their per-stock contributions: high-impact early trees
dominate Euclidean distance, low-impact late trees barely contribute.

Also extracts leaves_dense (leaf indices) for the dominant-splits
analysis in phase 3 (tracing tree paths back to feature/threshold pairs).

No checkpointing: save once at the end. np.savez_compressed on the
(n_long, 25000) matrix takes ~77 sec; checkpointing every 500 trees
would dominate runtime. Single-shot save keeps total runtime under
10 min. If interrupted, restart from zero (acceptable: 8 min extract).

Output:
- artefacts/leaf_signatures.npz with keys:
    'leaf_values'        float32 (n_long, 25000)  per-tree contributions
    'leaves_raw'         int16   (n_long, 25000)  raw leaf node ids
    'leaves_dense'       int8    (n_long, 25000)  dense [0, n_leaves_t) indices
    'n_leaves_per_tree'  int16   (25000,)
    'date'               datetime64 (n_long,)
    'permno'             int64   (n_long,)
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leaf_clustering_helpers import (
    route_all_stocks_through_tree,
    count_leaves,
    leaf_id_remap,
)

ARTEFACTS = 'artefacts/cs_artefacts_data.pkl'
TREES_PATH = 'artefacts/pi_verify_trees_seeds50.pkl'
OUT_PATH = 'artefacts/leaf_signatures.npz'
PRINT_EVERY = 1000   # progress print every N trees


def main():
    # ---- Load ----
    print('Loading artefacts...')
    with open(ARTEFACTS, 'rb') as f:
        art = pickle.load(f)
    test = art['test']
    features = list(art['FEATURES'])
    X = np.asarray(art['X_te_s'], dtype=np.float64)

    with open(TREES_PATH, 'rb') as f:
        trees_wrapped, seeds = pickle.load(f)
    n_trees = len(trees_wrapped)
    print(f'  {n_trees} trees across {len(seeds)} seeds')

    # ---- Identify long-leg stock-months ----
    df = test.copy()
    df['leg'] = 'middle'
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        df.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    long_idx = np.where(df['leg'].values == 'long')[0]
    n_long = len(long_idx)
    print(f'  long-leg stock-months: {n_long}')

    # ---- Pre-compute leaf metadata ----
    print('Precomputing leaf counts and remaps...', flush=True)
    n_leaves_per_tree = np.array(
        [count_leaves(t['tree']) for t in trees_wrapped],
        dtype=np.int16,
    )
    remaps = [leaf_id_remap(t['tree']) for t in trees_wrapped]

    # ---- Allocate result matrices (compact dtypes, Fortran order) ----
    # leaf_values float32: ~8.5 GB
    # leaves_raw  int16:  ~4.3 GB (XGBoost node ids fit easily in int16)
    # leaves_dense int8:  ~2.1 GB (depth-4 trees have <=16 leaves)
    #
    # CRITICAL: order='F' (column-major). The hot loop writes one full
    # column per tree (M[:, ti] = ...). With row-major (C) layout, each
    # such write touches 85599 cache lines that are 25000*itemsize bytes
    # apart — measured at 2.3 writes/sec, total 30+ min. With column-
    # major (F) layout, each column is contiguous and writes go at
    # 2900+ /sec (1300x faster, total ~3 min).
    leaf_values = np.zeros((n_long, n_trees), dtype=np.float32, order='F')
    leaves_raw = np.zeros((n_long, n_trees), dtype=np.int16, order='F')
    leaves_dense = np.zeros((n_long, n_trees), dtype=np.int8, order='F')

    # ---- Date / permno arrays ----
    date_arr = df.iloc[long_idx]['date'].values.astype('datetime64[ns]')
    permno_arr = df.iloc[long_idx]['permno'].values.astype(np.int64)

    # ---- Extract (no checkpointing — single save at end) ----
    print(f'Extracting {n_trees} trees...', flush=True)
    t0 = time.time()
    X_long = X[long_idx]

    for ti in range(n_trees):
        tree_dict = trees_wrapped[ti]['tree']
        remap = remaps[ti]

        # Vectorised: route ALL long-leg stocks through this tree at once
        raw_leaves = route_all_stocks_through_tree(X_long, tree_dict, features)
        leaves_raw[:, ti] = raw_leaves

        # Per-stock leaf-value lookup
        leaf_value_map = {
            nid: float(node['value'])
            for nid, node in tree_dict.items() if node.get('leaf', False)
        }
        # Vectorised lookup via numpy
        unique_leaves = np.unique(raw_leaves)
        for ul in unique_leaves:
            mask = raw_leaves == ul
            leaves_dense[mask, ti] = remap[int(ul)]
            leaf_values[mask, ti] = leaf_value_map[int(ul)]

        if (ti + 1) % PRINT_EVERY == 0 or ti == n_trees - 1:
            elapsed = time.time() - t0
            done = ti + 1
            rate = done / max(elapsed, 1e-6)
            eta = (n_trees - done) / max(rate, 1e-6)
            print(f'  tree {done}/{n_trees}  '
                  f'elapsed {elapsed:.0f}s  rate {rate:.1f}/s  '
                  f'ETA {eta:.0f}s', flush=True)

    # ---- Single save at end ----
    print(f'\nSaving to {OUT_PATH} (compressed save takes ~80s)...', flush=True)
    t_save = time.time()
    np.savez_compressed(
        OUT_PATH,
        leaf_values=leaf_values,
        leaves_raw=leaves_raw,
        leaves_dense=leaves_dense,
        n_leaves_per_tree=n_leaves_per_tree,
        date=date_arr,
        permno=permno_arr,
    )
    print(f'  saved in {time.time()-t_save:.0f}s', flush=True)

    # ---- Summary ----
    col_std = leaf_values.std(axis=0)
    print(f'\nDone. leaf_values shape: {leaf_values.shape}', flush=True)
    print(f'  per-tree std: min={col_std.min():.5f}, '
          f'median={np.median(col_std):.5f}, '
          f'max={col_std.max():.5f}', flush=True)
    print(f'  zero-variance trees: {(col_std == 0).sum()}', flush=True)
    print(f'  total leaves across ensemble: {n_leaves_per_tree.sum()}', flush=True)


if __name__ == '__main__':
    main()
