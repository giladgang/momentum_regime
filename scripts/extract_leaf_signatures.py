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

Checkpointing: saves every SAVE_EVERY trees. If interrupted, restarting
the script resumes from the last completed tree.

Output:
- artefacts/leaf_signatures.npz with keys:
    'leaf_values'        float32 (n_long, 25000)  per-tree contributions
    'leaves_raw'         int32   (n_long, 25000)  raw leaf node ids
    'leaves_dense'       int32   (n_long, 25000)  dense [0, n_leaves_t) indices
    'n_leaves_per_tree'  int32   (25000,)
    'date'               datetime64 (n_long,)
    'permno'             int64   (n_long,)
    'last_tree_completed' int     scalar
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
SAVE_EVERY = 500   # checkpoint after every N trees


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
    print('Precomputing leaf counts and remaps...')
    n_leaves_per_tree = np.array(
        [count_leaves(t['tree']) for t in trees_wrapped],
        dtype=np.int32,
    )
    remaps = [leaf_id_remap(t['tree']) for t in trees_wrapped]

    # ---- Resume-or-init ----
    if os.path.exists(OUT_PATH):
        d = np.load(OUT_PATH, allow_pickle=False)
        if (
            d['leaf_values'].shape == (n_long, n_trees)
            and 'last_tree_completed' in d.files
        ):
            leaf_values = d['leaf_values'].copy()
            leaves_raw = d['leaves_raw'].copy()
            leaves_dense = d['leaves_dense'].copy()
            last_completed = int(d['last_tree_completed'])
            print(f'Resuming: trees 0..{last_completed} already done. '
                  f'Continuing from tree {last_completed + 1}.')
        else:
            print(f'Existing {OUT_PATH} has wrong shape; starting from scratch.')
            leaf_values = np.zeros((n_long, n_trees), dtype=np.float32)
            leaves_raw = np.zeros((n_long, n_trees), dtype=np.int32)
            leaves_dense = np.zeros((n_long, n_trees), dtype=np.int32)
            last_completed = -1
    else:
        leaf_values = np.zeros((n_long, n_trees), dtype=np.float32)
        leaves_raw = np.zeros((n_long, n_trees), dtype=np.int32)
        leaves_dense = np.zeros((n_long, n_trees), dtype=np.int32)
        last_completed = -1

    # ---- Date / permno arrays ----
    date_arr = df.iloc[long_idx]['date'].values.astype('datetime64[ns]')
    permno_arr = df.iloc[long_idx]['permno'].values.astype(np.int64)

    # ---- Extract ----
    if last_completed + 1 >= n_trees:
        print('All trees already extracted. Nothing to do.')
        return

    print(f'Extracting trees {last_completed + 1}..{n_trees - 1}...')
    t0 = time.time()
    last_save = time.time()
    X_long = X[long_idx]   # subset once

    for ti in range(last_completed + 1, n_trees):
        tree_dict = trees_wrapped[ti]['tree']
        remap = remaps[ti]

        # Vectorised: route ALL long-leg stocks through this tree at once
        raw_leaves = route_all_stocks_through_tree(X_long, tree_dict, features)
        leaves_raw[:, ti] = raw_leaves

        # Per-stock leaf value lookup (vectorised via dict comprehension)
        leaf_value_map = {
            nid: float(node['value'])
            for nid, node in tree_dict.items() if node.get('leaf', False)
        }
        leaves_dense[:, ti] = np.array([remap[int(nid)] for nid in raw_leaves],
                                       dtype=np.int32)
        leaf_values[:, ti] = np.array([leaf_value_map[int(nid)] for nid in raw_leaves],
                                      dtype=np.float32)

        if (ti + 1) % SAVE_EVERY == 0 or ti == n_trees - 1:
            np.savez_compressed(
                OUT_PATH,
                leaf_values=leaf_values,
                leaves_raw=leaves_raw,
                leaves_dense=leaves_dense,
                n_leaves_per_tree=n_leaves_per_tree,
                date=date_arr,
                permno=permno_arr,
                last_tree_completed=np.array(ti, dtype=np.int64),
            )
            elapsed = time.time() - t0
            since_last = time.time() - last_save
            last_save = time.time()
            done = ti + 1
            todo = n_trees - done
            rate = done / max(elapsed, 1e-6) if last_completed < 0 else (done - last_completed - 1) / max(elapsed, 1e-6)
            eta = todo / max(rate, 1e-6)
            print(f'  checkpoint at tree {ti + 1}/{n_trees}  '
                  f'elapsed {elapsed:.0f}s  +{since_last:.0f}s since last  '
                  f'ETA {eta:.0f}s')

    # ---- Final summary ----
    col_std = leaf_values.std(axis=0)
    print(f'\nDone. leaf_values shape: {leaf_values.shape}')
    print(f'  per-tree std: min={col_std.min():.5f}, '
          f'median={np.median(col_std):.5f}, '
          f'max={col_std.max():.5f}')
    print(f'  zero-variance trees (every long stock got same value): '
          f'{(col_std == 0).sum()}')
    print(f'  total leaves across ensemble: {n_leaves_per_tree.sum()}')


if __name__ == '__main__':
    main()
