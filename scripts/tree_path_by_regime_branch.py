"""
tree_path_by_regime_branch.py
==============================
For trees that split on pi_filter at the root: what momentum combos
follow in the calm branch vs the panic branch?

This reveals the regime-conditional decision logic: the tree asks
"what regime?" first, then asks different momentum questions
depending on the answer.

Usage:
    python -u scripts/tree_path_by_regime_branch.py
"""

import numpy as np
import pandas as pd
import pickle, sys, os, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading artefacts ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
all_trees = art['all_trees']  # 500 trees from saved model
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

test_pi['leg'] = 'middle'
for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10: continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test_pi.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test_pi.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

X_test = test_pi[FEATURES].values.astype(float)
regime_arr = test_pi['regime'].values
leg_arr = test_pi['leg'].values

# Find trees that split on pi_filter at root
pi_root_trees = []
for tree in all_trees:
    root = tree[0]
    if not root['leaf'] and root['feature'] == 'pi_filter':
        pi_root_trees.append(tree)

print(f"Trees with pi_filter at root: {len(pi_root_trees)} / {len(all_trees)}")

def trace_after_root(tree, feature_values):
    """
    Trace stock through a pi_filter-root tree.
    Returns: (pi_direction, momentum_path, splits, leaf_value)
    pi_direction: 'low' (calm branch) or 'high' (panic branch)
    momentum_path: tuple of momentum splits after the root
    """
    root = tree[0]
    pi_idx = FEATURES.index('pi_filter')
    pi_val = feature_values[pi_idx]
    thresh = root['threshold']

    if pi_val < thresh:
        pi_direction = 'low'
        node_id = root['yes']
    else:
        pi_direction = 'high'
        node_id = root['no']

    # Now trace the rest, recording only momentum splits
    mom_path = []
    splits = []
    while True:
        node = tree[node_id]
        if node['leaf']:
            return pi_direction, tuple(mom_path), splits, node['value']
        feat = node['feature']
        feat_idx = FEATURES.index(feat)
        val = feature_values[feat_idx]
        thresh_node = node['threshold']
        if val < thresh_node:
            direction = 'L'
            node_id = node['yes']
        else:
            direction = 'H'
            node_id = node['no']
        if feat != 'pi_filter':
            mom_path.append(f"{feat.replace('mom_','m')}_{direction}")
            splits.append({
                'feature': feat,
                'threshold': thresh_node,
                'stock_value': val,
                'direction': direction,
            })

SAMPLE_SIZE = 5000

groups = {
    'Calm Long': (regime_arr == 'Calm') & (leg_arr == 'long'),
    'Calm Short': (regime_arr == 'Calm') & (leg_arr == 'short'),
    'Panic Long': (regime_arr == 'Panic') & (leg_arr == 'long'),
    'Panic Short': (regime_arr == 'Panic') & (leg_arr == 'short'),
}

for gname, gmask in groups.items():
    indices = np.where(gmask)[0]
    if len(indices) > SAMPLE_SIZE:
        rng = np.random.RandomState(42)
        indices = rng.choice(indices, SAMPLE_SIZE, replace=False)

    X_group = X_test[indices]
    n_stocks = len(X_group)

    print(f"\n{'='*70}")
    print(f"  {gname}: {n_stocks:,} stocks x {len(pi_root_trees)} pi-root trees")
    print(f"{'='*70}")

    # Separate counters for calm branch and panic branch
    branch_data = {
        'low': {'path_counter': Counter(), 'path_leaves': defaultdict(list),
                'path_splits': defaultdict(lambda: defaultdict(lambda: {'thresholds': [], 'values': []})),
                'count': 0},
        'high': {'path_counter': Counter(), 'path_leaves': defaultdict(list),
                 'path_splits': defaultdict(lambda: defaultdict(lambda: {'thresholds': [], 'values': []})),
                 'count': 0},
    }

    t0 = time.time()
    for tree_idx, tree in enumerate(pi_root_trees):
        if tree_idx % 50 == 0 and tree_idx > 0:
            print(f"    Tree {tree_idx}/{len(pi_root_trees)}...", flush=True)
        for stock_idx in range(n_stocks):
            pi_dir, path, splits, leaf_val = trace_after_root(tree, X_group[stock_idx])
            if not path:
                continue
            bd = branch_data[pi_dir]
            bd['count'] += 1
            bd['path_counter'][path] += 1
            if len(bd['path_leaves'][path]) < 5000:
                bd['path_leaves'][path].append(leaf_val)
            for s in splits:
                feat = s['feature']
                entry = bd['path_splits'][path][feat]
                if len(entry['thresholds']) < 5000:
                    entry['thresholds'].append(s['threshold'])
                    entry['values'].append(s['stock_value'])

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s")

    for branch_name, branch_label in [('low', 'CALM BRANCH (pi_filter < threshold)'),
                                       ('high', 'PANIC BRANCH (pi_filter >= threshold)')]:
        bd = branch_data[branch_name]
        total = bd['count']
        if total == 0:
            print(f"\n  {branch_label}: no traces")
            continue

        print(f"\n  {branch_label} ({total:,} traces):")

        for rank, (path, count) in enumerate(bd['path_counter'].most_common(8)):
            if len(path) < 1:
                continue
            pct = count / total * 100
            avg_leaf = np.mean(bd['path_leaves'][path]) * 100
            readable = ' -> '.join([p.replace('_L', '<').replace('_H', '>=') for p in path])

            print(f"\n    #{rank+1}: {readable}")
            print(f"         Count: {count:,} ({pct:.1f}%)  Avg leaf: {avg_leaf:+.4f}%")
            for step in path:
                feat_name = 'mom_' + step.split('_')[0][1:]
                if feat_name in bd['path_splits'][path]:
                    s = bd['path_splits'][path][feat_name]
                    feat_short = feat_name.replace('mom_', 'm')
                    dir_label = 'below' if step.endswith('_L') else 'above'
                    print(f"         {feat_short}: stock at {np.median(s['values'])*100:+.1f}%, "
                          f"{dir_label} threshold {np.median(s['thresholds'])*100:+.1f}%")

print("\nDone.")
