"""
tree_path_momentum_only.py
===========================
Trace stocks through MOMENTUM-HEAVY trees only (>50% momentum splits).
Find the most common momentum decision paths for each group.

Usage:
    python -u scripts/tree_path_momentum_only.py
"""

import numpy as np
import pandas as pd
import pickle, sys, os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading artefacts ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
all_trees = art['all_trees']
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

# Filter to momentum-heavy trees only
mom_trees = []
for tree in all_trees:
    pi_splits = sum(1 for n in tree.values() if not n['leaf'] and n['feature'] == 'pi_filter')
    mom_splits = sum(1 for n in tree.values() if not n['leaf'] and n['feature'] != 'pi_filter')
    total = pi_splits + mom_splits
    if total > 0 and mom_splits / total > 0.5:
        mom_trees.append(tree)

print(f"Momentum-heavy trees: {len(mom_trees)} / {len(all_trees)}")

# Trace function -- records only momentum splits, skips pi_filter splits
def trace_stock_mom_only(tree, feature_values, max_depth=4):
    node_id = 0
    mom_path = []
    while True:
        node = tree[node_id]
        if node['leaf']:
            return tuple(mom_path), node['value']
        feat = node['feature']
        thresh = node['threshold']
        feat_idx = FEATURES.index(feat)
        val = feature_values[feat_idx]
        if val < thresh:
            node_id = node['yes']
            direction = 'L'
        else:
            node_id = node['no']
            direction = 'H'
        # Only record momentum splits
        if feat != 'pi_filter':
            mom_path.append(f"{feat.replace('mom_','m')}_{direction}")

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
    print(f"  {gname}: {n_stocks:,} stocks x {len(mom_trees)} momentum trees")
    print(f"{'='*70}")

    path_counter = Counter()
    path_leaves = defaultdict(list)

    for tree_idx, tree in enumerate(mom_trees):
        if tree_idx % 100 == 0:
            print(f"  Tree {tree_idx}/{len(mom_trees)}...", flush=True)
        for stock_idx in range(n_stocks):
            path, leaf_val = trace_stock_mom_only(tree, X_group[stock_idx])
            if path:  # only count non-empty momentum paths
                path_counter[path] += 1
                if len(path_leaves[path]) < 10000:  # cap memory
                    path_leaves[path].append(leaf_val)

    total = n_stocks * len(mom_trees)

    print(f"\n  TOP MOMENTUM PATHS (pi_filter splits excluded):")
    print(f"  Total traces: {total:,}, Unique patterns: {len(path_counter)}")

    for path, count in path_counter.most_common(15):
        pct = count / total * 100
        avg_leaf = np.mean(path_leaves[path]) * 100
        readable = ' -> '.join([p.replace('_L', '<').replace('_H', '>=') for p in path])
        print(f"    {readable:<55s} {count:>7,} ({pct:>5.1f}%)  leaf={avg_leaf:+.3f}%")

print("\nDone.")
