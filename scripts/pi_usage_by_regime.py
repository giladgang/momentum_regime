"""
pi_usage_by_regime.py
=====================
For stocks in panic vs calm months, what fraction of their tree paths
actually encounter a pi_filter split?
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test['regime'] = np.where(test['pi_month'] >= 0.5, 'Panic', 'Calm')

xgb_model = joblib.load('artefacts/cs_artefacts_xgb.pkl')

# Parse trees
dump = xgb_model.get_booster().get_dump(dump_format='text')
all_trees = []
for raw in dump:
    nodes = {}
    for line in raw.strip().split('\n'):
        line_stripped = line.strip()
        node_id = int(re.match(r'(\d+):', line_stripped).group(1))
        if 'leaf' in line_stripped:
            val = float(re.search(r'leaf=([-\d.e+]+)', line_stripped).group(1))
            nodes[node_id] = dict(leaf=True, value=val)
        else:
            m = re.search(r'\[f(\d+)<([-\d.e+]+)\].*yes=(\d+),no=(\d+)', line_stripped)
            fidx = int(m.group(1))
            nodes[node_id] = dict(leaf=False,
                                  feature=FEATURES[fidx],
                                  threshold=float(m.group(2)),
                                  yes=int(m.group(3)), no=int(m.group(4)))
    all_trees.append(nodes)

print(f"Trees: {len(all_trees)}")

X_test = test[FEATURES].values.astype(float)
regime_arr = test['regime'].values

# Sample stocks
SAMPLE = 2000
rng = np.random.RandomState(42)

for regime in ['Calm', 'Panic']:
    indices = np.where(regime_arr == regime)[0]
    if len(indices) > SAMPLE:
        indices = rng.choice(indices, SAMPLE, replace=False)

    X_group = X_test[indices]

    total_paths = 0
    paths_with_pi = 0
    paths_pi_at_root = 0

    # Also track: per tree, does it have pi_filter?
    trees_with_pi = 0
    trees_encountered_pi = 0  # trees where at least one sampled stock hit pi_filter

    per_tree_pi_count = []

    for tree_idx, tree in enumerate(all_trees):
        tree_has_pi = any(n.get('feature') == 'pi_filter' for n in tree.values() if not n.get('leaf', False))
        if tree_has_pi:
            trees_with_pi += 1

        tree_pi_hits = 0
        for stock_idx in range(len(X_group)):
            node_id = 0
            hit_pi = False
            hit_pi_root = False
            while True:
                node = tree[node_id]
                if node['leaf']:
                    break
                feat = node['feature']
                feat_idx = FEATURES.index(feat)
                if feat == 'pi_filter':
                    hit_pi = True
                    if node_id == 0:
                        hit_pi_root = True
                if X_group[stock_idx, feat_idx] < node['threshold']:
                    node_id = node['yes']
                else:
                    node_id = node['no']

            total_paths += 1
            if hit_pi:
                paths_with_pi += 1
                tree_pi_hits += 1
            if hit_pi_root:
                paths_pi_at_root += 1

        if tree_pi_hits > 0:
            trees_encountered_pi += 1
        per_tree_pi_count.append(tree_pi_hits)

    pct_paths = paths_with_pi / total_paths * 100
    pct_root = paths_pi_at_root / total_paths * 100

    print(f"\n{regime}:")
    print(f"  Total paths: {total_paths:,}")
    print(f"  Paths hitting pi_filter: {paths_with_pi:,} ({pct_paths:.1f}%)")
    print(f"  Paths with pi at root:   {paths_pi_at_root:,} ({pct_root:.1f}%)")
    print(f"  Trees containing pi:     {trees_with_pi}/{len(all_trees)}")
    print(f"  Trees where stocks hit pi: {trees_encountered_pi}/{len(all_trees)}")

    # Distribution of pi_filter hit rate per tree
    arr = np.array(per_tree_pi_count)
    pct_per_tree = arr / len(X_group) * 100
    print(f"  Per-tree hit rate: mean={pct_per_tree.mean():.1f}%, "
          f"median={np.median(pct_per_tree):.1f}%, "
          f"min={pct_per_tree.min():.1f}%, max={pct_per_tree.max():.1f}%")

    # Breakdown: 0%, 1-50%, 50-99%, 100%
    print(f"  Trees where 0% of stocks hit pi:     {(arr == 0).sum()}")
    print(f"  Trees where 1-50% hit pi:            {((arr > 0) & (arr < len(X_group)//2)).sum()}")
    print(f"  Trees where 50-99% hit pi:           {((arr >= len(X_group)//2) & (arr < len(X_group))).sum()}")
    print(f"  Trees where 100% of stocks hit pi:   {(arr == len(X_group)).sum()}")

print("\nDone.")
