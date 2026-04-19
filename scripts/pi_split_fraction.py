"""
pi_split_fraction.py
====================
Check what fraction of tree paths split on pi_filter,
and compare combo frequencies for paths WITH vs WITHOUT pi_filter splits.
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, MOM_FEATURES)

GROUPS = {
    'mom_1': 'S', 'mom_2': 'S', 'mom_3': 'S',
    'mom_4': 'M', 'mom_5': 'M', 'mom_6': 'M',
    'mom_7': 'I', 'mom_8': 'I', 'mom_9': 'I',
    'mom_10': 'L', 'mom_11': 'L', 'mom_12': 'L',
}

print("Loading data ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test['regime'] = np.where(test['pi_month'] >= 0.5, 'Panic', 'Calm')

test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

X_test = test[FEATURES].values.astype(float)
regime_arr = test['regime'].values
leg_arr = test['leg'].values

# Train a smaller set of models (10 seeds = 5000 trees, enough for this check)
from xgboost import XGBRegressor

REDUCED = MOM_FEATURES + ['pi_filter']
X_train = train[REDUCED].values.astype(float)
y_train = train['ret_fwd'].values.astype(float)

def parse_trees(booster, feature_names):
    dump = booster.get_dump(dump_format='text')
    trees = []
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
                                      feature=feature_names[fidx],
                                      threshold=float(m.group(2)),
                                      yes=int(m.group(3)), no=int(m.group(4)))
        trees.append(nodes)
    return trees

print("Training 10 models (5000 trees) ...")
all_trees = []
for xs in range(1, 11):
    xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                       learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                       colsample_bytree=COLSAMPLE, tree_method='hist',
                       random_state=xs, verbosity=0)
    xgb.fit(X_train, y_train)
    all_trees.extend(parse_trees(xgb.get_booster(), FEATURES))
print(f"Total trees: {len(all_trees)}")

# Check: how many trees have pi_filter as a split feature ANYWHERE in the tree?
pi_in_tree = 0
for tree in all_trees:
    has_pi = any(n.get('feature') == 'pi_filter' for n in tree.values() if not n.get('leaf', False))
    if has_pi:
        pi_in_tree += 1
print(f"\nTrees with pi_filter split anywhere: {pi_in_tree}/{len(all_trees)} ({pi_in_tree/len(all_trees)*100:.1f}%)")

# Check: at what depth does pi_filter typically split?
pi_depths = []
for tree in all_trees:
    for node_id, node in tree.items():
        if not node.get('leaf', False) and node.get('feature') == 'pi_filter':
            # compute depth by tracing from root
            depth = 0
            cur = 0
            path = [0]
            # simple: just count based on node structure
            # Actually, let's find depth differently
            break
    # simpler: check root
    root = tree[0]
    if not root.get('leaf', False) and root.get('feature') == 'pi_filter':
        pi_depths.append(0)

print(f"Trees with pi_filter at ROOT: {len(pi_depths)}/{len(all_trees)} ({len(pi_depths)/len(all_trees)*100:.1f}%)")

# Trace stocks and separate paths with/without pi_filter
print("\nTracing stocks ...")

def trace_stock(tree, feature_values):
    """Returns (combo_frozenset, had_pi_split, leaf_value)"""
    node_id = 0
    groups_used = set()
    had_pi = False
    while True:
        node = tree[node_id]
        if node['leaf']:
            return frozenset(groups_used), had_pi, node['value']
        feat = node['feature']
        feat_idx = FEATURES.index(feat)
        if feature_values[feat_idx] < node['threshold']:
            node_id = node['yes']
        else:
            node_id = node['no']
        if feat == 'pi_filter':
            had_pi = True
        elif feat in GROUPS:
            groups_used.add(GROUPS[feat])

SAMPLE = 2000
groups_def = {
    'Calm Long': (regime_arr == 'Calm') & (leg_arr == 'long'),
    'Calm Short': (regime_arr == 'Calm') & (leg_arr == 'short'),
    'Panic Long': (regime_arr == 'Panic') & (leg_arr == 'long'),
    'Panic Short': (regime_arr == 'Panic') & (leg_arr == 'short'),
}

rng = np.random.RandomState(42)

for gname, gmask in groups_def.items():
    indices = np.where(gmask)[0]
    if len(indices) > SAMPLE:
        indices = rng.choice(indices, SAMPLE, replace=False)

    X_group = X_test[indices]

    total_paths = 0
    paths_with_pi = 0
    paths_without_pi = 0

    combo_with_pi = Counter()
    combo_without_pi = Counter()

    leaf_with_pi = []
    leaf_without_pi = []

    for tree in all_trees:
        for stock_idx in range(len(X_group)):
            combo, had_pi, leaf_val = trace_stock(tree, X_group[stock_idx])
            if not combo:
                continue
            total_paths += 1
            if had_pi:
                paths_with_pi += 1
                combo_with_pi[combo] += 1
                leaf_with_pi.append(leaf_val)
            else:
                paths_without_pi += 1
                combo_without_pi[combo] += 1
                leaf_without_pi.append(leaf_val)

    pct_with_pi = paths_with_pi / total_paths * 100 if total_paths > 0 else 0

    print(f"\n{'='*60}")
    print(f"  {gname}")
    print(f"{'='*60}")
    print(f"  Total momentum paths: {total_paths:,}")
    print(f"  With pi_filter split: {paths_with_pi:,} ({pct_with_pi:.1f}%)")
    print(f"  Without pi_filter:    {paths_without_pi:,} ({100-pct_with_pi:.1f}%)")

    if leaf_with_pi:
        print(f"  Avg leaf (with pi):    {np.mean(leaf_with_pi):+.6f}")
    if leaf_without_pi:
        print(f"  Avg leaf (without pi): {np.mean(leaf_without_pi):+.6f}")

    # Top 5 combos WITH pi_filter
    total_with = sum(combo_with_pi.values())
    print(f"\n  Top combos WITH pi_filter split:")
    for combo, count in combo_with_pi.most_common(5):
        combo_str = '+'.join(sorted(combo, key=lambda g: ['S','M','I','L'].index(g)))
        print(f"    {combo_str}: {count/total_with*100:.1f}%")

    # Top 5 combos WITHOUT pi_filter
    total_without = sum(combo_without_pi.values())
    print(f"  Top combos WITHOUT pi_filter split:")
    for combo, count in combo_without_pi.most_common(5):
        combo_str = '+'.join(sorted(combo, key=lambda g: ['S','M','I','L'].index(g)))
        print(f"    {combo_str}: {count/total_without*100:.1f}%")

print("\nDone.")
