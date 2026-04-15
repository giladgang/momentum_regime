"""
tree_path_analysis.py
=====================
Analyzes XGBoost decision paths to understand how the model uses momentum
features to select stocks in each regime.

Approach:
1. Classify trees as pi_filter-heavy (>50% pi splits) or momentum-heavy
2. Trace stocks through momentum-heavy trees only
3. Record momentum-only paths (skipping pi_filter splits)
4. For each group (calm long/short, panic long/short), find most common
   paths and report thresholds, stock values, and leaf predictions

Outputs:
- tree_path_results.pkl: full results dict
- tree_path_results.csv: summary table
- Console: detailed path analysis

Usage:
    python -u scripts/tree_path_analysis.py [--production]

    Without --production: uses 500 saved trees (validation, ~5 min)
    With --production: retrains all 50 seeds for 25,000 trees (~15 min)
"""

import numpy as np
import pandas as pd
import pickle, sys, os, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_SIZE = 5000  # stocks per group (set to None for full population)
MAX_DEPTH = 4


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("[ 1/5 ] Loading data ...")

with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

# Assign portfolio legs using production scores
test_pi['leg'] = 'middle'
for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test_pi.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test_pi.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

X_test = test_pi[FEATURES].values.astype(float)
regime_arr = test_pi['regime'].values
leg_arr = test_pi['leg'].values

print(f"  Test stocks: {len(X_test):,}")
print(f"  Calm long: {((regime_arr == 'Calm') & (leg_arr == 'long')).sum():,}")
print(f"  Calm short: {((regime_arr == 'Calm') & (leg_arr == 'short')).sum():,}")
print(f"  Panic long: {((regime_arr == 'Panic') & (leg_arr == 'long')).sum():,}")
print(f"  Panic short: {((regime_arr == 'Panic') & (leg_arr == 'short')).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GET TREES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 2/5 ] Loading trees ...")

production_mode = '--production' in sys.argv

if production_mode:
    import re
    from xgboost import XGBRegressor
    from config import (N_ESTIMATORS, MAX_DEPTH as XGB_DEPTH, LEARNING_RATE,
                        SUBSAMPLE, COLSAMPLE, XGB_SEEDS, MOM_FEATURES)

    REDUCED = MOM_FEATURES + ['pi_filter']
    train = art['train'].copy()
    X_train = train[REDUCED].values.astype(float)
    y_train = train['ret_fwd'].values.astype(float)

    def parse_trees(booster, feature_names):
        dump = booster.get_dump(dump_format='text')
        trees = []
        for raw in dump:
            nodes = {}
            for line in raw.strip().split('\n'):
                depth = len(line) - len(line.lstrip('\t'))
                line = line.strip()
                node_id = int(re.match(r'(\d+):', line).group(1))
                if 'leaf' in line:
                    val = float(re.search(r'leaf=([-\d.e+]+)', line).group(1))
                    nodes[node_id] = dict(leaf=True, value=val, depth=depth)
                else:
                    m = re.search(r'\[f(\d+)<([-\d.e+]+)\].*yes=(\d+),no=(\d+)', line)
                    fidx = int(m.group(1))
                    nodes[node_id] = dict(leaf=False, depth=depth,
                                          feature=feature_names[fidx],
                                          threshold=float(m.group(2)),
                                          yes=int(m.group(3)), no=int(m.group(4)))
            trees.append(nodes)
        return trees

    all_trees = []
    for i, xs in enumerate(XGB_SEEDS):
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=XGB_DEPTH,
                           learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE, tree_method='hist',
                           random_state=xs, verbosity=0)
        xgb.fit(X_train, y_train)
        all_trees.extend(parse_trees(xgb.get_booster(), FEATURES))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/50 seeds ({len(all_trees)} trees)")

    print(f"  Total trees: {len(all_trees)} (production)")
else:
    all_trees = art['all_trees']
    print(f"  Total trees: {len(all_trees)} (validation, saved model)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFY TREES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 3/5 ] Classifying trees ...")

mom_trees = []
pi_trees = []
tree_stats = {'pi_splits': 0, 'mom_splits': 0}

for tree in all_trees:
    pi_splits = sum(1 for n in tree.values() if not n['leaf'] and n['feature'] == 'pi_filter')
    mom_splits = sum(1 for n in tree.values() if not n['leaf'] and n['feature'] != 'pi_filter')
    tree_stats['pi_splits'] += pi_splits
    tree_stats['mom_splits'] += mom_splits
    total = pi_splits + mom_splits
    if total > 0 and mom_splits / total > 0.5:
        mom_trees.append(tree)
    else:
        pi_trees.append(tree)

total_splits = tree_stats['pi_splits'] + tree_stats['mom_splits']
print(f"  Pi-filter heavy trees: {len(pi_trees)} ({len(pi_trees)/len(all_trees)*100:.1f}%)")
print(f"  Momentum heavy trees:  {len(mom_trees)} ({len(mom_trees)/len(all_trees)*100:.1f}%)")
print(f"  Total splits: {total_splits} "
      f"(pi_filter: {tree_stats['pi_splits']} ({tree_stats['pi_splits']/total_splits*100:.1f}%), "
      f"momentum: {tree_stats['mom_splits']} ({tree_stats['mom_splits']/total_splits*100:.1f}%))")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRACE STOCKS THROUGH MOMENTUM TREES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 4/5 ] Tracing stocks through momentum trees ...")

def trace_stock(tree, feature_values):
    """
    Trace a single stock through a tree.
    Returns: (momentum_path, split_details, leaf_value)
    - momentum_path: tuple of 'mX_L' or 'mX_H' for momentum splits only
    - split_details: list of dicts with feature, threshold, stock value
    - leaf_value: the tree's prediction for this stock
    """
    node_id = 0
    mom_path = []
    split_details = []
    while True:
        node = tree[node_id]
        if node['leaf']:
            return tuple(mom_path), split_details, node['value']
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
        if feat != 'pi_filter':
            mom_path.append(f"{feat.replace('mom_', 'm')}_{direction}")
            split_details.append({
                'feature': feat,
                'threshold': thresh,
                'stock_value': val,
                'direction': direction,
            })


groups = {
    'Calm Long': (regime_arr == 'Calm') & (leg_arr == 'long'),
    'Calm Short': (regime_arr == 'Calm') & (leg_arr == 'short'),
    'Panic Long': (regime_arr == 'Panic') & (leg_arr == 'long'),
    'Panic Short': (regime_arr == 'Panic') & (leg_arr == 'short'),
}

all_results = {}

for gname, gmask in groups.items():
    indices = np.where(gmask)[0]
    if SAMPLE_SIZE and len(indices) > SAMPLE_SIZE:
        rng = np.random.RandomState(42)
        indices = rng.choice(indices, SAMPLE_SIZE, replace=False)

    X_group = X_test[indices]
    n_stocks = len(X_group)
    n_trees = len(mom_trees)
    total_traces = n_stocks * n_trees

    print(f"\n  {gname}: {n_stocks:,} stocks x {n_trees} trees = {total_traces:,} traces")

    path_counter = Counter()
    path_leaves = defaultdict(list)
    path_splits = defaultdict(lambda: defaultdict(lambda: {'thresholds': [], 'values': []}))

    t0 = time.time()
    for tree_idx, tree in enumerate(mom_trees):
        if tree_idx % 100 == 0 and tree_idx > 0:
            elapsed = time.time() - t0
            rate = tree_idx / elapsed
            remaining = (n_trees - tree_idx) / rate
            print(f"    Tree {tree_idx}/{n_trees} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", flush=True)

        for stock_idx in range(n_stocks):
            path, splits, leaf_val = trace_stock(tree, X_group[stock_idx])
            if not path:
                continue
            path_counter[path] += 1
            if len(path_leaves[path]) < 10000:
                path_leaves[path].append(leaf_val)
            for s in splits:
                feat = s['feature']
                entry = path_splits[path][feat]
                if len(entry['thresholds']) < 10000:
                    entry['thresholds'].append(s['threshold'])
                    entry['values'].append(s['stock_value'])

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s")

    # Store results
    top_paths = []
    for path, count in path_counter.most_common(20):
        pct = count / total_traces * 100
        avg_leaf = np.mean(path_leaves[path]) * 100

        split_info = []
        for step in path:
            feat_name = 'mom_' + step.split('_')[0][1:]
            if feat_name in path_splits[path]:
                s = path_splits[path][feat_name]
                split_info.append({
                    'feature': feat_name,
                    'direction': step.split('_')[1],
                    'median_threshold': np.median(s['thresholds']),
                    'median_stock_value': np.median(s['values']),
                })

        top_paths.append({
            'path': path,
            'path_readable': ' -> '.join(
                [p.replace('_L', '<').replace('_H', '>=') for p in path]),
            'count': count,
            'pct': pct,
            'avg_leaf_pct': avg_leaf,
            'splits': split_info,
        })

    all_results[gname] = top_paths


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORT AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 5/5 ] Results ...")

for gname, paths in all_results.items():
    print(f"\n{'='*70}")
    print(f"  {gname}")
    print(f"{'='*70}")

    for rank, p in enumerate(paths[:10]):
        print(f"\n  #{rank+1}: {p['path_readable']}")
        print(f"       Count: {p['count']:,} ({p['pct']:.1f}%)  "
              f"Avg leaf: {p['avg_leaf_pct']:+.4f}%")
        for s in p['splits']:
            feat_short = s['feature'].replace('mom_', 'm')
            dir_label = 'below' if s['direction'] == 'L' else 'above'
            print(f"       {feat_short}: stock at {s['median_stock_value']*100:+.1f}%, "
                  f"{dir_label} threshold {s['median_threshold']*100:+.1f}%")

# Save
with open('tree_path_results.pkl', 'wb') as f:
    pickle.dump({
        'results': all_results,
        'tree_stats': tree_stats,
        'n_mom_trees': len(mom_trees),
        'n_pi_trees': len(pi_trees),
        'n_total_trees': len(all_trees),
        'sample_size': SAMPLE_SIZE,
        'mode': 'production' if production_mode else 'validation',
    }, f)

# CSV summary
rows = []
for gname, paths in all_results.items():
    for p in paths[:10]:
        rows.append({
            'group': gname,
            'path': p['path_readable'],
            'count': p['count'],
            'pct': p['pct'],
            'avg_leaf_pct': p['avg_leaf_pct'],
        })
pd.DataFrame(rows).to_csv('tree_path_results.csv', index=False, float_format='%.4f')

print("\nSaved: tree_path_results.pkl, tree_path_results.csv")
print("Done.")
