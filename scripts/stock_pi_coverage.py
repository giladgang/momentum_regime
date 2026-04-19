"""
stock_pi_coverage.py
====================
For each stock-month in the test set, count how many of the 500 production
trees route that stock through a pi_filter split. Check if any stock
never encounters pi_filter.
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

# Sample stocks (full test set is too large)
SAMPLE = 5000
rng = np.random.RandomState(42)
indices = rng.choice(len(X_test), SAMPLE, replace=False)
X_sample = X_test[indices]

print(f"Tracing {SAMPLE} stocks through {len(all_trees)} trees ...")

pi_hit_counts = np.zeros(SAMPLE, dtype=int)

for tree in all_trees:
    for stock_idx in range(SAMPLE):
        node_id = 0
        while True:
            node = tree[node_id]
            if node['leaf']:
                break
            feat = node['feature']
            feat_idx = FEATURES.index(feat)
            if feat == 'pi_filter':
                pi_hit_counts[stock_idx] += 1
                # Don't break -- count the tree, not every split
                # Actually we want: did this stock hit pi_filter in this tree?
                # Let me just flag it
            if X_sample[stock_idx, feat_idx] < node['threshold']:
                node_id = node['yes']
            else:
                node_id = node['no']

# pi_hit_counts counts total pi_filter splits encountered (can be >1 per tree)
# Let's also count trees (not splits)
print("\nRe-tracing to count trees (not splits) ...")
trees_with_pi = np.zeros(SAMPLE, dtype=int)

for tree in all_trees:
    for stock_idx in range(SAMPLE):
        node_id = 0
        hit_pi = False
        while True:
            node = tree[node_id]
            if node['leaf']:
                break
            feat = node['feature']
            feat_idx = FEATURES.index(feat)
            if feat == 'pi_filter':
                hit_pi = True
            if X_sample[stock_idx, feat_idx] < node['threshold']:
                node_id = node['yes']
            else:
                node_id = node['no']
        if hit_pi:
            trees_with_pi[stock_idx] += 1

print(f"\nResults ({SAMPLE} stocks, {len(all_trees)} trees):")
print(f"  Trees with pi_filter per stock:")
print(f"    Mean:   {trees_with_pi.mean():.1f} / {len(all_trees)} ({trees_with_pi.mean()/len(all_trees)*100:.1f}%)")
print(f"    Median: {np.median(trees_with_pi):.0f} ({np.median(trees_with_pi)/len(all_trees)*100:.1f}%)")
print(f"    Min:    {trees_with_pi.min()} ({trees_with_pi.min()/len(all_trees)*100:.1f}%)")
print(f"    Max:    {trees_with_pi.max()} ({trees_with_pi.max()/len(all_trees)*100:.1f}%)")
print(f"    Std:    {trees_with_pi.std():.1f}")

print(f"\n  Stocks that NEVER hit pi_filter: {(trees_with_pi == 0).sum()}")
print(f"  Stocks hitting pi in <10% of trees: {(trees_with_pi < 50).sum()}")
print(f"  Stocks hitting pi in 10-50% of trees: {((trees_with_pi >= 50) & (trees_with_pi < 250)).sum()}")
print(f"  Stocks hitting pi in 50-90% of trees: {((trees_with_pi >= 250) & (trees_with_pi < 450)).sum()}")
print(f"  Stocks hitting pi in >90% of trees: {(trees_with_pi >= 450).sum()}")

# Percentile distribution
print(f"\n  Percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(trees_with_pi, p)
    print(f"    P{p:>2d}: {val:.0f} trees ({val/len(all_trees)*100:.1f}%)")

print("\nDone.")
