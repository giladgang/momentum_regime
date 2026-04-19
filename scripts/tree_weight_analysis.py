"""
tree_weight_analysis.py
=======================
Check if trees with pi_filter splits are concentrated early or late,
and compute the effective weight (leaf magnitude) of pi_filter vs non-pi trees.
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading production model ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
FEATURES = art['FEATURES']

xgb_model = joblib.load('artefacts/cs_artefacts_xgb.pkl')

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

all_trees = parse_trees(xgb_model.get_booster(), FEATURES)
print(f"Total trees: {len(all_trees)}")

# ── For each tree: has_pi, tree index, avg absolute leaf value ──
tree_stats = []
for i, tree in enumerate(all_trees):
    has_pi = any(n.get('feature') == 'pi_filter' for n in tree.values() if not n.get('leaf', False))
    leaf_vals = [abs(n['value']) for n in tree.values() if n.get('leaf', False)]
    avg_leaf = np.mean(leaf_vals) if leaf_vals else 0
    max_leaf = np.max(leaf_vals) if leaf_vals else 0
    tree_stats.append({
        'tree_idx': i,
        'has_pi': has_pi,
        'avg_abs_leaf': avg_leaf,
        'max_abs_leaf': max_leaf,
        'n_leaves': len(leaf_vals),
    })

df = pd.DataFrame(tree_stats)

# ── Where are pi_filter trees in the sequence? ──
print("\n" + "="*60)
print("  1. PI_FILTER TREES BY POSITION IN SEQUENCE")
print("="*60)

n_trees = len(df)
for start, end, label in [(0, 100, 'Trees 1-100'), (100, 200, 'Trees 101-200'),
                           (200, 300, 'Trees 201-300'), (300, 400, 'Trees 301-400'),
                           (400, 500, 'Trees 401-500')]:
    subset = df[(df['tree_idx'] >= start) & (df['tree_idx'] < end)]
    n_pi = subset['has_pi'].sum()
    print(f"  {label}: {n_pi}/{len(subset)} have pi_filter ({n_pi/len(subset)*100:.1f}%)")

# ── Leaf magnitude comparison ──
print("\n" + "="*60)
print("  2. LEAF MAGNITUDE: PI_FILTER TREES vs NON-PI TREES")
print("="*60)

pi_trees = df[df['has_pi']]
no_pi_trees = df[~df['has_pi']]

print(f"  Trees WITH pi_filter ({len(pi_trees)}):")
print(f"    Avg |leaf|: {pi_trees['avg_abs_leaf'].mean():.8f}")
print(f"    Max |leaf|: {pi_trees['max_abs_leaf'].mean():.8f}")

print(f"  Trees WITHOUT pi_filter ({len(no_pi_trees)}):")
print(f"    Avg |leaf|: {no_pi_trees['avg_abs_leaf'].mean():.8f}")
print(f"    Max |leaf|: {no_pi_trees['max_abs_leaf'].mean():.8f}")

ratio = pi_trees['avg_abs_leaf'].mean() / no_pi_trees['avg_abs_leaf'].mean()
print(f"\n  Ratio (pi / no-pi): {ratio:.2f}x")

# ── Leaf magnitude by tree position ──
print("\n" + "="*60)
print("  3. LEAF MAGNITUDE BY POSITION (ALL TREES)")
print("="*60)

for start, end, label in [(0, 50, 'Trees 1-50'), (50, 100, 'Trees 51-100'),
                           (100, 200, 'Trees 101-200'), (200, 300, 'Trees 201-300'),
                           (300, 400, 'Trees 301-400'), (400, 500, 'Trees 401-500')]:
    subset = df[(df['tree_idx'] >= start) & (df['tree_idx'] < end)]
    print(f"  {label}: avg |leaf| = {subset['avg_abs_leaf'].mean():.8f}, "
          f"max |leaf| = {subset['max_abs_leaf'].mean():.8f}")

# ── Total prediction weight ──
print("\n" + "="*60)
print("  4. TOTAL PREDICTION WEIGHT (sum of avg |leaf| across trees)")
print("="*60)

total_weight = df['avg_abs_leaf'].sum()
pi_weight = pi_trees['avg_abs_leaf'].sum()
no_pi_weight = no_pi_trees['avg_abs_leaf'].sum()

print(f"  Total weight: {total_weight:.6f}")
print(f"  Pi trees:     {pi_weight:.6f} ({pi_weight/total_weight*100:.1f}%)")
print(f"  Non-pi trees: {no_pi_weight:.6f} ({no_pi_weight/total_weight*100:.1f}%)")
print(f"  (Pi trees are {len(pi_trees)/len(df)*100:.1f}% of trees but {pi_weight/total_weight*100:.1f}% of weight)")

print("\nDone.")
