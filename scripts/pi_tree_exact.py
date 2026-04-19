"""
pi_tree_exact.py
================
Exact count: for each of the 500 production trees, does it contain
a pi_filter split? No sampling, no stock tracing -- just tree structure.
"""

import re, sys, os, joblib
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
FEATURES = art['FEATURES']

xgb_model = joblib.load('artefacts/cs_artefacts_xgb.pkl')
dump = xgb_model.get_booster().get_dump(dump_format='text')

total = len(dump)
has_pi = 0
pi_at_root = 0
pi_count_per_tree = []

for raw in dump:
    tree_has_pi = False
    root_is_pi = False
    n_pi_splits = 0
    for line in raw.strip().split('\n'):
        line_stripped = line.strip()
        if 'leaf' in line_stripped:
            continue
        m = re.search(r'\[f(\d+)<', line_stripped)
        if m:
            fidx = int(m.group(1))
            feat = FEATURES[fidx] if fidx < len(FEATURES) else f'f{fidx}'
            if feat == 'pi_filter':
                tree_has_pi = True
                n_pi_splits += 1
                node_id = int(re.match(r'(\d+):', line_stripped).group(1))
                if node_id == 0:
                    root_is_pi = True

    if tree_has_pi:
        has_pi += 1
    if root_is_pi:
        pi_at_root += 1
    pi_count_per_tree.append(n_pi_splits)

print(f"Production model (seed 50): {total} trees")
print(f"  Trees with pi_filter split:    {has_pi}/{total} ({has_pi/total*100:.1f}%)")
print(f"  Trees with pi_filter at root:  {pi_at_root}/{total} ({pi_at_root/total*100:.1f}%)")
print(f"  Trees WITHOUT pi_filter:       {total - has_pi}/{total} ({(total-has_pi)/total*100:.1f}%)")
print(f"\n  Pi_filter splits per tree (among trees that have it):")
import numpy as np
arr = np.array([x for x in pi_count_per_tree if x > 0])
print(f"    Mean: {arr.mean():.1f}")
print(f"    1 split:  {(arr == 1).sum()}")
print(f"    2 splits: {(arr == 2).sum()}")
print(f"    3 splits: {(arr == 3).sum()}")
print(f"    4+ splits: {(arr >= 4).sum()}")

# Also check all 50 seeds
print(f"\n--- Checking all 50 production seeds ---")
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, XGB_SEEDS, MOM_FEATURES, CS_FEATURES)
from xgboost import XGBRegressor

train = art['train']
X_train = train[CS_FEATURES].values.astype(float)
y_train = train['ret_fwd'].values.astype(float)

seed_stats = []
for xs in XGB_SEEDS:
    xgb_i = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                          learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                          colsample_bytree=COLSAMPLE, tree_method='hist',
                          random_state=xs, verbosity=0)
    xgb_i.fit(X_train, y_train)
    d = xgb_i.get_booster().get_dump(dump_format='text')
    count = 0
    for raw in d:
        for line in raw.strip().split('\n'):
            line_stripped = line.strip()
            if 'leaf' in line_stripped:
                continue
            m2 = re.search(r'\[f(\d+)<', line_stripped)
            if m2:
                fidx = int(m2.group(1))
                feat = CS_FEATURES[fidx] if fidx < len(CS_FEATURES) else f'f{fidx}'
                if feat == 'pi_filter':
                    count += 1
                    break
        else:
            continue
        count  # tree has pi
    # Simpler: count trees with pi
    n_with_pi = 0
    for raw in d:
        for line in raw.strip().split('\n'):
            if 'leaf' in line.strip():
                continue
            m2 = re.search(r'\[f(\d+)<', line.strip())
            if m2:
                fidx = int(m2.group(1))
                feat = CS_FEATURES[fidx] if fidx < len(CS_FEATURES) else f'f{fidx}'
                if feat == 'pi_filter':
                    n_with_pi += 1
                    break
    seed_stats.append(n_with_pi)
    if xs % 10 == 0:
        print(f"  Seed {xs}: {n_with_pi}/{len(d)} trees with pi_filter ({n_with_pi/len(d)*100:.1f}%)")

print(f"\nAcross all 50 seeds:")
print(f"  Mean: {np.mean(seed_stats):.1f} / 500 ({np.mean(seed_stats)/500*100:.1f}%)")
print(f"  Min:  {np.min(seed_stats)} ({np.min(seed_stats)/500*100:.1f}%)")
print(f"  Max:  {np.max(seed_stats)} ({np.max(seed_stats)/500*100:.1f}%)")

print("\nDone.")
