"""
tree_combo_analysis.py
=======================
For each stock in each group (calm long/short, panic long/short),
trace through all trees and record:
- Which momentum features were split on (the "combo")
- The stock's z-score at each momentum feature in the combo

Count combos and report average z-scores for each.

Uses production 50-seed model (25,000 trees).

Usage:
    python -u scripts/tree_combo_analysis.py
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, time
from collections import Counter, defaultdict
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, XGB_SEEDS, MOM_FEATURES)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("[ 1/4 ] Loading data ...")

with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
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

# Precompute monthly z-scores for all momentum features
print("  Computing monthly z-scores ...")
mom_cols = [f'mom_{h}' for h in range(1, 13)]
z_scores = np.zeros((len(test_pi), 12), dtype=float)

for date, grp in test_pi.groupby('date'):
    idx = grp.index
    for j, col in enumerate(mom_cols):
        vals = grp[col].values.astype(float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std > 0:
            z_scores[idx, j] = (vals - mean) / std

print(f"  Test stocks: {len(X_test):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN ALL 50 SEEDS AND PARSE TREES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 2/4 ] Training and parsing trees ...")

REDUCED = MOM_FEATURES + ['pi_filter']
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
COMBO_SEEDS = list(range(1, 101))  # 100 seeds x 500 trees = 50,000 trees
for i, xs in enumerate(COMBO_SEEDS):
    xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                       learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                       colsample_bytree=COLSAMPLE, tree_method='hist',
                       random_state=xs, verbosity=0)
    xgb.fit(X_train, y_train)
    all_trees.extend(parse_trees(xgb.get_booster(), FEATURES))
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/100 seeds ({len(all_trees)} trees)")

print(f"  Total trees: {len(all_trees)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRACE STOCKS AND COLLECT COMBOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 3/4 ] Tracing stocks through trees ...")

def get_mom_combo(tree, feature_values):
    """
    Walk stock through tree. Return frozenset of momentum features
    that were split on (ignoring pi_filter splits).
    """
    node_id = 0
    mom_features_used = set()
    while True:
        node = tree[node_id]
        if node['leaf']:
            return frozenset(mom_features_used)
        feat = node['feature']
        feat_idx = FEATURES.index(feat)
        if feature_values[feat_idx] < node['threshold']:
            node_id = node['yes']
        else:
            node_id = node['no']
        if feat != 'pi_filter':
            mom_features_used.add(feat)

SAMPLE_SIZE = 5000

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
    Z_group = z_scores[indices]  # z-scores for these stocks
    n_stocks = len(X_group)
    n_trees = len(all_trees)

    print(f"\n  {gname}: {n_stocks:,} stocks x {n_trees:,} trees = {n_stocks*n_trees:,} traces")

    # Count combos and accumulate z-scores
    combo_counter = Counter()
    # For each combo, for each feature in the combo, accumulate z-scores
    combo_zscores = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    for tree_idx, tree in enumerate(all_trees):
        if tree_idx % 5000 == 0 and tree_idx > 0:
            elapsed = time.time() - t0
            rate = tree_idx / elapsed
            remaining = (n_trees - tree_idx) / rate
            print(f"    Tree {tree_idx}/{n_trees} ({elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining)", flush=True)

        for stock_idx in range(n_stocks):
            combo = get_mom_combo(tree, X_group[stock_idx])
            if not combo:
                continue
            combo_counter[combo] += 1

            # Record z-scores (cap to avoid memory explosion)
            if combo_counter[combo] <= 50000:
                for feat in combo:
                    h = int(feat.replace('mom_', '')) - 1  # 0-indexed
                    combo_zscores[combo][feat].append(Z_group[stock_idx, h])

    elapsed = time.time() - t0
    total = n_stocks * n_trees
    print(f"    Done in {elapsed:.0f}s. Unique combos: {len(combo_counter)}")

    # Store top results
    top_combos = []
    for combo, count in combo_counter.most_common(20):
        pct = count / total * 100
        combo_sorted = sorted(combo, key=lambda f: int(f.replace('mom_', '')))
        z_info = {}
        for feat in combo_sorted:
            zvals = combo_zscores[combo][feat]
            z_info[feat] = {
                'mean_z': np.mean(zvals),
                'median_z': np.median(zvals),
                'std_z': np.std(zvals),
            }
        top_combos.append({
            'combo': combo_sorted,
            'count': count,
            'pct': pct,
            'z_info': z_info,
        })

    all_results[gname] = top_combos

# ══════════════════════════════════════════════════════════════════════════════
# 4. REPORT AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 4/4 ] Results ...")

for gname, combos in all_results.items():
    print(f"\n{'='*70}")
    print(f"  {gname}")
    print(f"{'='*70}")

    for rank, c in enumerate(combos[:15]):
        combo_str = ' + '.join([f.replace('mom_', 'm') for f in c['combo']])
        print(f"\n  #{rank+1}: {combo_str}  ({c['count']:,} times, {c['pct']:.1f}%)")
        for feat in c['combo']:
            z = c['z_info'][feat]
            feat_short = feat.replace('mom_', 'm')
            print(f"       {feat_short}: avg z-score = {z['mean_z']:+.2f}  "
                  f"(median {z['median_z']:+.2f}, std {z['std_z']:.2f})")

# Save
with open('tree_combo_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# CSV summary
rows = []
for gname, combos in all_results.items():
    for c in combos[:15]:
        combo_str = ' + '.join([f.replace('mom_', 'm') for f in c['combo']])
        z_str = '; '.join([f"{f.replace('mom_','m')}={c['z_info'][f]['mean_z']:+.2f}"
                           for f in c['combo']])
        rows.append({
            'group': gname,
            'combo': combo_str,
            'count': c['count'],
            'pct': c['pct'],
            'avg_z_scores': z_str,
        })
pd.DataFrame(rows).to_csv('tree_combo_results.csv', index=False, float_format='%.4f')

print("\nSaved: tree_combo_results.pkl, tree_combo_results.csv")
print("Done.")
