"""
combo_redesign.py
=================
Redesigned tree path analysis that captures regime-dependent behavior.

Key changes from tree_combo_grouped.py:
1. Uses production model (not freshly trained models)
2. Filters to paths that split on pi_filter (regime-aware paths only)
3. Records the average leaf value (prediction) per combo per group
4. Records the split direction at each momentum feature

Produces a table showing: same tree structure, different predictions by regime.
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, time, joblib
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MOM_FEATURES, TABLES_DIR

GROUPS = {
    'mom_1': 'S', 'mom_2': 'S', 'mom_3': 'S',
    'mom_4': 'M', 'mom_5': 'M', 'mom_6': 'M',
    'mom_7': 'I', 'mom_8': 'I', 'mom_9': 'I',
    'mom_10': 'L', 'mom_11': 'L', 'mom_12': 'L',
}
GROUP_ORDER = ['S', 'M', 'I', 'L']
GROUP_NAMES = {'S': '1-3mo', 'M': '4-6mo', 'I': '7-9mo', 'L': '10-12mo'}

# ── Load ──
print("Loading artefacts ...")
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

# ── Load production model ──
print("Loading production XGBoost model ...")
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
print(f"Production model trees: {len(all_trees)}")

# Check pi_filter presence
pi_in_tree = sum(1 for tree in all_trees
                 if any(n.get('feature') == 'pi_filter'
                        for n in tree.values() if not n.get('leaf', False)))
print(f"Trees with pi_filter: {pi_in_tree}/{len(all_trees)} ({pi_in_tree/len(all_trees)*100:.1f}%)")

# ── Trace stocks ──

def trace_stock(tree, feature_values):
    """Returns (combo, had_pi, leaf_value, split_directions)."""
    node_id = 0
    groups_used = set()
    had_pi = False
    directions = {}  # group -> 'below' or 'above'
    while True:
        node = tree[node_id]
        if node['leaf']:
            return frozenset(groups_used), had_pi, node['value'], directions
        feat = node['feature']
        feat_idx = FEATURES.index(feat)
        went_left = feature_values[feat_idx] < node['threshold']
        if went_left:
            node_id = node['yes']
        else:
            node_id = node['no']
        if feat == 'pi_filter':
            had_pi = True
        elif feat in GROUPS:
            g = GROUPS[feat]
            groups_used.add(g)
            directions[g] = 'below' if went_left else 'above'

SAMPLE = 3000
rng = np.random.RandomState(42)

groups_def = {
    'Calm Long': (regime_arr == 'Calm') & (leg_arr == 'long'),
    'Calm Short': (regime_arr == 'Calm') & (leg_arr == 'short'),
    'Panic Long': (regime_arr == 'Panic') & (leg_arr == 'long'),
    'Panic Short': (regime_arr == 'Panic') & (leg_arr == 'short'),
}

# Collect: for each group, combo -> list of leaf values and directions
results = {}

for gname, gmask in groups_def.items():
    indices = np.where(gmask)[0]
    if len(indices) > SAMPLE:
        indices = rng.choice(indices, SAMPLE, replace=False)
    X_group = X_test[indices]

    combo_leaves = defaultdict(list)       # combo -> [leaf_values]
    combo_directions = defaultdict(lambda: defaultdict(Counter))  # combo -> group -> Counter(below/above)
    total_pi_paths = 0
    total_paths = 0

    print(f"\n  Tracing {gname}: {len(X_group)} stocks x {len(all_trees)} trees ...")
    t0 = time.time()

    for tree in all_trees:
        for stock_idx in range(len(X_group)):
            combo, had_pi, leaf_val, directions = trace_stock(tree, X_group[stock_idx])
            if not combo:
                continue
            total_paths += 1
            if not had_pi:
                continue
            total_pi_paths += 1
            combo_leaves[combo].append(leaf_val)
            for g, d in directions.items():
                combo_directions[combo][g][d] += 1

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s. Pi-paths: {total_pi_paths:,}/{total_paths:,} ({total_pi_paths/total_paths*100:.1f}%)")

    results[gname] = {
        'combo_leaves': dict(combo_leaves),
        'combo_directions': {k: dict(v) for k, v in combo_directions.items()},
        'total_pi_paths': total_pi_paths,
    }

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Average leaf value by combo and group (pi_filter paths only)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  ANALYSIS 1: Average leaf value by combo (pi_filter paths only)")
print("="*80)

# Get all combos that appear in all 4 groups
all_combos = set()
for gname, res in results.items():
    all_combos.update(res['combo_leaves'].keys())

# Sort by frequency in Calm Long
combo_freq = {}
for combo in all_combos:
    cl = len(results['Calm Long']['combo_leaves'].get(combo, []))
    combo_freq[combo] = cl
sorted_combos = sorted(all_combos, key=lambda c: combo_freq[c], reverse=True)

print(f"\n{'Combo':<12s} | {'Calm Long':>10s} {'Calm Short':>11s} | {'Panic Long':>11s} {'Panic Short':>12s} | {'Count':>6s}")
print("-" * 80)

table_rows = []
for combo in sorted_combos[:15]:
    combo_str = '+'.join(sorted(combo, key=lambda g: GROUP_ORDER.index(g)))
    vals = {}
    counts = {}
    for gname in ['Calm Long', 'Calm Short', 'Panic Long', 'Panic Short']:
        leaves = results[gname]['combo_leaves'].get(combo, [])
        vals[gname] = np.mean(leaves) if leaves else 0
        counts[gname] = len(leaves)

    total_count = sum(counts.values())
    print(f"{combo_str:<12s} | {vals['Calm Long']:>+10.6f} {vals['Calm Short']:>+11.6f} | "
          f"{vals['Panic Long']:>+11.6f} {vals['Panic Short']:>+12.6f} | {total_count:>6,}")

    table_rows.append({
        'combo': combo_str,
        'cl': vals['Calm Long'],
        'cs': vals['Calm Short'],
        'pl': vals['Panic Long'],
        'ps': vals['Panic Short'],
        'count': total_count,
    })

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Split direction by combo and group
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  ANALYSIS 2: % going 'above' threshold at each horizon group")
print("  (pi_filter paths only, top combos)")
print("="*80)

for combo in sorted_combos[:10]:
    combo_str = '+'.join(sorted(combo, key=lambda g: GROUP_ORDER.index(g)))
    print(f"\n  Combo: {combo_str}")
    print(f"  {'Group':<8s} | {'Calm Long':>10s} {'Calm Short':>11s} | {'Panic Long':>11s} {'Panic Short':>12s}")
    print(f"  " + "-" * 60)

    for g in GROUP_ORDER:
        if g not in combo:
            continue
        vals = []
        for gname in ['Calm Long', 'Calm Short', 'Panic Long', 'Panic Short']:
            dirs = results[gname]['combo_directions'].get(combo, {}).get(g, Counter())
            total = dirs['above'] + dirs['below']
            pct_above = dirs['above'] / total * 100 if total > 0 else 0
            vals.append(pct_above)
        print(f"  {g:<8s} | {vals[0]:>9.1f}% {vals[1]:>10.1f}% | {vals[2]:>10.1f}% {vals[3]:>11.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Summary table - average leaf by horizon group (not combo)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  ANALYSIS 3: Average leaf value by horizon group (aggregated)")
print("  (across all combos that include that group, pi_filter paths only)")
print("="*80)

print(f"\n{'Group':<8s} | {'Calm Long':>10s} {'Calm Short':>11s} | {'Panic Long':>11s} {'Panic Short':>12s}")
print("-" * 60)

for g in GROUP_ORDER:
    vals = {}
    for gname in ['Calm Long', 'Calm Short', 'Panic Long', 'Panic Short']:
        all_leaves = []
        for combo, leaves in results[gname]['combo_leaves'].items():
            if g in combo:
                all_leaves.extend(leaves)
        vals[gname] = np.mean(all_leaves) if all_leaves else 0
    print(f"{g:<8s} | {vals['Calm Long']:>+10.6f} {vals['Calm Short']:>+11.6f} | "
          f"{vals['Panic Long']:>+11.6f} {vals['Panic Short']:>+12.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: % above threshold by horizon group (aggregated)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("  ANALYSIS 4: % going 'above' threshold by horizon group (aggregated)")
print("="*80)

print(f"\n{'Group':<8s} | {'Calm Long':>10s} {'Calm Short':>11s} | {'Panic Long':>11s} {'Panic Short':>12s}")
print("-" * 60)

for g in GROUP_ORDER:
    vals = []
    for gname in ['Calm Long', 'Calm Short', 'Panic Long', 'Panic Short']:
        total_above = 0
        total_all = 0
        for combo, dirs_dict in results[gname]['combo_directions'].items():
            if g in dirs_dict:
                total_above += dirs_dict[g]['above']
                total_all += dirs_dict[g]['above'] + dirs_dict[g]['below']
        pct = total_above / total_all * 100 if total_all > 0 else 0
        vals.append(pct)
    print(f"{g:<8s} | {vals[0]:>9.1f}% {vals[1]:>10.1f}% | {vals[2]:>10.1f}% {vals[3]:>11.1f}%")

print("\nDone.")
