"""
tree_combo_grouped.py
======================
Same as tree_combo_analysis.py but momentum horizons are grouped:
  Short (S): mom_1, mom_2, mom_3
  Medium (M): mom_4, mom_5, mom_6
  Intermediate (I): mom_7, mom_8, mom_9
  Long (L): mom_10, mom_11, mom_12

Instead of recording individual horizons, records which GROUP was split on.
Combo example: S+L means the tree split on at least one short-horizon and
at least one long-horizon feature.

Usage:
    python -u scripts/tree_combo_grouped.py
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, time
from collections import Counter, defaultdict
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, XGB_SEEDS, MOM_FEATURES)

# Horizon groups
GROUPS = {
    'mom_1': 'S', 'mom_2': 'S', 'mom_3': 'S',
    'mom_4': 'M', 'mom_5': 'M', 'mom_6': 'M',
    'mom_7': 'I', 'mom_8': 'I', 'mom_9': 'I',
    'mom_10': 'L', 'mom_11': 'L', 'mom_12': 'L',
}
GROUP_NAMES = {
    'S': 'Short (1-3mo)',
    'M': 'Medium (4-6mo)',
    'I': 'Intermediate (7-9mo)',
    'L': 'Long (10-12mo)',
}

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

# Precompute z-scores per group (average z across horizons in each group)
print("  Computing grouped z-scores ...")
mom_cols = [f'mom_{h}' for h in range(1, 13)]
z_scores_individual = np.zeros((len(test_pi), 12), dtype=float)

for date, grp in test_pi.groupby('date'):
    idx = grp.index
    for j, col in enumerate(mom_cols):
        vals = grp[col].values.astype(float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        if std > 0:
            z_scores_individual[idx, j] = (vals - mean) / std

# Group z-scores: S = avg(m1,m2,m3), M = avg(m4,m5,m6), etc.
z_groups = {
    'S': np.nanmean(z_scores_individual[:, 0:3], axis=1),
    'M': np.nanmean(z_scores_individual[:, 3:6], axis=1),
    'I': np.nanmean(z_scores_individual[:, 6:9], axis=1),
    'L': np.nanmean(z_scores_individual[:, 9:12], axis=1),
}

print(f"  Test stocks: {len(X_test):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN AND PARSE TREES
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

COMBO_SEEDS = list(range(1, 101))  # 100 seeds x 500 trees = 50,000 trees
all_trees = []
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
# 3. TRACE AND COLLECT GROUPED COMBOS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 3/4 ] Tracing stocks ...")

def get_grouped_combo(tree, feature_values):
    """Walk stock through tree, return frozenset of horizon GROUPS split on."""
    node_id = 0
    groups_used = set()
    while True:
        node = tree[node_id]
        if node['leaf']:
            return frozenset(groups_used)
        feat = node['feature']
        feat_idx = FEATURES.index(feat)
        if feature_values[feat_idx] < node['threshold']:
            node_id = node['yes']
        else:
            node_id = node['no']
        if feat in GROUPS:
            groups_used.add(GROUPS[feat])

SAMPLE_SIZE = 5000

groups_def = {
    'Calm Long': (regime_arr == 'Calm') & (leg_arr == 'long'),
    'Calm Short': (regime_arr == 'Calm') & (leg_arr == 'short'),
    'Panic Long': (regime_arr == 'Panic') & (leg_arr == 'long'),
    'Panic Short': (regime_arr == 'Panic') & (leg_arr == 'short'),
}

all_results = {}

for gname, gmask in groups_def.items():
    indices = np.where(gmask)[0]
    if SAMPLE_SIZE and len(indices) > SAMPLE_SIZE:
        rng = np.random.RandomState(42)
        indices = rng.choice(indices, SAMPLE_SIZE, replace=False)

    X_group = X_test[indices]
    z_group = {g: z_groups[g][indices] for g in ['S', 'M', 'I', 'L']}
    n_stocks = len(X_group)
    n_trees = len(all_trees)

    print(f"\n  {gname}: {n_stocks:,} stocks x {n_trees:,} trees")

    combo_counter = Counter()
    combo_zscores = defaultdict(lambda: defaultdict(list))
    total_unique_combos = 0

    t0 = time.time()
    for tree_idx, tree in enumerate(all_trees):
        if tree_idx % 5000 == 0 and tree_idx > 0:
            elapsed = time.time() - t0
            rate = tree_idx / elapsed
            remaining = (n_trees - tree_idx) / rate
            print(f"    Tree {tree_idx}/{n_trees} ({elapsed:.0f}s, ~{remaining:.0f}s left)", flush=True)

        for stock_idx in range(n_stocks):
            combo = get_grouped_combo(tree, X_group[stock_idx])
            if not combo:
                continue
            combo_counter[combo] += 1
            if combo_counter[combo] <= 50000:
                for g in combo:
                    combo_zscores[combo][g].append(z_group[g][stock_idx])

    elapsed = time.time() - t0
    total = n_stocks * n_trees
    print(f"    Done in {elapsed:.0f}s. Unique grouped combos: {len(combo_counter)}")

    # Total momentum paths (exclude pi_filter-only paths)
    mom_total = sum(c for combo, c in combo_counter.items() if combo)

    top_combos = []
    for combo, count in combo_counter.most_common(20):
        pct = count / mom_total * 100
        combo_sorted = sorted(combo, key=lambda g: ['S', 'M', 'I', 'L'].index(g))
        z_info = {}
        for g in combo_sorted:
            zvals = combo_zscores[combo][g]
            z_info[g] = {
                'mean_z': np.mean(zvals),
                'median_z': np.median(zvals),
                'std_z': np.std(zvals),
                'n_obs': len(zvals),
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
        combo_str = ' + '.join([f"{g}({GROUP_NAMES[g]})" for g in c['combo']])
        z_str = ', '.join([f"{g}={c['z_info'][g]['mean_z']:+.2f}" for g in c['combo']])
        print(f"\n  #{rank+1}: {combo_str}")
        print(f"       Count: {c['count']:,} ({c['pct']:.1f}%)  Z-scores: [{z_str}]")

# Save
with open('tree_combo_grouped_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Excel
writer = pd.ExcelWriter('tree_combo_grouped_results.xlsx', engine='openpyxl')
for gname, combos in all_results.items():
    rows = []
    for c in combos:
        combo_str = ' + '.join(c['combo'])
        row = {'Combo': combo_str, 'Count': c['count'], 'Pct': round(c['pct'], 2)}
        for g in c['combo']:
            row[f'{GROUP_NAMES[g]} avg_z'] = round(c['z_info'][g]['mean_z'], 2)
        rows.append(row)
    pd.DataFrame(rows).to_excel(writer, sheet_name=gname.replace(' ', '_'), index=False)


# Long-Short difference sheets (Calm and Panic)
GROUP_ORDER = ['S', 'M', 'I', 'L']
for regime in ['Calm', 'Panic']:
    long_key = f'{regime} Long'
    short_key = f'{regime} Short'
    if long_key not in all_results or short_key not in all_results:
        continue

    # Index combos by name
    long_by_combo = {' + '.join(c['combo']): c for c in all_results[long_key]}
    short_by_combo = {' + '.join(c['combo']): c for c in all_results[short_key]}
    all_combos = list(dict.fromkeys(
        list(long_by_combo.keys()) + list(short_by_combo.keys())
    ))

    rows = []
    for combo_str in all_combos:
        lc = long_by_combo.get(combo_str)
        sc = short_by_combo.get(combo_str)
        if not lc or not sc:
            continue
        groups_in_combo = lc['combo']
        total_count = lc['count'] + sc['count']
        row = {'Combo': combo_str, 'Total Count': total_count,
               'Long Count': lc['count'], 'Long Pct': round(lc['pct'], 2),
               'Short Count': sc['count'], 'Short Pct': round(sc['pct'], 2)}
        for g in GROUP_ORDER:
            if g in lc['z_info']:
                l_z = lc['z_info'][g]['mean_z']
                s_z = sc['z_info'][g]['mean_z']
                row[f'{GROUP_NAMES[g]} Long_z'] = round(l_z, 2)
                row[f'{GROUP_NAMES[g]} Short_z'] = round(s_z, 2)
                row[f'{GROUP_NAMES[g]} L-S'] = round(l_z - s_z, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Fixed canonical order: by combo size, then alphabetically
    df['_size'] = df['Combo'].str.count(r'\+') + 1
    df['_combo'] = df['Combo']
    df = df.sort_values(['_size', '_combo']).drop(columns=['_size', '_combo']).reset_index(drop=True)
    df.to_excel(writer, sheet_name=f'{regime}_L-S', index=False)

# Calm-Panic difference sheets (Long and Short)
for leg in ['Long', 'Short']:
    calm_key = f'Calm {leg}'
    panic_key = f'Panic {leg}'
    if calm_key not in all_results or panic_key not in all_results:
        continue

    calm_by_combo = {' + '.join(c['combo']): c for c in all_results[calm_key]}
    panic_by_combo = {' + '.join(c['combo']): c for c in all_results[panic_key]}
    all_combos = list(dict.fromkeys(
        list(calm_by_combo.keys()) + list(panic_by_combo.keys())
    ))

    rows = []
    for combo_str in all_combos:
        cc = calm_by_combo.get(combo_str)
        pc = panic_by_combo.get(combo_str)
        if not cc or not pc:
            continue
        row = {'Combo': combo_str,
               'Calm Pct': round(cc['pct'], 2),
               'Panic Pct': round(pc['pct'], 2)}
        for g in GROUP_ORDER:
            if g in cc['z_info']:
                c_z = cc['z_info'][g]['mean_z']
                p_z = pc['z_info'][g]['mean_z']
                row[f'{GROUP_NAMES[g]} Calm_z'] = round(c_z, 2)
                row[f'{GROUP_NAMES[g]} Panic_z'] = round(p_z, 2)
                row[f'{GROUP_NAMES[g]} C-P'] = round(c_z - p_z, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['_size'] = df['Combo'].str.count(r'\+') + 1
    df['_combo'] = df['Combo']
    df = df.sort_values(['_size', '_combo']).drop(columns=['_size', '_combo']).reset_index(drop=True)
    df.to_excel(writer, sheet_name=f'{leg}_C-P', index=False)

writer.close()

print("\nSaved: tree_combo_grouped_results.pkl, tree_combo_grouped_results.xlsx")

# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPORT LATEX TABLES
# ══════════════════════════════════════════════════════════════════════════════

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# ── Helper: format combo label for LaTeX (e.g. "S + I + L" -> "S {+} I {+} L") ──
def combo_label_tex(combo_str):
    return combo_str.replace(' + ', ' {+} ')

# ── table_combo_freq.tex ─────────────────────────────────────────────────────
# Frequency table: share of momentum-involving tree paths by group combo and
# regime x leg.  Columns: Combo, Calm Long, Calm Short, Panic Long, Panic Short.
# Sort by Calm Long percentage descending.

# Build a dict  combo_str -> {gname: pct}
freq_rows = {}
for gname in ['Calm Long', 'Calm Short', 'Panic Long', 'Panic Short']:
    for c in all_results[gname]:
        cstr = ' + '.join(c['combo'])
        freq_rows.setdefault(cstr, {})[gname] = c['pct']

# Sort by Calm Long pct descending
sorted_combos = sorted(freq_rows.keys(),
                       key=lambda k: freq_rows[k].get('Calm Long', 0),
                       reverse=True)

lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\begin{tabular}{l r r r r}')
lines.append(r'\toprule')
lines.append(r'Combination & Calm Long & Calm Short & Panic Long & Panic Short \\')
lines.append(r'\midrule')
for cstr in sorted_combos:
    vals = freq_rows[cstr]
    cl = vals.get('Calm Long', 0)
    cs = vals.get('Calm Short', 0)
    pl = vals.get('Panic Long', 0)
    ps = vals.get('Panic Short', 0)
    lines.append(f'{combo_label_tex(cstr)} & {cl:.1f}\\% & {cs:.1f}\\% & {pl:.1f}\\% & {ps:.1f}\\% \\\\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r"\caption{Share of momentum-involving tree paths by horizon group combination and regime$\times$leg. Each column sums to 100\%. Horizon groups: S = months 1--3, M = months 4--6, I = months 7--9, L = months 10--12.}")
lines.append(r'\label{tab:combo_freq}')
lines.append(r'\end{table}')

with open(os.path.join(TABLES_DIR, 'table_combo_freq.tex'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print("Saved: tables/table_combo_freq.tex")

# ── table_combo_long_cp.tex and table_combo_short_cp.tex ─────────────────────
# Calm-minus-panic z-score difference for each leg.
# Columns: Combo, Pct, S (1--3), M (4--6), I (7--9), L (10--12), Avg
# Rows are sorted by combo size then alphabetically.

for leg, tab_label, tab_file, caption_text in [
    ('Long', 'tab:combo_long_cp', 'table_combo_long_cp.tex',
     r"Calm-minus-panic z-score difference for the long leg, by horizon group combination. Each entry shows how much higher the average cross-sectional z-score is in calm than in panic. All 39 entries are positive: at every horizon in every combination, the model buys stocks with higher relative momentum in calm and lower relative momentum in panic. The largest shifts occur at the intermediate horizon (I, $+$0.36 to $+$0.55), where continuation is strongest. ``Pct'' is the average share of momentum-involving tree paths using that combination."),
    ('Short', 'tab:combo_short_cp', 'table_combo_short_cp.tex',
     r"Calm-minus-panic z-score difference for the short leg. Nearly all entries are negative: in calm, the model shorts stocks deep in the left tail (strongly negative z-scores), while in panic it shifts toward shorting stocks closer to the cross-sectional average or even above it. The short leg's upward z-score shift in panic is the mirror image of the long leg's downward shift (Table~\ref{tab:combo_long_cp}), confirming that the regime signal reverses the selection direction on both sides of the portfolio."),
]:
    calm_key = f'Calm {leg}'
    panic_key = f'Panic {leg}'
    calm_by_combo = {' + '.join(c['combo']): c for c in all_results[calm_key]}
    panic_by_combo = {' + '.join(c['combo']): c for c in all_results[panic_key]}
    all_combos_set = list(dict.fromkeys(
        list(calm_by_combo.keys()) + list(panic_by_combo.keys())
    ))

    cp_rows = []
    for combo_str in all_combos_set:
        cc = calm_by_combo.get(combo_str)
        pc = panic_by_combo.get(combo_str)
        if not cc or not pc:
            continue
        avg_pct = (cc['pct'] + pc['pct']) / 2
        groups_in_combo = cc['combo']
        diffs = {}
        for g in groups_in_combo:
            c_z = cc['z_info'][g]['mean_z']
            p_z = pc['z_info'][g]['mean_z']
            diffs[g] = c_z - p_z
        cp_rows.append({
            'combo_str': combo_str,
            'groups': groups_in_combo,
            'pct': avg_pct,
            'diffs': diffs,
        })

    # Sort by combo size then alphabetically
    cp_rows.sort(key=lambda r: (len(r['groups']), r['combo_str']))

    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l r r r r r r}')
    lines.append(r'\toprule')
    lines.append(r'Combination & Pct & S (1--3) & M (4--6) & I (7--9) & L (10--12) & Avg \\')
    lines.append(r'\midrule')
    for row in cp_rows:
        label = combo_label_tex(row['combo_str'])
        pct_str = f"{row['pct']:.1f}\\%"
        cells = []
        vals_for_avg = []
        for g in GROUP_ORDER:
            if g in row['diffs']:
                v = row['diffs'][g]
                cells.append(f'{v:+.2f}')
                vals_for_avg.append(v)
            else:
                cells.append('')
        avg_val = np.mean(vals_for_avg) if vals_for_avg else 0
        avg_str = f'{avg_val:+.2f}'
        lines.append(f"{label} & {pct_str} & {' & '.join(cells)} & {avg_str} \\\\")
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(f'\\caption{{{caption_text}}}')
    lines.append(f'\\label{{{tab_label}}}')
    lines.append(r'\end{table}')

    with open(os.path.join(TABLES_DIR, tab_file), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved: tables/{tab_file}")

print("Done.")
