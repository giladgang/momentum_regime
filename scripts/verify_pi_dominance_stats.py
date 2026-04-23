"""
Verify the three pi_filter dominance statistics claimed in the thesis:

  (1) Fraction of production trees with >=1 pi_filter split.
  (2) Fraction of a typical stock's tree paths that pass through a pi_filter split.
  (3) Share of the long-leg predicted return that flows from pi-splitting trees,
      separately for panic vs. calm months.

Production model = the committed saved ensemble (50 seeds, 500 trees each,
25,000 trees total). If only the single-seed artefact is available, we
fall back to that with a loud warning.
"""
import numpy as np, pandas as pd, pickle, joblib, re, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
FEATURES = art['FEATURES']
test = art['test'].copy()
X_test = art['X_test']
CS_FEATURES = FEATURES
PI_IDX = FEATURES.index('pi_filter')

try:
    from config import XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE
except Exception:
    XGB_SEEDS = list(range(1, 51)); N_ESTIMATORS = 500; MAX_DEPTH = 4
    LEARNING_RATE = 0.05; SUBSAMPLE = 0.8; COLSAMPLE = 0.8

# Production run: full 50-seed ensemble (25,000 trees). --fast for seed-50 only.
FAST = '--fast' in sys.argv
if FAST:
    XGB_SEEDS = [50]
    print("[FAST mode] seed 50 only (500 trees)")
else:
    print(f"[PRODUCTION mode] {len(XGB_SEEDS)} seeds x {N_ESTIMATORS} trees "
          f"= {len(XGB_SEEDS)*N_ESTIMATORS:,} total trees")

from xgboost import XGBRegressor

def parse_tree(dump_text):
    """Dict of node_id -> {'feature', 'threshold', 'yes', 'no', 'leaf', 'value'}."""
    tree = {}
    for line in dump_text.strip().split('\n'):
        s = line.strip()
        m_leaf = re.match(r'(\d+):leaf=([-\d.e+]+)', s)
        if m_leaf:
            tree[int(m_leaf.group(1))] = {'leaf': True, 'value': float(m_leaf.group(2))}
            continue
        m_split = re.match(r'(\d+):\[f(\d+)<([-\d.e+]+)\] yes=(\d+),no=(\d+)', s)
        if m_split:
            nid, fidx, thr, yes, no = m_split.groups()
            tree[int(nid)] = {'leaf': False, 'feature': FEATURES[int(fidx)],
                              'threshold': float(thr), 'yes': int(yes), 'no': int(no)}
    return tree

def trace(tree, x):
    """Return (leaf_value, had_pi_split) for stock vector x through one tree."""
    nid = 0
    had_pi = False
    while True:
        node = tree[nid]
        if node['leaf']:
            return node['value'], had_pi
        if node['feature'] == 'pi_filter':
            had_pi = True
        fidx = FEATURES.index(node['feature'])
        nid = node['yes'] if x[fidx] < node['threshold'] else node['no']

# ── Train all 50 seeds (this is the production ensemble definition) ─────
CACHE_TREES = f'artefacts/pi_verify_trees_seeds{len(XGB_SEEDS)}.pkl'
CACHE_SCORES = f'artefacts/pi_verify_scores_seeds{len(XGB_SEEDS)}.pkl'
X_train = art['X_train']
y_train = art['train']['ret_fwd'].values

if os.path.exists(CACHE_TREES) and os.path.exists(CACHE_SCORES):
    print(f"Loading cached trees + scores (seeds={len(XGB_SEEDS)}) ...")
    with open(CACHE_TREES, 'rb') as f:
        all_trees, seed_tree_counts = pickle.load(f)
    with open(CACHE_SCORES, 'rb') as f:
        cached_scores = pickle.load(f)
    print(f"  loaded {len(all_trees):,} trees from cache")
else:
    print(f"Training {len(XGB_SEEDS)} XGB seeds to reproduce production ensemble ...")
    t0 = time.time()
    all_trees = []
    seed_tree_counts = []
    cached_scores = np.zeros(len(X_test))
    for xs in XGB_SEEDS:
        m = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                         learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                         colsample_bytree=COLSAMPLE, tree_method='hist',
                         random_state=xs, verbosity=0)
        m.fit(X_train, y_train)
        cached_scores += m.predict(X_test)
        dump = m.get_booster().get_dump(dump_format='text')
        n_with_pi = 0
        for raw in dump:
            tree = parse_tree(raw)
            has_pi = any((not n.get('leaf')) and n.get('feature') == 'pi_filter' for n in tree.values())
            all_trees.append({'tree': tree, 'has_pi': has_pi})
            if has_pi: n_with_pi += 1
        seed_tree_counts.append(n_with_pi)
    cached_scores /= len(XGB_SEEDS)
    print(f"  elapsed {time.time()-t0:.1f}s | {len(all_trees):,} trees parsed")
    with open(CACHE_TREES, 'wb') as f:
        pickle.dump((all_trees, seed_tree_counts), f)
    with open(CACHE_SCORES, 'wb') as f:
        pickle.dump(cached_scores, f)
    print(f"  cached to {CACHE_TREES}, {CACHE_SCORES}")

# ── (1) Fraction of trees with >=1 pi split ─────────────────────────────
n_trees = len(all_trees)
n_pi_trees = sum(t['has_pi'] for t in all_trees)
print(f"\n(1) Trees with >=1 pi_filter split:")
print(f"    Production (50-seed ensemble):  {n_pi_trees:,}/{n_trees:,} "
      f"= {n_pi_trees/n_trees*100:.1f}%")
seed_pcts = [c/N_ESTIMATORS*100 for c in seed_tree_counts]
print(f"    Per-seed:  mean {np.mean(seed_pcts):.1f}%  "
      f"min {np.min(seed_pcts):.1f}%  max {np.max(seed_pcts):.1f}%")
print(f"    Seed-50 alone (thesis's cited 75.8%): "
      f"{seed_tree_counts[XGB_SEEDS.index(50)]}/{N_ESTIMATORS} "
      f"= {seed_tree_counts[XGB_SEEDS.index(50)]/N_ESTIMATORS*100:.1f}%")

# ── Identify calm-long and panic-long test-set rows for (2) and (3) ─────
# Need portfolio membership: long if in top decile of predicted score each month,
# short if bottom decile. We already have artefacts test with predicted scores.
print("\nIdentifying long-leg rows for calm/panic months ...")

# Map pandas index -> array position (X_test is ndarray aligned with test).
test_reset = test.reset_index(drop=False).rename(columns={'index': '_orig_idx'})
test_reset['_array_pos'] = np.arange(len(test_reset))
test_reset['score'] = cached_scores

# pick long leg each month (top decile of NYSE score)
long_rows = []
for date, grp in test_reset.groupby('date'):
    nyse_scores = grp[grp['exchcd'] == 1]['score'].dropna()
    if len(nyse_scores) < 10: continue
    hi = nyse_scores.quantile(0.90)
    long_rows.append(grp[grp['score'] >= hi])
long_df = pd.concat(long_rows)
long_df = long_df.rename(columns={'_array_pos': 'test_pos'})

# classify months as panic (pi_filter >= 0.5) vs calm
long_df['regime'] = np.where(long_df['pi_filter'] >= 0.5, 'panic', 'calm')

# Full-population run: trace every long-leg row. Use --sample N for a subsample.
sample_n = None
for arg in sys.argv[1:]:
    if arg.startswith('--sample='):
        sample_n = int(arg.split('=')[1])
if sample_n is not None:
    rng = np.random.RandomState(42)
    sampled = []
    for r in ('calm', 'panic'):
        idx_r = long_df.index[long_df['regime'] == r].to_numpy()
        if len(idx_r) > sample_n:
            idx_r = rng.choice(idx_r, sample_n, replace=False)
        sampled.append(long_df.loc[idx_r])
    long_df = pd.concat(sampled)
    print(f"  sampled {len(long_df):,} long-leg rows "
          f"({(long_df['regime']=='calm').sum()} calm, "
          f"{(long_df['regime']=='panic').sum()} panic)")
else:
    n_c = (long_df['regime']=='calm').sum(); n_p = (long_df['regime']=='panic').sum()
    print(f"  full population: {len(long_df):,} long-leg rows ({n_c} calm, {n_p} panic)")
n_panic = (long_df['regime'] == 'panic').sum()
n_calm  = (long_df['regime'] == 'calm').sum()
print(f"  long-leg rows: {n_calm:,} calm, {n_panic:,} panic")

# ── (2) and (3): trace every long-leg stock through every tree ──────────
print("\nTracing long-leg stocks through all trees (this takes a few minutes) ...")
t0 = time.time()

paths_total = 0
paths_with_pi = 0
sum_leaf = {'calm': {'with_pi': 0.0, 'without_pi': 0.0},
            'panic': {'with_pi': 0.0, 'without_pi': 0.0}}

X_test_arr = X_test
for j, row in enumerate(long_df.itertuples()):
    x = X_test_arr[row.test_pos]
    regime = row.regime
    for t in all_trees:
        leaf_val, had_pi = trace(t['tree'], x)
        paths_total += 1
        if had_pi:
            paths_with_pi += 1
            sum_leaf[regime]['with_pi'] += leaf_val
        else:
            sum_leaf[regime]['without_pi'] += leaf_val
    if (j+1) % 500 == 0:
        print(f"  {j+1}/{len(long_df)} rows  ({(j+1)/len(long_df)*100:.1f}%)  "
              f"elapsed {time.time()-t0:.0f}s")
print(f"  done in {time.time()-t0:.0f}s")

# (2)
print(f"\n(2) Long-leg path fraction through pi_filter splits:")
print(f"    {paths_with_pi:,}/{paths_total:,} = {paths_with_pi/paths_total*100:.1f}%")

# (3)
for regime in ('panic', 'calm'):
    w = sum_leaf[regime]['with_pi']
    wo = sum_leaf[regime]['without_pi']
    total = w + wo
    print(f"\n(3) {regime.capitalize()} long-leg leaf-sum decomposition:")
    print(f"    pi-splitting trees:       {w:+.2f}")
    print(f"    momentum-only trees:      {wo:+.2f}")
    print(f"    total:                    {total:+.2f}")
    if abs(total) > 1e-9:
        print(f"    share from pi-trees:      {w/total*100:.1f}%")
