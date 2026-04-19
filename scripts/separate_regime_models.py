"""
separate_regime_models.py
=========================
Train separate XGBoost models on calm-only and panic-only training data
using only momentum features (no pi_filter). Compare tree structures,
combo frequencies, split directions, and stock selection.
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, joblib
from collections import Counter, defaultdict
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, XGB_SEEDS, MOM_FEATURES, TRADING_FEE)

GROUPS = {
    'mom_1': 'S', 'mom_2': 'S', 'mom_3': 'S',
    'mom_4': 'M', 'mom_5': 'M', 'mom_6': 'M',
    'mom_7': 'I', 'mom_8': 'I', 'mom_9': 'I',
    'mom_10': 'L', 'mom_11': 'L', 'mom_12': 'L',
}
GROUP_ORDER = ['S', 'M', 'I', 'L']

# ── Load ──
print("Loading data ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
train = art['train'].copy()
test = art['test'].copy()
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

for df in [train, test]:
    df_merged = df.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
        columns={'pi_filter': 'pi_month'}), on='date', how='left')
    df['pi_month'] = df_merged['pi_month'].values
    df['regime'] = np.where(df['pi_month'] >= 0.5, 'Panic', 'Calm')

# ── Train separate models ──
print("\nTraining separate models ...")

FEATURES_MOM = MOM_FEATURES  # 12 momentum features only

results = {}
for regime in ['Calm', 'Panic']:
    train_r = train[train['regime'] == regime]
    X_tr = train_r[FEATURES_MOM].values.astype(float)
    y_tr = train_r['ret_fwd'].values.astype(float)

    # Handle NaN
    for j in range(X_tr.shape[1]):
        col_med = np.nanmedian(X_tr[:, j])
        X_tr[np.isnan(X_tr[:, j]), j] = col_med

    print(f"\n  {regime}: {len(train_r):,} training observations")

    # Train 10-seed ensemble
    all_preds = np.zeros(len(test))
    all_trees = []

    for xs in range(1, 11):
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                           learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE, tree_method='hist',
                           random_state=xs, verbosity=0)
        xgb.fit(X_tr, y_tr)

        X_te = test[FEATURES_MOM].values.astype(float)
        for j in range(X_te.shape[1]):
            col_med = np.nanmedian(X_te[:, j])
            X_te[np.isnan(X_te[:, j]), j] = col_med

        all_preds += xgb.predict(X_te)

        # Parse trees from first seed only for structure analysis
        if xs == 1:
            dump = xgb.get_booster().get_dump(dump_format='text')
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
                                              feature=FEATURES_MOM[fidx],
                                              threshold=float(m.group(2)),
                                              yes=int(m.group(3)), no=int(m.group(4)))
                all_trees.append(nodes)

    all_preds /= 10
    test[f'score_{regime.lower()}'] = all_preds
    results[regime] = {'trees': all_trees, 'n_trees': len(all_trees)}
    print(f"    Trees parsed: {len(all_trees)}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE IMPORTANCE (root split frequency)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. ROOT SPLIT FEATURE FREQUENCY")
print("="*60)

for regime in ['Calm', 'Panic']:
    root_features = Counter()
    for tree in results[regime]['trees']:
        root = tree[0]
        if not root.get('leaf', False):
            feat = root['feature']
            g = GROUPS.get(feat, feat)
            root_features[g] += 1
    total = sum(root_features.values())
    print(f"\n  {regime} model root splits:")
    for g in GROUP_ORDER:
        count = root_features.get(g, 0)
        print(f"    {g}: {count}/{total} ({count/total*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. COMBO FREQUENCIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. COMBO FREQUENCIES (Calm model vs Panic model)")
print("="*60)

def trace_stock(tree, feature_values, feature_names):
    node_id = 0
    groups_used = set()
    directions = {}
    while True:
        node = tree[node_id]
        if node['leaf']:
            return frozenset(groups_used), node['value'], directions
        feat = node['feature']
        feat_idx = feature_names.index(feat)
        went_left = feature_values[feat_idx] < node['threshold']
        node_id = node['yes'] if went_left else node['no']
        if feat in GROUPS:
            g = GROUPS[feat]
            groups_used.add(g)
            directions[g] = 'below' if went_left else 'above'

# Use test stocks from the SAME regime as the model
X_te = test[FEATURES_MOM].values.astype(float)
for j in range(X_te.shape[1]):
    col_med = np.nanmedian(X_te[:, j])
    X_te[np.isnan(X_te[:, j]), j] = col_med

SAMPLE = 3000
rng = np.random.RandomState(42)

for regime in ['Calm', 'Panic']:
    regime_mask = test['regime'] == regime
    indices = np.where(regime_mask)[0]
    if len(indices) > SAMPLE:
        indices = rng.choice(indices, SAMPLE, replace=False)

    X_group = X_te[indices]
    trees = results[regime]['trees']

    combo_counter = Counter()
    combo_directions = defaultdict(lambda: defaultdict(Counter))

    for tree in trees:
        for stock_idx in range(len(X_group)):
            combo, leaf_val, directions = trace_stock(tree, X_group[stock_idx], FEATURES_MOM)
            if combo:
                combo_counter[combo] += 1
                for g, d in directions.items():
                    combo_directions[combo][g][d] += 1

    total = sum(combo_counter.values())
    print(f"\n  {regime} model (applied to {regime} test stocks):")
    for combo, count in combo_counter.most_common(10):
        combo_str = '+'.join(sorted(combo, key=lambda g: GROUP_ORDER.index(g)))
        pct = count / total * 100
        print(f"    {combo_str}: {pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SPLIT DIRECTION: % going 'above' threshold
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. SPLIT DIRECTION: % ABOVE THRESHOLD BY GROUP")
print("  (Each model applied to its own regime's test stocks)")
print("="*60)

for regime in ['Calm', 'Panic']:
    regime_mask = test['regime'] == regime
    indices = np.where(regime_mask)[0]
    if len(indices) > SAMPLE:
        indices = rng.choice(indices, SAMPLE, replace=False)

    X_group = X_te[indices]
    trees = results[regime]['trees']

    group_above = defaultdict(int)
    group_total = defaultdict(int)

    for tree in trees:
        for stock_idx in range(len(X_group)):
            combo, leaf_val, directions = trace_stock(tree, X_group[stock_idx], FEATURES_MOM)
            for g, d in directions.items():
                group_total[g] += 1
                if d == 'above':
                    group_above[g] += 1

    print(f"\n  {regime} model:")
    for g in GROUP_ORDER:
        if group_total[g] > 0:
            pct = group_above[g] / group_total[g] * 100
            print(f"    {g}: {pct:.1f}% above threshold")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-APPLY: Calm model on panic stocks, Panic model on calm stocks
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4. CROSS-APPLICATION: Each model on BOTH regimes' stocks")
print("="*60)

for model_regime in ['Calm', 'Panic']:
    trees = results[model_regime]['trees']
    print(f"\n  {model_regime} MODEL applied to:")

    for stock_regime in ['Calm', 'Panic']:
        regime_mask = test['regime'] == stock_regime
        indices = np.where(regime_mask)[0]
        if len(indices) > SAMPLE:
            indices = rng.choice(indices, SAMPLE, replace=False)

        X_group = X_te[indices]
        group_above = defaultdict(int)
        group_total = defaultdict(int)

        for tree in trees:
            for stock_idx in range(len(X_group)):
                combo, leaf_val, directions = trace_stock(tree, X_group[stock_idx], FEATURES_MOM)
                for g, d in directions.items():
                    group_total[g] += 1
                    if d == 'above':
                        group_above[g] += 1

        print(f"    {stock_regime} stocks: ", end='')
        for g in GROUP_ORDER:
            if group_total[g] > 0:
                pct = group_above[g] / group_total[g] * 100
                print(f"{g}={pct:.1f}% ", end='')
        print()

# ══════════════════════════════════════════════════════════════════════════════
# 5. PORTFOLIO PERFORMANCE: each regime model on its own regime
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  5. PORTFOLIO PERFORMANCE")
print("="*60)

def long_short_port(df_data, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_data.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0)) for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0)) for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    return pd.DataFrame(monthly).set_index('date')['ret']

# Calm model on all months
r_calm_model = long_short_port(test, 'score_calm')
# Panic model on all months
r_panic_model = long_short_port(test, 'score_panic')
# Production model
r_prod = long_short_port(test, 'score_xgb')

for name, r in [('Calm model (all months)', r_calm_model),
                ('Panic model (all months)', r_panic_model),
                ('Production model', r_prod)]:
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    print(f"  {name}: Sharpe={sharpe:.2f}, Ann.Ret={ann_ret:.1%}, N={len(r)}")

# Regime-conditional performance
pi_test = test[['date', 'pi_month']].drop_duplicates('date').set_index('date')

for name, r in [('Calm model', r_calm_model), ('Panic model', r_panic_model), ('Production', r_prod)]:
    pi = pi_test.reindex(r.index)['pi_month']
    calm_r = r[pi < 0.5]
    panic_r = r[pi >= 0.5]
    calm_sh = calm_r.mean() / calm_r.std() * np.sqrt(12) if len(calm_r) > 1 and calm_r.std() > 0 else 0
    panic_sh = panic_r.mean() / panic_r.std() * np.sqrt(12) if len(panic_r) > 1 and panic_r.std() > 0 else 0
    print(f"  {name}: Calm Sharpe={calm_sh:.2f} ({len(calm_r)}mo), Panic Sharpe={panic_sh:.2f} ({len(panic_r)}mo)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Z-SCORE PROFILES: what each regime model selects
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  6. Z-SCORE PROFILES OF SELECTED STOCKS")
print("="*60)

for model_name, score_col in [('Calm model', 'score_calm'), ('Panic model', 'score_panic')]:
    for stock_regime in ['Calm', 'Panic']:
        regime_data = test[test['regime'] == stock_regime]

        # Assign legs for this model's scores
        zs = []
        for date, grp in regime_data.groupby('date'):
            nyse = grp[grp['exchcd'] == 1][score_col].dropna()
            if len(nyse) < 10:
                continue
            hi = nyse.quantile(0.90)
            longs = grp[grp[score_col] >= hi]
            if len(longs) == 0:
                continue
            for h in range(1, 13):
                col = f'mom_{h}'
                cs_mean = grp[col].mean()
                cs_std = grp[col].std()
                if cs_std > 0:
                    z = (longs[col].mean() - cs_mean) / cs_std
                    zs.append({'horizon': h, 'z': z})

        zdf = pd.DataFrame(zs)
        print(f"\n  {model_name} → {stock_regime} stocks (LONG leg z-scores):")
        for h in range(1, 13):
            z = zdf[zdf['horizon'] == h]['z'].mean()
            print(f"    mom_{h:>2d}: {z:+.3f}")

print("\nDone.")
