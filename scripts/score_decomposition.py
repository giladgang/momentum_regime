"""
score_decomposition.py
======================
Decompose each stock's XGBoost prediction into the contribution from
trees that split on pi_filter vs trees that don't.

For stocks in the long and short legs: how much of their score came
from regime-aware trees?
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

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')
test = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test['regime'] = np.where(test['pi_month'] >= 0.5, 'Panic', 'Calm')

# Assign legs
test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

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
regime_arr = test['regime'].values
leg_arr = test['leg'].values

# Sample from each group
SAMPLE = 2000
rng = np.random.RandomState(42)

groups_def = {
    'Calm Long': (regime_arr == 'Calm') & (leg_arr == 'long'),
    'Calm Short': (regime_arr == 'Calm') & (leg_arr == 'short'),
    'Panic Long': (regime_arr == 'Panic') & (leg_arr == 'long'),
    'Panic Short': (regime_arr == 'Panic') & (leg_arr == 'short'),
}

for gname, gmask in groups_def.items():
    indices = np.where(gmask)[0]
    if len(indices) > SAMPLE:
        indices = rng.choice(indices, SAMPLE, replace=False)

    X_group = X_test[indices]
    n = len(X_group)

    score_pi = np.zeros(n)      # sum of leaf values from trees where stock hit pi_filter
    score_no_pi = np.zeros(n)   # sum of leaf values from trees where stock did NOT hit pi_filter
    count_pi = np.zeros(n, dtype=int)
    count_no_pi = np.zeros(n, dtype=int)

    for tree in all_trees:
        for stock_idx in range(n):
            node_id = 0
            hit_pi = False
            while True:
                node = tree[node_id]
                if node['leaf']:
                    if hit_pi:
                        score_pi[stock_idx] += node['value']
                        count_pi[stock_idx] += 1
                    else:
                        score_no_pi[stock_idx] += node['value']
                        count_no_pi[stock_idx] += 1
                    break
                feat = node['feature']
                feat_idx = FEATURES.index(feat)
                if feat == 'pi_filter':
                    hit_pi = True
                if X_group[stock_idx, feat_idx] < node['threshold']:
                    node_id = node['yes']
                else:
                    node_id = node['no']

    total_score = score_pi + score_no_pi

    print(f"\n{'='*60}")
    print(f"  {gname} ({n} stocks)")
    print(f"{'='*60}")
    print(f"  Average total score:      {total_score.mean():+.6f}")
    print(f"  From pi_filter trees:     {score_pi.mean():+.6f} ({score_pi.mean()/total_score.mean()*100:.1f}% of total)")
    print(f"  From non-pi trees:        {score_no_pi.mean():+.6f} ({score_no_pi.mean()/total_score.mean()*100:.1f}% of total)")
    print(f"  Avg trees with pi:        {count_pi.mean():.0f} / 500")
    print(f"  Avg trees without pi:     {count_no_pi.mean():.0f} / 500")

    # What fraction of the DIFFERENCE between long and short comes from pi trees?
    if 'Long' in gname:
        print(f"\n  Score from pi trees:    min={score_pi.min():+.6f}  max={score_pi.max():+.6f}")
        print(f"  Score from non-pi trees: min={score_no_pi.min():+.6f}  max={score_no_pi.max():+.6f}")

print(f"\n{'='*60}")
print(f"  SCORE DECOMPOSITION: Long - Short spread")
print(f"{'='*60}")

for regime in ['Calm', 'Panic']:
    long_key = f'{regime} Long'
    short_key = f'{regime} Short'

    # Recompute for the means we already have
    # We need to re-access the data from the loop above
    # Simpler: just print the differences from the means

print("\n(See per-group scores above to compute spreads)")
print("Done.")
