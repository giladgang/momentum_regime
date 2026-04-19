"""
zscore_pi_only.py
=================
Reproduce the z-score + |SHAP| figure but only for stocks whose
prediction is dominated by pi_filter trees (>50% of score from pi trees).

Compares: all stocks vs pi-dominated stocks.
"""

import numpy as np
import pandas as pd
import pickle, re, sys, os, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
FEATURES = art['FEATURES']
shap_values = art['shap_values']

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

# Load production model and parse trees
xgb_model = joblib.load('artefacts/cs_artefacts_xgb.pkl')
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

# ── Compute pi_score fraction for all long/short stocks ──
X_test = test[FEATURES].values.astype(float)
regime_arr = test['regime'].values
leg_arr = test['leg'].values

# Only compute for long/short stocks (not middle)
ls_mask = (leg_arr == 'long') | (leg_arr == 'short')
ls_indices = np.where(ls_mask)[0]

print(f"Computing pi_score fraction for {len(ls_indices):,} long/short stocks ...")

score_pi = np.zeros(len(X_test))
score_no_pi = np.zeros(len(X_test))

# Process in batches to manage memory
BATCH = 5000
n_batches = (len(ls_indices) + BATCH - 1) // BATCH

for b in range(n_batches):
    start = b * BATCH
    end = min((b + 1) * BATCH, len(ls_indices))
    batch_indices = ls_indices[start:end]
    X_batch = X_test[batch_indices]

    for tree in all_trees:
        for i, global_idx in enumerate(batch_indices):
            node_id = 0
            hit_pi = False
            while True:
                node = tree[node_id]
                if node['leaf']:
                    if hit_pi:
                        score_pi[global_idx] += node['value']
                    else:
                        score_no_pi[global_idx] += node['value']
                    break
                feat = node['feature']
                feat_idx = FEATURES.index(feat)
                if feat == 'pi_filter':
                    hit_pi = True
                if X_batch[i, feat_idx] < node['threshold']:
                    node_id = node['yes']
                else:
                    node_id = node['no']

    if (b + 1) % 5 == 0 or b == n_batches - 1:
        print(f"  Batch {b+1}/{n_batches} done")

total_score = score_pi + score_no_pi
# Fraction of score from pi trees (handle division by zero)
pi_frac = np.where(np.abs(total_score) > 1e-10,
                   np.abs(score_pi) / (np.abs(score_pi) + np.abs(score_no_pi)),
                   0.5)

test['pi_frac'] = pi_frac
test['pi_dominated'] = pi_frac > 0.5

print(f"\nPi-dominated stocks (>50% score from pi trees):")
for regime in ['Calm', 'Panic']:
    for leg in ['long', 'short']:
        mask = (test['regime'] == regime) & (test['leg'] == leg)
        n_total = mask.sum()
        n_pi = (mask & test['pi_dominated']).sum()
        print(f"  {regime} {leg}: {n_pi:,}/{n_total:,} ({n_pi/n_total*100:.1f}%)")

# ── Compute z-scores and SHAP for pi-dominated stocks ──
horizons = list(range(1, 13))
mom_indices = [FEATURES.index(f'mom_{h}') for h in horizons]

x = np.arange(1, 13)
w = 0.35

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes[0, 0].sharey(axes[0, 1])

for col_idx, regime_val in enumerate(['Calm', 'Panic']):
    ax_z = axes[0, col_idx]
    ax_s = axes[1, col_idx]

    for leg_val, color, marker in [('long', '#2196F3', 'o'), ('short', '#E53935', 's')]:
        # Z-scores for pi-dominated stocks only
        zs_mean = []
        zs_std = []
        for h in horizons:
            col = f'mom_{h}'
            mask = (test['regime'] == regime_val) & test['pi_dominated']
            monthly_z = []
            for date, grp in test[mask].groupby('date'):
                leg_mask = grp['leg'] == leg_val
                if leg_mask.sum() == 0:
                    continue
                cs_mean = test.loc[test['date'] == date, col].mean()
                cs_std = test.loc[test['date'] == date, col].std()
                if cs_std > 0:
                    stock_zs = (grp.loc[leg_mask, col] - cs_mean) / cs_std
                    monthly_z.append(stock_zs.mean())
            zs_mean.append(np.mean(monthly_z) if monthly_z else 0)
            zs_std.append(np.std(monthly_z) / np.sqrt(len(monthly_z)) if len(monthly_z) > 1 else 0)

        zs_mean = np.array(zs_mean)
        zs_std = np.array(zs_std)

        ax_z.plot(x, zs_mean, f'{marker}-', color=color, linewidth=2.5,
                  markersize=8, label=f'{leg_val.title()} leg', zorder=5)
        ax_z.fill_between(x, zs_mean - zs_std, zs_mean + zs_std,
                          alpha=0.12, color=color)

        # SHAP for pi-dominated stocks
        shap_mask = (test['regime'] == regime_val).values & test['pi_dominated'].values & (test['leg'] == leg_val).values
        abs_shap = np.abs(shap_values[shap_mask][:, mom_indices]).mean(axis=0)
        pct_shap = abs_shap / abs_shap.sum() * 100

        offset = -w/2 if leg_val == 'long' else w/2
        ax_s.bar(x + offset, pct_shap, w, label=f'{leg_val.title()} leg',
                 color=color, alpha=0.85, edgecolor='white')

    ax_z.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    n = int((test.drop_duplicates('date')['regime'] == regime_val).sum())
    n_pi = int((test[test['pi_dominated']].drop_duplicates('date')['regime'] == regime_val).sum())
    ax_z.set_title(f'{regime_val} ({n} months, pi-dominated stocks only)',
                   fontsize=12, fontweight='bold')
    ax_z.legend(fontsize=9)
    ax_z.grid(alpha=0.3)
    if col_idx == 0:
        ax_z.set_ylabel('Z-score vs cross-section', fontsize=11)

    ax_s.set_title(f'{regime_val}', fontsize=12, fontweight='bold')
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([f'{h}' for h in horizons], fontsize=10)
    ax_s.set_xlabel('Momentum lookback horizon (months)', fontsize=11)
    ax_s.legend(fontsize=9)
    ax_s.grid(axis='y', alpha=0.3)
    if col_idx == 0:
        ax_s.set_ylabel('Share of momentum |SHAP| (%)', fontsize=11)

axes[1, 0].sharey(axes[1, 1])

plt.suptitle('Momentum term structure (pi_filter-dominated stocks only)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig('plots/zscore_and_absshap_pi_only.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved: plots/zscore_and_absshap_pi_only.png")

print("Done.")
