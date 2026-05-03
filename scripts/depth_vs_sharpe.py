"""
Annualised return vs tree depth, showing that multi-dimensional conditioning
(depth >= 3) is required for full performance.

Recomputes from production artefacts (50 seeds per depth).
Results saved to depth_results.csv for reproducibility.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pickle, sys, os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_ESTIMATORS, LEARNING_RATE, SUBSAMPLE, COLSAMPLE,
                    XGB_SEEDS, TRADING_FEE)

# Load artefacts
print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)

train = art['train'].copy()
test = art['test'].copy()
FEATURES = art['FEATURES']
X_train = art['X_train']
X_test = art['X_test']
y_train = art['y_train']

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
        lme = longs['me'].sum()
        nlw = (longs.set_index('permno')['me'] / lme).to_dict()
        rl = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        nsw = (shorts.set_index('permno')['me'] / sme).to_dict()
        rs = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(nlw.get(p,0)-prev_lw.get(p,0)) for p in set(nlw)|set(prev_lw))/2
        ts = sum(abs(nsw.get(p,0)-prev_sw.get(p,0)) for p in set(nsw)|set(prev_sw))/2
        monthly.append({'date': date, 'ret': rl - rs - fee*(tl+ts)})
        prev_lw, prev_sw = nlw, nsw
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

# Compute for each depth
depths_list = [1, 2, 3, 4, 5, 6]
results = []

for depth in depths_list:
    print(f"  Depth {depth} ...", flush=True)
    preds = np.zeros(len(X_test))
    for xs in XGB_SEEDS:
        model = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=depth,
                             learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                             colsample_bytree=COLSAMPLE, tree_method='hist',
                             random_state=xs, verbosity=0)
        model.fit(X_train, y_train)
        preds += model.predict(X_test)
    preds /= len(XGB_SEEDS)

    test[f'score_d{depth}'] = preds
    r = long_short_port(test, f'score_d{depth}')
    ann_ret = ((1 + r).prod() ** (12 / len(r)) - 1) * 100
    ann_vol = r.std() * np.sqrt(12) * 100
    sharpe = r.mean() / r.std() * np.sqrt(12)
    results.append({'depth': depth, 'ann_ret': round(ann_ret, 1),
                    'ann_vol': round(ann_vol, 1), 'sharpe': round(sharpe, 2)})
    print(f"    Ann.Ret={ann_ret:.1f}%  Ann.Vol={ann_vol:.1f}%  Sharpe={sharpe:.2f}")

# Save results
df = pd.DataFrame(results)
df.to_csv('results/thesis/depth_results.csv', index=False)
print("Saved: depth_results.csv")

returns = df['ann_ret'].values
depths = df['depth'].values

# Plot
fig, ax = plt.subplots(figsize=(8, 5.5))

colors = ['#E53935' if d < 3 else '#2196F3' if d <= 4 else '#90A4AE' for d in depths]
bars = ax.bar(depths, returns, color=colors, alpha=0.85, edgecolor='white', width=0.6)

for bar, r in zip(bars, returns):
    ax.text(bar.get_x() + bar.get_width()/2, r + 0.4,
            f'{r:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(depths)
ax.set_xticklabels([f'{d}' for d in depths], fontsize=11)
ax.set_xlabel('Maximum tree depth (order of feature interactions)', fontsize=12)
ax.set_ylabel('Out-of-sample annualised return (%)', fontsize=12)
ax.set_title('Tree depth and the regime-momentum interaction',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, max(returns) + 5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig('plots/thesis/depth_vs_sharpe.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: depth_vs_sharpe.png")

from PIL import Image
img = Image.open('plots/thesis/depth_vs_sharpe.png')
img.save('plots/depth_vs_sharpe.pdf', 'PDF', resolution=150)
print("Saved: plots/depth_vs_sharpe.pdf")
