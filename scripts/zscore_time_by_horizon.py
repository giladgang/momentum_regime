"""
zscore_time_by_horizon.py
=========================
Exports a monthly table with rows = date and columns = mom_1..mom_12,
filled with the mean cross-sectional z-score of long-leg stocks at that
horizon. No regime split.

Output:
- results/zscore_long_by_month.csv
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()

test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

horizons = list(range(1, 13))
rows = []
for date, grp in test.groupby('date'):
    row = {'date': date}
    long_mask = grp['leg'] == 'long'
    if long_mask.sum() == 0:
        for h in horizons:
            row[f'mom_{h}'] = np.nan
        rows.append(row)
        continue
    for h in horizons:
        col = f'mom_{h}'
        mean = grp[col].mean()
        std = grp[col].std()
        if std > 0:
            zs = (grp.loc[long_mask, col] - mean) / std
            row[f'mom_{h}'] = zs.mean()
        else:
            row[f'mom_{h}'] = np.nan
    rows.append(row)

df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
out_path = 'results/thesis/zscore_long_by_month.csv'
df.to_csv(out_path, index=False, float_format='%.4f')
print(f"Saved: {out_path}  ({len(df)} rows x {len(horizons)} horizons)")
print(df.head())
