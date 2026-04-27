"""
zscore_time_by_horizon.py
=========================
Exports three monthly tables with rows = date and columns = mom_1..mom_12,
filled with mean cross-sectional z-scores at each horizon for the long leg,
short leg, and long-minus-short spread. No regime split here — that's a
downstream view.

This is the plotly-free generator for the three by-month z-score CSVs that
feed §5.2 of the thesis. The earlier `zscore_long_short_heatmap.py` and
`zscore_longshort_heatmap.py` scripts produced these CSVs as a side effect
but require plotly (not in requirements.lock); this script does just the
CSV computation with stdlib + pandas + numpy and is therefore portable.

Outputs:
- results/thesis/zscore_long_by_month.csv
- results/thesis/zscore_short_by_month.csv
- results/thesis/zscore_longshort_by_month.csv

Usage:
    python scripts/zscore_time_by_horizon.py
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

# Assign legs by NYSE-breakpoint deciles of the production XGB score
test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

horizons = list(range(1, 13))


def _leg_zscore_row(grp, leg_mask, h):
    """Mean cross-sectional z-score of `leg_mask` stocks at horizon h within
    this monthly group. NaN if no stocks in the leg or zero cross-sectional std."""
    col = f'mom_{h}'
    mean = grp[col].mean()
    std = grp[col].std()
    if std == 0 or not np.isfinite(std):
        return np.nan
    if leg_mask.sum() == 0:
        return np.nan
    zs = (grp.loc[leg_mask, col] - mean) / std
    return zs.mean()


long_rows, short_rows, ls_rows = [], [], []
for date, grp in test.groupby('date'):
    long_mask = grp['leg'] == 'long'
    short_mask = grp['leg'] == 'short'

    long_row = {'date': date}
    short_row = {'date': date}
    ls_row = {'date': date}

    for h in horizons:
        z_long = _leg_zscore_row(grp, long_mask, h)
        z_short = _leg_zscore_row(grp, short_mask, h)

        long_row[f'mom_{h}'] = z_long
        short_row[f'mom_{h}'] = z_short
        # L-S spread: difference of the two leg-mean z-scores (NaN-safe)
        if np.isfinite(z_long) and np.isfinite(z_short):
            ls_row[f'mom_{h}'] = z_long - z_short
        else:
            ls_row[f'mom_{h}'] = np.nan

    long_rows.append(long_row)
    short_rows.append(short_row)
    ls_rows.append(ls_row)


def _save(rows, name):
    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    out_path = f'results/thesis/{name}.csv'
    os.makedirs('results/thesis', exist_ok=True)
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"Saved: {out_path}  ({len(df)} rows x {len(horizons)} horizons)")
    return df


long_df  = _save(long_rows,  'zscore_long_by_month')
short_df = _save(short_rows, 'zscore_short_by_month')
ls_df    = _save(ls_rows,    'zscore_longshort_by_month')

print()
print("Long leg z-scores (head):")
print(long_df.head())
print()
print("Short leg z-scores (head):")
print(short_df.head())
print()
print("L-S spread (head):")
print(ls_df.head())
