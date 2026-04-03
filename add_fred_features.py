"""
add_fred_features.py
====================
Adds TERM, CPS, HY_OAS to existing data/panel.parquet (which was built from WRDS).
No WRDS credentials needed -- only pulls from FRED.

Then standardizes using train-only statistics (pre-2011), matching import_wrds.py convention.
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web

panel = pd.read_parquet('data/panel.parquet')
print(f"Loaded panel: {len(panel)} rows, columns: {sorted(panel.columns.tolist())}")

# Check if features already exist
if 'TERM_z' in panel.columns and 'CPS_z' in panel.columns and 'HY_OAS_z' in panel.columns:
    print("Features already exist in panel -- exiting.")
    exit(0)

# --- Pull FRED data ---
print("Pulling FRED data...")

# TERM: 10Y - 2Y Treasury yield spread
t10y = web.DataReader('DGS10', 'fred', start='1970-01-01', end='2025-12-31')
t2y  = web.DataReader('DGS2',  'fred', start='1970-01-01', end='2025-12-31')
term = (t10y['DGS10'] - t2y['DGS2']).resample('ME').mean().to_frame('TERM')
term.index = term.index.to_period('M')
term = term.reset_index().rename(columns={'DATE': 'year_month'})
print(f"  TERM: {len(term)} months, {term['TERM'].notna().sum()} non-null")

# CPS: commercial paper spread (3-mo financial CP - 3-mo T-bill)
cp = web.DataReader('DCPF3M', 'fred', start='1970-01-01', end='2025-12-31')
tb = web.DataReader('DTB3',   'fred', start='1970-01-01', end='2025-12-31')
cps = (cp['DCPF3M'] - tb['DTB3']).resample('ME').mean().to_frame('CPS')
cps.index = cps.index.to_period('M')
cps = cps.reset_index().rename(columns={'DATE': 'year_month'})
print(f"  CPS:  {len(cps)} months, {cps['CPS'].notna().sum()} non-null")

# HY_OAS: ICE BofA high-yield OAS
hy = web.DataReader('BAMLH0A0HYM2', 'fred', start='1970-01-01', end='2025-12-31')
hy = hy.resample('ME').mean()
hy.columns = ['HY_OAS']
hy.index = hy.index.to_period('M')
hy = hy.reset_index().rename(columns={'DATE': 'year_month'})
print(f"  HY_OAS: {len(hy)} months, {hy['HY_OAS'].notna().sum()} non-null")

# --- Merge onto panel ---
# Drop old columns if they exist (raw and _z versions)
for col in ['TERM', 'CPS', 'HY_OAS', 'TERM_z', 'CPS_z', 'HY_OAS_z']:
    if col in panel.columns:
        panel.drop(columns=[col], inplace=True)

panel = (panel
    .merge(term[['year_month', 'TERM']], on='year_month', how='left')
    .merge(cps[['year_month', 'CPS']],   on='year_month', how='left')
    .merge(hy[['year_month', 'HY_OAS']], on='year_month', how='left')
)

# --- Winsorize using train-only bounds ---
train_mask = panel['date'] < '2011-01-01'
for col in ['TERM', 'CPS', 'HY_OAS']:
    vals = panel[col].astype(float)
    lo = vals[train_mask].dropna().quantile(0.01)
    hi = vals[train_mask].dropna().quantile(0.99)
    panel[col] = vals.clip(lo, hi)

# --- Standardize using train-only stats ---
for col in ['TERM', 'CPS', 'HY_OAS']:
    train_vals = panel.loc[train_mask, col].dropna()
    col_mean = train_vals.mean()
    col_std  = train_vals.std()
    panel[f'{col}_z'] = (panel[col] - col_mean) / col_std
    print(f"  {col}_z: train mean={col_mean:.4f}, std={col_std:.4f}, "
          f"non-null total={panel[f'{col}_z'].notna().sum()}")

# --- Save ---
panel.to_parquet('data/panel.parquet', index=False)
print(f"\nSaved data/panel.parquet ({len(panel)} rows, {len(panel.columns)} cols)")
print(f"Columns: {sorted(panel.columns.tolist())}")

# Verify new features exist in the right date range
hmm_cols = ['VOL_z', 'TERM_z', 'CPS_z', 'HY_OAS_z']
valid = panel.dropna(subset=hmm_cols)
print(f"\nRows with all 4 HMM features non-null: {len(valid)}")
print(f"  Date range: {valid['date'].min().date()} -> {valid['date'].max().date()}")
