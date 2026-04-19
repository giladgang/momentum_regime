"""
zscore_diagnostic.py
====================
Verify z-score computation and check if cross-sectional dispersion
explains why long/short z-scores converge at longer horizons in panic.
"""

import numpy as np
import pandas as pd
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)

test = art['test'].copy()
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

horizons = list(range(1, 13))
panic = test[test['regime'] == 'Panic']

# ══════════════════════════════════════════════════════════════════════════════
# 1. Cross-sectional std by horizon in panic
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. CROSS-SECTIONAL STD BY HORIZON (PANIC MONTHS)")
print("="*60)

print(f"  {'Horizon':>8s} {'Mean CS std':>12s} {'Mean CS mean':>13s} {'Long raw':>10s} {'Short raw':>10s} {'Raw diff':>10s} {'Z diff':>8s}")
print("  " + "-"*75)

for h in horizons:
    col = f'mom_{h}'
    monthly_stds = []
    monthly_means = []
    monthly_long_raw = []
    monthly_short_raw = []
    monthly_z_long = []
    monthly_z_short = []

    for date, grp in panic.groupby('date'):
        cs_mean = grp[col].mean()
        cs_std = grp[col].std()
        if cs_std == 0:
            continue
        monthly_stds.append(cs_std)
        monthly_means.append(cs_mean)

        long_vals = grp.loc[grp['leg'] == 'long', col]
        short_vals = grp.loc[grp['leg'] == 'short', col]

        if len(long_vals) > 0:
            monthly_long_raw.append(long_vals.mean())
            monthly_z_long.append(((long_vals - cs_mean) / cs_std).mean())
        if len(short_vals) > 0:
            monthly_short_raw.append(short_vals.mean())
            monthly_z_short.append(((short_vals - cs_mean) / cs_std).mean())

    avg_std = np.mean(monthly_stds)
    avg_mean = np.mean(monthly_means)
    avg_long = np.mean(monthly_long_raw)
    avg_short = np.mean(monthly_short_raw)
    avg_z_long = np.mean(monthly_z_long)
    avg_z_short = np.mean(monthly_z_short)

    print(f"  mom_{h:>2d}   {avg_std:>11.4f}  {avg_mean:>12.4f}  {avg_long:>+9.4f}  {avg_short:>+9.4f}  {avg_long-avg_short:>+9.4f}  {avg_z_long-avg_z_short:>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Same for calm
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. CROSS-SECTIONAL STD BY HORIZON (CALM MONTHS)")
print("="*60)

calm = test[test['regime'] == 'Calm']

print(f"  {'Horizon':>8s} {'Mean CS std':>12s} {'Mean CS mean':>13s} {'Long raw':>10s} {'Short raw':>10s} {'Raw diff':>10s} {'Z diff':>8s}")
print("  " + "-"*75)

for h in horizons:
    col = f'mom_{h}'
    monthly_stds = []
    monthly_means = []
    monthly_long_raw = []
    monthly_short_raw = []
    monthly_z_long = []
    monthly_z_short = []

    for date, grp in calm.groupby('date'):
        cs_mean = grp[col].mean()
        cs_std = grp[col].std()
        if cs_std == 0:
            continue
        monthly_stds.append(cs_std)
        monthly_means.append(cs_mean)

        long_vals = grp.loc[grp['leg'] == 'long', col]
        short_vals = grp.loc[grp['leg'] == 'short', col]

        if len(long_vals) > 0:
            monthly_long_raw.append(long_vals.mean())
            monthly_z_long.append(((long_vals - cs_mean) / cs_std).mean())
        if len(short_vals) > 0:
            monthly_short_raw.append(short_vals.mean())
            monthly_z_short.append(((short_vals - cs_mean) / cs_std).mean())

    avg_std = np.mean(monthly_stds)
    avg_mean = np.mean(monthly_means)
    avg_long = np.mean(monthly_long_raw)
    avg_short = np.mean(monthly_short_raw)
    avg_z_long = np.mean(monthly_z_long)
    avg_z_short = np.mean(monthly_z_short)

    print(f"  mom_{h:>2d}   {avg_std:>11.4f}  {avg_mean:>12.4f}  {avg_long:>+9.4f}  {avg_short:>+9.4f}  {avg_long-avg_short:>+9.4f}  {avg_z_long-avg_z_short:>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Manual verification: pick one panic month and show everything
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. MANUAL VERIFICATION: ONE PANIC MONTH")
print("="*60)

panic_dates = panic['date'].unique()
sample_date = sorted(panic_dates)[len(panic_dates)//2]  # middle panic month
grp = test[test['date'] == sample_date]

print(f"  Date: {sample_date}")
print(f"  Total stocks: {len(grp)}")
print(f"  Long: {(grp['leg']=='long').sum()}, Short: {(grp['leg']=='short').sum()}")
print(f"  pi_filter: {grp['pi_month'].iloc[0]:.3f}")

print(f"\n  {'Horizon':>8s} {'CS mean':>10s} {'CS std':>10s} {'Long mean':>10s} {'Short mean':>11s} {'Long z':>8s} {'Short z':>8s}")
print("  " + "-"*68)

for h in horizons:
    col = f'mom_{h}'
    cs_mean = grp[col].mean()
    cs_std = grp[col].std()
    long_mean = grp.loc[grp['leg'] == 'long', col].mean()
    short_mean = grp.loc[grp['leg'] == 'short', col].mean()
    long_z = (long_mean - cs_mean) / cs_std if cs_std > 0 else 0
    short_z = (short_mean - cs_mean) / cs_std if cs_std > 0 else 0
    print(f"  mom_{h:>2d}   {cs_mean:>+9.4f}  {cs_std:>9.4f}  {long_mean:>+9.4f}  {short_mean:>+10.4f}  {long_z:>+7.3f}  {short_z:>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Score correlation with each momentum horizon in panic
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4. CORRELATION: score_xgb vs mom_h IN PANIC")
print("="*60)

from scipy.stats import spearmanr

print(f"  {'Horizon':>8s} {'Spearman r':>11s}")
print("  " + "-"*22)
for h in horizons:
    col = f'mom_{h}'
    monthly_corrs = []
    for date, grp in panic.groupby('date'):
        valid = grp[['score_xgb', col]].dropna()
        if len(valid) < 30:
            continue
        rho, _ = spearmanr(valid['score_xgb'], valid[col])
        monthly_corrs.append(rho)
    print(f"  mom_{h:>2d}   {np.mean(monthly_corrs):>+10.4f}")

# Also pi_filter
print(f"  pi_filt  {'N/A (same for all stocks)':>20s}")

print("\nDone.")
