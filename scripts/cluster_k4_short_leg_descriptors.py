"""
cluster_k4_short_leg_descriptors.py
====================================
Computes short-leg per-cluster descriptors for the K=4 z-curve clustering,
parallel to the long-leg descriptors used in thesis §5.2.

Short-leg picks are reconstructed on-the-fly using the NYSE P10 breakpoint on
score_xgb (identical method to shap_horizon_long_short_final.py lines 42-49
and zscore_time_by_horizon.py lines 40-45).

Per-cluster output:
- short_z_centroid_mom_1..12  : cluster mean of the monthly short-leg z-score
                                 at each horizon (z relative to cross-section)
- short_pick_mom_1..12_mean   : cluster mean of the raw mean short-leg momentum
                                 at each horizon (decimal)
- short_basket_size           : mean number of short-leg stocks per month

Verification cross-checks:
1. Exactly 4 rows (one per cluster)
2. Sum of (n_months * short_basket_size) approx matches total short-leg
   stock-months counted independently across all 167 months.
3. Per-cluster correlation between short-leg z-curve and negated long-leg
   z-curve — we expect < 1.0 especially for cluster 3 (panic-dominant).
4. Cluster 3 month-1 short-leg z-value: should be elevated (~+0.30) per
   existing §5.2.2 panic asymmetry documentation.

Output:
    results/thesis/cluster_k4_short_leg_descriptors.csv

Usage:
    python scripts/cluster_k4_short_leg_descriptors.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_THESIS_DIR

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
LABELS_PATH = os.path.join(RESULTS_THESIS_DIR, 'zscore_l2_k4_labels.csv')
OUTPUT_PATH = os.path.join(RESULTS_THESIS_DIR, 'cluster_k4_short_leg_descriptors.csv')
LONG_DESCRIPTORS_PATH = os.path.join(RESULTS_THESIS_DIR, 'cluster_k4_descriptor_table.csv')

print("Loading artefacts...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()

labels = pd.read_csv(LABELS_PATH, parse_dates=['date'])
print(f"  Loaded {len(labels)} month labels, {labels['cluster'].nunique()} clusters")
print(f"  Cluster distribution: {labels['cluster'].value_counts().sort_index().to_dict()}")

long_desc = pd.read_csv(LONG_DESCRIPTORS_PATH)
print(f"  Loaded long-leg descriptors for {len(long_desc)} clusters")

# ---------------------------------------------------------------------------
# Assign short/long legs using NYSE P10 breakpoint on score_xgb
# Identical to zscore_time_by_horizon.py lines 40-45
# ---------------------------------------------------------------------------
print("\nAssigning legs (NYSE P10 breakpoint on score_xgb)...")
test['leg'] = 'middle'
skipped = 0
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        skipped += 1
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
print(f"  Skipped {skipped} months with <10 NYSE stocks")
print(f"  Short picks: {(test['leg'] == 'short').sum():,} stock-months")
print(f"  Long  picks: {(test['leg'] == 'long').sum():,} stock-months")

# ---------------------------------------------------------------------------
# Compute per-month short-leg z-curve
# z_h = (mean_picks(mom_h) - mean_all(mom_h)) / std_all(mom_h)
# ---------------------------------------------------------------------------
horizons = list(range(1, 13))

def _short_zcurve_row(grp, short_mask):
    """Monthly cross-sectional z-curve for short-leg picks at horizons 1..12."""
    zs = {}
    raws = {}
    for h in horizons:
        col = f'mom_{h}'
        vals = grp[col].dropna()
        mean_all = vals.mean()
        std_all = vals.std()
        short_vals = grp.loc[short_mask[short_mask].index, col].dropna()
        if std_all == 0 or not np.isfinite(std_all) or len(short_vals) == 0:
            zs[h] = np.nan
            raws[h] = np.nan
        else:
            zs[h] = ((short_vals - mean_all) / std_all).mean()
            raws[h] = short_vals.mean()
    return zs, raws


print("\nComputing per-month short-leg z-curves...")
monthly_rows = []
for date, grp in test.groupby('date'):
    short_mask = grp['leg'] == 'short'
    n_short = short_mask.sum()
    if n_short == 0:
        continue
    zs, raws = _short_zcurve_row(grp, short_mask)
    row = {'date': date, 'n_short': n_short}
    for h in horizons:
        row[f'short_z_mom_{h}'] = zs[h]
        row[f'short_pick_mom_{h}'] = raws[h]
    monthly_rows.append(row)

monthly_df = pd.DataFrame(monthly_rows)
monthly_df = monthly_df.merge(labels, on='date', how='left')
print(f"  Computed z-curves for {len(monthly_df)} months")

# ---------------------------------------------------------------------------
# Independent verification: total short-leg stock-months across all 167 months
# ---------------------------------------------------------------------------
total_short_stock_months_independent = (test['leg'] == 'short').sum()
print(f"\nIndependent total short-leg stock-months (all 167 months): "
      f"{total_short_stock_months_independent:,}")

# ---------------------------------------------------------------------------
# Aggregate per cluster
# ---------------------------------------------------------------------------
print("\nAggregating per cluster...")
cluster_rows = []
for k in sorted(labels['cluster'].unique()):
    sub = monthly_df[monthly_df['cluster'] == k]
    n_months = len(sub)

    short_basket_size = sub['n_short'].mean()

    short_z_centroid = {}
    short_pick_mean = {}
    for h in horizons:
        short_z_centroid[h] = sub[f'short_z_mom_{h}'].mean()
        short_pick_mean[h] = sub[f'short_pick_mom_{h}'].mean()

    row = {'cluster': k, 'n_months': n_months}
    for h in horizons:
        row[f'short_z_centroid_mom_{h}'] = short_z_centroid[h]
    for h in horizons:
        row[f'short_pick_mom_{h}_mean'] = short_pick_mean[h]
    row['short_basket_size'] = short_basket_size
    cluster_rows.append(row)

result_df = pd.DataFrame(cluster_rows)

# ---------------------------------------------------------------------------
# Verification cross-checks
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICATION CROSS-CHECKS")
print("=" * 60)

all_pass = True
concerns = []

# Check 1: exactly 4 rows
n_rows = len(result_df)
if n_rows == 4:
    print(f"[PASS] Row count: {n_rows} (expected 4)")
else:
    print(f"[FAIL] Row count: {n_rows} (expected 4)")
    all_pass = False

# Check 2: total short-leg stock-months consistency
computed_total = (result_df['n_months'] * result_df['short_basket_size']).sum()
print(f"\n[CHECK 2] Short-leg stock-months consistency:")
print(f"  Sum(n_months * short_basket_size) = {computed_total:.1f}")
print(f"  Independent count (all 167 months) = {total_short_stock_months_independent:,}")
rel_diff = abs(computed_total - total_short_stock_months_independent) / total_short_stock_months_independent
if rel_diff < 0.02:  # within 2%
    print(f"  Relative difference: {rel_diff:.4f} -> [PASS]")
else:
    print(f"  Relative difference: {rel_diff:.4f} -> [FAIL]")
    all_pass = False

# Check 3: correlation of short-leg z-curve vs negated long-leg z-curve
print(f"\n[CHECK 3] Per-cluster correlation: short-leg z vs -(long-leg z)")
long_z_cols = [f'z_centroid_mom_{h}' for h in horizons]
short_z_cols = [f'short_z_centroid_mom_{h}' for h in horizons]

for _, row in result_df.iterrows():
    k = int(row['cluster'])
    long_row = long_desc[long_desc['cluster'] == k].iloc[0]
    long_zvec = np.array([long_row[f'z_centroid_mom_{h}'] for h in horizons])
    short_zvec = np.array([row[f'short_z_centroid_mom_{h}'] for h in horizons])
    # Correlation between short-leg z and negated long-leg z
    corr = np.corrcoef(short_zvec, -long_zvec)[0, 1]
    asymmetry_flag = "[ASYMMETRIC]" if corr < 0.95 else "[SYMMETRIC]"
    print(f"  Cluster {k}: corr(short_z, -long_z) = {corr:+.4f}  {asymmetry_flag}")
    if k == 3 and corr >= 0.95:
        concerns.append(f"Cluster 3: short-leg z nearly mirrors long-leg (corr={corr:.4f}); "
                        "asymmetry expected from §5.2.2.")

# Check 4: Cluster 3 month-1 short-leg z value
k3_row = result_df[result_df['cluster'] == 3].iloc[0]
k3_mom1_z = k3_row['short_z_centroid_mom_1']
print(f"\n[CHECK 4] Cluster 3 (panic-dominant) short-leg z-curve month-1 value:")
print(f"  short_z_centroid_mom_1 = {k3_mom1_z:+.4f}")
long_k3 = long_desc[long_desc['cluster'] == 3].iloc[0]
long_k3_mom1 = long_k3['z_centroid_mom_1']
print(f"  long_z_centroid_mom_1  = {long_k3_mom1:+.4f}")
print(f"  Negated long           = {-long_k3_mom1:+.4f}")
asymm = k3_mom1_z - (-long_k3_mom1)
print(f"  Asymmetry (short_z - (-long_z)) at mom_1 = {asymm:+.4f}")
# §5.2.2 says panic aggregate ~+0.30 elevated month-1
if k3_mom1_z > 0.0:
    print(f"  [PASS] Cluster 3 month-1 short-leg z is positive ({k3_mom1_z:+.4f}), "
          f"consistent with §5.2.2 elevated asymmetry")
else:
    print(f"  [CONCERN] Cluster 3 month-1 short-leg z is not positive ({k3_mom1_z:+.4f}); "
          f"check §5.2.2 claim")
    concerns.append(f"Cluster 3 month-1 short-leg z = {k3_mom1_z:.4f} (not positive)")

# Per-cluster summary
print(f"\n[SUMMARY] Per-cluster short-leg z-curves (selected horizons):")
print(f"{'Cluster':>8s} {'n_months':>8s} {'basket':>8s} {'z_mom_1':>8s} "
      f"{'z_mom_6':>8s} {'z_mom_12':>9s}")
for _, row in result_df.iterrows():
    print(f"  {int(row['cluster']):>6d} {int(row['n_months']):>8d} "
          f"{row['short_basket_size']:>8.1f} "
          f"{row['short_z_centroid_mom_1']:>+8.4f} "
          f"{row['short_z_centroid_mom_6']:>+8.4f} "
          f"{row['short_z_centroid_mom_12']:>+9.4f}")

print(f"\n[SUMMARY] Raw short-leg pick momentum (decimal):")
print(f"{'Cluster':>8s} {'mom_1':>8s} {'mom_6':>8s} {'mom_12':>9s}")
for _, row in result_df.iterrows():
    print(f"  {int(row['cluster']):>6d} "
          f"{row['short_pick_mom_1_mean']:>+8.4f} "
          f"{row['short_pick_mom_6_mean']:>+8.4f} "
          f"{row['short_pick_mom_12_mean']:>+9.4f}")

print("\n" + "=" * 60)
if all_pass and not concerns:
    print("OVERALL: ALL CHECKS PASSED")
elif all_pass:
    print(f"OVERALL: PASS with {len(concerns)} concern(s):")
    for c in concerns:
        print(f"  - {c}")
else:
    print("OVERALL: SOME CHECKS FAILED -- review before committing")
print("=" * 60)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_THESIS_DIR, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False, float_format='%.8f')
print(f"\nWrote {OUTPUT_PATH}")
print(f"  Columns: {list(result_df.columns)}")
print(f"  Shape:   {result_df.shape}")
