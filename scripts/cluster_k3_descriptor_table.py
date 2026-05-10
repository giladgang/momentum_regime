"""
cluster_k3_descriptor_table.py
-------------------------------
Compute the consolidated per-cluster descriptor table for the K=3 aggregate
L2 z-score clusters.  Feeds §5.2 of the master's thesis.

Inputs
------
- results/thesis/zscore_l2_k3_labels.csv      167 rows (date, cluster)
- results/thesis/zscore_l2_k3_centroids.csv   existing centroids for cross-check
- results/thesis/zscore_long_by_month.csv      12-d z-curves
- results/thesis/picked_stock_feature_panel.csv 14 predictor features per month
- results/thesis/rule_path_labels.csv           long-leg picks (date, permno)
- artefacts/cs_artefacts_data.pkl              art['test'] cross-section panel
- results/thesis/fundamentals_returns.pkl      baseline_mom_pi -> returns (XGB)

Outputs
-------
- results/thesis/cluster_k3_descriptor_table.csv   3-row master table
- results/thesis/cluster_k3_member_dates.csv        167-row long-form dates
- results/thesis/cluster_k3_representative_dates.csv 15-row closest-to-centroid dates
- results/thesis/cluster_k3_year_frequency.csv       year x cluster pivot

Usage
-----
    python scripts/cluster_k3_descriptor_table.py
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_helpers import block_bootstrap_sharpe  # noqa: E402
from cluster_feature_search_helpers import picked_stock_fingerprint  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELS_CSV     = ROOT / "results/thesis/zscore_l2_k3_labels.csv"
CENTROIDS_CSV  = ROOT / "results/thesis/zscore_l2_k3_centroids.csv"
ZCURVE_CSV     = ROOT / "results/thesis/zscore_long_by_month.csv"
FEATURE_CSV    = ROOT / "results/thesis/picked_stock_feature_panel.csv"
PICKS_CSV      = ROOT / "results/thesis/rule_path_labels.csv"
CS_PKL         = ROOT / "artefacts/cs_artefacts_data.pkl"
RETURNS_PKL    = ROOT / "results/thesis/fundamentals_returns.pkl"

RES_DIR = ROOT / "results/thesis"

OUT_MASTER    = RES_DIR / "cluster_k3_descriptor_table.csv"
OUT_MEMBERS   = RES_DIR / "cluster_k3_member_dates.csv"
OUT_REPS      = RES_DIR / "cluster_k3_representative_dates.csv"
OUT_YEARFREQ  = RES_DIR / "cluster_k3_year_frequency.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K         = 3
HORIZONS  = list(range(1, 13))
MOM_COLS  = [f"mom_{h}" for h in HORIZONS]

BOOT_SZ   = 6
N_BOOT    = 5000
BOOT_SEED = 42

N_REPS_PER_CLUSTER = 5   # representative dates per cluster

# ---------------------------------------------------------------------------
# 1. Load all inputs
# ---------------------------------------------------------------------------
print("=== Loading inputs ===", flush=True)

labels_df = pd.read_csv(LABELS_CSV, parse_dates=["date"])
print(f"  Labels:  {labels_df.shape}  columns={labels_df.columns.tolist()}", flush=True)
assert labels_df.shape[0] == 167, f"Expected 167 label rows, got {labels_df.shape[0]}"

centroids_existing = pd.read_csv(CENTROIDS_CSV)
print(f"  Centroids CSV: {centroids_existing.shape}", flush=True)

zcurves_df = pd.read_csv(ZCURVE_CSV, parse_dates=["date"])
print(f"  Z-curves: {zcurves_df.shape}", flush=True)

feature_panel = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
print(f"  Feature panel: {feature_panel.shape}", flush=True)

picks_df = pd.read_csv(PICKS_CSV, parse_dates=["date"])
print(f"  Rule-path labels: {picks_df.shape}", flush=True)

with open(CS_PKL, "rb") as fh:
    art = pickle.load(fh)
test_df = art["test"].copy()
test_df["date"] = pd.to_datetime(test_df["date"])
print(f"  CS test panel: {test_df.shape}", flush=True)

with open(RETURNS_PKL, "rb") as fh:
    ret_data = pickle.load(fh)
m2_returns = ret_data["baseline_mom_pi"]["returns"]
m2_returns.index = pd.to_datetime(m2_returns.index)
print(f"  XGB returns: {len(m2_returns)} months", flush=True)

# ---------------------------------------------------------------------------
# 2. Merge labels with z-curves and feature panel
# ---------------------------------------------------------------------------
print("\n=== Building merged panel ===", flush=True)

# Z-curve indexed by date
z_indexed = zcurves_df.set_index("date")[MOM_COLS]

# Feature panel - keep only the 14 predictor columns + date (drop 'cluster' from FP)
PREDICTOR_COLS = [
    "pi_panic", "cs_mom_short", "cs_mom_mid", "cs_mom_long", "mom_overall",
    "cs_disp_short", "cs_disp_mid", "cs_disp_long", "pi_panic_freq_6mo",
    "pi_panic_freq_12mo", "past_sharpe_12mo", "cs_skew_short", "cs_skew_mid",
    "cs_skew_long",
]
feat_indexed = feature_panel[["date"] + PREDICTOR_COLS].set_index("date")

# Merge labels with feature panel and z-curves
labels_indexed = labels_df.set_index("date")["cluster"]
merged = pd.DataFrame({"cluster": labels_indexed})
merged = merged.join(feat_indexed, how="left")
merged = merged.join(z_indexed, how="left")
merged = merged.join(m2_returns.rename("m2_ret"), how="left")
merged.index.name = "date"
merged = merged.sort_index()

print(f"  Merged panel shape: {merged.shape}", flush=True)
print(f"  NaN in m2_ret: {merged['m2_ret'].isna().sum()}", flush=True)

# ---------------------------------------------------------------------------
# 3. Build picked-stock fingerprint for the long leg
# ---------------------------------------------------------------------------
print("\n=== Building picked-stock fingerprint ===", flush=True)

mom_panel = test_df[["date", "permno"] + MOM_COLS].copy()
fingerprint = picked_stock_fingerprint(
    picks_df[["date", "permno"]],
    mom_panel,
    MOM_COLS,
)
print(f"  Fingerprint shape: {fingerprint.shape}", flush=True)
print(f"  Fingerprint columns: {fingerprint.columns.tolist()}", flush=True)

# ---------------------------------------------------------------------------
# 4. Helper: max consecutive run of months in a cluster
# ---------------------------------------------------------------------------
def max_consecutive_run(date_list):
    """Longest run of consecutive calendar months in date_list."""
    if len(date_list) == 0:
        return 0
    sorted_dates = sorted(date_list)
    periods = [d.to_period("M") for d in sorted_dates]
    max_run = 1
    current_run = 1
    for i in range(1, len(periods)):
        diff = (periods[i] - periods[i - 1]).n
        if diff == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


# ---------------------------------------------------------------------------
# 5. Compute per-cluster statistics
# ---------------------------------------------------------------------------
print("\n=== Computing per-cluster statistics ===", flush=True)

descriptor_rows = []

for cl in range(K):
    cl_mask = merged["cluster"] == cl
    cl_df   = merged[cl_mask]
    cl_dates = cl_df.index.tolist()
    n_cl = len(cl_df)

    print(f"\n  Cluster {cl}: n={n_cl}", flush=True)

    # --- Identity ---
    # n_months already known

    # --- Regime context ---
    pi_panic_mean         = float(cl_df["pi_panic"].mean())
    pi_panic_freq_6mo_mean  = float(cl_df["pi_panic_freq_6mo"].dropna().mean())
    pi_panic_freq_12mo_mean = float(cl_df["pi_panic_freq_12mo"].dropna().mean())

    # --- Strategy outcomes ---
    m2_rs = cl_df["m2_ret"].dropna().values
    bbs = block_bootstrap_sharpe(m2_rs, block_size=BOOT_SZ, n_reps=N_BOOT, seed=BOOT_SEED)
    sharpe    = bbs["sharpe_point"]
    sharpe_lo = bbs["sharpe_lo95"]
    sharpe_hi = bbs["sharpe_hi95"]

    m2_all = cl_df["m2_ret"].values  # use all cluster months for hit rate / stats
    hit_rate     = float(np.mean(m2_all > 0))
    mean_return  = float(np.mean(m2_all))
    std_return   = float(np.std(m2_all, ddof=1))
    worst_return = float(np.min(m2_all))
    best_return  = float(np.max(m2_all))
    worst_date   = cl_df["m2_ret"].idxmin().strftime("%Y-%m-%d")
    best_date    = cl_df["m2_ret"].idxmax().strftime("%Y-%m-%d")

    # --- Cross-section context (from feature panel) ---
    cs_mom_short_mean = float(cl_df["cs_mom_short"].mean())
    cs_mom_mid_mean   = float(cl_df["cs_mom_mid"].mean())
    cs_mom_long_mean  = float(cl_df["cs_mom_long"].mean())
    mom_overall_mean  = float(cl_df["mom_overall"].mean())

    cs_disp_short_mean = float(cl_df["cs_disp_short"].mean())
    cs_disp_mid_mean   = float(cl_df["cs_disp_mid"].mean())
    cs_disp_long_mean  = float(cl_df["cs_disp_long"].mean())

    cs_skew_short_mean = float(cl_df["cs_skew_short"].mean())
    cs_skew_mid_mean   = float(cl_df["cs_skew_mid"].mean())
    cs_skew_long_mean  = float(cl_df["cs_skew_long"].mean())

    # --- Picked-stock fingerprint ---
    fp_cl = fingerprint.reindex(cl_dates).dropna(how="all")
    pick_mom_means = {
        f"pick_mom_{h}_mean": float(fp_cl[f"pick_mom_{h}"].mean()) if len(fp_cl) > 0 else float("nan")
        for h in HORIZONS
    }
    pick_disp_short_mean = float(fp_cl["pick_disp_short"].mean()) if len(fp_cl) > 0 else float("nan")
    pick_disp_mid_mean   = float(fp_cl["pick_disp_mid"].mean())   if len(fp_cl) > 0 else float("nan")
    pick_disp_long_mean  = float(fp_cl["pick_disp_long"].mean())  if len(fp_cl) > 0 else float("nan")

    # --- Z-centroid (12 values) ---
    z_cl = cl_df[MOM_COLS]
    z_centroid = z_cl.mean(axis=0)
    z_centroid_dict = {f"z_centroid_{c}": float(z_centroid[c]) for c in MOM_COLS}
    z_centroid_mean = float(z_centroid.mean())

    # --- Past strategy performance ---
    past_sharpe_12mo_mean = float(cl_df["past_sharpe_12mo"].dropna().mean())

    # --- Time / context ---
    min_date = cl_df.index.min().strftime("%Y-%m-%d")
    max_date = cl_df.index.max().strftime("%Y-%m-%d")
    max_consec = max_consecutive_run(cl_dates)

    row = {
        # Identity
        "cluster": cl,
        "n_months": n_cl,
        # Regime context
        "pi_panic_mean":            pi_panic_mean,
        "pi_panic_freq_6mo_mean":   pi_panic_freq_6mo_mean,
        "pi_panic_freq_12mo_mean":  pi_panic_freq_12mo_mean,
        # Strategy outcomes
        "sharpe":            sharpe,
        "sharpe_lo95":       sharpe_lo,
        "sharpe_hi95":       sharpe_hi,
        "hit_rate":          hit_rate,
        "mean_return":       mean_return,
        "std_return":        std_return,
        "worst_month_return": worst_return,
        "best_month_return":  best_return,
        "worst_month_date":   worst_date,
        "best_month_date":    best_date,
        # Cross-section context
        "cs_mom_short_mean": cs_mom_short_mean,
        "cs_mom_mid_mean":   cs_mom_mid_mean,
        "cs_mom_long_mean":  cs_mom_long_mean,
        "mom_overall_mean":  mom_overall_mean,
        "cs_disp_short_mean": cs_disp_short_mean,
        "cs_disp_mid_mean":   cs_disp_mid_mean,
        "cs_disp_long_mean":  cs_disp_long_mean,
        "cs_skew_short_mean": cs_skew_short_mean,
        "cs_skew_mid_mean":   cs_skew_mid_mean,
        "cs_skew_long_mean":  cs_skew_long_mean,
        # Picked-stock fingerprint
        **pick_mom_means,
        "pick_disp_short_mean": pick_disp_short_mean,
        "pick_disp_mid_mean":   pick_disp_mid_mean,
        "pick_disp_long_mean":  pick_disp_long_mean,
        # Z-centroid
        **z_centroid_dict,
        "z_centroid_mean": z_centroid_mean,
        # Past strategy performance
        "past_sharpe_12mo_mean": past_sharpe_12mo_mean,
        # Time / context
        "min_date": min_date,
        "max_date": max_date,
        "max_consecutive_months": max_consec,
    }
    descriptor_rows.append(row)

    print(f"    pi_panic_mean={pi_panic_mean:.4f}  sharpe={sharpe:.4f} "
          f"[{sharpe_lo:.4f}, {sharpe_hi:.4f}]", flush=True)
    print(f"    hit_rate={hit_rate:.3f}  mean_ret={mean_return:.4f}  "
          f"worst={worst_return:.4f}  best={best_return:.4f}", flush=True)
    print(f"    z_centroid_mean={z_centroid_mean:.6f}  max_consec={max_consec}", flush=True)

descriptor_df = pd.DataFrame(descriptor_rows)

print(f"\n  Descriptor table shape: {descriptor_df.shape}", flush=True)

# ---------------------------------------------------------------------------
# 6. Member dates CSV
# ---------------------------------------------------------------------------
print("\n=== Building member dates CSV ===", flush=True)

member_rows = []
for cl in range(K):
    cl_dates = labels_df[labels_df["cluster"] == cl]["date"]
    for d in cl_dates:
        member_rows.append({"cluster": cl, "date": d.strftime("%Y-%m-%d")})

member_df = pd.DataFrame(member_rows).sort_values(["cluster", "date"]).reset_index(drop=True)
print(f"  Member dates shape: {member_df.shape}", flush=True)

# ---------------------------------------------------------------------------
# 7. Representative dates (closest to centroid in z-curve space)
# ---------------------------------------------------------------------------
print("\n=== Building representative dates CSV ===", flush=True)

rep_rows = []
for cl in range(K):
    cl_mask  = merged["cluster"] == cl
    cl_z     = merged.loc[cl_mask, MOM_COLS].values
    cl_dates_arr = merged.index[cl_mask]

    centroid_vals = descriptor_df.loc[
        descriptor_df["cluster"] == cl,
        [f"z_centroid_{c}" for c in MOM_COLS]
    ].values[0]

    # L2 distance from each month's z-curve to the centroid
    dists = np.linalg.norm(cl_z - centroid_vals, axis=1)

    # Sort by distance, take top N_REPS_PER_CLUSTER
    top_idx = np.argsort(dists)[:N_REPS_PER_CLUSTER]
    for rank, idx in enumerate(top_idx, start=1):
        rep_rows.append({
            "cluster": cl,
            "rank": rank,
            "date": pd.Timestamp(cl_dates_arr[idx]).strftime("%Y-%m-%d"),
            "l2_distance_to_centroid": float(dists[idx]),
        })

rep_df = pd.DataFrame(rep_rows)
print(f"  Representative dates shape: {rep_df.shape}", flush=True)

# ---------------------------------------------------------------------------
# 8. Year-by-year frequency pivot
# ---------------------------------------------------------------------------
print("\n=== Building year-frequency pivot ===", flush=True)

labels_df["year"] = labels_df["date"].dt.year
year_freq = (
    labels_df.groupby(["year", "cluster"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={cl: f"cluster_{cl}" for cl in range(K)})
)
# Ensure all 3 cluster columns exist
for cl in range(K):
    col = f"cluster_{cl}"
    if col not in year_freq.columns:
        year_freq[col] = 0
year_freq = year_freq[[f"cluster_{cl}" for cl in range(K)]]
print(f"  Year-frequency shape: {year_freq.shape}", flush=True)

# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------
print("\n=== Saving outputs ===", flush=True)

descriptor_df.to_csv(OUT_MASTER, index=False)
print(f"  Saved: {OUT_MASTER}", flush=True)

member_df.to_csv(OUT_MEMBERS, index=False)
print(f"  Saved: {OUT_MEMBERS}", flush=True)

rep_df.to_csv(OUT_REPS, index=False)
print(f"  Saved: {OUT_REPS}", flush=True)

year_freq.to_csv(OUT_YEARFREQ)
print(f"  Saved: {OUT_YEARFREQ}", flush=True)

# ---------------------------------------------------------------------------
# 10. VERIFICATION
# ---------------------------------------------------------------------------
print("\n\n=== VERIFICATION ===", flush=True)
all_pass = True


def check(condition, label, detail):
    global all_pass
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
    print(f"  [{status}] {label}: {detail}", flush=True)
    return condition


# V1: Total months = 167
total_months = int(descriptor_df["n_months"].sum())
check(total_months == 167,
      "Total months = 167",
      f"n_months sum = {total_months}, expected 167")

# V2: Each date in exactly one cluster
date_counts = labels_df.groupby("date").size()
dup_count = int((date_counts > 1).sum())
check(dup_count == 0,
      "Each date in exactly one cluster",
      f"duplicate count = {dup_count}, expected 0")

# V3: Cluster labels are {0, 1, 2}
cluster_set = set(labels_df["cluster"].unique())
check(cluster_set == {0, 1, 2},
      "Cluster labels are {0, 1, 2}",
      f"set(clusters) = {cluster_set}, expected {{0,1,2}}")

# V4: pi_panic_mean matches existing centroids CSV
# The centroids CSV stores pi_bar = round(actual, 4), so compare at 4-decimal precision.
for cl in range(K):
    fresh = float(descriptor_df.loc[descriptor_df["cluster"] == cl, "pi_panic_mean"].values[0])
    existing = float(centroids_existing.loc[centroids_existing["cluster"] == cl, "pi_bar"].values[0])
    diff = abs(round(fresh, 4) - existing)
    check(diff <= 1e-6,
          f"pi_panic_mean matches centroids CSV (cluster {cl})",
          f"fresh={fresh:.6f}, round(fresh,4)={round(fresh,4)}, csv={existing}, diff={diff:.2e}, threshold 1e-6")

# V5: Sharpe matches existing centroids CSV
# The centroids CSV stores sharpe = round(actual, 4), so compare at 4-decimal precision.
for cl in range(K):
    fresh = float(descriptor_df.loc[descriptor_df["cluster"] == cl, "sharpe"].values[0])
    existing = float(centroids_existing.loc[centroids_existing["cluster"] == cl, "sharpe"].values[0])
    diff = abs(round(fresh, 4) - existing)
    check(diff <= 1e-6,
          f"Sharpe matches centroids CSV (cluster {cl})",
          f"fresh={fresh:.8f}, round(fresh,4)={round(fresh,4)}, csv={existing}, diff={diff:.2e}, threshold 1e-6")

# V6: Sharpe CI matches existing centroids CSV
# The centroids CSV stores sharpe_lo95/hi95 = round(actual, 4).
for cl in range(K):
    for key, csvkey in [("sharpe_lo95", "sharpe_lo95"), ("sharpe_hi95", "sharpe_hi95")]:
        fresh = float(descriptor_df.loc[descriptor_df["cluster"] == cl, key].values[0])
        existing = float(centroids_existing.loc[centroids_existing["cluster"] == cl, csvkey].values[0])
        diff = abs(round(fresh, 4) - existing)
        check(diff <= 1e-6,
              f"Sharpe {key} matches centroids CSV (cluster {cl})",
              f"fresh={fresh:.8f}, round(fresh,4)={round(fresh,4)}, csv={existing}, diff={diff:.2e}, threshold 1e-6")

# V7: Z-centroid matches existing centroids CSV
for cl in range(K):
    centroid_fresh = np.array([
        descriptor_df.loc[descriptor_df["cluster"] == cl, f"z_centroid_{c}"].values[0]
        for c in MOM_COLS
    ])
    centroid_existing = centroids_existing.loc[
        centroids_existing["cluster"] == cl, MOM_COLS
    ].values[0]
    max_diff = float(np.abs(centroid_fresh - centroid_existing).max())
    check(max_diff <= 1e-4,
          f"Z-centroid matches centroids CSV (cluster {cl})",
          f"max abs diff across 12 dims = {max_diff:.2e}, threshold 1e-4")

# V8: Hit rate in [0, 1]
hr_min = float(descriptor_df["hit_rate"].min())
hr_max = float(descriptor_df["hit_rate"].max())
check(0.0 <= hr_min and hr_max <= 1.0,
      "Hit rate in [0, 1]",
      f"min={hr_min:.4f}, max={hr_max:.4f}")

# V9: Mean return sign matches Sharpe sign per cluster
for cl in range(K):
    mr = float(descriptor_df.loc[descriptor_df["cluster"] == cl, "mean_return"].values[0])
    sh = float(descriptor_df.loc[descriptor_df["cluster"] == cl, "sharpe"].values[0])
    sign_ok = (mr > 0) == (sh > 0) or abs(mr) < 1e-10 or abs(sh) < 1e-10
    check(sign_ok,
          f"Mean return sign matches Sharpe sign (cluster {cl})",
          f"mean_return={mr:.4f}, sharpe={sh:.4f}")

# V10: worst <= mean <= best per cluster
for cl in range(K):
    row = descriptor_df[descriptor_df["cluster"] == cl].iloc[0]
    ok = row["worst_month_return"] <= row["mean_return"] <= row["best_month_return"]
    check(ok,
          f"Worst <= mean <= best (cluster {cl})",
          f"worst={row['worst_month_return']:.4f}, mean={row['mean_return']:.4f}, "
          f"best={row['best_month_return']:.4f}")

# V11: Member dates are unique within cluster
for cl in range(K):
    cl_member_dates = member_df[member_df["cluster"] == cl]["date"]
    dup = int(cl_member_dates.duplicated().sum())
    check(dup == 0,
          f"Member dates unique within cluster {cl}",
          f"duplicate count = {dup}")

# V12: Year-frequency rows sum across clusters = total months per year
year_freq_check = year_freq.copy()
year_totals = year_freq_check.sum(axis=1)
# Cross-check against actual label counts per year
actual_year_counts = labels_df.groupby(labels_df["date"].dt.year).size()
for yr in year_totals.index:
    expected = int(actual_year_counts.get(yr, 0))
    got = int(year_totals[yr])
    check(got == expected,
          f"Year {yr} frequency sum = total months",
          f"sum={got}, expected={expected}")

# V13: Representative dates: exactly 15 rows (3 clusters x 5 reps), no duplicate (cluster, rank)
n_rep_rows = len(rep_df)
check(n_rep_rows == 15,
      "Representative dates: 15 rows total",
      f"got {n_rep_rows}, expected 15")
dup_cr = int(rep_df.duplicated(subset=["cluster", "rank"]).sum())
check(dup_cr == 0,
      "Representative dates: no duplicate (cluster, rank)",
      f"duplicates = {dup_cr}")

# V14: All 4 output CSVs exist on disk
for path, name in [
    (OUT_MASTER,   "cluster_k3_descriptor_table.csv"),
    (OUT_MEMBERS,  "cluster_k3_member_dates.csv"),
    (OUT_REPS,     "cluster_k3_representative_dates.csv"),
    (OUT_YEARFREQ, "cluster_k3_year_frequency.csv"),
]:
    check(path.exists(),
          f"Output exists: {name}",
          f"path={path}")

# ---------------------------------------------------------------------------
# 11. Headline print
# ---------------------------------------------------------------------------
print("\n\n=== HEADLINE SUMMARY ===", flush=True)

rep_by_cluster = rep_df.groupby("cluster")["date"].apply(list).to_dict()

for cl in range(K):
    row = descriptor_df[descriptor_df["cluster"] == cl].iloc[0]
    rep_dates = rep_by_cluster.get(cl, [])

    # Find peak pick_mom horizon
    pick_mom_vals = np.array([row[f"pick_mom_{h}_mean"] for h in HORIZONS])
    peak_h = int(HORIZONS[np.argmax(pick_mom_vals)])
    peak_v = float(pick_mom_vals.max())

    print(f"\n=== CLUSTER {cl} (n={row['n_months']}) ===", flush=True)
    print(f"  pi_panic_mean = {row['pi_panic_mean']:.4f}, "
          f"pi_panic_freq_6mo = {row['pi_panic_freq_6mo_mean']:.4f}, "
          f"pi_panic_freq_12mo = {row['pi_panic_freq_12mo_mean']:.4f}, "
          f"past_sharpe_12mo = {row['past_sharpe_12mo_mean']:.4f}", flush=True)
    print(f"  Sharpe = {row['sharpe']:.4f} [{row['sharpe_lo95']:.4f}, {row['sharpe_hi95']:.4f}], "
          f"hit_rate = {row['hit_rate']:.3f}, "
          f"worst = {row['worst_month_return']:.4f} ({row['worst_month_date']}), "
          f"best = {row['best_month_return']:.4f} ({row['best_month_date']})", flush=True)
    print(f"  cs_mom_overall = {row['mom_overall_mean']:.4f}, "
          f"cs_disp_long = {row['cs_disp_long_mean']:.4f}, "
          f"pick_mom peak = {peak_v:.4f} at h={peak_h}", flush=True)
    print(f"  representative dates: {rep_dates}", flush=True)
    print(f"  max consecutive run = {row['max_consecutive_months']} months", flush=True)
    print(f"  date range: {row['min_date']} .. {row['max_date']}", flush=True)

# ---------------------------------------------------------------------------
# 12. Final status
# ---------------------------------------------------------------------------
print("\n" + ("=" * 60), flush=True)
if all_pass:
    print("OVERALL: ALL CHECKS PASSED", flush=True)
else:
    print("OVERALL: SOME CHECKS FAILED -- DO NOT COMMIT", flush=True)
print("=" * 60, flush=True)
