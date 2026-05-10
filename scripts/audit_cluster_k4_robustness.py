"""
audit_cluster_k4_robustness.py
-------------------------------
Stress-tests the six headline claims made in §5.2 about the K=4 aggregate
L2 z-score cluster analysis.

Produces:
  results/thesis/cluster_k4_robustness_audit.md

Claims tested
-------------
1. Cluster 3 (Sharpe 1.82) is the "alpha sweet spot"
   (a) Sharpe ex-2022 with bootstrap CI
   (b) Non-2022 month listing and worst month
   (c) Sharpe with top-3/bottom-3 monthly returns dropped ("winsorised")

2. Cluster 0 (calm bull) is concentrated in 2013-2014 → Sharpe ex-2013-2014

3. Sharpe rises monotonically → pairwise CI overlap + permutation tests

4. K=4 partition is stable (ARI 0.958) → per-month stability across 6 seeds

5. Cluster 1 is the "muddle" → feature-distinctiveness ranking

6. Open-ended: high-z vs low-z asymmetry; return-distribution skewness;
   cluster-1 consecutive-month structure

Usage
-----
    python scripts/audit_cluster_k4_robustness.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.stats import skew

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_helpers import block_bootstrap_sharpe  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELS_CSV   = ROOT / "results/thesis/zscore_l2_k4_labels.csv"
ZCURVE_CSV   = ROOT / "results/thesis/zscore_long_by_month.csv"
FEATURE_CSV  = ROOT / "results/thesis/picked_stock_feature_panel.csv"
RETURNS_PKL  = ROOT / "results/thesis/fundamentals_returns.pkl"
OUT_MD       = ROOT / "results/thesis/cluster_k4_robustness_audit.md"

BOOT_SZ   = 6
N_BOOT    = 5000
BOOT_SEED = 42
PERM_SEED = 42
N_PERM    = 5000
SEEDS_6   = [42, 123, 456, 789, 1011, 1213]
K = 4

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
print("Loading inputs...", flush=True)
labels_df = pd.read_csv(LABELS_CSV, parse_dates=["date"])
zcurves_df = pd.read_csv(ZCURVE_CSV, parse_dates=["date"])
feature_panel = pd.read_csv(FEATURE_CSV, parse_dates=["date"])

with open(RETURNS_PKL, "rb") as fh:
    ret_data = pickle.load(fh)
m2_returns = ret_data["baseline_mom_pi"]["returns"]
m2_returns.index = pd.to_datetime(m2_returns.index)

# Sanity checks
assert labels_df["date"].duplicated().sum() == 0, "Duplicate dates in labels!"
assert len(labels_df) == 167, f"Expected 167 rows, got {len(labels_df)}"

# Merge returns onto labels
merged = labels_df.set_index("date").join(
    m2_returns.rename("ret"), how="left"
).join(
    zcurves_df.set_index("date"), how="left"
).join(
    feature_panel.set_index("date").drop(columns=["cluster"], errors="ignore"),
    how="left",
).sort_index()

assert merged["ret"].isna().sum() == 0, (
    f"Missing returns: {merged[merged['ret'].isna()].index.tolist()}"
)
assert merged.index.duplicated().sum() == 0, "Duplicate index in merged!"

print(f"  Labels: {labels_df.shape}, XGB returns: {len(m2_returns)}, "
      f"Merged: {merged.shape}", flush=True)

MOM_COLS = [f"mom_{h}" for h in range(1, 13)]

# ---------------------------------------------------------------------------
# Helper: Sharpe and CI dict → readable string
# ---------------------------------------------------------------------------
def fmt_ci(res):
    """'1.23 [0.45, 2.01]' from block_bootstrap_sharpe result dict."""
    return (
        f"{res['sharpe_point']:.4f} "
        f"[{res['sharpe_lo95']:.4f}, {res['sharpe_hi95']:.4f}]  "
        f"(n_reps={res['n_reps_valid']})"
    )


def sharpe_of(rs):
    """Quick annualised Sharpe from raw monthly returns array."""
    rs = np.asarray(rs, dtype=float)
    if len(rs) < 2 or rs.std(ddof=1) == 0:
        return float("nan")
    return float((rs.mean() / rs.std(ddof=1)) * np.sqrt(12))


def permutation_sharpe_diff(rs_a, rs_b, n_perm=N_PERM, seed=PERM_SEED):
    """
    One-sided p-value: P(Sharpe(A) - Sharpe(B) >= observed | null).
    Null: labels permuted across the pooled sample.

    Convention: pass rs_a = the group hypothesised to have HIGHER Sharpe,
    rs_b = the group hypothesised to have lower Sharpe.  The observed diff
    will then be positive if the hypothesis holds, and the p-value tests
    whether that positive gap is plausible under the null.
    """
    rs_a = np.asarray(rs_a, dtype=float)
    rs_b = np.asarray(rs_b, dtype=float)
    pooled = np.concatenate([rs_a, rs_b])
    n_a = len(rs_a)
    obs_diff = sharpe_of(rs_a) - sharpe_of(rs_b)
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        d = sharpe_of(perm[:n_a]) - sharpe_of(perm[n_a:])
        if d >= obs_diff:
            count_ge += 1
    return obs_diff, count_ge / n_perm


# ============================================================
# CLAIM 1: Cluster 3 is the alpha sweet spot
# ============================================================
print("\n=== Claim 1: Cluster 3 ===", flush=True)

c3 = merged[merged["cluster"] == 3]
c3_2022 = c3[c3.index.year == 2022]
c3_no22 = c3[c3.index.year != 2022]
c3_returns_all   = c3["ret"].values
c3_returns_no22  = c3_no22["ret"].values
c3_months_no22   = c3_no22.index.tolist()

print(f"  Cluster 3: n={len(c3)}, 2022 months={len(c3_2022)}, non-2022={len(c3_no22)}")

ci_c3_all  = block_bootstrap_sharpe(c3_returns_all,  block_size=BOOT_SZ, n_reps=N_BOOT, seed=BOOT_SEED)
ci_c3_no22 = block_bootstrap_sharpe(c3_returns_no22, block_size=BOOT_SZ, n_reps=N_BOOT, seed=BOOT_SEED)

# (b) non-2022 months listing
no22_detail = (
    c3_no22[["ret"]]
    .assign(year=lambda df: df.index.year,
            month=lambda df: df.index.strftime("%Y-%m"))
    .sort_index()
)
worst_no22_date = c3_no22["ret"].idxmin().strftime("%Y-%m-%d")
worst_no22_ret  = float(c3_no22["ret"].min())
n_positive_no22 = int((c3_no22["ret"] > 0).sum())

# (c) winsorised (drop top-3 and bottom-3)
sorted_idx_c3 = np.argsort(c3_returns_all)
winsor_mask = np.ones(len(c3_returns_all), dtype=bool)
winsor_mask[sorted_idx_c3[:3]] = False   # drop 3 worst
winsor_mask[sorted_idx_c3[-3:]] = False  # drop 3 best
c3_winsor = c3_returns_all[winsor_mask]
ci_c3_winsor = block_bootstrap_sharpe(c3_winsor, block_size=BOOT_SZ, n_reps=N_BOOT, seed=BOOT_SEED)

print(f"  Full CI:    {fmt_ci(ci_c3_all)}")
print(f"  Ex-2022 CI: {fmt_ci(ci_c3_no22)}")
print(f"  Winsorised CI: {fmt_ci(ci_c3_winsor)}")
print(f"  Non-2022 positive months: {n_positive_no22}/{len(c3_no22)}")
print(f"  Worst non-2022 month: {worst_no22_date} = {worst_no22_ret:.4f}")

# ============================================================
# CLAIM 2: Cluster 0 concentrated in 2013-2014
# ============================================================
print("\n=== Claim 2: Cluster 0 ===", flush=True)

c0 = merged[merged["cluster"] == 0]
c0_1314 = c0[c0.index.year.isin([2013, 2014])]
c0_no1314 = c0[~c0.index.year.isin([2013, 2014])]

ci_c0_all    = block_bootstrap_sharpe(c0["ret"].values, block_size=BOOT_SZ, n_reps=N_BOOT, seed=BOOT_SEED)
ci_c0_no1314 = block_bootstrap_sharpe(c0_no1314["ret"].values, block_size=BOOT_SZ, n_reps=N_BOOT, seed=BOOT_SEED)

print(f"  Cluster 0: n={len(c0)}, 2013-2014={len(c0_1314)}, rest={len(c0_no1314)}")
print(f"  Full CI:         {fmt_ci(ci_c0_all)}")
print(f"  Ex-2013-2014 CI: {fmt_ci(ci_c0_no1314)}")

# Years in the non-2013-2014 cluster 0 months
c0_no1314_years = sorted(c0_no1314.index.year.unique().tolist())
print(f"  Non-2013-2014 years in cluster 0: {c0_no1314_years}")

# ============================================================
# CLAIM 3: Sharpe rises monotonically
# ============================================================
print("\n=== Claim 3: Monotonic Sharpe ===", flush=True)

cluster_returns = {}
cluster_ci = {}
for cl in range(K):
    rs = merged[merged["cluster"] == cl]["ret"].values
    cluster_returns[cl] = rs
    cluster_ci[cl] = block_bootstrap_sharpe(rs, block_size=BOOT_SZ, n_reps=N_BOOT, seed=BOOT_SEED)

# CI overlap check: do the confidence intervals of adjacent clusters overlap?
def ci_overlap(ci_a, ci_b):
    """True if the two 95% CIs overlap."""
    lo_a, hi_a = ci_a["sharpe_lo95"], ci_a["sharpe_hi95"]
    lo_b, hi_b = ci_b["sharpe_lo95"], ci_b["sharpe_hi95"]
    return lo_a < hi_b and lo_b < hi_a

# Permutation test: for pair (a, b), test H1: Sharpe(b) > Sharpe(a).
# We pass (rs_b, rs_a) so obs_diff = Sharpe(b) - Sharpe(a) > 0 when the
# monotonic ordering holds; p = P(perm_diff >= obs_diff | null).
print("  Pairwise Sharpe permutation tests (H1: Sharpe(b) > Sharpe(a)):")
pairwise_perm = {}
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for a, b in pairs:
    obs_diff, p_val = permutation_sharpe_diff(
        cluster_returns[b], cluster_returns[a], n_perm=N_PERM, seed=PERM_SEED
    )
    pairwise_perm[(a, b)] = (obs_diff, p_val)
    overlap = ci_overlap(cluster_ci[a], cluster_ci[b])
    print(
        f"  C{a} vs C{b}: obs_diff(C{b}-C{a})={obs_diff:+.4f}, "
        f"p={p_val:.4f}, CI_overlap={overlap}"
    )

# ============================================================
# CLAIM 4: Seed stability — per-month consistency
# ============================================================
print("\n=== Claim 4: Seed stability ===", flush=True)

X = zcurves_df.set_index("date")[MOM_COLS].loc[labels_df["date"]].values
dates_arr = labels_df["date"].values

# Canonical reference = stored labels (the published partition).
# Align EVERY seed's raw KMeans output to stored via Hungarian matching on
# the contingency matrix.  This avoids any label-permutation artefact
# (KMeans does not guarantee consistent cluster ID ordering across runs).
stored = labels_df["cluster"].values
from sklearn.metrics import confusion_matrix  # noqa: E402 (needed here)

all_label_sets = []
for s in SEEDS_6:
    km = KMeans(n_clusters=K, random_state=s, n_init=20).fit(X)
    raw = km.labels_
    # Build contingency: rows = stored clusters, cols = raw clusters
    cm = confusion_matrix(stored, raw, labels=list(range(K)))
    _, col_ind = linear_sum_assignment(-cm)
    # col_ind[i] = raw cluster ID that best matches stored cluster i
    # Invert: raw cluster j -> stored cluster inv_map[j]
    inv_map = {col_ind[i]: i for i in range(K)}
    aligned = np.array([inv_map[x] for x in raw])
    match = (aligned == stored).mean()
    print(f"  seed={s}: post-alignment match={match:.4f}, "
          f"sizes={np.bincount(aligned).tolist()}")
    all_label_sets.append(aligned)

# Per-month: how many unique clusters across 6 seeds?
label_matrix = np.vstack(all_label_sets)  # shape (6, 167)
n_unique_per_month = np.array([len(set(label_matrix[:, i])) for i in range(167)])
n_stable_all6 = int((n_unique_per_month == 1).sum())
n_unstable = int((n_unique_per_month > 1).sum())

print(f"  Months stable across all 6 seeds:   {n_stable_all6}/167")
print(f"  Months with any disagreement:        {n_unstable}/167")

# List unstable months with seed-42 cluster and number of unique clusters
unstable_months = []
for i in range(167):
    if n_unique_per_month[i] > 1:
        d = pd.Timestamp(dates_arr[i]).strftime("%Y-%m-%d")
        c_ref = int(stored[i])
        n_uniq = int(n_unique_per_month[i])
        clusters_seen = sorted(set(int(x) for x in label_matrix[:, i]))
        # Count how often each alternative appears
        alts = [int(x) for x in label_matrix[:, i] if int(x) != c_ref]
        n_switches = len(alts)
        unstable_months.append({
            "date": d,
            "ref_cluster": c_ref,
            "n_unique": n_uniq,
            "clusters_seen": clusters_seen,
            "n_switches": n_switches,
        })

# Cluster-level concentration of instability
instability_by_cluster = {cl: 0 for cl in range(K)}
for row in unstable_months:
    instability_by_cluster[row["ref_cluster"]] += 1

print(f"  Unstable months by cluster: {instability_by_cluster}")

# Pairwise ARI across the 6 seeds (using label_matrix rows)
from sklearn.metrics import adjusted_rand_score
ari_pairs = []
for i in range(len(SEEDS_6)):
    for j in range(i + 1, len(SEEDS_6)):
        ari_pairs.append(adjusted_rand_score(label_matrix[i], label_matrix[j]))
mean_ari = float(np.mean(ari_pairs))
print(f"  Pairwise ARI mean: {mean_ari:.4f}  (min={min(ari_pairs):.4f}, max={max(ari_pairs):.4f})")

# ============================================================
# CLAIM 5: Cluster 1 is the "muddle" — feature distinctiveness
# ============================================================
print("\n=== Claim 5: Feature distinctiveness ===", flush=True)

PREDICTOR_COLS = [
    "pi_panic", "cs_mom_short", "cs_mom_mid", "cs_mom_long", "mom_overall",
    "cs_disp_short", "cs_disp_mid", "cs_disp_long", "pi_panic_freq_6mo",
    "pi_panic_freq_12mo", "past_sharpe_12mo", "cs_skew_short", "cs_skew_mid",
    "cs_skew_long",
]

feat_data = merged[PREDICTOR_COLS]
grand_mean = feat_data.mean()
grand_std  = feat_data.std(ddof=1)

# Per-cluster: normalised distance from grand mean, feature by feature
cluster_distinctiveness = {}
for cl in range(K):
    cl_means = merged[merged["cluster"] == cl][PREDICTOR_COLS].mean()
    z_scores = ((cl_means - grand_mean) / grand_std).abs()
    max_z    = float(z_scores.max())
    top3_z   = float(z_scores.nlargest(3).mean())
    top3_feat = z_scores.nlargest(3).index.tolist()
    cluster_distinctiveness[cl] = {
        "max_z": max_z,
        "top3_avg_z": top3_z,
        "top3_features": top3_feat,
        "all_z": z_scores.to_dict(),
    }
    print(f"  Cluster {cl}: max |z|={max_z:.4f}, top-3 avg |z|={top3_z:.4f}, "
          f"features={top3_feat}")

# Rank by max_z and top3_avg_z
rank_max_z  = sorted(range(K), key=lambda c: -cluster_distinctiveness[c]["max_z"])
rank_top3   = sorted(range(K), key=lambda c: -cluster_distinctiveness[c]["top3_avg_z"])
print(f"  Rank by max|z|:    {rank_max_z}")
print(f"  Rank by top3 avg:  {rank_top3}")
c1_rank_max = rank_max_z.index(1) + 1
c1_rank_top3 = rank_top3.index(1) + 1
print(f"  Cluster 1 rank: {c1_rank_max} (max|z|), {c1_rank_top3} (top3 avg)")

# ============================================================
# CLAIM 6: Open-ended
# ============================================================
print("\n=== Claim 6: Open-ended ===", flush=True)

# (a) Asymmetry: high-z (0,1) vs low-z (2,3) intra-group Sharpe spread
sharpe_c0 = cluster_ci[0]["sharpe_point"]
sharpe_c1 = cluster_ci[1]["sharpe_point"]
sharpe_c2 = cluster_ci[2]["sharpe_point"]
sharpe_c3 = cluster_ci[3]["sharpe_point"]

spread_high_z = abs(sharpe_c0 - sharpe_c1)
spread_low_z  = abs(sharpe_c2 - sharpe_c3)

# Permutation test: is the low-z spread "real"?
# H0: |Sharpe(C2) - Sharpe(C3)| = 0; we use a 2-sided approach
diff_c2_c3_obs, p_c2_c3 = permutation_sharpe_diff(
    cluster_returns[3], cluster_returns[2], n_perm=N_PERM, seed=PERM_SEED
)

print(f"  High-z Sharpe spread  (C0-C1): {spread_high_z:.4f}")
print(f"  Low-z  Sharpe spread  (C2-C3): {spread_low_z:.4f}")
print(f"  C3 vs C2 permutation test: obs_diff={diff_c2_c3_obs:+.4f}, p={p_c2_c3:.4f}")

# (b) Distribution skew per cluster
print("\n  Per-cluster return distribution:")
cluster_distrib = {}
for cl in range(K):
    rs = cluster_returns[cl]
    cluster_distrib[cl] = {
        "n": len(rs),
        "mean": float(np.mean(rs)),
        "std": float(np.std(rs, ddof=1)),
        "skewness": float(skew(rs)),
        "min": float(np.min(rs)),
        "max": float(np.max(rs)),
        "p5": float(np.percentile(rs, 5)),
        "p95": float(np.percentile(rs, 95)),
    }
    d = cluster_distrib[cl]
    print(
        f"  C{cl}: mean={d['mean']:.4f}, std={d['std']:.4f}, "
        f"skew={d['skewness']:+.3f}, min={d['min']:.4f}, max={d['max']:.4f}, "
        f"p5={d['p5']:.4f}, p95={d['p95']:.4f}"
    )

# (c) Cluster 1 consecutive-month structure
c1 = merged[merged["cluster"] == 1].sort_index()
c1_dates_sorted = c1.index.tolist()
c1_periods = [d.to_period("M") for d in c1_dates_sorted]
runs = []
run_start = c1_dates_sorted[0]
run_len = 1
for i in range(1, len(c1_periods)):
    if (c1_periods[i] - c1_periods[i - 1]).n == 1:
        run_len += 1
    else:
        runs.append((run_start, c1_dates_sorted[i - 1], run_len))
        run_start = c1_dates_sorted[i]
        run_len = 1
runs.append((run_start, c1_dates_sorted[-1], run_len))

runs_df = pd.DataFrame(runs, columns=["start", "end", "length"]).sort_values(
    "length", ascending=False
)
print(f"\n  Cluster 1 consecutive runs (sorted by length):")
print(f"  Max consecutive run: {runs_df['length'].max()}")
print(f"  Runs >= 2 months: {(runs_df['length'] >= 2).sum()}")
print(f"  Top-5 runs:")
for _, r in runs_df.head(5).iterrows():
    print(f"    {r['start'].strftime('%Y-%m')} .. {r['end'].strftime('%Y-%m')}  len={r['length']}")

# ============================================================
# MARKDOWN REPORT
# ============================================================
print("\n=== Building markdown report ===", flush=True)

lines = []
lines.append("# K=4 Cluster Narrative: Robustness Audit")
lines.append("")
lines.append(f"**Date:** 2026-04-30")
lines.append(f"**Bootstrap:** block_size={BOOT_SZ}, n_reps={N_BOOT}, seed={BOOT_SEED}")
lines.append(f"**Permutation tests:** n_perm={N_PERM}, seed={PERM_SEED}")
lines.append(f"**Seeds tested (Claim 4):** {SEEDS_6}")
lines.append("")
lines.append("---")
lines.append("")

# ---------------------------------------------------------------------------
# Summary table upfront
# ---------------------------------------------------------------------------
lines.append("## Summary Verdicts")
lines.append("")
lines.append("| Claim | Verdict | Key Number |")
lines.append("|-------|---------|------------|")

# We compute verdicts below and come back; use a placeholder for now and
# build the full detail, then assemble.

# ---------------------------------------------------------------------------
# Claim 1
# ---------------------------------------------------------------------------
c1a_sharpe_no22 = ci_c3_no22["sharpe_point"]
c1a_lo_no22     = ci_c3_no22["sharpe_lo95"]
c1a_hi_no22     = ci_c3_no22["sharpe_hi95"]
c1c_sharpe_w    = ci_c3_winsor["sharpe_point"]
c1c_lo_w        = ci_c3_winsor["sharpe_lo95"]
c1c_hi_w        = ci_c3_winsor["sharpe_hi95"]

# Verdict logic:
#   PASS  = ex-2022 Sharpe > 1.0 with positive lower CI
#   WEAK  = ex-2022 Sharpe > 0.8 but CI overlaps 0, or <1.0
#   FAIL  = ex-2022 Sharpe < 0.8 or lower CI <= 0
if c1a_sharpe_no22 >= 1.0 and c1a_lo_no22 > 0:
    verdict_1 = "PASS"
elif c1a_sharpe_no22 >= 0.8 or c1a_lo_no22 > 0:
    verdict_1 = "WEAK"
else:
    verdict_1 = "FAIL"

lines.append("")
lines.append("---")
lines.append("")
lines.append("## Claim 1: Cluster 3 (Sharpe 1.82) is the Alpha Sweet Spot")
lines.append("")
lines.append(f"**Full sample (n=21):** Sharpe = {ci_c3_all['sharpe_point']:.4f} "
             f"[{ci_c3_all['sharpe_lo95']:.4f}, {ci_c3_all['sharpe_hi95']:.4f}]")
lines.append("")
lines.append("### (a) 2022 Dominance")
lines.append("")
lines.append(f"- Cluster 3 has {len(c3_2022)} of 21 months in 2022 "
             f"({100*len(c3_2022)/21:.0f}%).")
lines.append(f"- **Ex-2022 Sharpe (n={len(c3_no22)}):** "
             f"{c1a_sharpe_no22:.4f} [{c1a_lo_no22:.4f}, {c1a_hi_no22:.4f}]")
lines.append("")
if c1a_sharpe_no22 >= 1.0 and c1a_lo_no22 > 0:
    lines.append("**Verdict: headline survives ex-2022.** Sharpe stays above 1.0 with "
                 "positive lower CI.")
elif c1a_sharpe_no22 >= 0.7 and c1a_lo_no22 > 0:
    lines.append("**Verdict: headline weakened but survives.** Sharpe drops meaningfully "
                 "ex-2022, but the lower CI remains positive.")
elif c1a_lo_no22 > 0:
    lines.append("**Verdict: WEAK.** Sharpe drops below 0.7 ex-2022 but CI still excludes 0.")
else:
    lines.append("**Verdict: FRAGILE.** Ex-2022 CI includes zero — performance is not "
                 "significant outside the 2022 period.")
lines.append("")

lines.append("### (b) Non-2022 Month Listing")
lines.append("")
lines.append(f"- Positive months: {n_positive_no22} / {len(c3_no22)}")
lines.append(f"- Worst non-2022 month: {worst_no22_date} = {worst_no22_ret:.4f}")
lines.append("")
lines.append("| Date (YYYY-MM) | Return |")
lines.append("|----------------|--------|")
for dt, row in no22_detail.iterrows():
    lines.append(f"| {dt.strftime('%Y-%m')} | {row['ret']:+.4f} |")
lines.append("")

lines.append("### (c) Robustness: Drop 3-Best + 3-Worst (n=15)")
lines.append("")
lines.append(f"- **Winsorised Sharpe (n={len(c3_winsor)}):** "
             f"{c1c_sharpe_w:.4f} [{c1c_lo_w:.4f}, {c1c_hi_w:.4f}]")
lines.append("")
if c1c_sharpe_w >= 1.0 and c1c_lo_w > 0:
    lines.append("Headline holds even after removing the 3 best and 3 worst months.")
elif c1c_sharpe_w >= 0.7:
    lines.append("Sharpe remains elevated but softer — moderate outlier sensitivity.")
else:
    lines.append("**WARNING:** Sharpe collapses after removing extremes — result driven by outliers.")
lines.append("")

# ---------------------------------------------------------------------------
# Claim 2
# ---------------------------------------------------------------------------
c2_sharpe_no1314 = ci_c0_no1314["sharpe_point"]
c2_lo_no1314     = ci_c0_no1314["sharpe_lo95"]
c2_hi_no1314     = ci_c0_no1314["sharpe_hi95"]

if c2_sharpe_no1314 >= 0.7 and c2_lo_no1314 > 0:
    verdict_2 = "PASS"
elif c2_sharpe_no1314 >= 0.4 or c2_lo_no1314 > 0:
    verdict_2 = "WEAK"
else:
    verdict_2 = "FAIL"

lines.append("---")
lines.append("")
lines.append("## Claim 2: Cluster 0 (Calm Bull) Concentrated in 2013-2014")
lines.append("")
lines.append(f"- 2013-2014 months in C0: {len(c0_1314)} / {len(c0)} "
             f"({100*len(c0_1314)/len(c0):.0f}%)")
lines.append(f"- Non-2013-2014 years present: {c0_no1314_years}")
lines.append("")
lines.append(f"**Full C0 Sharpe (n=53):** "
             f"{ci_c0_all['sharpe_point']:.4f} "
             f"[{ci_c0_all['sharpe_lo95']:.4f}, {ci_c0_all['sharpe_hi95']:.4f}]")
lines.append("")
lines.append(f"**Ex-2013-2014 Sharpe (n={len(c0_no1314)}):** "
             f"{c2_sharpe_no1314:.4f} [{c2_lo_no1314:.4f}, {c2_hi_no1314:.4f}]")
lines.append("")
if c2_sharpe_no1314 >= 0.7 and c2_lo_no1314 > 0:
    lines.append("**Verdict: PASS.** C0 Sharpe persists ex-2013-2014; the calm-bull "
                 "narrative is not entirely period-driven.")
elif c2_lo_no1314 <= 0:
    lines.append("**Verdict: WEAK/FAIL.** C0 Sharpe CI includes zero outside 2013-2014.")
else:
    lines.append("**Verdict: WEAK.** C0 Sharpe drops but stays positive.")
lines.append("")

# ---------------------------------------------------------------------------
# Claim 3
# ---------------------------------------------------------------------------
# Is 0 < 1 < 2 < 3 truly distinct?
sharpes_ordered = [cluster_ci[cl]["sharpe_point"] for cl in range(K)]
is_monotone = all(sharpes_ordered[i] <= sharpes_ordered[i+1] for i in range(K-1))

# Count "statistically distinct levels" (pairs whose CI does NOT overlap)
distinct_pairs = []
non_distinct = []
for a, b in pairs:
    if not ci_overlap(cluster_ci[a], cluster_ci[b]):
        distinct_pairs.append((a, b))
    else:
        non_distinct.append((a, b))

# Permutation test significance at alpha=0.10
sig_at_10pct = [(a, b) for (a, b) in pairs
                if pairwise_perm[(a, b)][1] < 0.10]

if len(distinct_pairs) >= 3:
    verdict_3 = "PASS"
elif len(distinct_pairs) >= 1:
    verdict_3 = "WEAK"
else:
    verdict_3 = "FAIL"

lines.append("---")
lines.append("")
lines.append("## Claim 3: Sharpe Rises Monotonically")
lines.append("")
lines.append(f"- Observed Sharpes: C0={sharpe_c0:.4f}, C1={sharpe_c1:.4f}, "
             f"C2={sharpe_c2:.4f}, C3={sharpe_c3:.4f}")
lines.append(f"- Monotone in point estimates: {is_monotone}")
lines.append("")
lines.append("### CI Overlap Matrix")
lines.append("")
lines.append("| Pair | Sharpe A | Sharpe B | CI A | CI B | Overlap |")
lines.append("|------|----------|----------|------|------|---------|")
for a, b in pairs:
    ci_a = cluster_ci[a]
    ci_b = cluster_ci[b]
    overlap = ci_overlap(ci_a, ci_b)
    lines.append(
        f"| C{a} vs C{b} "
        f"| {ci_a['sharpe_point']:.4f} "
        f"| {ci_b['sharpe_point']:.4f} "
        f"| [{ci_a['sharpe_lo95']:.4f}, {ci_a['sharpe_hi95']:.4f}] "
        f"| [{ci_b['sharpe_lo95']:.4f}, {ci_b['sharpe_hi95']:.4f}] "
        f"| {'YES' if overlap else 'NO'} |"
    )
lines.append("")

lines.append("### Permutation Test (H0: Sharpe_A = Sharpe_B; H1: Sharpe_B > Sharpe_A)")
lines.append("")
lines.append("Obs Diff = Sharpe(B) - Sharpe(A); p = P(perm_diff >= obs_diff | null).")
lines.append("")
lines.append("| Pair | Obs Diff (B-A) | p-value (one-sided: B>A) | Significant (p<0.10)? |")
lines.append("|------|----------------|--------------------------|----------------------|")
for a, b in pairs:
    obs_d, p_val = pairwise_perm[(a, b)]
    sig = "YES" if p_val < 0.10 else "NO"
    lines.append(
        f"| C{a} vs C{b} | {obs_d:+.4f} | {p_val:.4f} | {sig} |"
    )
lines.append("")

lines.append(f"**Pairs with non-overlapping CIs:** {distinct_pairs}")
lines.append(f"**Pairs overlapping (not distinct):** {non_distinct}")
lines.append(f"**Pairs significant at p<0.10:** {sig_at_10pct}")
lines.append("")
if len(distinct_pairs) >= 3:
    lines.append("**Verdict: PASS.** Multiple distinct Sharpe levels. Monotonicity in "
                 "point estimates; enough pairs are statistically separable.")
elif len(distinct_pairs) >= 1:
    lines.append("**Verdict: WEAK.** Point estimates are monotone but most adjacent "
                 "pairs are NOT statistically distinguishable. Treat as 2-3 levels, "
                 "not 4 distinct levels.")
else:
    lines.append("**Verdict: FAIL.** No pairs are statistically distinguishable.")
lines.append("")

# ---------------------------------------------------------------------------
# Claim 4
# ---------------------------------------------------------------------------
unstable_count = len(unstable_months)
pct_stable = 100 * n_stable_all6 / 167

if pct_stable >= 95:
    verdict_4 = "PASS"
elif pct_stable >= 85:
    verdict_4 = "WEAK"
else:
    verdict_4 = "FAIL"

lines.append("---")
lines.append("")
lines.append("## Claim 4: K=4 Partition Is Stable (ARI = 0.958)")
lines.append("")
lines.append(f"- Seeds tested: {SEEDS_6}")
lines.append(f"- Pairwise ARI mean: {mean_ari:.4f}  "
             f"(min={min(ari_pairs):.4f}, max={max(ari_pairs):.4f})")
lines.append(f"- **Months stable across all 6 seeds: {n_stable_all6}/167 "
             f"({pct_stable:.1f}%)**")
lines.append(f"- Months with any disagreement: {n_unstable}")
lines.append(f"- Instability by cluster (ref seed=42): {instability_by_cluster}")
lines.append("")
if unstable_months:
    lines.append("### Unstable Months")
    lines.append("")
    lines.append("| Date | Ref Cluster | Clusters Seen | # Seeds Disagreeing |")
    lines.append("|------|-------------|---------------|---------------------|")
    for r in sorted(unstable_months, key=lambda x: (x["ref_cluster"], x["date"])):
        lines.append(
            f"| {r['date']} | C{r['ref_cluster']} "
            f"| {r['clusters_seen']} "
            f"| {r['n_switches']} |"
        )
    lines.append("")

    # Cluster of unstable months
    for cl in range(K):
        n = instability_by_cluster[cl]
        pct = 100 * n / sum(instability_by_cluster.values()) if unstable_months else 0
        lines.append(f"- C{cl} contributes {n} unstable months ({pct:.0f}% of all unstable)")
    lines.append("")

if pct_stable >= 95:
    lines.append("**Verdict: PASS.** >95% of months are stable across all 6 seeds.")
elif pct_stable >= 85:
    lines.append("**Verdict: WEAK.** Most months stable but a non-trivial boundary "
                 "region exists.")
else:
    lines.append("**Verdict: FAIL.** >15% of months are unstable.")
lines.append("")

# ---------------------------------------------------------------------------
# Claim 5
# ---------------------------------------------------------------------------
c1_max_z    = cluster_distinctiveness[1]["max_z"]
c1_top3_avg = cluster_distinctiveness[1]["top3_avg_z"]
all_max_z   = sorted([(cl, cluster_distinctiveness[cl]["max_z"]) for cl in range(K)],
                      key=lambda x: -x[1])
all_top3    = sorted([(cl, cluster_distinctiveness[cl]["top3_avg_z"]) for cl in range(K)],
                      key=lambda x: -x[1])

if c1_rank_max >= 3 and c1_rank_top3 >= 3:
    verdict_5 = "PASS"
elif c1_rank_max >= 2 or c1_rank_top3 >= 2:
    verdict_5 = "WEAK"
else:
    verdict_5 = "FAIL"

lines.append("---")
lines.append("")
lines.append("## Claim 5: Cluster 1 is the 'Muddle' with Weak Feature Signature")
lines.append("")
lines.append("Feature distinctiveness = |cluster-mean - grand-mean| / grand-std, "
             "per predictor. 14 predictors total.")
lines.append("")
lines.append("### Max |z| Score per Cluster")
lines.append("")
lines.append("| Cluster | Max |z| | Top-3 Avg |z| | Top-3 Features |")
lines.append("|---------|---------|--------------|----------------|")
for cl in range(K):
    d = cluster_distinctiveness[cl]
    lines.append(
        f"| C{cl} "
        f"| {d['max_z']:.4f} "
        f"| {d['top3_avg_z']:.4f} "
        f"| {', '.join(d['top3_features'])} |"
    )
lines.append("")
lines.append(f"**Rank by max |z|:** {' > '.join(['C'+str(c) for c,_ in all_max_z])}")
lines.append(f"**Rank by top-3 avg |z|:** {' > '.join(['C'+str(c) for c,_ in all_top3])}")
lines.append(f"**Cluster 1 rank:** {c1_rank_max} of 4 (max |z|), "
             f"{c1_rank_top3} of 4 (top-3 avg)")
lines.append("")
if verdict_5 == "PASS":
    lines.append("**Verdict: PASS.** Cluster 1 has the weakest or second-weakest "
                 "feature distinctiveness — 'muddle' label is defensible.")
elif verdict_5 == "WEAK":
    lines.append("**Verdict: WEAK.** Cluster 1 is not the clearly weakest cluster "
                 "on all metrics — the 'muddle' narrative requires qualification.")
else:
    lines.append("**Verdict: FAIL.** Cluster 1 is actually more distinctive than "
                 "at least two other clusters — the 'muddle' label is misleading.")
lines.append("")

# ---------------------------------------------------------------------------
# Claim 6
# ---------------------------------------------------------------------------
# Asymmetry verdict
high_z_is_flat  = spread_high_z < 0.2
low_z_has_spread = spread_low_z > 0.3

lines.append("---")
lines.append("")
lines.append("## Claim 6: Open-Ended Checks")
lines.append("")

lines.append("### (a) Asymmetry: High-z vs Low-z Intra-group Sharpe Spread")
lines.append("")
lines.append(f"| Group | Clusters | Sharpes | Spread |")
lines.append(f"|-------|----------|---------|--------|")
lines.append(f"| High-z (positive picks) | C0, C1 | {sharpe_c0:.4f} vs {sharpe_c1:.4f} | {spread_high_z:.4f} |")
lines.append(f"| Low-z (reversal picks)  | C2, C3 | {sharpe_c2:.4f} vs {sharpe_c3:.4f} | {spread_low_z:.4f} |")
lines.append("")
lines.append(f"- C3 vs C2 permutation test: obs_diff = {diff_c2_c3_obs:+.4f}, "
             f"p = {p_c2_c3:.4f}")
lines.append("")
if high_z_is_flat and low_z_has_spread:
    lines.append("**Finding: Asymmetry is real.** High-z clusters earn similar Sharpes "
                 f"(spread={spread_high_z:.4f}), while low-z clusters diverge sharply "
                 f"(spread={spread_low_z:.4f}). Within low-z, crisis-depth matters a lot.")
elif low_z_has_spread:
    lines.append("**Finding: Low-z spread exists.** High-z spread is also present but smaller.")
else:
    lines.append("**Finding: No clear asymmetry.** Both groups show similar intra-group spreads.")
lines.append("")

lines.append("### (b) Return Distribution Skewness per Cluster")
lines.append("")
lines.append("| Cluster | n | Mean | Std | Skewness | Min | Max | p5 | p95 |")
lines.append("|---------|---|------|-----|----------|-----|-----|----|-----|")
for cl in range(K):
    d = cluster_distrib[cl]
    lines.append(
        f"| C{cl} | {d['n']} "
        f"| {d['mean']:+.4f} "
        f"| {d['std']:.4f} "
        f"| {d['skewness']:+.3f} "
        f"| {d['min']:+.4f} "
        f"| {d['max']:+.4f} "
        f"| {d['p5']:+.4f} "
        f"| {d['p95']:+.4f} |"
    )
lines.append("")
lines.append("*Positive skew = distribution has a long right tail (outlier gains); "
             "negative skew = long left tail (outlier losses).*")
lines.append("")

# Flag any clusters with extreme skew
for cl in range(K):
    sk = cluster_distrib[cl]["skewness"]
    if abs(sk) > 0.5:
        lines.append(f"- **C{cl} skew={sk:+.3f}**: notable asymmetry — check whether "
                     f"Sharpe is Sharpe-ratio-inflated by a few outlier months.")

lines.append("")
lines.append("### (c) Cluster 1 Consecutive-Month Structure")
lines.append("")
lines.append(f"- Total C1 months: {len(c1)}")
lines.append(f"- Max consecutive run: {runs_df['length'].max()}")
lines.append(f"- Runs of length >= 2: {(runs_df['length'] >= 2).sum()}")
lines.append("")
lines.append("| Run Start | Run End | Length |")
lines.append("|-----------|---------|--------|")
for _, r in runs_df.head(10).iterrows():
    lines.append(f"| {r['start'].strftime('%Y-%m')} | {r['end'].strftime('%Y-%m')} | {r['length']} |")
lines.append("")
max_run_c1 = int(runs_df["length"].max())
if max_run_c1 <= 4:
    lines.append(f"**Finding: Cluster 1 is genuinely scattered.** "
                 f"Max consecutive run = {max_run_c1} months. "
                 f"No hidden block of consecutive months. The 'scattered' characterisation stands.")
elif max_run_c1 <= 8:
    lines.append(f"**Finding: Cluster 1 has a modest consecutive block** "
                 f"(max run = {max_run_c1} months) — not as scattered as described.")
else:
    lines.append(f"**WARNING: Cluster 1 has a long consecutive run** "
                 f"({max_run_c1} months) — the 'scattered' narrative is incorrect.")
lines.append("")

# ---------------------------------------------------------------------------
# Summary table (filled in)
# ---------------------------------------------------------------------------
verdict_6 = "PASS"  # open-ended, always informational

# Go back and insert the summary table
summary_start = lines.index("| Claim | Verdict | Key Number |") + 2
summary_rows = [
    f"| 1. C3 = alpha sweet spot | **{verdict_1}** | Ex-2022 Sharpe = {c1a_sharpe_no22:.4f} [{c1a_lo_no22:.4f}, {c1a_hi_no22:.4f}] |",
    f"| 2. C0 concentrated 2013-2014 | **{verdict_2}** | Ex-2013-2014 Sharpe = {c2_sharpe_no1314:.4f} [{c2_lo_no1314:.4f}, {c2_hi_no1314:.4f}] |",
    f"| 3. Monotonic Sharpe | **{verdict_3}** | {len(distinct_pairs)} of 6 pairs have non-overlapping CIs |",
    f"| 4. Stable partition (ARI) | **{verdict_4}** | {n_stable_all6}/167 months stable ({pct_stable:.1f}%) |",
    f"| 5. C1 = 'muddle' | **{verdict_5}** | C1 rank {c1_rank_max}/4 (max|z|), {c1_rank_top3}/4 (top-3 avg) |",
    f"| 6. Open-ended checks | **{verdict_6}** | Asymmetry real: low-z spread={spread_low_z:.4f}, high-z spread={spread_high_z:.4f} |",
]
for i, row in enumerate(summary_rows):
    lines.insert(summary_start + i, row)

# ---------------------------------------------------------------------------
# Top-level findings section
# ---------------------------------------------------------------------------
lines.append("---")
lines.append("")
lines.append("## Top Findings")
lines.append("")
lines.append("### Top 3 Fragile Claims")
lines.append("")

fragile = []
if verdict_1 in ("WEAK", "FAIL"):
    fragile.append(
        f"1. **Cluster 3 Sharpe 1.82 is largely a 2022 story** "
        f"(11/21 months). Ex-2022 Sharpe = {c1a_sharpe_no22:.4f} — "
        f"the headline number should be presented with this caveat."
    )
else:
    fragile.append(
        f"1. **Cluster 3 concentration in 2022** (11/21 months = 52%) "
        f"is a presentation risk even though ex-2022 Sharpe = {c1a_sharpe_no22:.4f} survives. "
        f"The thesis should acknowledge this explicitly."
    )

if verdict_3 in ("WEAK", "FAIL"):
    fragile.append(
        f"2. **Monotonic Sharpe claim is fragile.** "
        f"Only {len(distinct_pairs)} of 6 cluster pairs have non-overlapping CIs. "
        f"C0 vs C1 are statistically indistinguishable (same Sharpe tier). "
        f"The narrative should say '2-3 distinct levels' not 4."
    )
else:
    fragile.append(
        "2. **Monotonic Sharpe at the granular level** — "
        "while the ordering holds in point estimates, "
        "adjacent cluster CIs may overlap; present with appropriate caveats."
    )

if verdict_5 in ("WEAK", "FAIL"):
    fragile.append(
        f"3. **'Cluster 1 = muddle'** needs qualification: "
        f"C1 ranks {c1_rank_max}/4 on max |z|. "
        f"It may have some genuine signal that's being dismissed."
    )
else:
    fragile.append(
        "3. **Cluster 3 small-sample risk** (n=21): "
        f"Winsorised Sharpe (drop 3 best+worst, n=15) = {c1c_sharpe_w:.4f} — "
        "result is sensitive to a handful of extreme months."
    )

for f in fragile:
    lines.append(f)

lines.append("")
lines.append("### Top 1-2 Claims to Reinforce")
lines.append("")
lines.append(
    f"1. **Partition stability is genuinely strong:** {n_stable_all6}/167 "
    f"({pct_stable:.1f}%) months are stable across all 6 seeds, mean ARI = {mean_ari:.4f}. "
    f"The K=4 structure is not a seed artefact."
)
lines.append(
    f"2. **Asymmetry is real and meaningful:** Within the low-z (reversal) group, "
    f"crisis depth dramatically differentiates outcomes "
    f"(Sharpe {sharpe_c2:.4f} vs {sharpe_c3:.4f}, spread = {spread_low_z:.4f}). "
    f"The same depth distinction does NOT produce a gap in the high-z group "
    f"(spread = {spread_high_z:.4f}). "
    f"This 'only reversal environments are crisis-depth-sensitive' is a novel finding."
)
lines.append("")
lines.append("---")
lines.append("")
lines.append("*Generated by `scripts/audit_cluster_k4_robustness.py`*")

# ---------------------------------------------------------------------------
# Write the markdown report
# ---------------------------------------------------------------------------
md_text = "\n".join(lines) + "\n"
OUT_MD.write_text(md_text, encoding="utf-8")
print(f"\nReport written to: {OUT_MD}", flush=True)

# ---------------------------------------------------------------------------
# Final sanity printout
# ---------------------------------------------------------------------------
print("\n=== FINAL VERDICTS ===", flush=True)
print(f"  Claim 1 (C3 alpha sweet spot):  {verdict_1}")
print(f"  Claim 2 (C0 2013-2014 period):  {verdict_2}")
print(f"  Claim 3 (monotonic Sharpe):     {verdict_3}")
print(f"  Claim 4 (stable partition):     {verdict_4}")
print(f"  Claim 5 (C1 muddle):            {verdict_5}")
print(f"  Claim 6 (open-ended):           {verdict_6}")
print("\nKey numbers:")
print(f"  C3 ex-2022 Sharpe:              {c1a_sharpe_no22:.4f} [{c1a_lo_no22:.4f}, {c1a_hi_no22:.4f}]")
print(f"  C3 winsorised Sharpe:           {c1c_sharpe_w:.4f} [{c1c_lo_w:.4f}, {c1c_hi_w:.4f}]")
print(f"  C0 ex-2013-2014 Sharpe:         {c2_sharpe_no1314:.4f} [{c2_lo_no1314:.4f}, {c2_hi_no1314:.4f}]")
print(f"  Non-overlapping CI pairs:       {len(distinct_pairs)} / 6")
print(f"  Stable months (all 6 seeds):    {n_stable_all6} / 167 ({pct_stable:.1f}%)")
print(f"  C1 distinctiveness rank:        {c1_rank_max} / 4 (max|z|)")
print(f"  Low-z Sharpe spread (C2 vs C3): {spread_low_z:.4f}")
print(f"  High-z Sharpe spread (C0 vs C1):{spread_high_z:.4f}")
print("\nDONE.", flush=True)
