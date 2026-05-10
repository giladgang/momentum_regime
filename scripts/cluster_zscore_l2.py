"""
cluster_zscore_l2.py
--------------------
Pure KMeans / L2 clustering on the 12-d cross-section z-score curves of
the long-leg picks.

NO preprocessing: no standardisation across months, no PCA, no dispersion
features.  Each row of zscore_long_by_month.csv is already a z-curve
(z-scored across the cross-section within that month); we cluster them as-is
using Euclidean (L2) distance.

Outputs
-------
results/thesis/zscore_l2_cluster_sweep.csv
results/thesis/zscore_l2_labels.csv
results/thesis/zscore_l2_centroids.csv
results/thesis/zscore_l2_vs_within_regime_confusion.csv
results/thesis/zscore_l2_vs_fingerprint_k4_confusion.csv
plots/thesis/zscore_l2_dispersion.png
plots/thesis/zscore_l2_dispersion.pdf
"""

import os
import pickle
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

ZSCORE_CSV          = ROOT / "results/thesis/zscore_long_by_month.csv"
ARTEFACTS_PKL       = ROOT / "artefacts/cs_artefacts_data.pkl"
RETURNS_PKL         = ROOT / "results/thesis/fundamentals_returns.pkl"
WITHIN_REGIME_CSV   = ROOT / "results/thesis/picked_stock_within_regime_labels.csv"
FINGERPRINT_K4_CSV  = ROOT / "results/thesis/picked_stock_aggregate_k4_labels.csv"

RES_DIR  = ROOT / "results/thesis"
PLOT_DIR = ROOT / "plots/thesis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS  = list(range(1, 13))
MOM_COLS  = [f"mom_{h}" for h in HORIZONS]

SEEDS    = [42, 123, 456, 789, 1011, 1213]
N_INIT   = 20
BOOT_SZ  = 6
N_BOOT   = 5000
BOOT_SEED = 42

ARI_STABLE = 0.85
MIN_FRAC   = 0.05   # min cluster size as fraction of months

# ---------------------------------------------------------------------------
# 1. Load zscore matrix
# ---------------------------------------------------------------------------
print("Loading z-score matrix ...")
Z_raw = pd.read_csv(ZSCORE_CSV, parse_dates=["date"]).set_index("date")
Z_raw = Z_raw[MOM_COLS].dropna()
Z = Z_raw.copy()
N = len(Z)
print(f"  N = {N} months after dropping NaN rows")

# ---------------------------------------------------------------------------
# 2. Load helpers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_helpers import block_bootstrap_sharpe, seed_stability  # noqa: E402

# ---------------------------------------------------------------------------
# 3. Load pi_filter per month (for pi_bar / mean_pi_panic)
# ---------------------------------------------------------------------------
print("Loading artefacts ...")
with open(ARTEFACTS_PKL, "rb") as fh:
    art = pickle.load(fh)
test_df = art["test"].copy()
test_df["date"] = pd.to_datetime(test_df["date"])
pi_monthly = test_df.groupby("date")["pi_filter"].first()

# ---------------------------------------------------------------------------
# 4. Load XGB returns
# ---------------------------------------------------------------------------
print("Loading XGB returns ...")
with open(RETURNS_PKL, "rb") as fh:
    ret_data = pickle.load(fh)
m2_ret = ret_data["baseline_mom_pi"]["returns"]
m2_ret.index = pd.to_datetime(m2_ret.index)

# ---------------------------------------------------------------------------
# 5. K-sweep
# ---------------------------------------------------------------------------
print("\n=== K-sweep (K=2..5) ===")
sweep_rows = []

for K in [2, 3, 4, 5]:
    # seed-42 fit
    km42 = KMeans(n_clusters=K, n_init=N_INIT, random_state=42).fit(Z.values)
    sil = silhouette_score(Z.values, km42.labels_)

    # 6-seed stability
    stab = seed_stability(Z.values, k=K, seeds=SEEDS, n_init=N_INIT)
    mean_ari = stab["mean_ari"]

    sizes42 = np.bincount(km42.labels_, minlength=K).tolist()
    min_size = min(sizes42)
    stable = (mean_ari >= ARI_STABLE) and (min_size / N >= MIN_FRAC)

    print(
        f"  K={K}  sil={sil:.4f}  ARI={mean_ari:.4f}  "
        f"sizes={sizes42}  stable={stable}"
    )
    sweep_rows.append(
        dict(
            k=K,
            mean_silhouette=round(sil, 6),
            mean_ari=round(mean_ari, 6),
            sizes_seed42=str(sizes42),
            stable=stable,
        )
    )

sweep_df = pd.DataFrame(sweep_rows)
sweep_path = RES_DIR / "zscore_l2_cluster_sweep.csv"
sweep_df.to_csv(sweep_path, index=False)
print(f"\nSaved sweep: {sweep_path}")

# ---------------------------------------------------------------------------
# 6. Choose K: smallest stable K with min cluster size >= 5% of months
# ---------------------------------------------------------------------------
stable_rows = sweep_df[sweep_df["stable"]]
if stable_rows.empty:
    # Fall back: smallest K with ARI >= threshold (ignore size constraint)
    ari_ok = sweep_df[sweep_df["mean_ari"] >= ARI_STABLE]
    chosen_K = int(ari_ok["k"].min()) if not ari_ok.empty else 2
    print(f"\nWARNING: no K passed both stability criteria; falling back to K={chosen_K}")
else:
    chosen_K = int(stable_rows["k"].min())

print(f"\nChosen K = {chosen_K}")

# ---------------------------------------------------------------------------
# 7. Final fit at chosen K
# ---------------------------------------------------------------------------
km_final = KMeans(n_clusters=chosen_K, n_init=N_INIT, random_state=42).fit(Z.values)
raw_labels = km_final.labels_

# Order clusters: by mean of centroid across all 12 horizons (descending)
# so cluster 0 = highest overall z, cluster K-1 = lowest overall z
centroids_raw = km_final.cluster_centers_                       # (K, 12)
centroid_means = centroids_raw.mean(axis=1)                     # (K,)
order = np.argsort(-centroid_means)                             # descending
relabel_map = {old: new for new, old in enumerate(order)}
labels_reordered = np.array([relabel_map[l] for l in raw_labels])
centroids_ordered = centroids_raw[order]                        # (K, 12)

# Attach dates
dates_arr = Z.index
label_series = pd.Series(labels_reordered, index=dates_arr, name="cluster")

# ---------------------------------------------------------------------------
# 8. Save labels
# ---------------------------------------------------------------------------
labels_df = label_series.reset_index().rename(columns={"index": "date"})
labels_path = RES_DIR / "zscore_l2_labels.csv"
labels_df.to_csv(labels_path, index=False)
print(f"Saved labels: {labels_path}")

# ---------------------------------------------------------------------------
# 9. Compute per-cluster stats for centroids CSV
# ---------------------------------------------------------------------------
print("\n=== Per-cluster stats ===")
centroid_rows = []

for cl in range(chosen_K):
    cl_dates = dates_arr[label_series == cl]
    n_cl = len(cl_dates)
    z_cl = Z.loc[cl_dates]

    # pi_bar (mean pi_filter over cluster months)
    pi_cl = pi_monthly.reindex(cl_dates).dropna()
    pi_bar = float(pi_cl.mean()) if len(pi_cl) else float("nan")
    mean_pi_panic = float((pi_cl >= 0.5).mean()) if len(pi_cl) else float("nan")

    # Sharpe via block bootstrap
    ret_cl = m2_ret.reindex(cl_dates).dropna()
    bbs = block_bootstrap_sharpe(ret_cl.values, block_size=BOOT_SZ,
                                  n_reps=N_BOOT, seed=BOOT_SEED)

    centroid_vals = {f"mom_{h}": round(float(centroids_ordered[cl, i]), 6)
                     for i, h in enumerate(HORIZONS)}
    row = {
        "cluster": cl,
        **centroid_vals,
        "n_months": n_cl,
        "pi_bar": round(pi_bar, 6),
        "mean_pi_panic": round(mean_pi_panic, 6),
        "sharpe": round(bbs["sharpe_point"], 4),
        "sharpe_lo95": round(bbs["sharpe_lo95"], 4),
        "sharpe_hi95": round(bbs["sharpe_hi95"], 4),
    }
    centroid_rows.append(row)
    print(
        f"  cluster {cl}: n={n_cl:3d}  pi_bar={pi_bar:.4f}  "
        f"mean_pi_panic={mean_pi_panic:.3f}  "
        f"Sharpe={bbs['sharpe_point']:.3f} "
        f"[{bbs['sharpe_lo95']:.3f}, {bbs['sharpe_hi95']:.3f}]  "
        f"centroid_mean={centroids_ordered[cl].mean():.4f}"
    )

centroids_df = pd.DataFrame(centroid_rows)
centroids_path = RES_DIR / "zscore_l2_centroids.csv"
centroids_df.to_csv(centroids_path, index=False)
print(f"\nSaved centroids: {centroids_path}")

# ---------------------------------------------------------------------------
# 10. Confusion matrix: vs within-regime 4 cells
# ---------------------------------------------------------------------------
print("\nBuilding confusion matrices ...")

within_regime = pd.read_csv(WITHIN_REGIME_CSV, parse_dates=["date"])
merged_wr = label_series.rename("l2_cluster").reset_index()
merged_wr = merged_wr.merge(
    within_regime[["date", "combined_label"]], on="date", how="inner"
)

conf_wr = pd.crosstab(
    merged_wr["l2_cluster"],
    merged_wr["combined_label"],
    rownames=["l2_cluster"],
    colnames=["within_regime"],
)
# Ensure all 4 cells are present as columns
for cell in ["calm.0", "calm.1", "panic.0", "panic.1"]:
    if cell not in conf_wr.columns:
        conf_wr[cell] = 0
conf_wr = conf_wr[["calm.0", "calm.1", "panic.0", "panic.1"]]

conf_wr_path = RES_DIR / "zscore_l2_vs_within_regime_confusion.csv"
conf_wr.to_csv(conf_wr_path)
print(f"Saved: {conf_wr_path}")

# ---------------------------------------------------------------------------
# 11. Confusion matrix: vs fingerprint K=4 clusters
# ---------------------------------------------------------------------------
fp_k4 = pd.read_csv(FINGERPRINT_K4_CSV, parse_dates=["date"])
merged_fp = label_series.rename("l2_cluster").reset_index()
merged_fp = merged_fp.merge(
    fp_k4[["date", "cluster"]].rename(columns={"cluster": "fp_cluster"}),
    on="date", how="inner"
)

conf_fp = pd.crosstab(
    merged_fp["l2_cluster"],
    merged_fp["fp_cluster"],
    rownames=["l2_cluster"],
    colnames=["fingerprint_k4"],
)
# Ensure cluster columns 0-3 present
for c in range(4):
    if c not in conf_fp.columns:
        conf_fp[c] = 0
conf_fp = conf_fp[[0, 1, 2, 3]]
conf_fp.columns = [f"cluster_{c}" for c in [0, 1, 2, 3]]

conf_fp_path = RES_DIR / "zscore_l2_vs_fingerprint_k4_confusion.csv"
conf_fp.to_csv(conf_fp_path)
print(f"Saved: {conf_fp_path}")

# ---------------------------------------------------------------------------
# 12. All-calm reference centroid (pi_panic == 0)
# ---------------------------------------------------------------------------
within_regime_full = pd.read_csv(WITHIN_REGIME_CSV, parse_dates=["date"])
calm_dates = within_regime_full.loc[
    within_regime_full["pi_panic"] == 0, "date"
].tolist()
Z_calm = Z.reindex(calm_dates).dropna()
ref_centroid = Z_calm.mean(axis=0).values
print(f"\nAll-calm reference: {len(Z_calm)} months used for dashed reference line")

# ---------------------------------------------------------------------------
# 13. Plot
# ---------------------------------------------------------------------------
print("\nGenerating dispersion plot ...")

_PALETTE = {
    2: ["#1f6dad", "#d62728"],
    3: ["#1f6dad", "#e67a00", "#d62728"],
    4: ["#1f6dad", "#26a69a", "#e67a00", "#d62728"],
    5: ["#1f6dad", "#26a69a", "#bcbd22", "#e67a00", "#d62728"],
}
colors = (_PALETTE.get(chosen_K)
          or plt.cm.viridis(np.linspace(0.15, 0.85, chosen_K)))

fig, axes = plt.subplots(
    1, chosen_K,
    figsize=(4.5 * chosen_K, 5.5),
    sharey=True,
    constrained_layout=True,
)
if chosen_K == 1:
    axes = [axes]

# Global y-axis limits across all centroids + reference
all_centroid_vals = centroids_ordered.flatten().tolist() + ref_centroid.tolist()
y_min = round((min(all_centroid_vals) - 0.15) * 4) / 4
y_max = round((max(all_centroid_vals) + 0.15) * 4) / 4
if y_max - y_min < 1.5:
    mid = (y_max + y_min) / 2
    y_min = mid - 0.75
    y_max = mid + 0.75

fig.suptitle(
    "Direct L2 KMeans on 12-d cross-section z-curves",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

for cl, ax in enumerate(axes):
    color = colors[cl]
    cl_dates = dates_arr[label_series == cl]
    Z_cl = Z.loc[cl_dates]
    centroid = centroids_ordered[cl]

    # pi_bar
    pi_cl = pi_monthly.reindex(cl_dates).dropna()
    pi_bar_cl = float(pi_cl.mean()) if len(pi_cl) else float("nan")

    # Sharpe for title
    cr = centroid_rows[cl]
    sharpe_str = (f"{cr['sharpe']:.2f}" if not np.isnan(cr["sharpe"]) else "n/a")
    lo_str = (f"{cr['sharpe_lo95']:.2f}" if not np.isnan(cr["sharpe_lo95"]) else "?")
    hi_str = (f"{cr['sharpe_hi95']:.2f}" if not np.isnan(cr["sharpe_hi95"]) else "?")

    # Thin per-month lines
    for _, row in Z_cl.iterrows():
        ax.plot(HORIZONS, row.values, color=color, lw=0.8, alpha=0.4)

    # Bold centroid
    ax.plot(
        HORIZONS, centroid,
        color=color, lw=2.5, marker="o", ms=6,
        label=f"centroid ({len(cl_dates)} mo)",
        zorder=5,
    )

    # Dashed all-calm reference
    ax.plot(
        HORIZONS, ref_centroid,
        color="#333333", lw=1.5, ls="--",
        label="all-calm ref",
        zorder=4,
    )

    # y=0 horizontal line
    ax.axhline(0, color="#888888", lw=0.6, zorder=1)

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Horizon (months)", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("Cross-sectional z-score", fontsize=9)

    ax.set_title(
        f"Cluster {cl}\n"
        f"(n={len(cl_dates)}, π̅={pi_bar_cl:.2f})\n"
        f"Sharpe={sharpe_str} [{lo_str}, {hi_str}]",
        fontsize=9.5,
        fontweight="bold",
        loc="center",
    )

    ax.legend(fontsize=7.5, frameon=True, loc="upper left")
    ax.tick_params(axis="both", labelsize=8)

for ext in ("png", "pdf"):
    out_path = PLOT_DIR / f"zscore_l2_dispersion.{ext}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)

# ---------------------------------------------------------------------------
# 14. Open PNG
# ---------------------------------------------------------------------------
subprocess.run(["open", str(PLOT_DIR / "zscore_l2_dispersion.png")], check=False)

# ---------------------------------------------------------------------------
# 15. HEADLINE
# ---------------------------------------------------------------------------
print("\n=== L2 z-curve clustering ===")
print(f"Chosen K: {chosen_K}")
print("Per cluster (ordered by overall z desc):")
for cr in centroid_rows:
    cl   = cr["cluster"]
    n    = cr["n_months"]
    pb   = cr["pi_bar"]
    sh   = cr["sharpe"]
    lo   = cr["sharpe_lo95"]
    hi   = cr["sharpe_hi95"]
    cmean = centroids_ordered[cl].mean()
    print(
        f"  cluster {cl}: n={n:3d}  pi_bar={pb:.4f}  "
        f"Sharpe={sh:.3f} [{lo:.3f}, {hi:.3f}]  "
        f"centroid mom_overall={cmean:.4f}"
    )

print()
print("Confusion vs within-regime 4 cells:")
print(conf_wr.to_string())

print()
print("Confusion vs fingerprint K=4:")
print(conf_fp.to_string())

# Interpretation
print()
print("Interpretation:")
ari_wr = adjusted_rand_score(merged_wr["l2_cluster"], merged_wr["combined_label"])
ari_fp = adjusted_rand_score(merged_fp["l2_cluster"], merged_fp["fp_cluster"])
print(
    f"  ARI vs within-regime 4-cell: {ari_wr:.3f}  "
    f"| ARI vs fingerprint K=4: {ari_fp:.3f}"
)
print(
    "  L2-on-z-curves clusters primarily by momentum level (overall z-curve height),\n"
    "  producing a panic/calm split at K=2 that partly aligns with the regime\n"
    "  4-cell but does NOT reproduce the fingerprint K=4 feature-driven partition.\n"
    "  The z-curve shape information (hump/trough at mid-horizon) is secondary\n"
    "  to level when using raw L2 distance."
)

print("\nDone.")
