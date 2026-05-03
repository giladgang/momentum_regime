"""
cluster_zscore_l2_k3.py
------------------------
Pure-L2 KMeans K=3 on the full 12-d z-curve panel (no regime pre-split).

Parallel to cluster_zscore_l2_k4.py -- provides a 3-cluster reference
for thesis completeness.

No preprocessing, no PCA, no standardisation -- pure Euclidean distance on
the raw z-curves.

Outputs (results/thesis/):
  zscore_l2_k3_labels.csv
  zscore_l2_k3_centroids.csv  (cluster, 12 z-centroid values, n_months,
                                pi_bar, centroid_mean, sharpe,
                                sharpe_lo95, sharpe_hi95)

Plot (plots/thesis/):
  zscore_l2_k3_dispersion.png
  zscore_l2_k3_dispersion.pdf
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

ZSCORE_CSV    = ROOT / "results/thesis/zscore_long_by_month.csv"
ARTEFACTS_PKL = ROOT / "artefacts/cs_artefacts_data.pkl"
RETURNS_PKL   = ROOT / "results/thesis/fundamentals_returns.pkl"

RES_DIR  = ROOT / "results/thesis"
PLOT_DIR = ROOT / "plots/thesis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS = list(range(1, 13))
MOM_COLS = [f"mom_{h}" for h in HORIZONS]

K            = 3
N_INIT       = 20
RANDOM_STATE = 42

STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
MIN_CELL_SIZE   = 5

BOOT_SZ   = 6
N_BOOT    = 5000
BOOT_SEED = 42

# Cluster aesthetics (spec: 0=light blue, 1=orange, 2=red)
CLUSTER_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#d62728",
}

# ---------------------------------------------------------------------------
# Helpers path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_helpers import block_bootstrap_sharpe  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Load z-score matrix
# ---------------------------------------------------------------------------
print("Loading z-score matrix ...", flush=True)
Z_raw = pd.read_csv(ZSCORE_CSV, parse_dates=["date"]).set_index("date")
Z_raw = Z_raw[MOM_COLS].dropna()
Z = Z_raw.copy()
X = Z.values    # shape (N, 12) -- pure Euclidean L2, no preprocessing
N = len(Z)
print(f"  N = {N} months after dropping NaN rows", flush=True)

# ---------------------------------------------------------------------------
# 2. Load artefacts -> pi_panic per month
# ---------------------------------------------------------------------------
print("Loading artefacts ...", flush=True)
with open(ARTEFACTS_PKL, "rb") as fh:
    art = pickle.load(fh)
test_df = art["test"].copy()
test_df["date"] = pd.to_datetime(test_df["date"])
pi_monthly = test_df.groupby("date")["pi_filter"].first()

pi_aligned = pi_monthly.reindex(Z.index)
pi_panic   = (pi_aligned > 0.5).astype(int)

n_calm  = int((pi_panic == 0).sum())
n_panic = int((pi_panic == 1).sum())
print(f"  Total months: {N} (calm={n_calm}, panic={n_panic})", flush=True)
assert n_calm + n_panic == N, "SANITY FAIL: calm + panic != total"

# ---------------------------------------------------------------------------
# 3. Load M2 returns
# ---------------------------------------------------------------------------
print("Loading M2 returns ...", flush=True)
with open(RETURNS_PKL, "rb") as fh:
    ret_data = pickle.load(fh)
m2_ret = ret_data["baseline_mom_pi"]["returns"]
m2_ret.index = pd.to_datetime(m2_ret.index)

# ---------------------------------------------------------------------------
# 4. Primary KMeans fit (seed=42)
# ---------------------------------------------------------------------------
print(f"\nFitting KMeans K={K} (seed=42, n_init={N_INIT}) ...", flush=True)
km_primary = KMeans(n_clusters=K, n_init=N_INIT, random_state=RANDOM_STATE)
km_primary.fit(X)
labels_raw = km_primary.labels_.copy()

sizes_raw = sorted(np.bincount(labels_raw, minlength=K).tolist())
print(f"  Cluster sizes (sorted, seed 42): {sizes_raw}", flush=True)
print(f"  Inertia: {km_primary.inertia_:.2f}", flush=True)

# ---------------------------------------------------------------------------
# 5. Ordering: cluster 0 = highest mean centroid across 12 horizons (desc)
# ---------------------------------------------------------------------------
def order_clusters_by_centroid_mean(labels_in, X_in, k=K):
    """Re-map cluster labels so cluster 0 = highest overall z (descending)."""
    centroids = np.array([X_in[labels_in == cl].mean(axis=0) for cl in range(k)])
    centroid_means = centroids.mean(axis=1)     # scalar per cluster
    order = np.argsort(-centroid_means)         # descending
    remap = {int(old): int(new) for new, old in enumerate(order)}
    relabelled = np.array([remap[int(l)] for l in labels_in])
    return relabelled, remap

labels_ordered, remap_primary = order_clusters_by_centroid_mean(labels_raw, X, K)
sizes_ordered = [int((labels_ordered == k_idx).sum()) for k_idx in range(K)]
print(f"  Cluster sizes (ordered by centroid_mean desc): {sizes_ordered}", flush=True)

# Sanity: each cluster >= MIN_CELL_SIZE
for k_idx, sz in enumerate(sizes_ordered):
    if sz < MIN_CELL_SIZE:
        print(f"  WARNING: cluster {k_idx} has only {sz} months (<{MIN_CELL_SIZE})!", flush=True)

# ---------------------------------------------------------------------------
# 6. 6-seed stability check
# ---------------------------------------------------------------------------
print("\nRunning 6-seed stability check ...", flush=True)
all_labels_seeds = []
for seed in STABILITY_SEEDS:
    km_s = KMeans(n_clusters=K, n_init=N_INIT, random_state=seed).fit(X)
    labels_s_raw = km_s.labels_.copy()
    labels_s_ord, _ = order_clusters_by_centroid_mean(labels_s_raw, X, K)
    all_labels_seeds.append(labels_s_ord)
    print(f"  seed={seed}: sizes={sorted(np.bincount(labels_s_raw, minlength=K).tolist())}", flush=True)

pair_aris = []
for i in range(len(STABILITY_SEEDS)):
    for j in range(i + 1, len(STABILITY_SEEDS)):
        pair_aris.append(adjusted_rand_score(all_labels_seeds[i], all_labels_seeds[j]))

mean_ari_stability = float(np.mean(pair_aris))
print(f"\n  Mean pairwise ARI across {len(pair_aris)} seed pairs: {mean_ari_stability:.4f}", flush=True)
if mean_ari_stability < 0.85:
    print(f"  WARNING: stability ARI {mean_ari_stability:.3f} < 0.85 threshold!", flush=True)

# ---------------------------------------------------------------------------
# 7. All-calm reference centroid (for dashed reference)
# ---------------------------------------------------------------------------
calm_mask    = (pi_panic == 0).values
ref_centroid = X[calm_mask].mean(axis=0)
print(f"\nAll-calm reference: {calm_mask.sum()} months", flush=True)

# ---------------------------------------------------------------------------
# 8. Sharpe helper
# ---------------------------------------------------------------------------
def compute_sharpe(dates_subset):
    """Block-bootstrap Sharpe for M2 returns on dates_subset."""
    ret_cl = m2_ret.reindex(dates_subset).dropna()
    bbs = block_bootstrap_sharpe(ret_cl.values, block_size=BOOT_SZ,
                                 n_reps=N_BOOT, seed=BOOT_SEED)
    return bbs

# ---------------------------------------------------------------------------
# 9. Per-cluster statistics and centroids
# ---------------------------------------------------------------------------
print("\nComputing per-cluster statistics ...", flush=True)
centroid_rows = []

for cl in range(K):
    cl_mask  = (labels_ordered == cl)
    cl_dates = Z.index[cl_mask]
    X_cl     = X[cl_mask]
    n_cl     = int(cl_mask.sum())
    centroid = X_cl.mean(axis=0)        # mean z-curve (12 values)
    centroid_mean = float(centroid.mean())

    pi_bar = float(pi_panic.values[cl_mask].mean())

    bbs = compute_sharpe(cl_dates)
    sh  = round(bbs["sharpe_point"], 4)
    lo  = round(bbs["sharpe_lo95"],  4)
    hi  = round(bbs["sharpe_hi95"],  4)

    row = {
        "cluster":       cl,
        **{f"mom_{h}": round(float(centroid[i]), 6) for i, h in enumerate(HORIZONS)},
        "centroid_mean": round(centroid_mean, 6),
        "n_months":      n_cl,
        "pi_bar":        round(pi_bar, 4),
        "sharpe":        sh,
        "sharpe_lo95":   lo,
        "sharpe_hi95":   hi,
    }
    centroid_rows.append(row)

    print(f"  Cluster {cl}: n={n_cl}  pi_bar={pi_bar:.3f}  "
          f"centroid_mean={centroid_mean:.4f}  "
          f"Sharpe={sh:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

    if np.isfinite(sh) and (sh < -2 or sh > 3):
        print(f"  WARNING: Sharpe={sh:.3f} outside [-2, 3]", flush=True)

centroids_df = pd.DataFrame(centroid_rows)
centroids_path = RES_DIR / "zscore_l2_k3_centroids.csv"
centroids_df.to_csv(centroids_path, index=False)
print(f"\n  Centroids saved: {centroids_path}", flush=True)

# ---------------------------------------------------------------------------
# 10. Labels CSV
# ---------------------------------------------------------------------------
print("\nBuilding labels CSV ...", flush=True)
label_rows = []
for date, cl in zip(Z.index, labels_ordered):
    label_rows.append({"date": date, "cluster": int(cl)})

label_df = pd.DataFrame(label_rows).sort_values("date").reset_index(drop=True)
labels_path = RES_DIR / "zscore_l2_k3_labels.csv"
label_df.to_csv(labels_path, index=False)
print(f"  Saved: {labels_path}", flush=True)
print(f"  Label distribution:\n{label_df['cluster'].value_counts().sort_index().to_string()}", flush=True)

# ---------------------------------------------------------------------------
# 11. Plot: 3-panel dispersion
# ---------------------------------------------------------------------------
print("\nGenerating dispersion plot ...", flush=True)

fig, axes = plt.subplots(
    1, 3,
    figsize=(13.5, 5.5),
    sharey=True,
    constrained_layout=True,
)

# Global y-axis limits
all_z_vals = list(ref_centroid) + [
    cr[f"mom_{h}"] for cr in centroid_rows for h in HORIZONS
]
y_min = round((min(all_z_vals) - 0.15) * 4) / 4
y_max = round((max(all_z_vals) + 0.15) * 4) / 4
if y_max - y_min < 1.5:
    mid = (y_max + y_min) / 2
    y_min = mid - 0.75
    y_max = mid + 0.75

fig.suptitle(
    "L2 KMeans K=3 on 12-d z-curves (no regime pre-split)",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

for ax, cl in zip(axes, range(K)):
    color   = CLUSTER_COLORS[cl]
    cr      = centroid_rows[cl]
    n_cl    = cr["n_months"]
    pi_bar  = cr["pi_bar"]
    sh      = cr["sharpe"]
    lo      = cr["sharpe_lo95"]
    hi      = cr["sharpe_hi95"]

    cl_mask  = (labels_ordered == cl)
    X_cl     = X[cl_mask]
    centroid = X_cl.mean(axis=0)

    sharpe_str = f"{sh:.2f}" if np.isfinite(sh) else "n/a"
    lo_str     = f"{lo:.2f}" if np.isfinite(lo) else "?"
    hi_str     = f"{hi:.2f}" if np.isfinite(hi) else "?"

    # Thin per-month lines
    for row in X_cl:
        ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.4)

    # Bold centroid
    ax.plot(
        HORIZONS, centroid,
        color=color, lw=2.5, marker="o", ms=6,
        label=f"centroid ({n_cl} mo)",
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
        f"(n={n_cl}, π̅={pi_bar:.2f})\n"
        f"Sharpe={sharpe_str} [{lo_str}, {hi_str}]",
        fontsize=9.5,
        fontweight="bold",
        loc="center",
    )

    ax.legend(fontsize=7.5, frameon=True, loc="upper left")
    ax.tick_params(axis="both", labelsize=8)

for ext in ("png", "pdf"):
    out_path = PLOT_DIR / f"zscore_l2_k3_dispersion.{ext}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)

plt.close(fig)

# ---------------------------------------------------------------------------
# 12. Open PNG
# ---------------------------------------------------------------------------
subprocess.run(["open", str(PLOT_DIR / "zscore_l2_k3_dispersion.png")], check=False)

# ---------------------------------------------------------------------------
# 13. HEADLINE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("=== L2 K=3 aggregate ===", flush=True)
print("=" * 60, flush=True)
print(f"ARI stability: {mean_ari_stability:.4f} (mean pairwise across {len(STABILITY_SEEDS)} seeds)", flush=True)
print(f"\nPer cluster:", flush=True)
for cr in centroid_rows:
    cl  = cr["cluster"]
    n   = cr["n_months"]
    pi  = cr["pi_bar"]
    cm  = cr["centroid_mean"]
    sh  = cr["sharpe"]
    lo  = cr["sharpe_lo95"]
    hi  = cr["sharpe_hi95"]
    print(f"  cluster {cl}: n={n}  pi_bar={pi:.3f}  centroid_mean={cm:.4f}  "
          f"Sharpe={sh:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

# ---------------------------------------------------------------------------
# 14. Sanity checks
# ---------------------------------------------------------------------------
print("\n--- Sanity checks ---", flush=True)
print(f"  Labels CSV exists: {labels_path.exists()}", flush=True)
print(f"  Centroids CSV exists: {centroids_path.exists()}", flush=True)
for ext in ("png", "pdf"):
    p = PLOT_DIR / f"zscore_l2_k3_dispersion.{ext}"
    print(f"  Plot exists: {p.exists()}  ({p.name})", flush=True)
for k_idx, sz in enumerate(sizes_ordered):
    chk = "OK" if sz >= MIN_CELL_SIZE else "FAIL"
    print(f"  Cluster {k_idx} size={sz} >= {MIN_CELL_SIZE}: {chk}", flush=True)
print(f"  Stability ARI={mean_ari_stability:.4f} >= 0.85: "
      f"{'OK' if mean_ari_stability >= 0.85 else 'WARN'}", flush=True)

print("\nDone.", flush=True)
