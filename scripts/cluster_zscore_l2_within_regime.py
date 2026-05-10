"""
cluster_zscore_l2_within_regime.py
-----------------------------------
Pure-L2 KMeans on the 12-d cross-section z-curves, conditioned on regime.

Split months into calm (pi_panic=0) and panic (pi_panic=1) FIRST, then run
KMeans(K=2) within each subset.  No standardisation, no PCA, no dispersion
features -- pure Euclidean distance on the raw z-curves.

This is the L2-on-z-curves analogue of picked_stock_within_regime.py.

Outputs (results/thesis/):
  zscore_l2_within_calm_centroids.csv
  zscore_l2_within_panic_centroids.csv
  zscore_l2_within_regime_labels.csv
  zscore_l2_within_regime_vs_fingerprint_within_regime_confusion.csv

Plot (plots/thesis/):
  zscore_l2_within_regime_dispersion.png
  zscore_l2_within_regime_dispersion.pdf
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
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

ZSCORE_CSV          = ROOT / "results/thesis/zscore_long_by_month.csv"
ARTEFACTS_PKL       = ROOT / "artefacts/cs_artefacts_data.pkl"
RETURNS_PKL         = ROOT / "results/thesis/fundamentals_returns.pkl"
FINGERPRINT_WR_CSV  = ROOT / "results/thesis/picked_stock_within_regime_labels.csv"

RES_DIR  = ROOT / "results/thesis"
PLOT_DIR = ROOT / "plots/thesis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS  = list(range(1, 13))
MOM_COLS  = [f"mom_{h}" for h in HORIZONS]

STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
N_INIT          = 20
ARI_THRESHOLD   = 0.85  # spec: require >=0.85 stability
MIN_CELL_SIZE   = 5

BOOT_SZ   = 6
N_BOOT    = 5000
BOOT_SEED = 42

# Cell aesthetics (spec colours)
CELL_STYLE = {
    "calm-l2.0":  dict(color="#1f77b4", label="calm-l2.0 (high-z)"),
    "calm-l2.1":  dict(color="#9ecae1", label="calm-l2.1 (low-z)"),
    "panic-l2.0": dict(color="#ff7f0e", label="panic-l2.0 (high-z)"),
    "panic-l2.1": dict(color="#d62728", label="panic-l2.1 (low-z)"),
}
CELLS = ["calm-l2.0", "calm-l2.1", "panic-l2.0", "panic-l2.1"]

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

# Align to Z-curve dates
pi_aligned  = pi_monthly.reindex(Z.index)
pi_panic    = (pi_aligned > 0.5).astype(int)

n_calm  = int((pi_panic == 0).sum())
n_panic = int((pi_panic == 1).sum())
print(f"  Total months: {N} (calm={n_calm}, panic={n_panic})", flush=True)
assert n_calm + n_panic == N, "SANITY FAIL: calm + panic != total"

# ---------------------------------------------------------------------------
# 3. Load XGB returns
# ---------------------------------------------------------------------------
print("Loading XGB returns ...", flush=True)
with open(RETURNS_PKL, "rb") as fh:
    ret_data = pickle.load(fh)
m2_ret = ret_data["baseline_mom_pi"]["returns"]
m2_ret.index = pd.to_datetime(m2_ret.index)

# ---------------------------------------------------------------------------
# 4. Stability function: 6-seed KMeans ARI
# ---------------------------------------------------------------------------
def run_kmeans_stability(X, k=2):
    """Run KMeans K=2 across STABILITY_SEEDS on raw X (no preprocessing).

    Returns
    -------
    labels_seed42 : ndarray shape (n,)
    mean_ari      : float  -- mean pairwise ARI across all seed pairs
    mean_sil      : float  -- mean silhouette across seeds
    sizes         : list of int (sorted ascending) from seed-42 fit
    """
    all_labels = []
    sils = []
    for s in STABILITY_SEEDS:
        km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(X)
        all_labels.append(km.labels_.copy())
        sils.append(silhouette_score(X, km.labels_))

    pair_aris = []
    for i in range(len(STABILITY_SEEDS)):
        for j in range(i + 1, len(STABILITY_SEEDS)):
            pair_aris.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    mean_ari = float(np.mean(pair_aris))
    mean_sil = float(np.mean(sils))
    sizes    = sorted(np.bincount(all_labels[0], minlength=k).tolist())
    return all_labels[0], mean_ari, mean_sil, sizes


# ---------------------------------------------------------------------------
# 5. Ordering function: sub-cluster 0 = higher-z basket
# ---------------------------------------------------------------------------
def order_by_centroid_mean(labels, X, k=2):
    """Re-map labels so cluster 0 = higher mean centroid across 12 horizons.

    Returns re-labelled integer array of same length as `labels`.
    """
    centroids = np.array([X[labels == cl].mean(axis=0) for cl in range(k)])
    centroid_means = centroids.mean(axis=1)          # mean across 12 horizons
    order = np.argsort(-centroid_means)              # descending
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[int(l)] for l in labels])


# ---------------------------------------------------------------------------
# 6. Sharpe helper
# ---------------------------------------------------------------------------
def compute_sharpe(dates_subset):
    """Block-bootstrap Sharpe for the XGB returns on dates_subset."""
    ret_cl = m2_ret.reindex(dates_subset).dropna()
    bbs = block_bootstrap_sharpe(ret_cl.values, block_size=BOOT_SZ,
                                 n_reps=N_BOOT, seed=BOOT_SEED)
    return bbs


# ---------------------------------------------------------------------------
# 7. Per-regime processing
# ---------------------------------------------------------------------------
results = {}   # "calm" / "panic" -> dict of artefacts

for regime_name, mask_flag in [("calm", 0), ("panic", 1)]:
    regime_mask = (pi_panic == mask_flag).values
    dates_sub = Z.index[regime_mask]
    X_sub     = Z.values[regime_mask]          # shape (n_sub, 12) -- raw, no preprocessing
    n_sub     = int(regime_mask.sum())

    print(f"\n{'='*60}", flush=True)
    print(f"  REGIME: {regime_name.upper()}  (n={n_sub})", flush=True)
    print(f"{'='*60}", flush=True)

    if n_sub < 2 * MIN_CELL_SIZE:
        print(f"  WARNING: too few months ({n_sub}) to split into K=2 "
              f"with >=5 per cell; skipping.", flush=True)
        results[regime_name] = None
        continue

    # K=2 stability on raw z-curves
    labels_raw, mean_ari, mean_sil, sizes = run_kmeans_stability(X_sub, k=2)

    print(f"  KMeans K=2 (pure L2):", flush=True)
    print(f"    silhouette = {mean_sil:.3f}", flush=True)
    print(f"    mean ARI   = {mean_ari:.3f}", flush=True)
    print(f"    sizes (seed 42) = {sizes}", flush=True)

    if mean_ari < ARI_THRESHOLD:
        print(f"  WARNING: mean ARI {mean_ari:.3f} < {ARI_THRESHOLD} -- "
              f"sub-clusters not stable.", flush=True)

    for sz in sizes:
        if sz < MIN_CELL_SIZE:
            print(f"  WARNING: sub-cluster has only {sz} months (<{MIN_CELL_SIZE}); "
                  f"analysis may be unreliable.", flush=True)

    # Order: sub-cluster 0 = higher-z basket
    labels = order_by_centroid_mean(labels_raw, X_sub, k=2)

    # Centroids (mean z-curves per sub-cluster)
    centroid_rows = []
    for cl in [0, 1]:
        cl_mask   = (labels == cl)
        cl_dates  = dates_sub[cl_mask]
        n_cl      = int(cl_mask.sum())
        centroid  = X_sub[cl_mask].mean(axis=0)   # mean z-curve

        bbs = compute_sharpe(cl_dates)
        row = {
            "cluster":     cl,
            **{f"mom_{h}": round(float(centroid[i]), 6) for i, h in enumerate(HORIZONS)},
            "n_months":    n_cl,
            "pi_bar":      float(mask_flag),       # constant within regime
            "sharpe":      round(bbs["sharpe_point"], 4),
            "sharpe_lo95": round(bbs["sharpe_lo95"], 4),
            "sharpe_hi95": round(bbs["sharpe_hi95"], 4),
        }
        centroid_rows.append(row)

        print(f"\n    Sub-cluster {cl}: n={n_cl}  pi_bar={mask_flag}  "
              f"Sharpe={bbs['sharpe_point']:.3f} "
              f"[{bbs['sharpe_lo95']:.3f}, {bbs['sharpe_hi95']:.3f}]",
              flush=True)
        if np.isfinite(bbs["sharpe_point"]) and (bbs["sharpe_point"] < -2 or bbs["sharpe_point"] > 3):
            print(f"  WARNING: Sharpe={bbs['sharpe_point']:.3f} outside [-2, 3]", flush=True)

    centroids_df = pd.DataFrame(centroid_rows)
    centroids_path = RES_DIR / f"zscore_l2_within_{regime_name}_centroids.csv"
    centroids_df.to_csv(centroids_path, index=False)
    print(f"\n  Centroids saved: {centroids_path}", flush=True)

    results[regime_name] = {
        "n":          n_sub,
        "labels":     labels,
        "dates":      dates_sub,
        "pi_val":     mask_flag,
        "centroids":  centroid_rows,
        "mean_ari":   mean_ari,
        "mean_sil":   mean_sil,
        "sizes":      sizes,
        "X_sub":      X_sub,
    }


# ---------------------------------------------------------------------------
# 8. Per-month labels CSV
# ---------------------------------------------------------------------------
print("\nBuilding per-month label CSV ...", flush=True)
label_rows = []
for regime_name, mask_flag in [("calm", 0), ("panic", 1)]:
    r = results.get(regime_name)
    if r is None:
        continue
    for date, sub_cl in zip(r["dates"], r["labels"]):
        label_rows.append({
            "date":           date,
            "pi_panic":       mask_flag,
            "sub_cluster":    int(sub_cl),
            "combined_label": f"{regime_name}-l2.{int(sub_cl)}",
        })

label_df = pd.DataFrame(label_rows).sort_values("date").reset_index(drop=True)
labels_path = RES_DIR / "zscore_l2_within_regime_labels.csv"
label_df.to_csv(labels_path, index=False)
print(f"  Saved: {labels_path}", flush=True)
print(f"  Label distribution:\n{label_df['combined_label'].value_counts().to_string()}", flush=True)


# ---------------------------------------------------------------------------
# 9. All-calm reference centroid (for dashed reference)
# ---------------------------------------------------------------------------
calm_mask   = (pi_panic == 0).values
Z_calm      = Z.values[calm_mask]
ref_centroid = Z_calm.mean(axis=0)   # shape (12,)
print(f"\nAll-calm reference: {len(Z_calm)} months", flush=True)


# ---------------------------------------------------------------------------
# 10. Confusion matrix vs fingerprint within-regime labels
# ---------------------------------------------------------------------------
print("\nBuilding confusion matrix vs fingerprint within-regime ...", flush=True)
fp_wr = pd.read_csv(FINGERPRINT_WR_CSV, parse_dates=["date"])

merged = label_df.merge(
    fp_wr[["date", "combined_label"]].rename(columns={"combined_label": "fp_label"}),
    on="date", how="inner"
)
print(f"  Merged rows for confusion: {len(merged)}", flush=True)

# Rename L2 label column to avoid ambiguity
merged = merged.rename(columns={"combined_label": "l2_label"})

# Cross-tab: rows = L2-z cells, columns = fingerprint cells
fp_cells_expected = ["calm.0", "calm.1", "panic.0", "panic.1"]
conf = pd.crosstab(
    merged["l2_label"],
    merged["fp_label"],
    rownames=["l2_within_regime"],
    colnames=["fingerprint_within_regime"],
)
for c in fp_cells_expected:
    if c not in conf.columns:
        conf[c] = 0
conf = conf[fp_cells_expected]
# Ensure all L2 cells are rows
for cell in CELLS:
    if cell not in conf.index:
        conf.loc[cell] = 0
conf = conf.loc[CELLS]

conf_path = RES_DIR / "zscore_l2_within_regime_vs_fingerprint_within_regime_confusion.csv"
conf.to_csv(conf_path)
print(f"  Saved: {conf_path}", flush=True)
print(conf.to_string(), flush=True)

# ARI between the two 4-cell labellings
ari_vs_fp = adjusted_rand_score(merged["l2_label"], merged["fp_label"])
print(f"\n  ARI (L2-within-regime vs fingerprint-within-regime) = {ari_vs_fp:.4f}", flush=True)


# ---------------------------------------------------------------------------
# 11. Plot
# ---------------------------------------------------------------------------
print("\nGenerating dispersion plot ...", flush=True)

fig, axes = plt.subplots(
    1, 4,
    figsize=(18, 5.5),
    sharey=True,
    constrained_layout=True,
)

# Compute global y-axis limits
all_centroid_vals = list(ref_centroid)
for regime_name, mask_flag in [("calm", 0), ("panic", 1)]:
    r = results.get(regime_name)
    if r is None:
        continue
    for cr in r["centroids"]:
        all_centroid_vals.extend([cr[f"mom_{h}"] for h in HORIZONS])

y_min = round((min(all_centroid_vals) - 0.15) * 4) / 4
y_max = round((max(all_centroid_vals) + 0.15) * 4) / 4
if y_max - y_min < 1.5:
    mid = (y_max + y_min) / 2
    y_min = mid - 0.75
    y_max = mid + 0.75

fig.suptitle(
    "Pure-L2 KMeans on 12-d z-curves, within regime: 2 calm + 2 panic",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

for ax, cell in zip(axes, CELLS):
    style = CELL_STYLE[cell]
    color = style["color"]

    # Parse cell name: "calm-l2.0" -> regime="calm", sub_cl=0
    regime_part, sub_part = cell.rsplit("-l2.", 1)
    sub_cl = int(sub_part)
    regime_flag = 0 if regime_part == "calm" else 1

    r = results.get(regime_part)
    if r is None:
        ax.set_title(f"{cell}\n(NO DATA)")
        continue

    cl_mask  = (r["labels"] == sub_cl)
    cl_dates = r["dates"][cl_mask]
    X_cl     = r["X_sub"][cl_mask]       # shape (n_cl, 12)
    centroid = X_cl.mean(axis=0)
    n_cl     = int(cl_mask.sum())
    pi_bar   = float(regime_flag)         # constant within regime

    cr = r["centroids"][sub_cl]
    sh = cr["sharpe"]
    lo = cr["sharpe_lo95"]
    hi = cr["sharpe_hi95"]
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
        f"{cell}\n"
        f"(n={n_cl}, π̅={pi_bar:.0f})\n"
        f"Sharpe={sharpe_str} [{lo_str}, {hi_str}]",
        fontsize=9.5,
        fontweight="bold",
        loc="center",
    )

    ax.legend(fontsize=7.5, frameon=True, loc="upper left")
    ax.tick_params(axis="both", labelsize=8)

for ext in ("png", "pdf"):
    out_path = PLOT_DIR / f"zscore_l2_within_regime_dispersion.{ext}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)

plt.close(fig)

# ---------------------------------------------------------------------------
# 12. Open PNG
# ---------------------------------------------------------------------------
subprocess.run(["open", str(PLOT_DIR / "zscore_l2_within_regime_dispersion.png")],
               check=False)

# ---------------------------------------------------------------------------
# 13. HEADLINE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("=== L2 within-regime K=2 ===", flush=True)
print("=" * 60, flush=True)

for regime_name, mask_flag in [("calm", 0), ("panic", 1)]:
    r = results.get(regime_name)
    if r is None:
        print(f"\n{regime_name.upper()}: SKIPPED (too few months)", flush=True)
        continue
    print(f"\n{regime_name.upper()} (n={r['n']}):  ARI={r['mean_ari']:.3f},  "
          f"silhouette={r['mean_sil']:.3f}", flush=True)
    for cr in r["centroids"]:
        cl    = cr["cluster"]
        n_cl  = cr["n_months"]
        sh    = cr["sharpe"]
        lo    = cr["sharpe_lo95"]
        hi    = cr["sharpe_hi95"]
        label = "high-z" if cl == 0 else "low-z"
        print(f"  {regime_name}-l2.{cl} ({label}):  n={n_cl}  "
              f"Sharpe={sh:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

print("\nConfusion vs fingerprint within-regime:", flush=True)
print(conf.to_string(), flush=True)
print(f"\nARI = {ari_vs_fp:.4f}", flush=True)

# Interpretation
calm_r   = results.get("calm")
panic_r  = results.get("panic")
calm_aris  = [calm_r["mean_ari"]]  if calm_r  else []
panic_aris = [panic_r["mean_ari"]] if panic_r else []

# Check dominant fp cell per L2 cell to interpret alignment
if len(merged) > 0:
    per_l2 = conf.idxmax(axis=1)
    dominant_fp = per_l2.to_dict()
    # Assess whether L2 cells align with same-regime fp cells or cross regimes
    calm_fp_dominant  = {c: dominant_fp.get(c, "?") for c in ["calm-l2.0", "calm-l2.1"]}
    panic_fp_dominant = {c: dominant_fp.get(c, "?") for c in ["panic-l2.0", "panic-l2.1"]}
    print(f"\n  Dominant fp cell per L2 cell: {dominant_fp}", flush=True)

if ari_vs_fp > 0.5:
    interpretation = (
        "Pure-L2 within-regime clustering largely recovers the fingerprint "
        "within-regime cells, suggesting the z-curve height (level) is the "
        "primary signal even within regimes."
    )
elif ari_vs_fp > 0.2:
    interpretation = (
        "Pure-L2 within-regime clustering partially overlaps with the fingerprint "
        "within-regime cells but finds a different cut -- L2 height and fingerprint "
        "features capture partially distinct structure."
    )
else:
    interpretation = (
        "Pure-L2 within-regime clustering finds a largely different partition than "
        "the fingerprint within-regime cells -- z-curve level alone does not recover "
        "the feature-driven within-regime structure."
    )

print(f"\nInterpretation: {interpretation}", flush=True)

# ---------------------------------------------------------------------------
# 14. Sanity checks
# ---------------------------------------------------------------------------
print("\n--- Sanity checks ---", flush=True)
print(f"  Output CSV exists: {labels_path.exists()}", flush=True)
for regime_name in ["calm", "panic"]:
    cp = RES_DIR / f"zscore_l2_within_{regime_name}_centroids.csv"
    print(f"  Output CSV exists: {cp.exists()}  ({cp.name})", flush=True)
print(f"  Confusion CSV exists: {conf_path.exists()}", flush=True)
for ext in ("png", "pdf"):
    p = PLOT_DIR / f"zscore_l2_within_regime_dispersion.{ext}"
    print(f"  Plot exists: {p.exists()}  ({p.name})", flush=True)

print("\nDone.", flush=True)
