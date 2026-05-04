"""
plot_cluster_k4_individual_panels.py
------------------------------------
Generates four standalone per-cluster z-curve dispersion panels, one PNG/PDF
per cluster, for embedding directly above each cluster's prose paragraph in
latex/main_results.tex.

Reuses the same z-curve data, labels, and aesthetics as
scripts/cluster_zscore_l2_k4.py (the existing 4-panel figure generator).

Inputs:
    results/thesis/zscore_long_by_month.csv
    results/thesis/zscore_l2_k4_labels.csv
    results/thesis/zscore_l2_k4_centroids.csv
    artefacts/cs_artefacts_data.pkl  (for pi_panic -> all-calm reference)

Outputs (plots/thesis/):
    zscore_l2_k4_panel_c0.{png,pdf}
    zscore_l2_k4_panel_c1.{png,pdf}
    zscore_l2_k4_panel_c2.{png,pdf}
    zscore_l2_k4_panel_c3.{png,pdf}
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ZSCORE_CSV    = ROOT / "results/thesis/zscore_long_by_month.csv"
LABELS_CSV    = ROOT / "results/thesis/zscore_l2_k4_labels.csv"
CENTROIDS_CSV = ROOT / "results/thesis/zscore_l2_k4_centroids.csv"
PLOT_DIR = ROOT / "plots/thesis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = list(range(1, 13))
MOM_COLS = [f"mom_{h}" for h in HORIZONS]
K        = 4

CLUSTER_COLORS = {
    0: "#1f6dad",
    1: "#26a69a",
    2: "#e67a00",
    3: "#d62728",
}

CLUSTER_TITLES = {
    0: "Cluster 1 (calm continuation)",
    1: "Cluster 2 (mild continuation)",
    2: "Cluster 3 (post-panic recovery)",
    3: "Cluster 4 (deep crisis)",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading z-score matrix ...", flush=True)
Z = pd.read_csv(ZSCORE_CSV, parse_dates=["date"]).set_index("date")
Z = Z[MOM_COLS].dropna()
X = Z.values

print("Loading cluster labels ...", flush=True)
labels_df = pd.read_csv(LABELS_CSV, parse_dates=["date"]).set_index("date")
labels = labels_df.reindex(Z.index)["cluster"].astype(int).values

print("Loading centroids ...", flush=True)
centroids_df = pd.read_csv(CENTROIDS_CSV)

# ---------------------------------------------------------------------------
# Global y-limits (consistent across panels for visual comparability)
# ---------------------------------------------------------------------------
all_z_vals = list(X.flatten())
y_min = round((min(all_z_vals) - 0.15) * 4) / 4
y_max = round((max(all_z_vals) + 0.15) * 4) / 4

# ---------------------------------------------------------------------------
# Render one panel per cluster
# ---------------------------------------------------------------------------
for cl in range(K):
    cl_mask = (labels == cl)
    X_cl    = X[cl_mask]
    n_cl    = int(cl_mask.sum())
    centroid = X_cl.mean(axis=0)

    cr = centroids_df[centroids_df["cluster"] == cl].iloc[0]
    pi_bar = float(cr["pi_bar"])
    sh     = float(cr["sharpe"])
    lo     = float(cr["sharpe_lo95"])
    hi     = float(cr["sharpe_hi95"])

    color = CLUSTER_COLORS[cl]

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)

    for row in X_cl:
        ax.plot(HORIZONS, row, color=color, lw=0.8, alpha=0.4)

    ax.plot(
        HORIZONS, centroid,
        color=color, lw=2.5, marker="o", ms=6,
        label=f"centroid ({n_cl} mo)",
        zorder=5,
    )

    ax.axhline(0, color="#888888", lw=0.6, zorder=1)

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Horizon (months)", fontsize=10)
    ax.set_ylabel("Cross-sectional z-score", fontsize=10)

    sharpe_str = f"{sh:.2f}" if np.isfinite(sh) else "n/a"
    lo_str     = f"{lo:.2f}" if np.isfinite(lo) else "?"
    hi_str     = f"{hi:.2f}" if np.isfinite(hi) else "?"
    ax.set_title(
        f"{CLUSTER_TITLES[cl]} -- "
        f"n={n_cl}, " + r"$\bar\pi=$" + f"{pi_bar:.2f}, "
        f"Sharpe={sharpe_str} [{lo_str}, {hi_str}]",
        fontsize=10.5,
        fontweight="bold",
    )

    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    ax.tick_params(axis="both", labelsize=9)

    for ext in ("png", "pdf"):
        out = PLOT_DIR / f"zscore_l2_k4_panel_c{cl}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}", flush=True)
    plt.close(fig)

print("Done.", flush=True)
