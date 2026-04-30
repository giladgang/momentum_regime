"""
plot_calm0_vs_panic0_dispersion.py
-----------------------------------
2-panel side-by-side comparison of z-score dispersion curves for the two
largest "mom-up" sub-clusters: calm.0 (left) and panic.0 (right).

Outputs:
    plots/thesis/calm0_vs_panic0_dispersion.png
    plots/thesis/calm0_vs_panic0_dispersion.pdf
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing project helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from bootstrap_helpers import block_bootstrap_sharpe

# ---------------------------------------------------------------------------
# Paths (relative to repo root; run from repo root)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABELS_CSV   = os.path.join(ROOT, "results/thesis/picked_stock_within_regime_labels.csv")
ZSCORE_CSV   = os.path.join(ROOT, "results/thesis/zscore_long_by_month.csv")
ART_PKL      = os.path.join(ROOT, "artefacts/cs_artefacts_data.pkl")
RETURNS_PKL  = os.path.join(ROOT, "results/thesis/fundamentals_returns.pkl")
OUT_PNG      = os.path.join(ROOT, "plots/thesis/calm0_vs_panic0_dispersion.png")
OUT_PDF      = os.path.join(ROOT, "plots/thesis/calm0_vs_panic0_dispersion.pdf")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
labels = pd.read_csv(LABELS_CSV, parse_dates=["date"])
zdf    = pd.read_csv(ZSCORE_CSV, parse_dates=["date"])

# Horizon columns
hz_cols = [f"mom_{i}" for i in range(1, 13)]
horizons = np.arange(1, 13)

# Merge labels into zscore frame
zdf = zdf.merge(labels[["date", "pi_panic", "sub_cluster", "combined_label"]],
                on="date", how="left")

# ---------------------------------------------------------------------------
# All-calm reference: all months where pi_panic == 0
# ---------------------------------------------------------------------------
calm_mask = zdf["pi_panic"] == 0
all_calm_z = zdf.loc[calm_mask, hz_cols].values          # shape (n_calm, 12)
all_calm_centroid = all_calm_z.mean(axis=0)

# ---------------------------------------------------------------------------
# Sub-cluster selections
# ---------------------------------------------------------------------------
calm0_mask  = zdf["combined_label"] == "calm.0"
panic0_mask = zdf["combined_label"] == "panic.0"

calm0_z  = zdf.loc[calm0_mask,  hz_cols].values
panic0_z = zdf.loc[panic0_mask, hz_cols].values

calm0_centroid  = calm0_z.mean(axis=0)
panic0_centroid = panic0_z.mean(axis=0)

n_calm0  = calm0_z.shape[0]
n_panic0 = panic0_z.shape[0]

# pi_bar per cluster (mean pi_panic value = 0 for calm.0, 1 for panic.0;
# more useful: mean of pi_filter from the labels file -- but labels only has
# pi_panic as int flag.  Use that directly.)
pi_bar_calm0  = float(zdf.loc[calm0_mask,  "pi_panic"].mean())
pi_bar_panic0 = float(zdf.loc[panic0_mask, "pi_panic"].mean())

# ---------------------------------------------------------------------------
# Monthly returns for Sharpe computation
# ---------------------------------------------------------------------------
with open(RETURNS_PKL, "rb") as fh:
    fund_d = pickle.load(fh)

all_returns = fund_d["baseline_mom_pi"]["returns"]   # pd.Series indexed by date

def get_cluster_returns(mask_series):
    """Return the monthly strategy returns for the months in mask_series."""
    dates = zdf.loc[mask_series, "date"]
    dates = pd.to_datetime(dates)
    idx = all_returns.index
    common = idx[idx.isin(dates)]
    return all_returns.loc[common].values

rs_calm0  = get_cluster_returns(calm0_mask)
rs_panic0 = get_cluster_returns(panic0_mask)

bstrap_calm0  = block_bootstrap_sharpe(rs_calm0,  block_size=6, n_reps=5000, seed=42)
bstrap_panic0 = block_bootstrap_sharpe(rs_panic0, block_size=6, n_reps=5000, seed=42)

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
print(f"calm.0  : n={n_calm0}, pi_bar={pi_bar_calm0:.2f}, "
      f"Sharpe={bstrap_calm0['sharpe_point']:.2f} "
      f"[{bstrap_calm0['sharpe_lo95']:.2f}, {bstrap_calm0['sharpe_hi95']:.2f}]")
print(f"panic.0 : n={n_panic0}, pi_bar={pi_bar_panic0:.2f}, "
      f"Sharpe={bstrap_panic0['sharpe_point']:.2f} "
      f"[{bstrap_panic0['sharpe_lo95']:.2f}, {bstrap_panic0['sharpe_hi95']:.2f}]")
assert n_calm0  > 0, "calm.0 is empty"
assert n_panic0 > 0, "panic.0 is empty"

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
COLOR_CALM0  = "#1f77b4"
COLOR_PANIC0 = "#ff7f0e"
COLOR_REF    = "#333333"
COLOR_ZERO   = "#888888"

def title_str(cell, n, pi_bar, bs):
    s     = bs["sharpe_point"]
    lo    = bs["sharpe_lo95"]
    hi    = bs["sharpe_hi95"]
    return (f"{cell}\n"
            f"(n={n}, π̅={pi_bar:.2f})\n"
            f"Sharpe={s:.2f} [{lo:.2f}, {hi:.2f}]")

def draw_panel(ax, z_matrix, centroid, cell_color, cell_name, n, pi_bar, bs):
    """Draw one panel: thin monthly lines + bold centroid + reference."""
    # Thin lines for each month
    for row in z_matrix:
        ax.plot(horizons, row, color=cell_color, lw=0.8, alpha=0.4)

    # Bold centroid
    ax.plot(horizons, centroid, color=cell_color, lw=2.5,
            marker="o", ms=6, label=f"{cell_name} centroid")

    # Dashed reference (all-calm)
    ax.plot(horizons, all_calm_centroid, color=COLOR_REF, lw=1.5, ls="--",
            label="All-calm centroid (reference)")

    # y=0 line
    ax.axhline(0, color=COLOR_ZERO, lw=0.6)

    # Formatting
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Horizon (months)", fontsize=10)
    ax.set_title(title_str(cell_name, n, pi_bar, bs), fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=True, loc="upper left")

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)

draw_panel(axes[0], calm0_z,  calm0_centroid,
           COLOR_CALM0,  "calm.0",
           n_calm0,  pi_bar_calm0,  bstrap_calm0)

draw_panel(axes[1], panic0_z, panic0_centroid,
           COLOR_PANIC0, "panic.0",
           n_panic0, pi_bar_panic0, bstrap_panic0)

# Shared y-axis label only on the left
axes[0].set_ylabel("Cross-sectional z-score (long leg)", fontsize=10)

# Auto-scale is usually fine; clamp if needed
for ax in axes:
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(min(lo_y, -1.5), max(hi_y, 1.0))

fig.suptitle(
    'Same "mom-up" basket type, different regime: calm.0 vs panic.0',
    fontsize=12, y=1.01
)
fig.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
fig.savefig(OUT_PDF,           bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
