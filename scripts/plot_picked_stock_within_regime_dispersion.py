"""
plot_picked_stock_within_regime_dispersion.py
---------------------------------------------
4-panel z-score dispersion plot for the within-regime sub-clusters:
  calm.0, calm.1, panic.0, panic.1

Inputs
------
results/thesis/picked_stock_within_regime_labels.csv
results/thesis/zscore_long_by_month.csv
results/thesis/fundamentals_returns.pkl  (key: baseline_mom_pi -> returns)

Outputs
-------
plots/thesis/picked_stock_within_regime_dispersion.png
plots/thesis/picked_stock_within_regime_dispersion.pdf
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV  = ROOT / "results/thesis/picked_stock_within_regime_labels.csv"
ZSCORE_CSV  = ROOT / "results/thesis/zscore_long_by_month.csv"
RETURNS_PKL = ROOT / "results/thesis/fundamentals_returns.pkl"
OUT_DIR     = ROOT / "plots/thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load labels
# ---------------------------------------------------------------------------
labels = pd.read_csv(LABELS_CSV, parse_dates=["date"])
print("Labels columns:", list(labels.columns))
print("First 3 rows:")
print(labels.head(3).to_string())
print()

# Confirm expected columns exist
for col in ["date", "pi_panic", "sub_cluster", "combined_label"]:
    if col not in labels.columns:
        sys.exit(f"ERROR: expected column '{col}' not found in labels CSV")

# ---------------------------------------------------------------------------
# 2. Build cell_to_dates
# ---------------------------------------------------------------------------
CELLS = ["calm.0", "calm.1", "panic.0", "panic.1"]

cell_to_dates = {}
for cell in CELLS:
    rows = labels[labels["combined_label"] == cell]
    if rows.empty:
        sys.exit(f"ERROR: cell '{cell}' is empty — check labels CSV")
    cell_to_dates[cell] = sorted(rows["date"].tolist())
    print(f"  {cell}: {len(cell_to_dates[cell])} months")

print()

# ---------------------------------------------------------------------------
# 3. Load z-score panel
# ---------------------------------------------------------------------------
Z = pd.read_csv(ZSCORE_CSV, parse_dates=["date"]).set_index("date")
horizons = list(range(1, 13))   # 1..12

# ---------------------------------------------------------------------------
# 4. All-calm reference (pi_panic == 0)
# ---------------------------------------------------------------------------
calm_dates = labels[labels["pi_panic"] == 0]["date"].tolist()
Z_calm = Z.reindex(calm_dates).dropna()
ref_centroid = Z_calm.mean(axis=0).values   # shape (12,)
print(f"All-calm reference: {len(Z_calm)} months used")
print()

# ---------------------------------------------------------------------------
# 5. Load XGB returns for per-cell Sharpe
# ---------------------------------------------------------------------------
with open(RETURNS_PKL, "rb") as fh:
    ret_data = pickle.load(fh)

m2_ret = ret_data["baseline_mom_pi"]["returns"]   # pd.Series indexed by date
m2_ret.index = pd.to_datetime(m2_ret.index)

def annualised_sharpe(dates: list) -> float:
    """Monthly returns → annualised Sharpe (mean/std * sqrt(12))."""
    r = m2_ret.reindex(dates).dropna()
    if len(r) < 2:
        return float("nan")
    return (r.mean() / r.std(ddof=1)) * np.sqrt(12)

# ---------------------------------------------------------------------------
# 6. Cell aesthetics
# ---------------------------------------------------------------------------
CELL_STYLE = {
    "calm.0":  dict(color="#1f77b4", label="calm.0 (mom-up)"),
    "calm.1":  dict(color="#9ecae1", label="calm.1 (mom-down)"),
    "panic.0": dict(color="#ff7f0e", label="panic.0 (mom-up)"),
    "panic.1": dict(color="#d62728", label="panic.1 (mom-down)"),
}

# ---------------------------------------------------------------------------
# 7. Compute per-cell centroids and Sharpes
# ---------------------------------------------------------------------------
cell_info = {}
for cell in CELLS:
    dates = cell_to_dates[cell]
    Z_cell = Z.reindex(dates).dropna()
    centroid = Z_cell.mean(axis=0).values
    sharpe = annualised_sharpe(dates)
    pi_bar = labels[labels["combined_label"] == cell]["pi_panic"].mean()
    cell_info[cell] = dict(
        dates=dates,
        Z_cell=Z_cell,
        centroid=centroid,
        sharpe=sharpe,
        pi_bar=pi_bar,
        n=len(Z_cell),
    )
    print(f"  {cell}: n={len(Z_cell)}, pi_bar={pi_bar:.2f}, Sharpe={sharpe:.2f}")

print()

# ---------------------------------------------------------------------------
# 8. Build y-axis limits (include all centroids + reference)
# ---------------------------------------------------------------------------
all_vals = list(ref_centroid)
for cell in CELLS:
    all_vals.extend(cell_info[cell]["centroid"].tolist())

y_min = min(all_vals) - 0.15
y_max = max(all_vals) + 0.15
# round to neat limits
y_min = round(y_min * 4) / 4
y_max = round(y_max * 4) / 4
# ensure a minimum span
if y_max - y_min < 1.5:
    mid = (y_max + y_min) / 2
    y_min = mid - 0.75
    y_max = mid + 0.75

print(f"y-axis limits: ({y_min:.2f}, {y_max:.2f})")
print()

# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 4,
    figsize=(4.5 * 4, 5.5),
    sharey=True,
    constrained_layout=True,
)

fig.suptitle(
    "Picked-stock z-curves by sub-cluster (calm/panic × mom-up/down)",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)

for ax, cell in zip(axes, CELLS):
    info = cell_info[cell]
    style = CELL_STYLE[cell]
    color = style["color"]

    # Thin per-month lines
    for _, row in info["Z_cell"].iterrows():
        ax.plot(horizons, row.values, color=color, lw=0.8, alpha=0.4)

    # Bold centroid
    ax.plot(
        horizons, info["centroid"],
        color=color, lw=2.5, marker="o", ms=6,
        label=f"centroid ({info['n']} mo)",
        zorder=5,
    )

    # Dashed all-calm reference
    ax.plot(
        horizons, ref_centroid,
        color="#333333", lw=1.5, ls="--",
        label="all-calm ref",
        zorder=4,
    )

    # y=0 line
    ax.axhline(0, color="#888888", lw=0.6, zorder=1)

    # Axes formatting
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon (months)", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("Cross-sectional z-score", fontsize=9)

    # Title
    sharpe_str = f"{info['sharpe']:.2f}" if not np.isnan(info["sharpe"]) else "n/a"
    ax.set_title(
        f"{cell}\n"
        f"(n={info['n']}, π̅={info['pi_bar']:.2f})\n"
        f"Sharpe={sharpe_str}",
        fontsize=9.5,
        fontweight="bold",
        loc="center",
    )

    ax.legend(fontsize=7.5, frameon=True, loc="upper left")
    ax.tick_params(axis="both", labelsize=8)

# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
for ext in ("png", "pdf"):
    out_path = OUT_DIR / f"picked_stock_within_regime_dispersion.{ext}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------
print()
print("=== Panel summary ===")
for cell in CELLS:
    info = cell_info[cell]
    sharpe_str = f"{info['sharpe']:.2f}" if not np.isnan(info["sharpe"]) else "n/a"
    print(f"  {cell:8s}: n={info['n']:3d}, pi_bar={info['pi_bar']:.2f}, Sharpe={sharpe_str}")
