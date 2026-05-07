"""
plot_zscore_long_heatmap.py
===========================
Heatmap of raw long-leg z-curves across all 167 test-period months.

Purpose: motivate the clustering analysis by showing that the raw 167-month
z-curve landscape is hard to read patterns from without structure imposed.

Inputs:
  results/thesis/zscore_long_by_month.csv  (167 months x 12 horizons)
  results/thesis/zscore_l2_k4_labels.csv   (used ONLY to verify row count)

Outputs:
  plots/thesis/zscore_long_heatmap.png  (300 dpi)
  plots/thesis/zscore_long_heatmap.pdf

Design:
  - Single subplot, figsize ~ (10, 12)
  - X-axis: 12 horizons (mom_1 to mom_12)
  - Y-axis: 167 dates, chronological (oldest top, newest bottom), labels every 12m
  - Colour: RdBu_r (divergent), symmetric range based on data |max|
  - Colourbar labelled "long-leg cross-sectional z-score"
  - Title: "Long-leg z-curves over the test period (167 months)"
  - No row dendrogram, no clustering colouring

Usage:
    python scripts/plot_zscore_long_heatmap.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ZSCORE_CSV  = ROOT / "results" / "thesis" / "zscore_long_by_month.csv"
LABELS_CSV  = ROOT / "results" / "thesis" / "zscore_l2_k4_labels.csv"
OUT_DIR     = ROOT / "plots" / "thesis"
OUT_PNG     = OUT_DIR / "zscore_long_heatmap.png"
OUT_PDF     = OUT_DIR / "zscore_long_heatmap.pdf"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(ZSCORE_CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

labels = pd.read_csv(LABELS_CSV, parse_dates=["date"])

# ── Verification: row counts ──────────────────────────────────────────────────
n_rows  = len(df)
n_labels = len(labels)
print(f"zscore_long_by_month rows : {n_rows}")
print(f"zscore_l2_k4_labels rows  : {n_labels}")

assert n_rows == 167, f"Expected 167 rows, got {n_rows}"
# labels may have 167 or 168 rows depending on whether header counted; verify dates align
common = set(df["date"].dt.date) & set(labels["date"].dt.date)
print(f"Dates in common           : {len(common)}  (expected 167)")

# ── Build matrix (167 x 12) ───────────────────────────────────────────────────
horizon_cols = [f"mom_{h}" for h in range(1, 13)]
Z = df[horizon_cols].values.astype(float)   # (167, 12)

dates       = df["date"].tolist()
date_labels = [d.strftime("%Y-%m") for d in dates]

# ── Colour scale: symmetric around 0 ─────────────────────────────────────────
abs_max = np.nanmax(np.abs(Z))
vmin, vmax = -abs_max, abs_max
print(f"Z range: [{Z.min():.3f}, {Z.max():.3f}]  |max|={abs_max:.3f}  vmin/vmax=±{abs_max:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 12))

im = ax.imshow(
    Z,
    aspect="auto",
    origin="upper",           # oldest at top (row 0)
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    interpolation="nearest",
)

# ── X-axis: 12 horizons ───────────────────────────────────────────────────────
ax.set_xticks(range(12))
ax.set_xticklabels([f"mom_{h}" for h in range(1, 13)], fontsize=9, rotation=45, ha="right")
ax.set_xlabel("Momentum lookback horizon", fontsize=11)

# ── Y-axis: dates every 12 months ────────────────────────────────────────────
tick_positions = list(range(0, n_rows, 12))
tick_labels    = [date_labels[i] for i in tick_positions]
ax.yaxis.set_major_locator(FixedLocator(tick_positions))
ax.set_yticklabels(tick_labels, fontsize=8)
ax.set_ylabel("Month (chronological)", fontsize=11)

# ── Colourbar ─────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("long-leg cross-sectional z-score", fontsize=10)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(f"Long-leg z-curves over the test period ({n_rows} months)",
             fontsize=13, fontweight="bold", pad=12)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {OUT_PNG}  ({OUT_PNG.stat().st_size:,} bytes)")
print(f"Saved: {OUT_PDF}  ({OUT_PDF.stat().st_size:,} bytes)")

# ── Verification ──────────────────────────────────────────────────────────────
print("\n=== VERIFICATION (heatmap) ===")

v_rows = (n_rows == 167)
print(f"[H1] 167 rows (months) in data: {n_rows}  [{'PASS' if v_rows else 'FAIL'}]")

# Date axis: first date < last date (chronological)
v_chron = dates[0] < dates[-1]
print(f"[H2] Date axis chronological ({dates[0].date()} < {dates[-1].date()}): "
      f"[{'PASS' if v_chron else 'FAIL'}]")

v_png = OUT_PNG.stat().st_size > 0
v_pdf = OUT_PDF.stat().st_size > 0
print(f"[H3] PNG file size > 0: {OUT_PNG.stat().st_size:,} bytes  [{'PASS' if v_png else 'FAIL'}]")
print(f"[H4] PDF file size > 0: {OUT_PDF.stat().st_size:,} bytes  [{'PASS' if v_pdf else 'FAIL'}]")

print("=== Heatmap verification complete ===")
