"""
2026-06-02-spmo-mechanism-charts.py

Summary figure for "Why SPMO outperformed after 2024" study.
Three panels:
  A – Mechanism decomposition (POST 2024-25 excess)
  B – Idiosyncratic alpha vs UMD loading (rolling 36m)
  C – 2024 excess: top-10 name contributions
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "/Users/giladgang/momentum_regime/real_trade_analysis"
LADDER_CSV   = os.path.join(BASE, "results", "2026-06-02-ladder-marginals.csv")
REGR_CSV     = os.path.join(BASE, "results", "2026-06-02-regression.csv")
ATTR_CSV     = os.path.join(BASE, "results", "2026-06-02-attr2024-top.csv")
PLOT_STEM    = os.path.join(BASE, "plots",   "2026-06-02-spmo-mechanism")

# ── load data ───────────────────────────────────────────────────────────────
ladder = pd.read_csv(LADDER_CSV, index_col="period")
regr   = pd.read_csv(REGR_CSV, parse_dates=["date"])
attr   = pd.read_csv(ATTR_CSV, index_col=0)   # first col = ticker

# Exact period labels
pre_label  = "PRE <=2023"
post_label = "POST >=2024"

# Components to decompose (in display order)
components = ["selection", "weighting", "cap", "rebal_drift", "fee"]
comp_labels = ["Selection", "Weighting", "Cap\nConstr.", "Rebal /\nDrift", "Fee"]

# Convert to %
pre_vals  = ladder.loc[pre_label,  components].values * 100
post_vals = ladder.loc[post_label, components].values * 100
pre_total  = ladder.loc[pre_label,  "total"] * 100
post_total = ladder.loc[post_label, "total"] * 100

# ── figure layout ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle(
    "Why the SPMO replica beat the S&P 500 after 2024:\n"
    "low-turnover hold-and-drift, idiosyncratic alpha, concentrated names",
    fontsize=12, fontweight="bold", y=1.01
)

# ── Panel A: Mechanism decomposition ────────────────────────────────────────
ax = axes[0]
x = np.arange(len(components))
w = 0.35

# POST bars
colors_post = []
for c in components:
    colors_post.append("#c0392b" if c == "rebal_drift" else "#2980b9")

bars_post = ax.bar(x + w/2, post_vals, width=w, color=colors_post,
                   label="POST >=2024", zorder=3)

# PRE bars (faded)
bars_pre = ax.bar(x - w/2, pre_vals, width=w,
                  color=["#e8a09a" if c == "rebal_drift" else "#aed6f1"
                          for c in components],
                  alpha=0.7, label="PRE <=2023", zorder=3)

# Annotate POST values
for bar, v in zip(bars_post, post_vals):
    va = "bottom" if v >= 0 else "top"
    offset = 0.3 if v >= 0 else -0.3
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
            f"{v:+.1f}%", ha="center", va=va, fontsize=8, fontweight="bold")

# Total line / marker for POST
ax.axhline(0, color="black", linewidth=0.8, zorder=2)
ax.hlines(post_total, x[-1] + w/2 + 0.25, x[-1] + w/2 + 0.65,
          colors="black", linewidths=2, zorder=4)
ax.annotate(f"Total\n{post_total:+.1f}%",
            xy=(x[-1] + w/2 + 0.45, post_total),
            xytext=(x[-1] + w/2 + 0.45, post_total + 2),
            ha="center", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="black", lw=1))

ax.set_xticks(x)
ax.set_xticklabels(comp_labels, fontsize=9)
ax.set_ylabel("Annualised excess return (%)", fontsize=9)
ax.set_title(
    "A  |  Mechanism decomposition\n"
    "Rebal/Drift (semi-annual hold-and-drift)\n"
    "dominates the POST >=2024 excess",
    fontsize=9, loc="left"
)
ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.set_xlim(-0.7, len(components) - 0.3 + 1.0)

# ── Panel B: Rolling alpha and UMD loading ───────────────────────────────────
ax = axes[1]
ax2 = ax.twinx()

alpha_pct = regr["alpha_ann"] * 100
umd       = regr["umd"]

line_alpha, = ax.plot(regr["date"], alpha_pct, color="#2c7bb6", lw=1.8,
                      label="Rolling 36m alpha (%, left)")
line_umd,   = ax2.plot(regr["date"], umd, color="#d7191c", lw=1.4,
                        linestyle="--", label="UMD loading (right)")

ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.axvline(pd.Timestamp("2024-01-01"), color="gray", linewidth=0.8,
           linestyle=":", label="2024 start")

ax.set_ylabel("Rolling 36m alpha (ann., %)", fontsize=9, color="#2c7bb6")
ax2.set_ylabel("UMD beta loading", fontsize=9, color="#d7191c")
ax.tick_params(axis="y", labelcolor="#2c7bb6")
ax2.tick_params(axis="y", labelcolor="#d7191c")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

ax.set_title(
    "B  |  Idiosyncratic alpha, not momentum factor\n"
    "Alpha turns positive post-2018; UMD loading\n"
    "stable ~0.5 (not surging) into 2024-25",
    fontsize=9, loc="left"
)

# Combined legend
lines = [line_alpha, line_umd]
labs  = [l.get_label() for l in lines]
ax.legend(lines, labs, fontsize=8, loc="lower left")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xlabel("")

# ── Panel C: 2024 name contributions ────────────────────────────────────────
ax = axes[2]

# Sort descending (already sorted in CSV, but ensure it)
attr_sorted = attr["excess_contrib"].sort_values(ascending=False)
tickers = attr_sorted.index.tolist()
values  = (attr_sorted.values * 100).tolist()  # to %

# Horizontal bars, sorted descending → plot bottom-to-top for natural read
y = np.arange(len(tickers))
bar_colors = []
top3 = {"NVDA", "AVGO", "META"}
for t in tickers:
    bar_colors.append("#c0392b" if t in top3 else "#2980b9")

hbars = ax.barh(y, values, color=bar_colors, height=0.6, zorder=3)

# Invert y so largest is on top
ax.invert_yaxis()
ax.set_yticks(y)
ax.set_yticklabels(tickers, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)

# Annotate values
for bar, v in zip(hbars, values):
    ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
            f"{v:.2f}%", va="center", ha="left", fontsize=8)

# Mark top-3 total
top3_total = sum(v for t, v in zip(tickers, values) if t in top3)
ax.axvline(top3_total, color="#c0392b", linewidth=1.2, linestyle=":",
           label=f"Top-3 sum: {top3_total:.1f}%")

total_all = sum(values)
ax.axvline(total_all, color="black", linewidth=1.2, linestyle="--",
           label=f"Top-10 sum: {total_all:.1f}%")

ax.set_xlabel("2024 contribution to excess return (%)", fontsize=9)
ax.set_title(
    "C  |  2024 excess: a few names drive it all\n"
    "Top-3 (NVDA/AVGO/META) ≈ entire top-10 excess;\n"
    "concentration, not breadth",
    fontsize=9, loc="left"
)
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

# ── save ─────────────────────────────────────────────────────────────────────
plt.tight_layout()

for ext in ["pdf", "png"]:
    out = f"{PLOT_STEM}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
