"""Month-over-month % of XGB long-leg picks that are NEW (not in last month's basket).

Outputs: plots/diagnostic/xgb_pct_new_entries.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path("/Users/giladgang/momentum_regime")
PICKS = ROOT / "results" / "thesis" / "rule_path_labels.csv"
PI    = ROOT / "results" / "thesis" / "expanding_pi_filter_prod.csv"
OUT   = ROOT / "plots" / "diagnostic" / "xgb_pct_new_entries.png"

picks = pd.read_csv(PICKS, parse_dates=["date"])
date_to_set = {d: set(g["permno"]) for d, g in picks.groupby("date")}
dates = sorted(date_to_set)

rows = []
for prev, curr in zip(dates[:-1], dates[1:]):
    p_prev, p_curr = date_to_set[prev], date_to_set[curr]
    new = p_curr - p_prev
    rows.append({
        "date": curr,
        "n_picks": len(p_curr),
        "n_new": len(new),
        "pct_new": 100.0 * len(new) / len(p_curr),
    })
df = pd.DataFrame(rows)
df["pct_new_12m"] = df["pct_new"].rolling(12, min_periods=3).mean()

print(f"months: {len(df)}  mean %new: {df['pct_new'].mean():.1f}%  median: {df['pct_new'].median():.1f}%")
print(f"min: {df['pct_new'].min():.1f}% on {df.loc[df['pct_new'].idxmin(),'date'].date()}")
print(f"max: {df['pct_new'].max():.1f}% on {df.loc[df['pct_new'].idxmax(),'date'].date()}")

pi = pd.read_csv(PI, parse_dates=["date"])
pi = pi[(pi["date"] >= df["date"].min()) & (pi["date"] <= df["date"].max())]

fig, ax = plt.subplots(figsize=(11, 5.5))
for _, r in pi.iterrows():
    if r["pi_filter"] >= 0.5:
        ax.axvspan(r["date"] - pd.Timedelta(days=15), r["date"] + pd.Timedelta(days=15),
                   color="red", alpha=0.10, lw=0)

ax.plot(df["date"], df["pct_new"], color="#888", lw=0.9, label="month-over-month")
ax.plot(df["date"], df["pct_new_12m"], color="#cc0000", lw=2.2, label="12-month rolling mean")
ax.axhline(df["pct_new"].mean(), color="#1b7a1b", lw=1, ls="--", alpha=0.7,
           label=f"overall mean = {df['pct_new'].mean():.1f}%")

ax.set_ylim(0, max(df["pct_new"].max() * 1.05, 60))
ax.set_xlabel("Date")
ax.set_ylabel("% of long-leg picks that are NEW vs last month")
ax.set_title("XGB long-leg turnover — % of picks new each month (shaded = panic regime)")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
