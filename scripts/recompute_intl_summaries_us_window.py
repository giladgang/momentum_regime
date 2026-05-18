"""
One-off: recompute the international (UK, JP) strategy summaries on the
2011-01-31 -> 2024-11-29 sub-window (167 months), matching the US headline
sample. Existing intl_<region>_summary.csv files use the full
2011-2026 sample (183 months) which makes Sharpe/MDD comparisons against
the US numbers asymmetric.

Reads:  results/thesis/intl_uk_returns.csv
        results/thesis/intl_jp_returns.csv
Writes: results/thesis/intl_uk_summary_us_window.csv
        results/thesis/intl_jp_summary_us_window.csv
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import metrics

WINDOW_END = "2024-11-30"

for region in ("uk", "jp"):
    df = pd.read_csv(f"results/thesis/intl_{region}_returns.csv", parse_dates=["date"])
    df = df[df["date"] <= WINDOW_END]
    strats = ["market", "fixed_mom_12", "fixed_mom_1",
              "method0_formula", "method1_lr", "method1b_ridge", "method2_xgb"]
    rows = []
    for s in strats:
        r = df[s].dropna()
        ann_ret, ann_vol, sharpe, mdd = metrics(r)
        total_ret = float((1 + r).prod() - 1) if len(r) else float("nan")
        rows.append({"strategy": s, "n_months": len(r),
                     "ann_ret": ann_ret, "ann_vol": ann_vol,
                     "sharpe": sharpe, "max_dd": mdd, "total_ret": total_ret})
    out = pd.DataFrame(rows)
    out_path = f"results/thesis/intl_{region}_summary_us_window.csv"
    out.to_csv(out_path, index=False)
    print(f"\n=== {region.upper()} 2011-2024 window ({out_path}) ===")
    print(out.round(4).to_string(index=False))
