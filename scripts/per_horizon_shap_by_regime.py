"""
per_horizon_shap_by_regime.py
==============================
Aggregate SHAP values per momentum horizon, separately for {calm, panic} regimes
and {long, short} legs.

Reproduces / verifies the existing thesis claims at lines 76 and 98 of
latex/main_results.tex:
  - Calm long-leg: month 11 = 18.5%, month 12 = 18.0% of long-leg momentum SHAP
  - Calm long-leg: months 8-12 together = 67%
  - Panic long-leg: months 7-12 together ~ 63%
  - Short side month 1 SHAP: 10.1% in panic vs 5.0% in calm

Method:
  1. Load art['shap_values'] (N, 13): features = mom_1..mom_12, pi_filter
  2. Assign regime (calm | panic) per stock-month via pi_filter >= 0.5
  3. Assign leg (long | short | middle) per stock-month via NYSE P90/P10
     breakpoints of score_xgb (same method as production strategy)
  4. For each (regime, leg) combo:
     - filter shap_values rows
     - sum |SHAP| per horizon
     - compute momentum-only share: |SHAP_h| / sum_h |SHAP_h|
     - compute pi_share: |SHAP_pi| / (|SHAP_pi| + sum_h |SHAP_h|)

Outputs:
  results/thesis/per_horizon_shap_by_regime.csv

Usage:
    python scripts/per_horizon_shap_by_regime.py
"""

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTEFACTS = ROOT / "artefacts" / "cs_artefacts_data.pkl"
OUT_CSV    = ROOT / "results" / "thesis" / "per_horizon_shap_by_regime.csv"

# ── Load artefacts ────────────────────────────────────────────────────────────
print("Loading artefacts ...", flush=True)
with open(ARTEFACTS, "rb") as f:
    art = pickle.load(f)

test       = art["test"].copy()
shap_vals  = art["shap_values"]   # (N, 13): mom_1..mom_12, pi_filter
FEATURES   = art["FEATURES"]      # ['mom_1',...,'mom_12','pi_filter']

print(f"  shap_values shape : {shap_vals.shape}")
print(f"  FEATURES          : {FEATURES}")
print(f"  test rows         : {len(test)}")

# Confirm feature layout
assert FEATURES[-1] == "pi_filter", f"Unexpected last feature: {FEATURES[-1]}"
mom_cols = [f"mom_{h}" for h in range(1, 13)]
assert FEATURES[:12] == mom_cols, f"Unexpected mom feature order"

mom_indices = list(range(12))   # columns 0-11 of shap_vals
pi_idx      = 12                 # column 12

# ── Assign regime per stock-month ─────────────────────────────────────────────
# pi_filter is a stock-month column in test (same value for all stocks in a month).
test["regime"] = np.where(test["pi_filter"] >= 0.5, "panic", "calm")

# ── Assign leg per stock-month (NYSE P90/P10 breakpoints of score_xgb) ────────
print("Assigning legs via NYSE P90/P10 breakpoints ...", flush=True)
test["leg"] = "middle"
for date, grp in test.groupby("date"):
    nyse = grp[grp["exchcd"] == 1]["score_xgb"].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    long_idx  = grp.index[grp["score_xgb"] >= hi]
    short_idx = grp.index[grp["score_xgb"] <= lo]
    test.loc[long_idx,  "leg"] = "long"
    test.loc[short_idx, "leg"] = "short"

print(f"  long  : {(test['leg']=='long').sum():,}")
print(f"  short : {(test['leg']=='short').sum():,}")
print(f"  middle: {(test['leg']=='middle').sum():,}")

# ── Aggregate |SHAP| per (regime, leg, horizon) ───────────────────────────────
regime_arr = test["regime"].values
leg_arr    = test["leg"].values

rows = []
pi_rows = []  # sidecar for pi_share per (regime, leg)

for regime in ["calm", "panic"]:
    for leg in ["long", "short"]:
        mask = (regime_arr == regime) & (leg_arr == leg)
        n    = mask.sum()
        if n == 0:
            continue
        sv_sub   = shap_vals[mask]   # (n_rows, 13)

        # Sum |SHAP| per momentum horizon
        abs_mom  = np.abs(sv_sub[:, :12])   # (n_rows, 12)
        sum_mom  = abs_mom.sum(axis=0)       # (12,)  total per horizon
        total_mom = sum_mom.sum()

        abs_pi   = np.abs(sv_sub[:, pi_idx]).sum()
        total_all = total_mom + abs_pi

        # Per-horizon share of momentum-only |SHAP|
        if total_mom > 0:
            mom_share = sum_mom / total_mom
        else:
            mom_share = np.zeros(12)

        # pi_share
        if total_all > 0:
            pi_share = float(abs_pi / total_all)
        else:
            pi_share = float("nan")

        for h_idx, h in enumerate(range(1, 13)):
            rows.append({
                "regime":    regime,
                "leg":       leg,
                "horizon":   h,
                "mom_share": float(mom_share[h_idx]),
                "n_rows":    n,
            })

        pi_rows.append({
            "regime":   regime,
            "leg":      leg,
            "pi_share": pi_share,
            "n_rows":   n,
        })

        print(f"\n  ({regime}, {leg}) n={n:,}  pi_share={pi_share:.4f}")
        for h_idx, h in enumerate(range(1, 13)):
            print(f"    mom_{h:02d}: {mom_share[h_idx]*100:5.1f}%")

# ── Save output ───────────────────────────────────────────────────────────────
out_df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT_CSV, index=False, float_format="%.6f")
print(f"\nWrote {OUT_CSV}  ({len(out_df)} rows)")

pi_df = pd.DataFrame(pi_rows)
print("\n=== pi_share per (regime, leg) ===")
print(pi_df.to_string(index=False, float_format="%.4f"))

# ── Verification block ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("VERIFICATION BLOCK — per_horizon_shap_by_regime")
print("="*60)

def get_share(df, regime, leg, horizon):
    row = df[(df["regime"] == regime) & (df["leg"] == leg) & (df["horizon"] == horizon)]
    if len(row) == 0:
        return float("nan")
    return float(row["mom_share"].iloc[0])

def get_share_sum(df, regime, leg, horizons):
    return sum(get_share(df, regime, leg, h) for h in horizons)

# 1. Calm long-leg month 11 ≈ 18.5%
target, tol = 0.185, 0.02
actual = get_share(out_df, "calm", "long", 11)
diff   = abs(actual - target)
status = "PASS" if diff <= tol else "FAIL"
print(f"[C1] Calm long-leg month 11 share ≈ 18.5% (±2pp): "
      f"actual={actual*100:.1f}%  diff={diff*100:.1f}pp  [{status}]")

# 2. Calm long-leg month 12 ≈ 18.0%
target, tol = 0.180, 0.02
actual = get_share(out_df, "calm", "long", 12)
diff   = abs(actual - target)
status = "PASS" if diff <= tol else "FAIL"
print(f"[C2] Calm long-leg month 12 share ≈ 18.0% (±2pp): "
      f"actual={actual*100:.1f}%  diff={diff*100:.1f}pp  [{status}]")

# 3. Calm long-leg months 8-12 ≈ 67%
target, tol = 0.67, 0.03
actual = get_share_sum(out_df, "calm", "long", range(8, 13))
diff   = abs(actual - target)
status = "PASS" if diff <= tol else "FAIL"
print(f"[C3] Calm long-leg months 8-12 sum ≈ 67% (±3pp): "
      f"actual={actual*100:.1f}%  diff={diff*100:.1f}pp  [{status}]")

# 4. Panic long-leg months 7-12 ≈ 63%
target, tol = 0.63, 0.03
actual = get_share_sum(out_df, "panic", "long", range(7, 13))
diff   = abs(actual - target)
status = "PASS" if diff <= tol else "FAIL"
print(f"[C4] Panic long-leg months 7-12 sum ≈ 63% (±3pp): "
      f"actual={actual*100:.1f}%  diff={diff*100:.1f}pp  [{status}]")

# 5. Panic short-leg month 1 ≈ 10.1%
target, tol = 0.101, 0.02
actual = get_share(out_df, "panic", "short", 1)
diff   = abs(actual - target)
status = "PASS" if diff <= tol else "FAIL"
print(f"[C5] Panic short-leg month 1 share ≈ 10.1% (±2pp): "
      f"actual={actual*100:.1f}%  diff={diff*100:.1f}pp  [{status}]")

# 6. Calm short-leg month 1 ≈ 5.0%
target, tol = 0.050, 0.02
actual = get_share(out_df, "calm", "short", 1)
diff   = abs(actual - target)
status = "PASS" if diff <= tol else "FAIL"
print(f"[C6] Calm short-leg month 1 share ≈ 5.0% (±2pp): "
      f"actual={actual*100:.1f}%  diff={diff*100:.1f}pp  [{status}]")

# 7. All shares per (regime, leg) sum to 1.000 ± 0.001
all_pass = True
for regime in ["calm", "panic"]:
    for leg in ["long", "short"]:
        sub   = out_df[(out_df["regime"] == regime) & (out_df["leg"] == leg)]
        total = sub["mom_share"].sum()
        ok    = abs(total - 1.0) <= 0.001
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"[C7] {regime}/{leg} shares sum = {total:.6f}  [{status}]")

# Extra cross-check: aggregate pi_share ≈ 0.46
print()
all_sv_abs    = np.abs(shap_vals)
total_all_abs = all_sv_abs.sum()
pi_share_agg  = float(all_sv_abs[:, pi_idx].sum() / total_all_abs)
diff = abs(pi_share_agg - 0.46)
status = "PASS" if diff <= 0.05 else "FAIL"
print(f"[CX] Aggregate pi_share ≈ 0.46 (±0.05): "
      f"actual={pi_share_agg:.4f}  diff={diff:.4f}  [{status}]")

print("="*60)
print("Verification complete.")
