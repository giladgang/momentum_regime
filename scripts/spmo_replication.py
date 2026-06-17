"""SPMO replication on CRSP universe vs XGB long-only.

Methodology (S&P 500 Momentum Index):
  - Universe: top-500-by-me (S&P 500 proxy; CRSP has no membership flag)
  - Signal: 12-month price change excluding most recent month ("13-1 momentum"),
            scaled by standard deviation of returns over the same 12-month
            window. APPROXIMATION: S&P uses daily returns; we have monthly
            only, so we use monthly stdev * sqrt(12) over 12 months.
  - Z-score the signal cross-sectionally, winsorize at +-3, then map
    z -> momentum_score:  z >= 0 -> 1+z;  z < 0 -> 1/(1-z)
  - Picks: top 100 by raw z (long-only)
  - Weight: market cap * momentum_score, normalized, capped at min(9%, 3x cap-share)
  - Rebal: monthly variant (apples-to-apples vs XGB) AND semi-annual
           (March/September, matches the actual ETF)

XGB long-only: top-decile long leg from results/thesis/rule_path_labels.csv,
               value-weighted by market equity, monthly rebal.

Test period: 2011-01-31 → 2024-11-29 (matches XGB out-of-sample window).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/giladgang/momentum_regime")
CRSP = ROOT / "data" / "crsp_msf_raw.parquet"
PICKS = ROOT / "results" / "thesis" / "rule_path_labels.csv"

TEST_START = pd.Timestamp("2011-01-01")
TEST_END   = pd.Timestamp("2024-12-31")
TOP_UNIVERSE = 500
N_PICKS = 100
VOL_LOOKBACK = 12   # S&P 500 Momentum: vol over the same 12m window as the signal
ANN = 12


def metrics(r: pd.Series) -> dict:
    r = pd.Series(r).dropna()
    mu, sd = r.mean(), r.std()
    sharpe = mu / sd * np.sqrt(ANN) if sd > 0 else 0.0
    ann_ret = (1 + r).prod() ** (ANN / len(r)) - 1 if len(r) else float("nan")
    ann_vol = sd * np.sqrt(ANN)
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() if len(r) else float("nan")
    return dict(n=len(r), sharpe=sharpe, ann_ret=ann_ret, ann_vol=ann_vol, mdd=mdd)


print("loading CRSP...")
df = pd.read_parquet(CRSP)
df = df[df["shrcd"].isin([10, 11])]
df = df[df["exchcd"].isin([1, 2, 3])]
df = df[df["prc"].abs() > 1.0]
df = df[["permno", "date", "ret_adj", "me"]].dropna(subset=["ret_adj"])
df = df.sort_values(["permno", "date"]).reset_index(drop=True)
df["ret_fwd"] = df.groupby("permno")["ret_adj"].shift(-1)

print("computing 12m skip-1 return + same-window vol (annualized)...")
df["log_ret"] = np.log1p(df["ret_adj"])
g = df.groupby("permno", group_keys=False)
df["lr12"] = g["log_ret"].apply(lambda s: s.rolling(12).sum().shift(1))
df["signal_12"] = np.expm1(df["lr12"])
df["vol_12"] = g["ret_adj"].apply(
    lambda s: s.rolling(VOL_LOOKBACK, min_periods=10).std().shift(1)
) * np.sqrt(ANN)
df["rmom_12"] = df["signal_12"] / df["vol_12"]


def build_spmo(rebal_months=None):
    """rebal_months=None => monthly; or list like [3,9] for semi-annual."""
    monthly = []
    holdings: dict[int, float] = {}
    for date, grp in df.groupby("date"):
        if date < TEST_START or date > TEST_END:
            continue
        do_rebal = (rebal_months is None) or (date.month in rebal_months) or (not holdings)
        if do_rebal:
            u = grp.dropna(subset=["me", "rmom_12"]).copy()
            if len(u) < 200:
                if holdings:
                    held = grp[grp["permno"].isin(holdings)].copy()
                    held["w"] = held["permno"].map(holdings)
                    r = (held["w"] * held["ret_fwd"]).sum()
                    monthly.append({"date": date, "ret": r})
                continue
            u = u.nlargest(TOP_UNIVERSE, "me")
            u["raw_score"] = (u["rmom_12"] - u["rmom_12"].mean()) / u["rmom_12"].std()
            # winsorize z at +-3, then S&P piecewise transform
            z = u["raw_score"].clip(-3, 3)
            u["mom_score"] = np.where(z >= 0, 1.0 + z, 1.0 / (1.0 - z))
            picks = u.nlargest(N_PICKS, "raw_score").copy()
            # market cap proportions inside the eligible (top-500) universe
            cap_share = picks["me"] / u["me"].sum()
            w_raw = picks["me"] * picks["mom_score"]
            w = w_raw / w_raw.sum()
            # S&P security cap: min(9%, 3 * cap-share); iterate once to redistribute
            cap = np.minimum(0.09, 3.0 * cap_share.values)
            for _ in range(3):
                over = w.values > cap
                if not over.any():
                    break
                excess = (w.values[over] - cap[over]).sum()
                w_vals = w.values.copy()
                w_vals[over] = cap[over]
                under = ~over
                if under.any() and excess > 0:
                    w_vals[under] += excess * (w_vals[under] / w_vals[under].sum())
                w = pd.Series(w_vals, index=w.index)
            picks["w"] = w.values
            holdings = dict(zip(picks["permno"], picks["w"]))
        held = grp[grp["permno"].isin(holdings)].copy()
        held["w"] = held["permno"].map(holdings)
        r = (held["w"] * held["ret_fwd"]).sum()
        monthly.append({"date": date, "ret": r})
        new_raw = {}
        for _, row in held.iterrows():
            if pd.notna(row["ret_fwd"]):
                new_raw[row["permno"]] = holdings[row["permno"]] * (1 + row["ret_fwd"])
        tot = sum(new_raw.values())
        if tot > 0:
            holdings = {p: w / tot for p, w in new_raw.items()}
    return pd.DataFrame(monthly).set_index("date")["ret"]


def build_xgb_long_only():
    picks = pd.read_csv(PICKS, parse_dates=["date"])
    picks = picks.drop_duplicates(["date", "permno"])
    me = df[["date", "permno", "me", "ret_fwd"]]
    j = picks.merge(me, on=["date", "permno"], how="left").dropna(subset=["me", "ret_fwd"])
    rows = []
    for date, grp in j.groupby("date"):
        if date < TEST_START or date > TEST_END:
            continue
        w = grp["me"] / grp["me"].sum()
        r = (w * grp["ret_fwd"]).sum()
        rows.append({"date": date, "ret": r})
    return pd.DataFrame(rows).set_index("date")["ret"]


print("\nbuilding SPMO monthly rebal...")
spmo_m = build_spmo(rebal_months=None)
print(f"  months: {len(spmo_m)}")

print("building SPMO semi-annual rebal (Mar/Sep)...")
spmo_h = build_spmo(rebal_months=[3, 9])
print(f"  months: {len(spmo_h)}")

print("building XGB long-only (from picks CSV)...")
xgb_lo = build_xgb_long_only()
print(f"  months: {len(xgb_lo)}")

print("\n" + "=" * 64)
print("LONG-ONLY COMPARISON  (test period 2011-01 → 2024-11)")
print("=" * 64)
print(f"{'Strategy':<28} {'N':>4} {'Sharpe':>8} {'AnnRet':>8} {'AnnVol':>8} {'MDD':>8}")
print("-" * 64)
for name, r in [
    ("SPMO replication (monthly)", spmo_m),
    ("SPMO replication (Mar/Sep)", spmo_h),
    ("XGB long-only",              xgb_lo),
]:
    m = metrics(r)
    print(f"{name:<28} {m['n']:>4} {m['sharpe']:>8.2f} {m['ann_ret']:>8.1%} {m['ann_vol']:>8.1%} {m['mdd']:>8.1%}")

print("\n(XGB long-only published Sharpe in table_alt_targets.tex: 1.107)")

OUT = ROOT / "results" / "thesis" / "spmo_vs_xgb_long_only.csv"
out_df = pd.concat([spmo_m.rename("spmo_monthly"),
                    spmo_h.rename("spmo_semiannual"),
                    xgb_lo.rename("xgb_long_only")], axis=1)
out_df.to_csv(OUT)
print(f"\nwrote {OUT}")
