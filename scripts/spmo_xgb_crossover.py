"""SPMO vs XGB: 2x2 ablation isolating SIGNAL vs UNIVERSE.

Four portfolios, each with its METHOD'S native construction (so we hold
weighting + rebal fixed within each method and vary only the universe):

  Method   | Universe = top-500-by-me (S&P 500 proxy) | Universe = full CRSP
  ---------|------------------------------------------|----------------------
  SPMO     | (1) SPMO baseline                        | (2) SPMO on full CRSP
  XGB      | (3) XGB on top-500                       | (4) XGB baseline

SPMO method: top-100 by SPMO score; weight = me x mom_score (winsorized
piecewise); rebal Mar/Sep.

XGB method: top-100 by XGB score; weight = me (value-weighted); monthly
rebal.

Reading the table:
  (1) -> (2)  : effect of dropping the universe restriction (SPMO signal)
  (3) -> (4)  : effect of widening universe (XGB signal)
  (1) -> (3)  : on same universe, signal swap (SPMO -> XGB)
  (2) -> (4)  : on same universe, signal swap (SPMO -> XGB) -- full CRSP
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path("/Users/giladgang/momentum_regime")
CRSP = ROOT / "data" / "crsp_msf_raw.parquet"
XGB_MODEL = ROOT / "artefacts" / "cs_artefacts_xgb.pkl"
XGB_DATA  = ROOT / "artefacts" / "cs_artefacts_data.pkl"

TEST_START = pd.Timestamp("2011-01-01")
TEST_END   = pd.Timestamp("2024-12-31")
TOP_UNIVERSE = 500
N_PICKS = 100
VOL_LOOKBACK = 12
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


print("loading CRSP + computing SPMO signal...")
df = pd.read_parquet(CRSP)
df = df[df["shrcd"].isin([10, 11])]
df = df[df["exchcd"].isin([1, 2, 3])]
df = df[df["prc"].abs() > 1.0]
df = df[["permno", "date", "ret_adj", "me"]].dropna(subset=["ret_adj"])
df = df.sort_values(["permno", "date"]).reset_index(drop=True)
df["ret_fwd"] = df.groupby("permno")["ret_adj"].shift(-1)

g = df.groupby("permno", group_keys=False)
df["log_ret"] = np.log1p(df["ret_adj"])
df["lr12"] = g["log_ret"].apply(lambda s: s.rolling(12).sum().shift(1))
df["signal_12"] = np.expm1(df["lr12"])
df["vol_12"] = g["ret_adj"].apply(
    lambda s: s.rolling(VOL_LOOKBACK, min_periods=10).std().shift(1)
) * np.sqrt(ANN)
df["spmo_signal"] = df["signal_12"] / df["vol_12"]

print("loading XGB model + features, scoring test panel...")
with open(XGB_MODEL, "rb") as f:
    xgb = pickle.load(f)
with open(XGB_DATA, "rb") as f:
    data = pickle.load(f)
test = data["test"][["date", "permno"]].copy()
test["xgb_score"] = xgb.predict(data["X_te_s"])
print(f"  XGB scores: {len(test):,} (date, permno) observations")

df = df.merge(test, on=["date", "permno"], how="left")


def run_port(score_col, universe="top500", weighting="me_x_score", rebal="semi", label=""):
    """Build a long-only portfolio.

    score_col  : 'spmo_signal' or 'xgb_score'
    universe   : 'top500' or 'full'
    weighting  : 'me_x_score' (SPMO style) or 'me' (value-weighted)
    rebal      : 'semi' (Mar/Sep) or 'monthly'
    """
    monthly, holdings = [], {}
    for date, grp in df.groupby("date"):
        if date < TEST_START or date > TEST_END:
            continue
        do_rebal = (rebal == "monthly") or (date.month in (3, 9)) or (not holdings)
        if do_rebal:
            u = grp.dropna(subset=["me", score_col]).copy()
            if universe == "top500":
                if len(u) < 200:
                    if holdings:
                        held = grp[grp["permno"].isin(holdings)].copy()
                        held["w"] = held["permno"].map(holdings)
                        r = (held["w"] * held["ret_fwd"]).sum()
                        monthly.append({"date": date, "ret": r})
                    continue
                u = u.nlargest(TOP_UNIVERSE, "me")
            if len(u) < N_PICKS:
                continue
            mu, sd = u[score_col].mean(), u[score_col].std()
            if sd <= 0:
                continue
            z = ((u[score_col] - mu) / sd).clip(-3, 3)
            u["mom_score"] = np.where(z >= 0, 1.0 + z, 1.0 / (1.0 - z))
            picks = u.nlargest(N_PICKS, score_col).copy()
            if weighting == "me_x_score":
                cap_share = picks["me"] / u["me"].sum()
                w_raw = picks["me"] * picks["mom_score"]
                w = (w_raw / w_raw.sum()).values
                cap = np.minimum(0.09, 3.0 * cap_share.values)
                for _ in range(3):
                    over = w > cap
                    if not over.any():
                        break
                    excess = (w[over] - cap[over]).sum()
                    w[over] = cap[over]
                    under = ~over
                    if under.any():
                        w[under] += excess * (w[under] / w[under].sum())
                picks["w"] = w
            else:  # value-weighted
                picks["w"] = picks["me"] / picks["me"].sum()
            holdings = dict(zip(picks["permno"], picks["w"]))
        held = grp[grp["permno"].isin(holdings)].copy()
        held["w"] = held["permno"].map(holdings)
        r = (held["w"] * held["ret_fwd"]).sum()
        monthly.append({"date": date, "ret": r})
        new_raw = {p: holdings[p] * (1 + row["ret_fwd"])
                   for p, row in held.set_index("permno").iterrows()
                   if pd.notna(row["ret_fwd"])}
        tot = sum(new_raw.values())
        if tot > 0:
            holdings = {p: w / tot for p, w in new_raw.items()}
    return pd.DataFrame(monthly).set_index("date")["ret"].rename(label)


print("\nbuilding 4 portfolios...")
spmo_top500 = run_port("spmo_signal", universe="top500", weighting="me_x_score", rebal="semi", label="SPMO/top500")
spmo_full   = run_port("spmo_signal", universe="full",   weighting="me_x_score", rebal="semi", label="SPMO/full CRSP")
xgb_top500  = run_port("xgb_score",   universe="top500", weighting="me",         rebal="monthly", label="XGB/top500")
xgb_full    = run_port("xgb_score",   universe="full",   weighting="me",         rebal="monthly", label="XGB/full CRSP")

print("\n" + "=" * 76)
print("SPMO vs XGB: SIGNAL x UNIVERSE  (long-only, 2011-01 -> 2024-11)")
print("=" * 76)
print(f"{'Strategy':<28} {'N':>4} {'Sharpe':>8} {'AnnRet':>8} {'AnnVol':>8} {'MDD':>8}")
print("-" * 76)
for series in [spmo_top500, spmo_full, xgb_top500, xgb_full]:
    m = metrics(series)
    print(f"{series.name:<28} {m['n']:>4} {m['sharpe']:>8.2f} {m['ann_ret']:>8.1%} {m['ann_vol']:>8.1%} {m['mdd']:>8.1%}")

print("\nReading the 2x2:")
print("  (SPMO/top500 -> SPMO/full):  universe widens for SPMO signal")
print("  (XGB/top500  -> XGB/full):   universe widens for XGB signal")
print("  (SPMO/top500 -> XGB/top500): signal swap on S&P 500 proxy")
print("  (SPMO/full   -> XGB/full):   signal swap on full CRSP")

OUT = ROOT / "results" / "thesis" / "spmo_xgb_crossover.csv"
pd.concat([spmo_top500, spmo_full, xgb_top500, xgb_full], axis=1).to_csv(OUT)
print(f"\nwrote {OUT}")
