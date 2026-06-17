# Why did SPMO (momentum) beat the market — and what it means

*Exploratory investigation, real_trade_analysis (production untouched). Data: real CRSP,
legacy 2011-2024 spliced with CRSP v2 through Dec-2025. SPMO = validated replica of the
S&P 500 Momentum index (top-quintile risk-adjusted momentum, float-cap x score weighting,
9% cap, semi-annual Mar/Sep rebalance); "market" = cap-weighted Top-500 (S&P 500 proxy).*

## Headline

Over 2011-2025 SPMO and the market are a **risk-adjusted tie** (Sharpe ~1.0 each). SPMO
tracked the market for **13 years (2011-2023 excess ~0)** and then earned **essentially its
entire lifetime lead in a single year, 2024 (+23 pts vs the market)**. That 2024 win was a
**lucky conjunction of conditions captured by a do-nothing structure**, not a durable edge,
and it gives **no crisis protection**. Details and figures below.

---

## 1. The break-out happened in 2024, not 2025
The TradingView "SPMO breaks away" look is misleading: the +129% spike is a bad data print
(SPMO trades ~$100-130, not the $697 pre-market tick), and cumulative charts make an early
lead look like it widens later (compounding). The additive, artifact-free view shows the
lead was **earned in 2024 (+18 pts), with ~0 for the prior decade and +4 in 2025.**
- Annual excess: 2024 **+23%**, 2025 **+4%**, 2011-2023 avg **+0.3%/yr**.
- Figures: `02_when_the_lead_was_earned`, `01_breakout_spmo_vs_sp500`.
- Source: `2026-06-02-when-breakaway.py`, `2026-06-02-spmo-vs-sp500-thru2025.py`.

## 2. The mechanism is low-turnover "hold-and-drift", NOT selection or weighting
A counterfactual ladder (rebuild SPMO one design choice at a time; the full rung reproduces
the real replica exactly). Post-2024 excess (+13.5%/yr) decomposes as:

| component | PRE 2011-23 | POST 2024-25 |
|---|---:|---:|
| momentum selection | -1.5% | -0.6% |
| cap x score weighting | +1.4% | -2.5% |
| 9% cap | -0.2% | +0.6% |
| **rebalance / hold-and-drift** | +0.4% | **+16.2%** |
| fee | -0.1% | -0.1% |

**The entire win is the semi-annual, low-turnover hold that lets winners run.** Selection
and the cap x score weighting actually *detracted*; the cap was ~neutral. A monthly-rebalanced
clone held *more* NVIDIA (avg 18% vs 9%) yet beat the market by only +2% -- it churned the
gains away (~346%/yr turnover vs SPMO's ~132%). **This revises the earlier "cap x score
concentration" deep-research story: it is rebalancing inertia, not the weighting scheme.**
- Figure: `05_mechanism_3panel` (panel A). Source: `2026-06-02-spmo-mechanism-ladder.py`, `results/2026-06-02-ladder-marginals.csv`.

## 3. Idiosyncratic, not the momentum factor
FF5+breadth regression with a post-2024 dummy: alpha jump **+12.3%/yr (t=2.43, p=0.015)**;
the UMD (momentum factor) loading does **not** absorb it (+0.29 post vs +0.38 pre). So the win
is idiosyncratic to this product, not an expression of the diversified momentum premium.
- Figure: `05_mechanism_3panel` (panel B). Source: `2026-06-02-spmo-mechanism-regression.py`, `results/2026-06-02-regression.csv`.

## 4. Three stocks were the whole story
Exact name attribution of 2024 excess: **NVDA +3.9%, AVGO +3.3%, META +3.0%** -- the top-3
~= the entire +17.9% linear excess; the rest of the book nets slightly negative beyond them.
- Figures: `05_mechanism_3panel` (panel C), `04_why2024_name_attribution`. Source: `2026-06-02-why-spmo-2024.py`, `results/2026-06-02-attr2024-top.csv`.

## 5. Why 2024 specifically: four independent conditions aligned exactly once
SPMO (buy past winners, hold) needs all of: a wide cross-sectional **dispersion** (spread to
capture), **persistent winners** (they keep winning), **big concentrated bets** (the held
names soar), and **breadth** (the rest of the book doesn't drag). These are largely
independent and **2024 is the only year in the sample all four were favorable**:
- 2020 had dispersion+persistence+magnitude but **breadth was negative** (rest dragged).
- 2017 had breadth+magnitude but **dispersion was low**.
- 2023 had dispersion+magnitude but **persistence collapsed** (winners reshuffled).
- **Breadth was the rare binding condition -- 2024 is the only positive-breadth year.**
- Figure: `03_why2024_four_conditions_heatmap` (the capstone). Source: `2026-06-02-why2024-heatmap.py`, `results/2026-06-02-why2024-heatmap.csv`.

## 6. Momentum is a weak harvester of the available return
- **Perfect-foresight ceiling:** same SPMO rules but picking the top-20% by *future* return
  would have returned **80%/yr** (Sharpe 3.45) vs SPMO's 16%. Momentum captures a sliver.
  Even in 2024: market +26%, SPMO +48%, oracle +110% (SPMO got <half the achievable).
- Figures: `06_oracle_perfect_foresight_ceiling`, `07_oracle_vs_spmo_yearly`. Source: `2026-06-02-oracle-vs-spmo.py`.
- **Why:** the top performers barely repeat. Year-over-year retention of the top-20% list:
  **size ~91%** (biggest companies are near-permanent) but **momentum/return only ~18-20%**
  (best performers are a revolving door). Momentum bets on the *transient* attribute.
- Figure: `08_persistence_size_vs_momentum_vs_return`. Source: `2026-06-02-retention-3way.py`.

## 7. Dispersion and persistence are independent
A wide winner-loser spread does NOT imply the winners persist (corr ~+0.18). 2023 had high
dispersion yet the lowest winner survival (the leadership flipped). Momentum needs *both*
high -- and they rarely coincide. So "high dispersion helps momentum" is only half the story.
- Figure: `09_sp500_dispersion_thru2025` (dispersion context, elevated 2024-25). Source: `2026-06-02-dispersion-extend.py`, `2026-06-02-survival-vs-dispersion.py`.

## 8. Combining CAPM + momentum barely helps
A market + momentum blend lifts Sharpe only **1.01 -> 1.05** (max at ~50/50) and *lowers*
return (diluting with the market). The two are 0.88 correlated, so there is almost nothing to
diversify. Not a meaningful improvement.
- Figure: `11_capm_plus_momentum_blend`. Source: `2026-06-02-capm-plus-momentum.py`.

## 9. Crisis / tail behavior: no protection
SPMO is long-only equity (beta ~1, 0.88 corr) and **falls with the market in every crisis**
(2015-16, Q4-18, COVID, 2022). It is mildly defensive on average down-months (down-capture
0.85, worst month -10% vs market -12%) but its **max drawdown is slightly deeper (-26% vs
-24%)** due to momentum-crash timing (it lags the sharp recovery legs). It is a risk, not a
refuge. (Note: the long-*short* XGB would behave very differently -- it can short into a
crash -- but the long-only momentum book cannot hedge.)
- Figure: `10_crisis_tail_profile`. Source: `2026-06-02-tail-profile-spmo.py`.

## 10. Durability verdict: cannot be claimed
Post-2024 excess is a ~2.3-sigma outlier, BUT the 95% block-bootstrap CI on the monthly
excess is **[-2.4%, +18.9%] (spans zero, n=23 months ~ 1-2 episodes)**, and the edge is **not
cleanly regime-conditional** (U-shaped across dispersion terciles). Consistent with a
favorable realized draw, not a structural edge.
- Source: `2026-06-02-spmo-durability.py`, `results/2026-06-02-durability.csv`.

---

## Bottom line for the thesis
SPMO tracked the market for 13 years, then earned its entire lifetime lead in **one year
(2024)** when four largely-independent conditions happened to align and a do-nothing
(low-turnover) structure held three soaring AI mega-caps. It is **idiosyncratic concentration
captured by inertia, not a durable momentum premium**, it leaves most of the achievable
return on the table, and it offers **no crisis protection**. This independently corroborates
the thesis's central finding that momentum edges here are **time-concentrated and
regime/luck-dependent**, not steady premia.

## Caveats
- Market-cap unit mismatch at the legacy/v2 splice ($millions vs $thousands) corrupted one
  month (Dec-2024); size-based charts use a single consistent v2 source to avoid it; all
  within-year cap-weighted results (SPMO/market returns) are unaffected.
- Tail analysis is 2011-2024 (no GFC-scale event in window). Oracle is a hindsight upper bound.
- Crisis chart is the long-ONLY book; the headline long-short would look different on the downside.

## Figure index (figures/)
01 breakout SPMO vs S&P 500 (thru 2025) | 02 when the lead was earned | 03 why-2024 four-conditions heatmap (capstone) |
04 why-2024 name attribution | 05 mechanism 3-panel (ladder / regression / attribution) | 06 oracle ceiling |
07 oracle vs SPMO yearly | 08 persistence: size vs momentum vs return | 09 S&P dispersion thru 2025 |
10 crisis tail profile | 11 CAPM + momentum blend.

Additional exploratory figures in `../plots/` (2026-06-02-*): oracle-gap-per-rebal, topsize-churn-v2,
spmo-rotation, top-returners-2024, topreturn-persistence, spmo-turnover.
