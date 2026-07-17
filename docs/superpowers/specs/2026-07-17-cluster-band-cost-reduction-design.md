# Cluster-conditioned banding for transaction-cost reduction — design

**Date:** 2026-07-17
**Status:** design, pending approval
**Owner:** Gilad Gang (advisor: Marc Stam)

## Question

Can the thesis's momentum-shape clusters be used to expand and contract the
no-trade band — trade less in some clusters, more in others — so as to reduce
transaction costs on the 7 benchmark strategies?

**Cost is the outcome variable.** Not information ratio. This is the pivot that
makes the study well-posed: cost is measured directly (turnover x spread, precise
to under a basis point), whereas cost-driven differences in IR are provably
undetectable on this book (cost moves IR by <=0.03 against a 0.054 noise floor —
a 162x gap, established by clean-room rebuild 2026-07-17).

## Sample — the thesis test set ONLY

**2011-01-01 .. 2025-11-30, 179 months.** (`TRAIN_END`/`SPLIT_DATE` = 2011-01-01;
`EVAL_YEARS` = 2011..2025. RESULTS_LOG.md updated to 179 by Gilad 2026-07-17.)

Consequence, stated up front: 2011-2025 is the low-spread era. Measured top-1000
median half-spread by decade: 51.3bp (1990s) / 3.98bp (2000s) / 1.28bp (2010s) /
1.58bp (2020s). Restricting to the test set removes the entire high-spread era,
where nearly all cost and all cost savings live. This is not a flaw in the design;
it is the design's central finding (see "Expected result").

## The cost budget (the whole pie) — measured, test set

| strategy | turnover/yr | COST bp/yr | a 42% NMV-scale cut saves |
|---|---|---|---|
| momentum | 4.03 | 13.5 | 5.7 |
| reversal | 11.02 | 38.4 | 16.1 |
| lowvol | 2.59 | 6.8 | 2.9 |
| xgb | 9.08 | 31.2 | 13.1 |
| value | 0.53 | 2.0 | 0.8 |
| profitability | 0.51 | 1.5 | 0.6 |
| investment | 1.17 | 3.8 | 1.6 |
| **mean** | **4.13** | **13.9** | **5.8** |

Implied test-window effective trade-weighted half-spread ~3.4bp across all seven.
Corroboration: `xgb` exists only from 2011 and shows 1.72bp trade-weighted,
consistent with the 1.28bp 2010s median.

**Everything this study can achieve is a fraction of the COST column.**

## Identification — can we know the cluster in real time?

Cluster = KMeans(k=4) on the book's momentum term-structure z-curve
(`mom_1..mom_12`, cross-sectionally z-scored), centroids fit on data strictly
before month t, labels ordered by z-level (0 = most winner-tilted .. 3 = deepest
loser). The curve is built from past returns, so it is observable at t.

Measured real-time (PIT) vs hindsight agreement, 73.4% overall:

| hindsight says | real-time agreed |
|---|---|
| C0 | 69% |
| C1 | **44%** — coin flip, unusable |
| C2 | 76% |
| C3 | **99%** — 92% precision |

**C3 is the one cluster that is genuinely knowable in real time.** This is a
necessary condition for tradability and it is a finding in its own right.

### PIT cluster distribution on the test set

| cluster | months | share | verdict |
|---|---|---|---|
| C0 | 1 | 0.6% | untestable (n=1) |
| C1 | 26 | 14.5% | unusable (44% identification) |
| C2 | 113 | 63.1% | "widen in C2" IS a uniform band |
| C3 | **39** | **21.8%** | the only real treatment |

C3's 39 months fall in **19 distinct episodes** (longest 9, in 2016). Effective
n = 19, not 39. Inference must use block bootstrap at episode scale.

## The ceiling (the interpretive frame)

Cost = sum_t (turnover_t x spread_t). A state-conditional band can only change
*when* you trade, so it can only exploit **spread variation across states**.

Detrended (year-demeaned log) spread faced by the book, by cluster:

| cluster | raw (bp) | detrended ratio |
|---|---|---|
| C0 | 42.32 | 0.990 |
| C1 | 26.23 | 1.028 |
| C2 | 3.42 | 1.001 |
| C3 | 11.33 | 0.981 |
| | **12x** | **1.047x** |

The raw 12x cluster spread gap is **pure calendar** — C0 months live in the 1990s,
C2 months in the 2010s. Detrended, spreads vary 4.7% across clusters.

**=> Cost saving from cluster timing alone is capped at ~1.9%.** Arithmetic, not
an estimate. (Contrast: the HMM regime carries real spread information, 1.19x at
t=6.09, because spreads track market stress; clusters describe book shape, not
stress. Hence the difference.)

But turnover DOES vary by cluster — C0 0.282 / C1 0.326 / C2 0.333 / **C3 0.366**.
C3 is the highest-trading state. So widening in C3 removes real turnover. The live
question is therefore not "can clusters time spreads" (no, 1.9%) but **"does
widening in high-turnover clusters cut cost more efficiently than widening
uniformly?"** Both sides measured in cost and turnover — precise, no IR.

## Design

### Exhibit 1 — Identification
The PIT-vs-hindsight table and the test-set cluster distribution. Gates
everything: C0/C1 results are null by construction and must be labelled as such.

### Exhibit 2 — Ceiling
Raw-vs-detrended spread-by-cluster table; the 1.9% bound derived.

### Exhibit 3 — The 4x8 grid (the core)
Per strategy, for cluster c in {0,1,2,3} and band level
E in {10, 15, 20, 25, 30, 40, 60, 100}: widen the band to E **only in cluster-c
months**, baseline E=10 elsewhere. `E=10` is the no-op band (top-100 of top-1000
= top 10%), i.e. plain monthly rebalancing; level 10 therefore recovers baseline
and is the internal control.

Every cell reported **three** ways:
1. **raw cost saved** (bp/yr)
2. **cost saved per month of widening** — because raw saving is mechanically
   proportional to the cluster's share of months, so C2 (63%) "wins" by being
   common. That is arithmetic, not insight, and normalising kills it.
3. **turnover removed**

Return impact rides along as a **reported column**, never a test statistic.

### Exhibit 3b — Expand AND contract at matched turnover (the pure-timing arm)

Exhibit 3 only widens. Contracting alone raises cost, so "expand and contract" is
only meaningful as **reallocation**: contract the band where trading is cheap,
expand it where trading is dear, holding **total turnover fixed by construction**.
This isolates the spread-timing channel — the one the 1.9% ceiling bounds — and
tests it directly rather than by inference.

Construction: choose per-cluster widths `E_c = 10 * m_c` with the multiplier
vector `m` tilted toward widening in high-spread clusters and tightening in
low-spread clusters, then solve a single global scale so realised total turnover
matches the uniform-band turnover at each of the 8 levels. Compare cost at that
matched turnover. Any cost difference is **pure timing** and must be <= ~1.9%.

Note C0 has n=1 on the test set, so the "contract" leg is exercised almost
entirely through C1 (26 months, and only 44% identifiable). Report this as a
power limitation, not a result.

### Exhibit 4 — Uniform frontier (the matched test)
Uniform band E in {10,12,15,18,22,26,30,35,40,50,60,80,100} traces a curve of
(turnover removed, cost saved). Interpolate it. Then ask whether each cluster cell
sits **above** that curve **at the same turnover removed**. That is the only
meaning "more efficient" can have here, and both axes are precise.

### Exhibit 5 — Reconciliation
Does the measured cluster-minus-uniform gap fall inside the 1.9% bound?
**Falsification condition, stated in advance:** if it lands materially outside the
bound, the ceiling arithmetic is wrong and the study has found something real.

## Method

- **PIT labels**: expanding refit (`_pit_month_labels`), centroids from pre-t data.
- **Costing**: DNMV / Novy-Marx-Velikov effective half-spread, `cost = sum |dw| * hs`
  (`banding_study.price_ledger`). `hs` is decimal in source; bp in the loaded panel.
- **Baseline**: no-op band E=10 = monthly rebalance.
- **Inference**: block bootstrap at **episode** scale (19 C3 episodes, not 39
  months). Report intervals on the statistic actually being claimed.

## Defects to fix before/while building (found 2026-07-17)

1. **`_pit_month_labels` burn-in silently defaults to 0.** Months before the
   24-month history threshold are labelled C0 without assignment (1992-1993 in the
   full sample). Moot for a 2011+ test set but must not be inherited silently —
   assert the test window is fully assigned.
2. **Mean-CI reported beside an IR delta.** `paired_block_bootstrap` returns a CI
   of `mean(a-b)`; `gp_bands.py:127-129`, `cluster_bands.py:180-185`,
   `var_band_test.py:103-108`, `ml_band.py:96-104` all print `excl0` from that CI
   next to an IR-difference point estimate. Different statistics. **This study
   reports cost, so the CI must be on the cost difference.** Do not repeat.
3. **`_boot_p` returns 0.0 for an identically-zero difference**, which then passes
   BH-FDR and prints "significant" on "no difference at all" (source of the fake
   2/35 survivors in `cluster_bands.csv`). Guard it.

## Expected result — stated in advance, both directions publishable

The pie is 13.9bp/yr on average. C3 covers 21.8% of months. The spread-timing
ceiling is 1.9%. So the honest prior is that a C3 band saves **~1-3bp/yr on
momentum** and the cluster curve lies **on** the uniform frontier, not above it.

If so, the result is not "clusters fail" — it is the sharper and more useful
**"on a modern large-cap long-only book there is nothing left to cut."** 14bp/yr
is not a constraint any practitioner manages around, and that is a direct,
quantified answer to the implementability critique: costs are not what binds here.

If the cluster curve lies **above** the uniform frontier by more than the 1.9%
bound permits, the ceiling is wrong and that is a genuine finding.

## Out of scope

- Any IR / risk-adjusted-return significance claim (the 162x problem).
- The return channel — whether C3 stocks rebound — is a separate question with its
  own power characteristics; not this study.
- Novelty relative to Novy-Marx-Velikov (2016 RFS), NMV (2019 FAJ), Detzel-
  Novy-Marx-Velikov (2023 JF), Arnott-Li-Linnainmaa (2024 FAJ). **Unresolved and
  independent of this study's outcome.** Marc must adjudicate.
