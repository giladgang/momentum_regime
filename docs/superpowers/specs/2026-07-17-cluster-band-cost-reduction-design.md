# Cluster-conditioned banding for transaction-cost reduction — design

**Date:** 2026-07-17
**Status:** design, pending approval
**Owner:** Gilad Gang (advisor: Marc Stam)

## Question

Using the thesis's k=4 momentum-shape clusters, can we widen (or tighten) the
no-trade band in each cluster — trade less in some market states, more in others
— to reduce the transaction cost of the 7 benchmark strategies? Test each cluster
one at a time.

**Cost is the outcome variable, not information ratio.** Cost is measured directly
(turnover x spread, precise to under a basis point). Cost-driven differences in IR
are provably undetectable on this book — cost moves IR by <=0.03 against a 0.054
noise floor, a 162x gap (clean-room rebuild, 2026-07-17). Measuring cost directly
sidesteps that entirely.

## Sample

**2011-01 .. 2024-11, 167 months.** This is the thesis test set AND the native
coverage of the thesis cluster labels (which end 2024-11, the thesis data cutoff).

Consequence, stated up front: 2011-2024 is the low-spread era. Measured top-1000
median half-spread by decade: 51.3bp (1990s) / 3.98bp (2000s) / 1.28bp (2010s) /
1.58bp (2020s). The whole sample sits in the ~1.3-1.6bp world. This is not a flaw;
it is the study's central quantitative finding (see "Expected result").

Extension to 2025-11 was considered and deferred: the thesis clusters derive from
NYSE-decile splits of the PRODUCTION XGB score (`scripts/zscore_time_by_horizon.py`,
`artefacts/cs_artefacts_data.pkl`), so labelling 2025 requires re-scoring the
thesis XGB model on 2025 stocks and rebuilding the long-short z-curve feature — a
mini-pipeline with divergence risk, buying ~3 extra C3 months. Not worth it for a
result bounded at ~3%. Can be added later as an isolated task if 2011-2025
coverage is wanted for cross-exhibit consistency.

## Cluster labels — the thesis object (authoritative)

**Source: `results/thesis/cluster_k4_member_dates.csv`** (date -> cluster in
{0,1,2,3}), with frozen centroids in
`results/thesis/cluster_k4_descriptor_table.csv` (`z_centroid_mom_1..12`).

- **k=4, fit on the long-short book's cross-sectional momentum term-structure**
  (XGB-decile legs, z-scored per horizon). Distinct object from the paper's
  long-only book; do NOT re-derive from `paper/src/clusters.py`.
- Cluster identities from the frozen centroids (z_centroid across mom_1..12):
  **C0 = winner** (+0.26..+0.80), C1 = mild-winner (~+0.1..+0.3),
  C2 = mild-loser (~-0.3), **C3 = deep loser** (-0.73..-0.87 across all horizons).
- **These labels are a FULL-SAMPLE (2011-2024) fit** => conditioning on them uses
  information not available in real time. Every result is therefore an **in-sample
  UPPER BOUND** on the cost a cluster-timed band could save: "with perfect
  hindsight knowledge of the cluster, widening in cluster c saves at most X."
  If even the hindsight upper bound is below the ceiling (which the ~3% bound
  predicts), no real-time version can do better — a clean, strong negative.
  This framing was chosen deliberately (Gilad, 2026-07-17): use the thesis
  clusters as-is, report the upper bound.

### Per-cluster facts on the momentum test set (measured, thesis labels)

| thesis cluster | months (mom) | book momentum | turnover/yr | raw half-spread | cost bp/yr |
|---|---|---|---|---|---|
| C0 (winner) | 37 | 0.275 | 3.68 | 1.30bp | 12.1 |
| C1 | 44 | 0.189 | 4.25 | 1.29bp | 13.7 |
| C2 | 20 | 0.326 | 4.59 | 1.50bp | 16.7 |
| C3 (deep loser) | 17 | 0.046 | 3.85 | 1.60bp | 14.2 |

(118 of the 167 months carry a label on the momentum strategy; cluster counts
vary slightly by strategy. C3 is the smallest cell — 17 months — so inference on
C3 is the least powered.)

Note two things the labels changed from an earlier draft built on re-derived
clusters:
1. **C3 is NOT the highest-turnover cluster** (C2 is, 4.59 vs 3.85). The rationale
   "widen in C3 because it trades most" does NOT hold on the thesis labels. C3's
   case rests on it being the highest-SPREAD state (1.60bp), i.e. trade less where
   trading is dearest.
2. The "99% real-time identifiable" figure was for the re-derived PIT clusters,
   NOT these thesis labels. It does not apply here and is removed.

## The ceiling (the interpretive frame — and, per the novelty check, the actual contribution)

Cost = sum_t (turnover_t x spread_t). A state-conditional band can only change
*when* you trade, so it can only exploit **spread variation across states**.

Detrended (year-demeaned log) half-spread faced by the book across the thesis
clusters: **max/min = 1.052x => cost saving from cluster timing is capped at
~2.9%.** Arithmetic, not an estimate. The raw cross-cluster spread differences are
mostly calendar (spreads fell ~40x over the full sample; within 2011-2024 the
residual trend still loads on cluster).

Contrast: the HMM regime carries real spread information (1.19x, t=6.09) because
spreads track market stress; clusters describe book shape, not stress. Hence the
cluster ceiling is even tighter than the regime ceiling.

**The novelty check (2026-07-17) makes the ceiling the paper's contribution, not
a caveat.** Regime/cluster-conditioned no-trade banding for cross-sectional equity
factors is genuinely unclaimed as a method — NMV banding is static (one knob
away); GP (2013) and CDS (2020) use quadratic-impact costs under which no
no-trade band exists at all, so this is NOT a special case of their frameworks.
But "novel method, null result" is weak on its own; a referee will ask "is the
null your specific labels or fundamental?" **Only a ceiling framed as a GENERAL
bound answers that** — the maximum cost any state-conditioning of factor execution
can save over static banding. Lead with the ceiling; the cluster grid confirms it.

## Design — exhibits

### Exhibit 1 — Cluster labels & cost budget
The frozen-centroid identities, the per-cluster table above, and the total cost
budget per strategy (13.9 bp/yr mean; 1.5-38 bp/yr range). States the whole pie:
everything downstream is a fraction of this.

### Exhibit 2 — The ceiling
The raw-vs-detrended spread-by-cluster table; the ~2.9% bound derived; the
regime-vs-cluster contrast; the general-bound framing.

### Exhibit 3 — The 4x8 grid (the core; every cluster, one by one)
For each strategy, each cluster c in {0,1,2,3}, and each band level
E in {10, 15, 20, 25, 30, 40, 60, 100}: widen the band to E **only in cluster-c
months**, baseline E=10 (= no-op band = plain monthly rebalance) elsewhere.
Level 10 recovers baseline and is the internal control. 7 x 4 x 8 = 224 cells.

Each cell reported **three** ways:
1. **raw cost saved** (bp/yr)
2. **cost saved per month of widening** — raw saving is mechanically proportional
   to a cluster's share of months, so a common cluster "wins" by frequency alone.
   Normalising per treated-month removes that artifact.
3. **turnover removed**

Return impact rides along as a **reported column**, never a test statistic.

### Exhibit 3b — Expand AND contract at matched turnover (pure-timing arm)
Widening alone only ever cuts cost; "expand and contract" is meaningful only as
reallocation: contract the band where trading is cheap, expand where it is dear,
holding **total turnover fixed by construction**. Choose per-cluster multipliers
`m_c` tilted to widen high-spread clusters / tighten low-spread ones, solve one
global scale so realised total turnover matches the uniform band at each level,
compare cost. Any difference is **pure timing** and must be <= ~2.9%.
(C0 is winner-tilted and low-spread, so the "contract" leg mostly acts through
C0/C1; report as a power limitation, not a result.)

### Exhibit 4 — Uniform frontier (the matched test)
Uniform band E in {10,12,15,18,22,26,30,35,40,50,60,80,100} traces a curve of
(turnover removed, cost saved). Interpolate it. Ask whether each cluster cell sits
**above** that curve at the **same turnover removed** — the only meaning "more
efficient" can have. Both axes precise.

### Exhibit 5 — Reconciliation
Does the measured cluster-minus-uniform gap fall inside the ~2.9% bound?
**Falsification condition, stated in advance:** if it lands materially outside the
bound, the ceiling arithmetic is wrong and the study has found something real.

## Method

- **Labels**: join `cluster_k4_member_dates.csv` (month-end aligned) to each
  strategy frame. In-sample by construction (see above); do not re-derive.
- **Costing**: DNMV / Novy-Marx-Velikov effective half-spread,
  `cost = sum |dw| * hs` (`banding_study.price_ledger`). `hs` decimal in source,
  bp in the loaded panel.
- **Baseline**: no-op band E=10 = monthly rebalance.
- **Engine**: `execution.simulate` with the existing `cluster_band` policy
  (arg = (E_0..E_3)); a "widen cluster c only" cell is
  `arg = (10,..,E at index c,..,10)`.
- **Inference**: block bootstrap on the **cost** difference (the statistic being
  claimed), block >= 12 to respect episode clustering (C3's 17 months are few
  episodes). Report intervals on cost, never borrow an IR CI.

## Defects to fix before/while building (found 2026-07-17)

1. **Mean-CI reported beside an IR delta.** `paired_block_bootstrap` returns a CI
   of `mean(a-b)`; `gp_bands.py:127-129`, `cluster_bands.py:180-185`,
   `var_band_test.py:103-108`, `ml_band.py:96-104` all print `excl0` from that CI
   next to an IR-difference point estimate — different statistics. This study
   reports COST, so the CI must be on the cost difference. Do not repeat the bug.
2. **`_boot_p` returns 0.0 for an identically-zero difference**, which then passes
   BH-FDR and prints "significant" on "no difference at all" (the fake 2/35
   survivors in `cluster_bands.csv` were exactly this). Guard it.
3. **`_pit_month_labels` burn-in defaults to 0** — irrelevant here (thesis labels
   used directly, not `_pit_month_labels`), but do not accidentally reintroduce it.

## Expected result — stated in advance, both directions publishable

Pie ~13.9 bp/yr average. C3 covers ~14% of labelled months. Spread-timing ceiling
~2.9%. Honest prior: widening in any single cluster saves ~1-3 bp/yr on the
higher-turnover strategies and the cluster curve lies **on** the uniform frontier,
not above it.

If so, the result is not "clusters fail" — it is **"on a modern large-cap
long-only book there is nothing left to cut, and here is the general bound that
says why."** 14 bp/yr is not a constraint any practitioner manages around; that is
a direct, quantified answer to Marc's implementability critique.

If a cluster cell beats the uniform frontier by more than ~2.9% at matched
turnover, the ceiling is wrong and that is a genuine finding.

## Out of scope

- Any IR / risk-adjusted-return significance claim (the 162x problem).
- The return channel (whether C3 stocks rebound) — separate question, separate
  power profile; not this study.
- **Novelty adjudication.** The METHOD is genuinely unclaimed (verified 2026-07-17:
  not prior art in NMV static banding, not a special case of GP 2013 / CDS 2020).
  But a null carries only if the ceiling is the contribution. Whether the finished
  package clears JPM/FRL is Marc's call, and DanielJagannathanKim2019 (regime
  momentum) is in the bib — the SIGNAL story is pre-empted; the EXECUTION story is
  not. This study delivers the execution/ceiling result; it does not resolve venue.
