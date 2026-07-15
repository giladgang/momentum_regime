# Regime-Conditional Banding Across Strategies — Design

**Date**: 2026-07-15
**Branch**: `paper/applied-study`
**Status**: design approved in brainstorming (Sections 1–3); awaiting written-spec review
**Companion to**: `2026-07-13-applied-paper-design.md`, `2026-07-14-s6-regime-banding-design.md`

---

## 1. Motivation and contribution

The applied study so far centers on one model (the regime-aware XGB selector). Gilad's
reframing: **XGB is just one strategy that happens to profit in these market conditions.**
The transferable, publishable contribution is the *mechanism* on top of it —
**regime-conditional banding**: taking a strategy's target portfolio and rebalancing it
less aggressively (a no-trade band), with the band width conditioned on the market regime.

This directly answers Marc Stam's two live points (email thread through 2026-07-15):

1. **Implementability of a panic-concentrated edge.** Marc's concern: trading in panic is
   costly and hard (wide spreads, thin liquidity, one-sided sell-offs). A crash/recovery
   decomposition of the applied model (run 2026-07-15 on `paper/results/walk_returns.csv`;
   scratch script to be promoted into `paper/` as part of this work) shows the active return
   is **entirely a recovery-phase phenomenon** (panic-recovery active +1.65%/mo, t=2.1;
   panic-crash active -0.96%/mo; calm ~0). So the profitable action is *holding through the crash and being
   positioned for the rebound*, not trading into the storm. Regime-conditional banding
   operationalizes exactly that: trade little (wide band) while the market falls; re-rank
   (tight band) as recovery begins.
2. **The "investable narrative" / cost-mitigation angle** Marc pushed from the first email:
   a strategy that adds value by *lowering* turnover, not by beating the market gross.

The framing follows Detzel, Novy-Marx & Velikov (2023, JF), "Model Comparison with
Transaction Costs" (in `~/Downloads`): gross alpha does not indicate a tradable opportunity;
only strategies that survive costs expand the *achievable* frontier. The banding methodology
follows Novy-Marx & Velikov (2016, RFS, "A Taxonomy of Anomalies and Their Trading Costs")
and (2019, FAJ, "Comparing Cost-Mitigation Techniques").

**Target outlet**: practitioner journal (JPM / FRL), consistent with the collaboration plan.

## 2. Research question, claim, and pre-registered gates

**Question**: Is regime-conditional banding a *general* cost-mitigation technique — does it
improve net-of-cost performance across a spectrum of standard strategies, not just XGB?

**Pre-registered expectations** (registered before running, per the S6 discipline; expected
directions recorded in `paper/results/PAPER_NOTES.md` before results):

- **G1** — Regime-conditional banding raises net-of-cost information ratio versus the
  *static* (regime-agnostic) optimal band for a majority of the 7 strategies at the measured
  spread cost level; the paired improvement's bootstrap CI excludes zero.
- **G2** — The improvement is **larger for high-turnover / high-cost strategies**
  (low-volatility, short-term reversal) than for low-cost strategies (value). This is the
  core prediction and mirrors DNMV's own theme that high-cost signals are where cost
  mitigation matters.
- **G3** — Combining the 7 strategies into one net-of-cost book, the achievable frontier
  (max net Sharpe) is higher under regime-conditional banding than under static banding.

A null result on G1/G2 is itself reportable ("regime timing of the band does not beat a
well-chosen static band"), so the study is not conditioned on confirming the claim.

## 3. Experimental design

**Unit of analysis = one cell:** `(strategy, banding policy, cost level)`. Each cell runs
through the existing `paper/execution.py::simulate()` engine and is priced net of cost.

**Banding tiers** (per strategy):

1. **No band / monthly** — full rebalance to target each period. Upper turnover bound.
2. **Static optimal band** — buy/hold band (NMV 2016/2019), a single band width chosen to
   maximize net-of-cost IR, *regime-agnostic*. The literature baseline.
3. **Regime-conditional band** — band width switches on the regime. The **direction is
   explored as a grid** (calm-width x panic-width), reporting which cells sit on the
   net-of-cost frontier, rather than assumed. Includes a **3-state crash/recovery variant**:
   wide band while the market is still falling (do not churn in illiquidity), tighter band
   once recovery begins (re-rank into the rebound).

**Cost levels** (columns): `0 bp`; the **measured stock-level spread** (S5 half-spreads,
calm ~1.3 / panic ~1.5 bp, regime-conditional); a **panic-stress multiple (3–5x)** applied to
the panic spread to pre-empt Marc's illiquidity critique.

**Headline** (frontier summary): combine the 7 strategies into one book and report max net
Sharpe under no-band vs static-band vs regime-band. Primary construction is a transparent
**IR-weighted combined book**; a full net-of-cost mean-variance solve (DNMV eq. A.10, numeric)
is a robustness row.

## 4. Strategies and signals

Each strategy is expressed as a **stock-level score** (higher = long side), fed to the same
engine. Sign and cadence follow the source literature.

| Strategy | Signal (long side) | Cadence | Data source |
|---|---|---|---|
| Momentum (12–1) | cumulative return t-12 to t-2, high | monthly | on disk (`stocks.parquet` `mom_*`) |
| Short-term reversal | prior-month return, **low** | monthly | on disk (`mom_1`) |
| Low-volatility (IVOL) | FF3-residual vol, prior ~90 trading days, **low** | monthly | daily `dsf_v2_*` (on disk) |
| XGB (regime-ML) | `score_pi`, high | monthly | on disk (`paper/results/xsec`) |
| Value (B/M) | book equity / market equity, high | annual (June) | **Compustat pull (gated)** |
| Profitability | cash-based operating profitability (Ball et al. 2016), high; plain operating profitability as robustness | annual (June) | **Compustat pull (gated)** |
| Investment | asset growth (ΔAT), **low** | annual (June) | **Compustat pull (gated)** |

**Long-only** portfolios (per the applied setting). Banding reduces trading *on top of* each
strategy's native rebalance cadence (fundamentals annual-June; the rest monthly).

## 5. Data and the WRDS pull — HARD APPROVAL GATE

> **GATE (must not be skipped): No WRDS / Compustat session runs without Gilad's explicit
> per-session approval.** Gilad has authorized WRDS access with advance notification. Before
> any pull, present (a) exact tables/fields/date range, (b) estimated row count / runtime,
> (c) where output lands, and wait for an explicit "yes". Per repo policy, WRDS is never
> initiated unprompted.

**Pull spec (to be approved before execution):**

- **Table**: `comp.funda` (annual Compustat North America), `1985–2025` (≥5 years of history
  before 1990 for lagged asset growth and PIT alignment). Standard filters:
  `indfmt='INDL', datafmt='STD', popsrc='D', consol='C'`.
- **Fields**:
  - Book equity: `ceq, txditc, pstkrv, pstkl, pstk, seq, at, lt`.
  - Profitability: `revt, cogs, xsga, xint` (operating); `+ che, rect, invt, ap, xacc` for
    the cash-based variant.
  - Investment: `at` (current and lagged).
- **Link**: `crsp.ccmxpf_linktable` (`linktype in ('LU','LC')`, primary; `linkprim in ('P','C')`),
  merging gvkey↔permno onto the permnos already in the panel. Respect `linkdt`/`linkenddt`.
- **PIT discipline**: accounting data assumed known with a **6-month reporting lag**; June-t
  rebalance uses the fiscal-year-end that is at least 6 months stale (standard FF timing).
  No look-ahead; verify per-firm.
- **Market equity** for B/M: CRSP `me` already in the panel (FF December-ME convention).

**No new data** is required for momentum, reversal, low-vol, or XGB — those build from
`stocks.parquet`, the daily `dsf_v2_*` files, and existing `xsec` scores.

## 6. Banding methods (detail)

- **Static optimal band**: buy/hold rule — a stock enters when its rank crosses into the
  decile; it is held until its rank falls outside a keep-band `E` (percentile). One `E`
  chosen to maximize net-of-cost IR over the sample. This is `nmv_band` in `execution.py`
  with a single regime-agnostic `E`.
- **Regime-conditional band**: `E` switches on the regime. Two-state form
  (`E_calm, E_panic`) already exists (`nmv_band`, `pi_band`). Three-state crash/recovery form
  is **new** and must be added: `E_calm, E_crash, E_recovery`.
- **Grid**: sweep (`E_calm` x `E_panic`) and the 3-state (`E_calm, E_crash, E_recovery`) over
  a coarse grid; report the net-of-cost frontier and the best cell per strategy.

## 7. Regime model and crash/recovery labeling

- **Regime signal**: the main applied model — single input **market drawdown (DD)**,
  `pi >= 0.5` = panic. Unchanged (per project "main model only" direction).
- **Crash vs recovery** (within panic), defined on the market's *own* drawdown path
  (exogenous to any strategy): a panic month is **recovery** if the market drawdown is
  healing (`dd_t > dd_{t-1}`, equivalently market return > 0 that month), else **crash**.
  Both definitions gave identical buckets in the 2026-07-15 decomposition; use the drawdown
  trajectory as primary and report the market-sign version as a robustness check.
- Labels use only information available at formation (contemporaneous drawdown state), so
  the 3-state band is implementable, not look-ahead.

## 8. Cost model and evaluation

- **Cost model** (DNMV eq. A.4–A.7, already in `simulate()`): one-way stock cost `c_it`;
  `TC_t = sum_i |w_it - w~_{i,t-1}| * c_it`; `net = gross - TC`; turnover `= 1/2 sum |Δw|`.
  Long-only, so costs are paid on both buys and sells of the drifted book.
- **Regime-conditional costs**: `c_it` scaled to the panic spread in panic months; stress
  column multiplies the panic cost by 3–5x.
- **Per-cell metrics**: gross IR; net IR at each cost level; turnover (total / calm / panic);
  **breakeven cost (bp)**; net long-leg factor alpha (FF5-LO, FF6-LO) with block-bootstrap t.
- **Key statistic**: Δ(net IR) of regime-band minus static-band, with a **paired moving-block
  bootstrap CI** (`paper/execution.py::paired_block_bootstrap`).
- **Frontier summary**: IR-weighted combined book (headline) and a net-of-cost MVE solve
  (robustness). Trading-diversification netting across the 7 signals (DNMV eq. A.8–A.9)
  applies only in the combined-book frontier, not the per-strategy tables.

**Spread estimator**: primary = existing S5 quoted half-spreads. The DNMV Hasbrouck–Gibbs
Bayesian estimator on the daily data is an **optional** robustness deliverable, not a blocker
(decision: use existing S5 spreads).

## 9. Universe, period, portfolio construction

- **Universe**: large-cap. Primary top-1000; top-500 (S&P 500-scale) as robustness (a top-500
  walk is already running).
- **Period**: **1990–2025** (extends the 2011–2025 applied walk) — needed for fundamental-
  anomaly power and to span more regime episodes; the regime + XGB expanding-walk
  infrastructure already supports 1990–2025.
- **Weighting**: cap-weighted within the selected decile, with the existing weight cap logic
  (`_target_weights`, `cap`).
- **Expanding window**, time-ordered splits (no random splits, no look-ahead).

## 10. Code architecture

Reuse the existing generic engine; add strategy scoring and the 3-state band. All new code
under `paper/`, following the applied-study layout. Nothing under `scripts/`, `results/`,
`tables/`, `latex/`, `data/` is touched (thesis side stays clean).

- `paper/src/signals.py` (**new**) — one function per strategy returning a stock-level score
  panel with a common schema `(date, permno, me, score, pi, ret_fwd)`. Momentum / reversal /
  low-vol / XGB from existing data; value / profitability / investment from the (gated)
  Compustat merge. One clear interface; each signal independently testable.
- `paper/src/fundamentals.py` (**new**) — Compustat load + CCM link + PIT-lagged book equity,
  profitability, asset growth. Isolated so the WRDS dependency lives in one place.
- `paper/execution.py` (**extend**) — add the 3-state `regime_band` policy (`E_calm, E_crash,
  E_recovery`) alongside the existing `nmv_band` / `pi_band`. No change to the pricing/ledger
  core.
- `paper/banding_study.py` (**new**) — the driver: builds the cell matrix (strategy x band x
  cost), runs `simulate()` per cell, assembles per-strategy tables and the combined-book
  frontier, writes `paper/results/banding_study/`.
- `paper/config.py` (**extend**) — grids (`E` sweeps, cost levels, stress multiples), universe
  size, period, paths.
- **Tests** (`paper/tests/`): `test_signals.py` (each score's sign convention + schema + row
  counts on a fixture), `test_fundamentals.py` (PIT lag correctness — assert no look-ahead),
  `test_regime_band.py` (3-state band membership logic on a hand-built fixture; crash/recovery
  labeling). Mirror existing `test_banding.py` style.

## 11. Sequencing

Per Gilad's choice, **build the full 7-strategy set together** (not prototype-first). Ordering
within that:

1. Register G1–G3 expectations in `PAPER_NOTES.md`.
2. Build `signals.py` for the 4 no-WRDS strategies; validate against known applied numbers.
3. **[GATE]** Get WRDS approval; run the Compustat pull; build `fundamentals.py` + the 3
   fundamental signals.
4. Add the 3-state `regime_band`; run the full cell matrix (`banding_study.py`).
5. Assemble per-strategy tables + combined-book frontier; evaluate gates; write results.

## 12. Risks and open questions

- **Spread estimator mismatch** with DNMV (quoted vs Hasbrouck–Gibbs) — flagged; robustness
  only.
- **Low-vol at monthly resolution** — proper IVOL needs daily FF3 residuals; the `dsf_v2_*`
  daily files support this, but confirm coverage back to 1990.
- **PIT correctness** for fundamentals is the highest-risk item — the test asserts no
  look-ahead; spot-check specific firm-years.
- **Grid cost** — the cell matrix is (7 strategies x ~tens of band cells x 3 cost levels);
  cheap relative to the fold grid (seconds–minutes per cell), but log any coverage caps.
- **Regime label stability** at the crash/recovery boundary — report sensitivity to the
  labeling definition.

## 13. Out of scope (this study)

- Gârleanu–Pedersen optimal partial-adjustment trading (a possible future extension; not
  built here).
- Long-short construction (applied setting is long-only).
- Any edit to thesis `.tex` / `scripts/` / `results/` / `tables/` / `latex/` / `data/`.
