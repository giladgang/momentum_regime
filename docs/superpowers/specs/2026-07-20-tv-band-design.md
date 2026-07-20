# Term-Structure-Conditioned Time-Varying Band — Design

**Date**: 2026-07-20
**Branch**: `paper/applied-study`
**Status**: design approved in brainstorming; awaiting written-spec review
**Companion to**: `2026-07-15-regime-conditional-banding-study-design.md`, `2026-07-17-cluster-band-cost-reduction-design.md`

---

## 1. Motivation and what is new

The banding campaign so far has established a robust **null**: a band whose width is
conditioned on the market regime does not beat a well-chosen *static* band out of sample.
The relevant prior attempts and their outcomes:

- `pi_band` / `gp_bands` — 2-state and theory-derived regime widths: regularized WF 0/7; GP
  1/7 CI-excludes-0 (momentum, +0.066/yr, does not survive multiplicity).
- `train_band` — per-stock band `E = f(single momentum z-score, pi)`: regularization drives
  the slopes to 0 (reverts to static). 0/7.
- `cluster_bands` / `cluster_optimal_band` — band conditioned on the discrete momentum-shape
  cluster: 2/35 survive BH-FDR, both artifacts (Δ = 0.000 vs static).
- `ml_band` — learned-weight month band on **market-context** z-features (DD, DISP, REL_N,
  CS, TERM, LVIX, VOL, SKEW) + pi: 0/14 positive-and-significant.

The mechanism behind the null is documented in `turnover_diagnostic`: **trading is
regime-invariant** (XGB one-way turnover ≈ 0.74 calm vs 0.80 panic), so there is little
regime structure in the *trading* for a band to exploit.

**What this study adds that the priors did not test:** the band is a learned **month-level**
function of the **full 12-horizon momentum term structure** — both the long-leg z-curve (the
thesis clustering input) and the cross-sectional mean term structure — plus continuous
`pi_filter`. Every prior learned band used either a *single* momentum summary (`train_band`),
the *discrete* cluster label (`cluster_bands`), or *market-context* features (`ml_band`), never
the continuous 24-dim term-structure shape. The burden is on this richer input to do what the
coarser inputs could not.

**Honest prior**: the base rate for this family is null. The study is therefore staged
(Section 3) so the cheapest possible test can falsify the idea before the learned machinery is
built, and **either outcome is a clean paper result**: a positive result overturns the null on
a richer input set; a null extends the documented null to the richest input set and the
perfect-hindsight ceiling, strengthening the paper's "static band is enough" conclusion.

**Target outlet**: practitioner journal (JPM / FRL).

## 2. Research question and pre-registered decision rule

**Question**: Does a band whose entry/exit percentiles are a learned function of the 12-horizon
momentum term structure and `pi_filter` beat a static band, net of costs, out of sample?

**Pre-registered gates** (record expected directions in `paper/results/PAPER_NOTES.md`
*before* running Gate 1):

- **G0 (ceiling)** — The **in-sample feature-conditioned oracle** band beats the static band
  by more than the ~2.9% spread-timing artifact in annualized net IR. The oracle is a *function
  of the features* fit on the full sample (Section 3), **not** a per-month argmax — a per-month
  choice uses month identity (infinite capacity) and is not a valid ceiling for a feature
  function. G0 is a *necessary* condition: if the best in-sample function of these features
  cannot beat static, no learnable real-time band can, and Gate 1 is not run.
- **G1 (realizable)** — At least one trainer's **walk-forward OOS** learned band beats the
  static band in net IR, the paired block-bootstrap CI on (learned − static) excludes 0, and
  the win survives 1-SE regularization toward static.
- **G2 (mechanism, reported regardless)** — Independent recomputation of turnover in calm vs
  panic months, reproducing (or contradicting) the regime-invariant-trading diagnostic that
  explains the null.

A null on G0 or G1 is reportable and is the expected outcome; the study is not conditioned on
confirming G1.

## 3. Staged experimental design

**Gate 0 — ceiling test (cheap, no learning).** Build the clean-room engine, pass the trust
gate (Section 6), then compute the **feature-conditioned oracle** band — the best band that is
a *function of the features*, fit with full foreknowledge on the whole sample. Two variants,
both valid ceilings (neither uses month identity):
  1. **Bin oracle**: partition months into bins over the term-structure/`pi` feature space
     (e.g. the K=4 thesis clusters, or a coarse grid), assign each bin the single
     `(E_enter, E_exit)` maximizing that bin's in-sample net IR. Model-agnostic upper bound on
     what any function of these features can do.
  2. **Full-sample fit**: fit the Trainer-A flexible function on the entire sample (no OOS
     split), maximizing in-sample net IR. The best the chosen function class can do.
Report each variant's net-IR uplift vs the best static band. This mirrors
`cluster_optimal_band.py`'s "strongest case" logic applied to the continuous term-structure
input. **Proceed to Gate 1 only if G0 clears.**

**Gate 1 — learned real-time band (only if G0 clears).** Train the three method families
(Section 5), evaluate walk-forward OOS (Section 7), and test G1. If G0 clears but no trainer
reaches the ceiling OOS, the gap between ceiling and realized is itself the finding (signal is
real in-sample, not learnable in real time).

## 4. Band mechanic and portfolio construction

- **Universe / construction**: long-only, value-weighted, top-decile (`k = 10%`) book on
  `score_pi` (main model, DD regime), measured as **active return vs the value-weighted
  universe**. `EVAL_YEARS = 2011..2025`; OOS stitched from 2013 (60-month train guard).
- **Output**: per month the policy emits a pair `(E_enter,t, E_exit,t)` in rank-percentile
  units, with `E_enter ≤ E_exit`.
- **Mechanic (NMV hold-band, floating count)**: keep a held name while its `score_pi` rank%
  `≤ E_exit`; add a non-held name while rank% `≤ E_enter`. Holdings count floats between the
  two thresholds — no fill-to-k. This is the literature `nmv_band` behavior.
- **Static baseline**: identical mechanic frozen at `(E_enter = 10%, E_exit = E*)`, with `E*`
  grid-selected on the training window. The *only* difference under test is whether the pair
  moves with the state.
- **Turnover / cost**: drift-adjust prior holdings by realized return before retrading;
  `cost_t = Σ_i |Δw_i,t| · half_spread_i,t`. Primary cost = measured half-spreads; robustness
  column = flat 10 bp.

## 5. Features and the three trainers

**Features (25-dim, all PIT, standardized on the training window only):**

- 12 — long-leg **z-curve**: current decile picks' mean momentum at each horizon minus the
  cross-sectional mean, in cross-sectional-std units (the thesis clustering input).
- 12 — cross-sectional **mean term structure**: universe-mean momentum at each horizon.
- 1 — `pi_filter` (continuous).
- **Robustness arm (low-DOF)**: compress the 24 momentum inputs to level / slope / curvature
  (3) + `pi` = 4 inputs. Run alongside the full 25-dim inputs.

**Objective (all methods)**: annualized net-of-cost active IR,
`IR = mean(r_t)/std(r_t)·√12`, `r_t = (book−bench) − cost`. Net IR is the *selection and
evaluation* metric for every method (rationale: banding is a small-mean / high-variance,
benchmark-relative problem — IR is the standardized effect size the significance test is built
on, and is leverage-invariant; raw active return is *reported alongside* but not optimized).

**Trainer A — linear-exp policy, fit by direct net-IR search.**
`E_enter,t, E_exit,t = clip(base · exp(w·x_t), lo, hi)` with a shared/parallel weight map per
output. Parameters `(base_enter, base_exit, w_enter, w_exit)` fit by **black-box optimization**
(CMA-ES, Nelder-Mead fallback) maximizing *training-window* net IR directly. This is the
literal "maximize net IR" objective; low enough dimensional for black-box to be feasible.

**Trainer B — gradient-boosted trees (supervised).** A GBM maps `x_t → (E_enter,t, E_exit,t)`.
Because net IR is non-differentiable through the discrete rank band, the GBM cannot ingest it
directly; it is trained on a **proxy target**: the ex-post net-IR-optimal `(E_enter, E_exit)`
computed on a **trailing window** ending at each month (a single month's argmax is pure noise;
the trailing window is the smoothing choice). Selected/evaluated by OOS net IR like the others.

**Trainer C — ridge regressor (supervised).** Same trailing-window target as B, linear model.
The simplest supervised baseline; isolates how much of any B result is nonlinearity vs the
target definition.

## 6. Clean-room engine and trust gate

- New module `paper/tv_band.py`, importing **nothing** from `execution.py` / `banding_study.py`
  — only numpy / pandas + the raw artifacts. If the documented null is an engine bug,
  independent code on the same inputs disagrees.
- **Inputs (reused; data is not in doubt, and re-pulling hits WRDS — out of scope)**: the
  monthly `paper/results/xsec/xsec_*_DD.parquet` frames (`date, permno, me, pi, score_pi,
  mom_12, ret_fwd`) and `paper/results/s5/half_spreads.parquet` (`permno, ym, hs`). The
  12-horizon momentum inputs are read from `paper/results/data/stocks.parquet`
  (`CL.MOMS` columns) joined on `(date, permno)`.
- **Trust gate (blocking)**: the no-band `monthly` policy reproduces `walk_returns.csv`
  `strat_ret − bench_ret` for `rule_r`/`DD` to ≤ 1e-6 per month. No band result is computed or
  trusted until this passes.

## 7. Validation

- **Walk-forward**: expanding window; all policy parameters (static `E*`, Trainer A weights,
  Trainer B/C models and their trailing-window targets) re-fit **annually on prior months
  only**; realized year-Y active returns stitched into one OOS series (2013→2025).
- **Statistics**: paired block-bootstrap CI (block 12, 10,000 reps) on the (learned − static)
  net-IR difference. Report per trainer: OOS net IR of monthly / static / learned, the
  learned-minus-static delta, its CI, and whether the 1-SE-regularized fit still prefers the
  learned band over static.
- **Honesty guard**: the learned band is declared a win only if the delta is positive, its CI
  excludes 0, **and** it survives the 1-SE regularization toward static — the guard that
  neutralized the search asymmetry in the prior campaign.

## 8. Code architecture

`paper/tv_band.py`, self-contained, staged behind `--stage`:

- `load_inputs()` — build the monthly panel (xsec + 12-horizon moms + spreads); no external
  band code.
- `simulate(panel, policy) -> (monthly_df, ledger)` — the from-scratch loop: drift-adjust,
  members from `(E_enter, E_exit)`, VW target weights, gross active, turnover, per-trade
  ledger. Reproduces `monthly` gross (trust gate).
- `features(panel) -> x_t` — 25-dim (and 4-dim compressed) PIT feature matrix.
- `ceiling(...)` — Gate 0 perfect-hindsight band and its net-IR uplift vs static.
- `train_A/B/C(...)`, `walkforward(...)` — Gate 1 trainers and the stitched-OOS harness.
- `bootstrap_ci(...)`, `report(...)` — CI and `paper/results/banding_study/tv_band.{md,csv}`.

**Stages**: `--stage gate0` (trust gate + ceiling), `--stage gate1` (trainers + WF),
`--stage report`. Gate 1 runs only after Gate 0 clears.

## 9. Sequencing

1. `load_inputs` + `simulate` + **trust gate** — must pass before anything else.
2. **Gate 0** ceiling; record G0 outcome in `PAPER_NOTES.md`.
3. If G0 clears: register G1/G2 expectations, then build/run the three trainers and the WF
   harness.
4. **G2** mechanism diagnostic (independent calm-vs-panic turnover), reported regardless.
5. `report` + write results into the paper's banding section.

## 10. Risks and open questions

- **Dimensionality**: 25 inputs × 2 outputs is a large overfit surface for ~250 months / annual
  refits. Mitigations: the 1-SE-vs-static gate, the 4-dim compressed robustness arm, and G0 as
  a necessary pre-filter (no OOS run if the ceiling is absent).
- **Trailing-window target (B/C)**: window length is a free knob; report sensitivity (e.g.
  24/36/60 months) rather than tuning to the best.
- **Floating count**: a moving `E_enter` changes book breadth and therefore active risk, not
  just cost — net IR captures this, and raw active return / count are reported alongside so the
  mechanism is legible.
- **12-horizon momentum source**: confirm `stocks.parquet` carries all 12 horizons PIT-aligned
  to the xsec formation dates; if a horizon is missing, fall back to the horizons available in
  the xsec frames and note the reduced term structure.

## 11. Out of scope

- Re-pulling data from WRDS (data is not what is in doubt; account-lockout risk).
- Short-leg / long-short construction (paper is long-only).
- The other 6 strategies (main-model focus); a momentum-only reference anchor may be added if
  the engine's behavior needs a sanity check, but is not a headline.
- Liquidity-provision / cost-aware-substitution levers (separate, already-documented lines).
