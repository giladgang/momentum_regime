# Applied momentum-regime study — design spec

**Date:** 2026-07-13 (brainstorm 2026-07-12/13, all decisions Gilad's)
**Goal:** practitioner paper (JPM / FRL direction, with Marc Stam): does HMM
regime detection add net-of-cost value to a large-cap, LONG-ONLY momentum
portfolio — in particular by informing WHEN trading is worth paying for?
**Process rule:** run first, write later. No paper prose until the numbers
are in. This spec covers the pipeline through headline results + saved
cross-sections; the execution/TC frontier (S5) is designed in detail only
after the main run finishes (Gilad's call).

## Decisions log (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Sequencing | Applied study first; pre-registered thesis walk-forward chains stay parked (launch later, unchanged) |
| 2 | Universe | Top 1000 by point-in-time month-end market cap (Russell 1000 proxy); top-500 rerun as robustness |
| 3 | Yearly combo rule | Plain argmax (highest mean per-fold validation metric); RULE R (1-SE parsimony) computed alongside as a comparison row |
| 4 | Selection CV objective | Re-run fold grid in the applied setting: long-only top-decile IR vs the VW top-1000 benchmark |
| 5 | TC / execution channel | Built AFTER the main run as pure post-processing; the run must persist full monthly cross-sections to make that possible |
| 6 | Data | WRDS refresh through mid-2026 FIRST (Gilad runs/authorizes the pull); refreshed data is the PRIMARY dataset, not a bonus |
| 7 | Code home | New self-contained `paper/` package in this repo; thesis pipeline untouched |
| 8 | Mandate | Fully invested, always; no π-scaled cash overlay (benchmark-relative claim, not market timing) |
| 9 | Architecture | Port validated machinery into `paper/` with bit-reproduction gates (not importlib-wrapping, not an unanchored rewrite) |

## Package layout

```
paper/
  config.py        # ALL knobs: universe size, dates, seeds, cost grid, paths
  pipeline.py      # staged runner, resume-safe per stage, --smoke per stage
  src/
    data_import.py # S0: WRDS pull + build (ported from 2026-07-09-import-ext2025)
    universe.py    # S1: point-in-time top-N membership
    hmm.py         # ported bit-validated Gibbs + posterior-mean forward filter
    selection.py   # S2 fold cells + argmax / rule_r aggregation
    model.py       # XGB ensemble (frozen thesis hyperparameters)
    portfolio.py   # long-only top-decile VW construction + benchmark
    metrics.py     # IR, TE, beta, capture, MDD, turnover, capacity
  tests/           # pytest: bit-gates + schema smokes
  results/         # outputs (binaries gitignored, CSVs tracked)
```

Imports from repo root: `config.py` constants that must match production
(HMM seeds/priors, CRISIS_WINDOWS, XGB hyperparameters, MOM_FEATURES) and
`data/` inputs. Nothing under `scripts/`, `results/`, `tables/` is touched.

## Pipeline stages

**S0 — data refresh (Gilad runs the pull).** Parameterized re-import through
the latest CRSP month (~2026-04/05 given CRSP lag): same WRDS CIZ v2 filters,
`ret_adj := mthret` (delisting-integrated, no Shumway re-application), frozen
winsor bounds and train(<2011) z-stats, market return computed VW from
msf_v2/dsf_v2. Verification: classic-CRSP overlap checks exactly as the
ext2025 import (corr, p99 return diff), plus row-identity vs the existing
ext2025 parquets on their common window. Output: ONE integrated primary
panel + stock file, 1990-12 → latest, under `paper/results/data/`.

**S1 — universe.** Monthly point-in-time membership: common shares
(CIZ equivalents of shrcd 10/11), NYSE/AMEX/NASDAQ, |prc| > $1, rank by
month-end cap, keep top 1000. Momentum deciles are cut WITHIN this universe.
Sanity gate: membership covers ≈90%+ of total CRSP cap each month.

**S2 — selection fold grid (applied objective).** Candidate pool: the frozen
25 combos of `experiments/results/wf_pool.csv` (pre-2011-informed
restriction, disclosed). Folds: biennial 1-7 (validations 1997-2011) +
annual 101-115 (fold 100+k trains < Jan 2010+k, validates year 2010+k;
115 validates 2025). Seeds: 3 HMM × 5 XGB per cell. Validation metric:
**IR of the long-only top-decile portfolio vs the VW top-1000 benchmark
over the fold's validation months** (IR = mean(active)/std(active)·√12).
≈ 25×22×3 = 1650 cells, append-per-cell, resume-safe. All folds re-run
under this objective — no reuse of the L/S thesis cells for selection.

**S3 — yearly walk (2011 → 2026 partial).** For each trading year Y:
selection evidence = folds with validation end ≤ Jan Y; argmax picks the
combo (rule_r recorded alongside); HMM: 50 seeds, panel [1990-12, Dec Y-1],
posterior-mean parameters, crisis-sign panic labels, causal forward filter
through year Y (filtered π only, never smoothed); XGB: 20 seeds on universe
stock-months < Jan Y; score year-Y universe months. Final year 2026 runs on
whatever months have ret_fwd (~Jan-Mar/Apr).
**Persisted per month (the S5 enabler):** full cross-section
{date, permno, cap, universe_flag, score, π} + per-(year,rule) monthly
portfolio and benchmark returns. Resume by (year, rule) with a
complete-block check (all expected months present, not row presence).

**S4 — headline results (gross).** Stitched 2011-2026 monthly series per
rule: active return, tracking error, IR, Sharpe, MDD, rolling 36m beta,
up/down capture, regime-conditional splits (π ≥ 0.5), subperiod table
(GFC-recovery is pre-window; COVID 2020, 2022, Apr-2025 tariff panic),
concentration stats (max weight, effective N, sector exposures).
Comparator set on the same universe/weighting: (i) VW top-1000 benchmark,
(ii) classic 12-1 momentum top decile (no ML), (iii) XGB without π,
(iv) XGB with π (the strategy). Fees are NOT netted here — S4 is gross;
costs enter in S5 per executed turnover under each policy.

**S5 — TC / execution analysis (post-run; detailed design deferred).**
From saved cross-sections only, no refits: one-way turnover accounting,
cost grid (5/10/20 bps) + breakeven bps, capacity table (days-to-trade for
a $1B book at 10% ADV participation), then the execution-policy frontier —
monthly full rebalance vs static banding vs π-conditional banding — over
the 4-comparator × policy grid. Marc's confirmed angle (value THROUGH lower
transaction costs) lives here.

**S6 — paper tables/figures.** Only after numbers are reviewed.

## Applied-integrity rules (binding)

1. XGB hyperparameters frozen at the thesis train-only choices; no retuning
   on the applied window, ever.
2. Every feature gets a real-time availability check during the build
   (market-derived DD/VOL/DISP/REL_N; published daily yields CS/TERM);
   convention stated in outputs: information through month-end t, positions
   formed at t close, return earned over t+1. If any feature is borderline,
   add a one-month-delay robustness row.
3. Selection is mechanical (argmax) on evidence available before the traded
   year; the pool, folds, seeds, and metric are fixed by this spec before
   any S2/S3 result is read.
4. Before the first S3 launch, a dated "registered expectations" block is
   appended to this spec (argmax vs rule_r behavior, π vs no-π ordering) —
   scored in the paper either way.
5. Full-table reporting: no comparator or policy dropped for looking bad.

## Verification gates (each blocks the next stage)

- `hmm.py` reproduces a canonical `prod_budget_*_pi.npz` bit-for-bit for the
  same seeds/features before S2 uses it.
- `selection.py` fold cell reproduces a known `trainonly` cell val_sharpe
  under the L/S config before the objective switches to applied IR.
- S0 overlap checks (above); S1 cap-coverage gate.
- Smoke mode per stage (reduced seeds/iters, separate outputs, deleted after
  inspection); pytest schema tests per module.
- Every stochastic output logs its seeds; reruns keep original seeds.

## Out of scope (disclosed in the paper, not silently missed)

Taxes; intraday execution modeling; round lots; borrow costs (moot long-only
— one sentence as a reason FOR long-only); TE-targeting / optimizers (we
report exposures instead); international replication; π-scaled cash overlay
(mandate is fully invested); short leg entirely.

## Compute plan

8-core/8GB machine, 6-7 workers, nohup + caffeinate, all stages resume-safe.
S2 ≈ 1-2 overnights; S3 ≈ 1 overnight (~25 distinct (year,combo) HMM fits at
50 seeds + fast XGB on ~1000-name cross-sections); S4/S5 minutes. The parked
thesis walk-forward chains are independent and can run on any idle night
once their three audit fixes are applied.

## Registered expectations (2026-07-13, before any S2/S3 result was read)

E1. argmax churns: >= 4 distinct combos across 2011-2025.
E2. Gross active value: IR(score_pi, argmax) > IR(score_nopi) on the stitched walk.
E3. Neither of those gross IRs exceeds 0.8 (large-cap momentum is weaker).
E4. Regime value concentrates in panic months: mean monthly active return of
    the argmax strategy in pi >= 0.5 months exceeds its calm-month mean.
E5. rule_r selects weakly-fewer distinct combos than argmax.
Scored HIT/MISS in the S4 report either way; no rule/combo adoption based on
these outcomes.

## Known expectations to manage

Momentum is historically weaker in large caps — the regime/timing angle,
not raw decile spread, carries the paper. Long-only truncates the classic
short-leg crash story; the relevant mechanism is the long-leg recovery tilt
(C4 cluster: losers get weight in recovery), which is exactly the
"reversal" effect Marc flagged as interesting.
