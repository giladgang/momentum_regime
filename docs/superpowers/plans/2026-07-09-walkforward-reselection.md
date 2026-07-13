# Walk-Forward Yearly Re-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure the OOS performance of the feature-selection PROCEDURE by
re-selecting the HMM feature set every year from trailing data only and
trading the selection forward through 2011-2024.

**Architecture:** Three-stage pipeline reusing validated harnesses: (1) extend
the train-only CV fold grid with annual folds 2011-2023 (reuses the 2026-06-21
trainonly module verbatim via importlib); (2) apply the two pre-committed
selection rules per year (pure aggregation, no compute); (3) per-year
evaluation of the selected combos with the bit-validated canonical HMM
machinery (2026-07-09 script) at the stabilized seed budget. All stages
append-per-cell and resume.

**Tech Stack:** Python 3.12 (.venv), numpy/scipy/pandas/xgboost, existing
experiments modules via importlib. Exploratory code (`experiments/`, per
CLAUDE.md) — verification gates are smoke runs + consistency probes, not
pytest (documented deviation from TDD, consistent with repo conventions).

## Global Constraints

- NO new pip dependencies (requirements.lock untouched).
- All outputs under `experiments/results/` prefixed `wf_`; production
  artefacts, `data/`, `tables/`, `results/` untouched.
- Panel window pinned to 1990-12-01 start (canonical window; audit finding).
- Resume-safety: every expensive unit persists on completion; re-launch skips.
- Seeds fixed and logged: selection folds seeds [0,1,2]; evaluation HMM seeds
  config.HMM_SEEDS[:50] (=1..50), XGB config.XGB_SEEDS[:20] (=1..20).
- Anti-p-hacking: results reported in full; no combo or rule adopted based on
  walk-forward outcomes; predictions below are registered before any result.

---

## PRE-REGISTERED SPEC (committed 2026-07-09, before any walk-forward result)

**Universe.** The 53 train-evaluable Pass-1 survivors (fixed candidate pool
from `trainonly_boost_pool_all53.csv`). ADR combos excluded (no canonical
prep exists — disclosed scope limit).

**Selection data for year Y** (Y = 2011..2024): all CV folds whose validation
window ends on or before Jan 1 of year Y:
- the 7 existing biennial folds (validations 1997-2011, cells already on disk
  at 10 seeds from the boosted sweep), plus
- 13 new annual folds: for t = 2010..2022, train through Dec t, validate
  calendar year t+1. New cells at 3 HMM seeds x 5 XGB seeds (seed noise SE
  ~0.05 << fold noise SE ~0.19; measured 2026-07-09).
Mixed fold lengths (2y pre-2011, 1y after) disclosed; every combo sees the
identical fold/seed grid, so cross-combo ranking is fair by construction.

**Rules walked forward** (both committed 2026-07-09 before canonical results):
- R1 (argmax): highest mean of per-fold means.
- R2 (RULE R): eligible = combos within 1 SE of the argmax (SE = fold-std of
  argmax / sqrt(n_folds)); choose smallest D; ties -> higher mean, then
  Pass-1 ESS.

**Evaluation for year Y:** the rule's selected combo, HMM trained on panel
rows [1990-12, Dec Y-1] with 50 seeds (stabilization: thesis seed-convergence
appendix shows plateau by k=30-50; 200 is overkill, <30 understates),
posterior-mean mu/Sigma/P, crisis-sign panic labels; XGB 20 seeds trained on
stock rows < Jan Y; predict year-Y months; long-short via src.utils
(NYSE P10/P90, value-weighted, 10bps; legs reset at year joins — small fee
overstatement at 14 joins, disclosed). Stitch 2011-2024 into one monthly
series per rule.

**Registered predictions** (scored after the run, reported either way):
1. R1 (argmax) churns: >=5 distinct combos across the 14 years, drawn mostly
   from the VOL/TERM/SKEW families.
2. R2 (parsimony) is stable: DD alone selected in >=10 of 14 years.
3. Neither walk reaches production's canonical 1.11; both land in 0.40-0.85.
4. Panic-month Sharpe of both walks < production's 1.53 (weak C4
   monetization, per the DD mechanism comparison).
5. R2 >= R1 on full-period Sharpe (argmax pays the winner's-curse tax yearly).

**Deliverables:** wf_fold_cells.csv (new fold cells), wf_selections.csv
(per-year per-rule choice + full ranking), wf_year_returns.csv (stitched
monthly returns), wf_summary.md (metrics, prediction scoring, comparison vs
production 1.11 / DD 0.72 canonical anchors).

---

### Task 1: Annual fold-cell generator

**Files:**
- Create: `experiments/2026-07-09-walkforward-folds.py`
- Output: `experiments/results/wf_fold_cells.csv` (schema identical to
  `hmm_featsel_trainonly.csv`)

**Interfaces:**
- Consumes: `trainonly` module (`experiments/2026-06-21-hmm-feature-select-trainonly.py`)
  via importlib: `eval_cell(panel, stocks, combo, fold, hmm_seed, xgb_seeds,
  n_iter, n_burnin, fee, xgb_n_jobs) -> dict|None`, `load_hmm_panel()`,
  `load_stock_panel()`, `completed_cells(path)`, `append_row(path, row)`.
- Produces: rows {combo, n_features, fold, hmm_seed, n_xgb_seeds, fee,
  val_sharpe, n_val_months, panic_agree, hmm_sec, xgb_sec} with fold ids
  101..113 (fold 100+k = train through Dec 2009+k, validate year 2010+k).

- [ ] **Step 1: Write the script** — annual FOLDS list
  `[(100+k, f'{2010+k}-01-01', f'{2011+k}-01-01') for k in range(1, 14)]`
  (fold 101: train<2011-01-01, validate 2011; ... fold 113: validate 2023);
  combos from `trainonly_boost_pool_all53.csv`; seeds [0,1,2]; XGB seeds
  [0..4]; 2000/500 iters; multiprocessing spawn pool (workers arg, default 6),
  reusing `trainonly._init_worker`/`_run_unit` with `sys.modules['trainonly']`
  registration (proven pattern from the boost driver).
- [ ] **Step 2: Smoke** — `--smoke`: 2 combos x fold 101 x 1 seed, 400/100
  iters, separate output file; expect 2 rows, plausible val_sharpe, then
  delete smoke file. Run and verify output before full launch.
- [ ] **Step 3: Full launch check** — enumerate units: expect
  53*13*3 = 2067 cells minus already-done; log ETA at measured cell rate.

### Task 2: Per-year selection (pure aggregation)

**Files:**
- Create: `experiments/2026-07-09-walkforward-select.py`
- Output: `experiments/results/wf_selections.csv`

**Interfaces:**
- Consumes: `hmm_featsel_trainonly.csv` + `hmm_featsel_trainonly_boost.csv`
  (biennial folds 1-7) + `wf_fold_cells.csv` (annual folds 101-113);
  `results/cv/hmm_feature_selection_pass1.csv` for ESS tie-break.
- Produces: rows {year, rule, combo, cv_mean, n_folds, n_eligible, margin}
  for rule in {argmax, rule_r}; year-Y selection uses folds with
  val_end <= Y-01-01 (biennial folds all qualify for Y>=2011; annual fold
  100+k qualifies for Y >= 2011+k).

- [ ] **Step 1: Write the script** — per-fold means (mean over seeds), per
  year: restrict folds, compute per-combo mean, apply R1 and R2 exactly as
  in `2026-07-05-trainonly-rule.py` (1-SE of argmax, min D, mean then ESS
  tie-break). Print the two chosen combos per year.
- [ ] **Step 2: Consistency probe** — for Y=2011 (biennial folds only,
  10-seed data) R2 must reproduce the locked RULE R verdict (DD) and R1 the
  boosted argmax (DD+VOL+REL_N). Any mismatch = bug; stop.

### Task 3: Per-year evaluation

**Files:**
- Create: `experiments/2026-07-09-walkforward-eval.py`
- Output: `experiments/results/wf_year_returns.csv` (append per year-rule),
  `experiments/results/wf_pi/<year>_<combo>.npz`

**Interfaces:**
- Consumes: canonical module (`experiments/2026-07-09-prod-budget-dd-vol-reln.py`)
  via importlib: `fit_seed(seed)` + `_init_worker(z_train, z_full, signs,
  n_iter, n_burnin)` + `forward_filter`; `wf_selections.csv`; stock frames
  = pd.concat of artefact train+test (full 1991-2024 panel with mom_1..12,
  ret_fwd, me, exchcd); `src.utils.long_short_port`.
- Produces: rows {date, year, rule, combo, ret, pi} — monthly LS returns for
  year Y months only.

- [ ] **Step 1: Write the script** — for each (year, rule) from
  wf_selections (dedupe identical (year, combo) across rules — compute once,
  emit for both): panel = parquet, window [1990-12, Y-01-01) for Z_train,
  Z_full through Dec Y (filter needs year-Y months only + history);
  crisis-sign mask on train window (config CRISIS_WINDOWS); 50-seed pool ->
  pi_avg; merge into stock frames; XGB 20 seeds on rows < Y-01-01; score
  year-Y rows; long_short_port on year-Y months (fee from config); append.
  Resume: skip (year, rule) pairs already in the CSV.
- [ ] **Step 2: Smoke** — `--smoke`: year 2011, rule_r combo only, 3 HMM
  seeds x 200 iters x 3 XGB seeds; expect 12 monthly rows, sane Sharpe;
  delete smoke rows.
- [ ] **Step 3: Anchor probe** — full-budget year-2024 run for combo
  DD+DISP+REL_N+CS trains on ~1990-2023: its pi over 2011-2023 should
  correlate > 0.95 with the canonical production pi (same features, window
  one year short). Assert and log.

### Task 4: Chain runner + report

**Files:**
- Create: scratchpad `wf_chain.sh` (waits for any in-flight canonical run,
  then: folds -> select -> eval), caffeinate attached, monitor armed.
- Create: `experiments/2026-07-09-walkforward-report.py` — stitch
  wf_year_returns, per-rule metrics (Sharpe/ret/vol/MDD, calm/panic split
  under each year's own pi), selection-path table (which combo each year),
  prediction scoring vs the registered spec, `wf_summary.md`.

- [ ] **Step 1:** write + launch chain (nohup, resume-safe), arm monitor.
- [ ] **Step 2:** on completion run report script; score the 5 registered
  predictions explicitly; deliver final table.

## Self-Review

- Spec coverage: universe/folds/seeds/rules/eval/predictions all mapped to
  Tasks 1-4. Gap: none found.
- Placeholders: none — code-bearing steps reference exact existing,
  session-validated functions and give exact fold/seed/schema values.
- Type consistency: fold ids ints (1-7 existing, 101-113 new); combo strings
  'DD+VOL+REL_N' style throughout; dates 'YYYY-01-01' strings at boundaries.
- Deviation note: TDD replaced by smoke+probe gates (Steps labeled) per
  CLAUDE.md experiments/ convention; each stage has an explicit
  verification gate before full compute.

## SPEC AMENDMENT 2026-07-09 ~17:05 (before any annual-fold result was read)

Universe restricted from 53 to 25 combos for compute (user request):
top-20 by mean CV over the pre-2011 folds (1-7) UNION all D<=2 combos
(parsimony candidates preserved). The restriction rule uses ONLY pre-2011
train-side information, so the walk remains causal; combos outside the 2011
short-list can never enter later (beam-search compromise, disclosed).
Registered predictions unchanged. ~20 fold-101 cells existed at amendment
time; none were read or used in forming the rule.
