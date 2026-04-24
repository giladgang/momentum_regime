# Next-Run Plan

Self-contained runbook. Any future Claude Code session (or Gilad himself)
can resume from here after a session disconnect. All gates, commands,
and decision branches are spelled out explicitly so nothing depends on
prior conversation context.

Written: 2026-04-24 (session with Claude Opus 4.7).

---

## Current state

### Committed locally, not pushed
- `97b35b6` — First test suite (tests/test_utils, test_config, test_cross_sectional_lookahead, test_portfolio_edge_cases, test_pipeline_smoke, test_reproducibility) + SWE hygiene (pytest.ini, CI, pre-flight in run_pipeline.py, `src/utils.py` `std < 1e-12` bug fix, `scripts/cross_sectional_model.py` metrics() dedup)
- `56a9fdf` — Initial CV scripts (`scripts/xgb_cv.py`, `scripts/hmm_cv.py`)
- `3062fac` — Test/engineering phase-1 (`tests/test_cv_scripts.py`, `tests/conftest.py`, `requirements.txt`, test_utils refactor + CI inclusion, smoke-path isolation for CV scripts)
- `220186c` — `scripts/apply_shumway_delisting.py` + `tests/test_shumway_delisting.py`

### Background compute
- **Expanding-window backtest** (PID 1752) — `python scripts/expanding_window_backtest.py --first-retrain-year 1995 --last-retrain-year 2024 --hmm-seeds 200 --xgb-seeds 50 --tag prod`, writing to `logs/expanding_prod.log`.
- Started 2026-04-24 12:18 PM Asia/Jerusalem. Expected ~17h total. **Still running as of plan write time.**

### Pending decisions from Gilad
- When the backtest finishes, a single "go" message authorises the full sequence below.
- Thesis `.tex` edits remain gated on explicit greenlight AFTER Gilad sees the new numbers (memory: `feedback_thesis_edits_await_data.md`).
- `git push` remains gated on Gilad's explicit approval.

---

## Gate conditions (hard stops)

| Gate | Rule |
|---|---|
| G1. Don't run heavy compute while backtest PID 1752 is alive | `ps -p 1752` must return no output before starting Step C. |
| G2. Don't overwrite existing results | Baseline snapshot (Step B) must complete successfully before any `run_pipeline.py` rerun. |
| G3. Don't edit thesis `.tex` until Gilad greenlights | Step L reports findings. NO `Edit` on any `latex/*.tex` file before Gilad says so. |
| G4. Don't push without explicit approval | `git push` only when Gilad says "push". |
| G5. Don't start the running `expanding_window_backtest.py` via any code path (the user has one live instance) | — |

---

## Execution sequence

### Step A. Verify backtest outputs

```bash
ls -la results/expanding_returns_prod.csv \
       results/expanding_pi_filter_prod.csv \
       results/expanding_summary_prod.csv
```

All three must exist and be non-empty. If missing, the backtest was killed or crashed; read `logs/expanding_prod.log` to diagnose. Do not proceed.

### Step B. Snapshot baseline

Create a pristine archive of pre-Shumway state so every comparison later is fair:

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP="baseline_pre_shumway_${STAMP}"
mkdir -p "${SNAP}"
cp -r artefacts "${SNAP}/artefacts"
cp -r tables    "${SNAP}/tables"
cp -r plots     "${SNAP}/plots"
cp -r results   "${SNAP}/results"
cp    data/panel_with_regimes.parquet "${SNAP}/"
echo "Snapshot: ${SNAP}"
ls -la "${SNAP}"
```

Verify the snapshot by listing contents.

### Step C. Apply Shumway

```bash
python scripts/apply_shumway_delisting.py --force
```

Expected output: "Backup created: data/crsp_msf_raw.parquet.bak_before_shumway" and "Panel written: data/crsp_msf_raw.parquet". Rule counts should be approximately: rule_1 = 1294, rule_2 = 2, rule_3 = 11, rule_4 = 1, rule_5 = ~2,081,177.

### Step D. Full pipeline rerun

```bash
python run_pipeline.py 2>&1 | tee logs/pipeline_post_shumway.log
```

Expected ~2-3 hours. The pre-flight tests (steps 95-98 in `run_pipeline.py:115`) run first — if any fail, abort and investigate. Then Steps 1 (skipped, HMM already fit), 2 onward run.

### Step E. Run leg-betas analysis

```bash
python scripts/leg_betas_by_regime.py 2>&1 | tee logs/leg_betas.log
```

Outputs:
- `results/leg_betas_by_regime.csv` — static calm vs panic leg betas per strategy
- `tables/table_leg_betas.tex`
- `plots/leg_betas_rolling.pdf`

**Key question to evaluate after this runs:** does Panel B (M2) show a clean inversion? Specifically:
- In **calm** regime: long-leg beta > short-leg beta (past winners are the high-beta basket).
- In **panic** regime: long-leg beta < short-leg beta (after the selection direction flips, M2 is now long past losers — the high-beta basket — and short past winners).

If the calm/panic flip is numerically unambiguous (e.g., at least ~0.3 beta gap in each direction and the sign flips), that's "clean inversion." Otherwise, the story reframes.

### Step F. Report to Gilad (first checkpoint — no thesis edits)

Compose a report with two parts:

**Part 1. Pipeline-rerun diff (Shumway impact).**
Compare baseline snapshot vs fresh outputs for:
- Headline Sharpe in `artefacts/cs_artefacts_data.pkl` strategies_lo['Method 2: XGB']
- Turnover (available in `results/` or recompute)
- SHAP percentages for pi_filter (baseline had 45%)
- Sub-period Sharpes from `tables/table_subperiod.tex`
- Factor alphas from the alpha tables

Expected shift: <0.05 Sharpe. If shift is >0.15 in either direction, flag as anomalous and investigate before continuing.

**Part 2. Leg-betas finding.**
Report Panel B (M2) numbers and classify:
- **Clean inversion:** recommend thesis edit bundle #1 + #2 + #3 (definitions below). #4 standalone either way.
- **No clean inversion:** recommend reframing #1 as "mechanism is cross-sectional re-ranking of momentum term structure, not beta-exposure channel" — also publishable, arguably more novel, but different narrative. Would replace #1 rather than coexist with it. #2 and #3 become optional depending on how the reframe lands.

**STOP here.** Do NOT modify any `latex/*.tex` file. Gilad reviews and greenlights.

### Step G. Calibrate regression tests

Re-run the existing test suite against new artefacts:

```bash
python -m pytest tests/test_pipeline_technical.py tests/test_thesis_consistency.py tests/test_expanding_window.py --tb=short 2>&1 | tee logs/regression_calibration.log
```

Any failing tests are either (a) genuinely stale bounds that need updating to match new artefacts, or (b) real regressions. Investigate each. Update bounds only when the shift is coherent with Shumway's expected effect (small, systematic). Do NOT loosen bounds to hide real regressions.

### Step H. Run CV smoke tests

```bash
python scripts/xgb_cv.py --smoke                              # ~30 sec
python -m pytest tests/test_cv_scripts.py -v                   # validate smoke CSV
python scripts/hmm_cv.py --smoke                              # ~5-10 min
python -m pytest tests/test_cv_scripts.py -v                   # validate hmm smoke CSV
```

Smoke CSVs land at `results/xgb_cv_smoke.csv` and `results/hmm_cv_smoke.csv` (new paths, never collide with full-run outputs).

### Step I. Full XGB CV

```bash
python scripts/xgb_cv.py > logs/xgb_cv.log 2>&1 &
XGB_PID=$!
echo "XGB CV running as PID ${XGB_PID}"
```

Expected ~2-5h. Monitor `logs/xgb_cv.log` and `results/xgb_cv_results.csv` (appends after each cell). Script is checkpointed — can be killed and resumed without loss.

Wait for completion:

```bash
wait ${XGB_PID}
```

### Step J. Full HMM CV

```bash
python scripts/hmm_cv.py > logs/hmm_cv.log 2>&1 &
HMM_PID=$!
echo "HMM CV running as PID ${HMM_PID}"
wait ${HMM_PID}
```

Expected ~15-30h. Same checkpointing. Results at `results/hmm_cv_features.csv`.

### Step K. Final report to Gilad (second checkpoint)

Compose final report:
- CV winners: read `results/xgb_cv_winner.json` and `results/hmm_cv_winner.json`
- Full pipeline-rerun diff (already in Step F report, update with any late numbers)
- Leg-betas finding and recommended thesis-edit shape
- List of every thesis `.tex` line that needs updating (for numerical consistency + the #1-#4 bundle)

**STOP.** Gilad reviews and explicitly greenlights each thesis edit. Implement edits only after greenlight. After edits land:

```bash
git pull --rebase    # sync with any Overleaf edits
git push
```

Only with Gilad's explicit "push".

---

## Thesis edit bundle (for the Step F / Step K recommendation)

Triggered by Step E's leg-betas finding. **Do NOT execute until Gilad greenlights.**

### Edit #1 — [HIGHEST VALUE] new subsection "Leg-level beta dynamics across regimes"

Location: `latex/main_results.tex`, right after Section `sec:dm_comparison`.

Structure:
1. State D&M's finding precisely (loser-decile beta spikes to 3-5 in bear markets; winner stays low; WML's bear-market beta ≈ −1.7; option-like asymmetry driven entirely by losers).
2. Replicate on 2011-2025 sample for Fixed 12-mo momentum. Show loser-leg beta > winner-leg beta, gap widening in panic.
3. Show M2's leg betas inverting in panic (long leg becomes high-beta basket).
4. Include one static table (calm-vs-panic leg betas from `tables/table_leg_betas.tex`) + one rolling-beta figure (`plots/leg_betas_rolling.pdf`, D&M Fig. 3 layout).

**If no clean inversion:** replace #1 with a reframing: "M2's outperformance in panic is not mediated by the D&M beta channel, suggesting the mechanism is specifically the cross-sectional re-ranking of the momentum term structure rather than exposure to a single high-beta basket."

### Edit #2 — [MEDIUM VALUE] expand beta discussion in literature_review.tex

Location: `latex/literature_review.tex:19` (currently one sentence).

Expand to three sentences covering:
1. D&M's rolling-beta finding (loser decile β → 3-5; winner stays ≤ 1).
2. Option-like asymmetry (D&M Table 4) — the beta spike is concentrated in up markets during bear regimes, making WML behave like a written call option.
3. Non-hedgeability (D&M's critique of Grundy-Martin) — ex-ante CAPM hedging doesn't fix it because realized beta correlates with contemporaneous market return.

Frames your contribution as attacking a specific asymmetric-beta channel that exposure-scaling strategies (Barroso-Santa-Clara) also target, but through a different mechanism.

### Edit #3 — [MEDIUM VALUE] sharpen conclusion's "two channels" framing

Location: `latex/conclusion.tex:53`.

Current version positions M2 as a complement to D&M-style exposure scaling. Sharpen to:

> Exposure scaling (Barroso-Santa-Clara, D&M dynamic strategy) takes the beta asymmetry as given and reduces gross exposure when crash risk is elevated. M2 attacks the asymmetry at the selection stage by re-ranking which stocks occupy each leg, so the high-beta basket moves from the short side to the long side precisely when the regime switches. The two channels operate on different margins and are naturally combinable.

### Edit #4 — [LOWER VALUE, STANDALONE] pi_filter → D&M bear-indicator link

Location: anywhere in `latex/methodology.tex` where pi_filter is introduced.

Add one sentence noting that pi_filter is a richer multivariate alternative to D&M's single binary bear-market indicator (past-24-month market return < 0) — same conceptual role but built from drawdown, dispersion, relative-strength, and credit-spread features rather than lagged market return alone.

This one is **independent of leg-betas finding**. Can land regardless of #1-#3 outcome.

---

## Files that should NOT be modified during execution

- `latex/*.tex` — any thesis narrative
- `data/*.parquet` — EXCEPT `crsp_msf_raw.parquet` which Step C modifies (backup exists)
- `results/expanding_*_prod.csv` — these are the current backtest's outputs; leave alone
- Anything under `baseline_pre_shumway_*/` after Step B — it's a pristine reference

## Files that WILL be modified (expected)

- `data/crsp_msf_raw.parquet` + backup (Step C)
- `artefacts/*.pkl` (Step D, regenerated by pipeline)
- `tables/*.tex` (Step D, regenerated)
- `plots/*.png`, `plots/*.pdf` (Step D + Step E, regenerated)
- `results/*.csv` (Step D + Step E, many files overwritten; baseline in snapshot)
- `results/xgb_cv_results.csv`, `results/hmm_cv_features.csv` (Steps I, J — new files)
- `results/xgb_cv_winner.json`, `results/hmm_cv_winner.json` (Steps I, J)
- `tables/table_xgb_cv.tex`, `tables/table_hmm_cv.tex` (Steps I, J)
- `logs/*.log` (Steps D, E, H, I, J)

## Recovery / rollback

If anything in Steps C-E goes wrong and you need the pre-Shumway state back:

```bash
cp data/crsp_msf_raw.parquet.bak_before_shumway data/crsp_msf_raw.parquet
cp -r baseline_pre_shumway_<stamp>/artefacts/*  artefacts/
cp -r baseline_pre_shumway_<stamp>/tables/*     tables/
cp -r baseline_pre_shumway_<stamp>/plots/*      plots/
cp -r baseline_pre_shumway_<stamp>/results/*    results/
cp    baseline_pre_shumway_<stamp>/panel_with_regimes.parquet data/
```

Then rerun pipeline without Shumway to sanity-check baseline restored.

## Session-recovery checklist (if this session disconnects)

A future Claude session should:

1. Read this file (`NEXT_RUN_PLAN.md`).
2. Check `git log --oneline -5` — most recent commit should include `220186c` or later.
3. Check `ps -p 1752` — if still alive, gate G1 applies; if dead, proceed with Step A.
4. Check `ls baseline_pre_shumway_*` — if exists, Step B is done; skip to next incomplete step.
5. Check `grep -c "Backup created" logs/pipeline_post_shumway.log` (if log exists) to determine if Step C completed.
6. Follow remaining steps in order.

Memory note pointer: `feedback_thesis_edits_await_data.md` (durable across sessions) locks the hard-stop on thesis `.tex` edits.
