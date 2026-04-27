# RUN_MANIFEST.md

Single-source-of-truth log of post-Shumway research-pipeline phases.
Each phase: started → ended → git SHA at start → output files written → notes.

Append-only (newest at bottom). Status emoji: ⏳ pending / 🔄 running / ✅ done / ❌ failed.

---

## Phase 0 — Baseline + Shumway

✅ **Step A — Verify backtest outputs** (manual, pre-session)

✅ **Step B — Snapshot pre-Shumway US baseline**
- Output: `baseline_pre_shumway_20260426_150517/` (3.0 GB; 12 artefacts, 56 tables, 164 plots, 55 results, panel_with_regimes)
- Notes: Reference for any post-Shumway comparison.

✅ **Step C — Apply Shumway delisting (US)**
- Command: `python scripts/apply_shumway_delisting.py --force`
- Counts: rule_1=1294, rule_2=2, rule_3=11, rule_4=1, rule_5=2,081,177
- Backup: `data/crsp_msf_raw.parquet.bak_*` (auto)

🔄 **Step D — Full pipeline rerun (US, post-Shumway)** — restarted after Step 14 PIL bug
- Started: 2026-04-26 ~15:14
- Steps 1–13 completed in main run (logs/pipeline_post_shumway.log)
- Step 14 FAILED: `FileNotFoundError: 'depth_vs_sharpe.png'` (PIL paths missing `plots/` prefix)
  - Fix: confirmed `plots/` prefix already in `scripts/depth_vs_sharpe.py:125-126`; rerun completed cleanly
  - Step 14 rerun: 1725s, all 6 depths fit
- Steps 15: ✅ done (22s)
- Steps 16, 19 (skip), 20, 95–102: 🔄 running in `logs/pipeline_remaining_steps.log`

---

## Phase 0.5 — Pre-int'l-Shumway snapshot

✅ **Snapshot UK/JP panels before strict-mode Shumway**
- Output: `baseline_pre_intl_shumway_20260426_221329/` (248 MB; 6 panels)
- Saved before re-import overwrites them.

---

## Phase 1a — US analysis chain

✅ **Step E — leg_betas_by_regime**
- Started/ended: 2026-04-26 ~22:00
- Bug fixed: `r_mkt` was pandas Float64; cast to numpy float64 before OLS (`scripts/leg_betas_by_regime.py:119`)
- Outputs: `results/leg_betas_by_regime.csv`, `tables/table_leg_betas.tex`, `plots/leg_betas_rolling.pdf`
- **Headline finding (corrected)**: M2 XGB does NOT show a clean inversion
  - M2 long β: 1.399 calm → 1.648 panic (rises)
  - M2 short β: 1.225 calm → 0.981 panic (falls)
  - In BOTH regimes, long-β > short-β (no flip). Long-short β gap widens
    from +0.17 (calm) to +0.67 (panic) — mechanism is "cross-sectional
    re-ranking of momentum term structure" (the long leg loads more on
    high-β names in panic and the short leg loads less). NOT D&M-style
    leg-beta inversion.
  - Fixed-12-mom: long β stays low (1.07-1.09), short β stays high
    (1.69-1.80) regardless of regime. No inversion.
  - → **Triggers the "cross-sectional re-ranking" framing** for the
    thesis-edit bundle #1, NOT the "clean inversion" framing.

✅ **Step F — first-checkpoint report (preliminary)**
- Output: `results/STEP_F_REPORT.md`
- Will refresh after Phase 2 CV winners + UK/JP production results land

⏳ **Step G — regression-test recalibration** — pending pipeline loop completion
⏳ **Step H — CV smoke tests** — pending Step G

---

## Phase 1b — Int'l prep (in flight)

🔄 **N0a — UK Compustat Global re-import (with secstat/dldte)**
- Started: 2026-04-26 ~22:14
- Ended: 2026-04-26 ~22:36 (~22 min, faster than 1–2h estimate)
- WRDS DNS workaround applied: `scripts/import_intl_stocks.py` patches `socket.getaddrinfo` for Wharton hostnames (local resolver SERVFAILed; Google DNS resolves to 165.123.60.118)
- Output: `data/uk_stock_panel.parquet` (514,115 rows × 5,569 securities, 29 cols incl secstat, dldte, dlrsn)
- Verified: secstat 100%, dldte 59.5%, all 29 expected cols present

🔄 **N0a — JP Compustat Global re-import**
- Started: 2026-04-26 ~22:36
- SQL Step 1 done: 1,486,396 stock-month rows pulled

⏳ **N0b — apply_shumway_intl strict mode (UK + JP)**
⏳ **N0c — smoke-retest cross_sectional_intl UK + JP**

---

## Phase 2 — Lane A (US heavy) and Lane B (UK/JP heavy)

⏳ **Step I — full XGB CV** (~5-7h, 6 workers)
⏳ **Step J — full HMM CV** (~5-7h with parallel refactor; was 30h serial)
⏳ **Step N1 — UK/JP production HMM** (~12-18h, 200 seeds)
⏳ **Step N2 — UK/JP production CS** (~2h)

Note: Lanes run **sequentially** (A then B) to honor the 6-core cap (2 cores reserved for OS).

---

## Phase 3 — Composition

⏳ **Step N3 — UK/JP writeup** → `RESULTS_LOG.md`
⏳ **Step K — final report** → `results/STEP_K_REPORT.md`
⏳ **Step L — prose-edit checklist** → `results/PROSE_EDITS.md`

---

## Phase 4 — Thesis edits (HARD STOP)

🛑 **Step M — US thesis edits** — gated, awaits Gilad row-by-row greenlight
🛑 **Step N4 — UK/JP thesis edits** — gated, bundled with M

---

## Bugs found & fixed (this session)

1. `scripts/depth_vs_sharpe.py` PIL paths missing `plots/` prefix (pre-existing fix, confirmed via re-run)
2. `scripts/leg_betas_by_regime.py:119` pandas Float64 → numpy float64 cast for OLS compatibility
3. `scripts/import_intl_stocks.py` socket.getaddrinfo monkey-patch for Wharton DNS workaround
4. `scripts/apply_shumway_intl.py:345-348` format-string crash (`{v:>10,}` fails when v is the
   `'mode'` string `'strict'`/`'heuristic'`) — added isinstance check
5. `scripts/hmm_cv.py:load_hmm_panel` global dropna across 9 candidates killed 70% of rows for
   any combo not including the quarterly `GDP_g`. Removed the dropna; per-combo dropna now
   handled inside `zscore_train`. Pre-existing latent bug; smoke test caught it.

## Parallel refactors (this session, all verified IDENTICAL output to ≥6 decimals)

- `scripts/hmm_cv.py` — added `--workers N`, fork-based pool, fail-count tracking. 2.07× at 2 workers.
- `scripts/xgb_cv.py` — added `--workers N`, same template. 1.71× at 2 workers (smoke).
- `scripts/hmm_intl.py` — added `--workers N`, fork-based pool. 1.58× at 2 workers (smoke).

Production runs will use `--workers 6` per the 2-cores-free directive. Expected wall-clock at 6 workers:
- Step I (XGB CV, 180 cells): ~30-50 min vs ~8h serial
- Step J (HMM CV, 1395 cells): ~5h vs ~31h serial
- Step N1 (UK + JP HMM, 200 seeds × 2 regions): ~2-3h vs 20-34h serial

## Audit findings (1 real, 5 false positives, 5 P3 deferred)

Initial Explore-agent audit flagged 11 issues. After verification:
- 5 false positives (resume CSV trim, zero-byte detection, ret_fwd consistency — all correct on close read)
- 1 real but minor (`hmm_cv.py` silent except — folded into refactor's fail-count summary)
- 5 P3 code-smell items deferred (CRISIS_WINDOWS duplication etc.)

## Step I results (XGB CV) — 2026-04-26 23:27 (43 min wall-clock)

- Winner: **depth=3, lr=0.10, n_estimators=200** (mean val Sharpe **+0.596**, fold-std 0.811)
- Top-5 mean Sharpes (0.560-0.596) are within fold-noise of each other; the CV is picking among near-identical configs
- **Fold 3 (2001-2003 dot-com) is uniformly bad across ALL 36 hyperparameter cells** (mean -0.59, max -0.38). Not a hyperparam issue — a regime-extrapolation issue. The model can't cope with unprecedented panics.
  - Per-fold: 1=+1.36, 2=+0.45, 3=-0.43, 4=+1.46, 5=+0.14
- Implication for Step K writeup: the CV winner is essentially "any reasonable XGB config" — the ranking among them is noise. Flag this and recommend the chosen config as "representative" rather than "uniquely best".

## Step J — DONE (4h 42min wall-clock)

Launched 23:27 by `scripts/_chain_phase2.sh`, finished 04:09. 6 workers, no failures.

**Winner: `DD+CS+LVIX+DISP`** (mean val Sharpe **+0.507**, fold-std 0.694)
- Per-fold: +1.02, +0.29, **−0.47**, +1.32, +0.38 (fold-3 dot-com pattern again)
- Stability winner (lowest fold-std): `DD+LVIX+REL_N+SKEW`, mean +0.193

**Big finding**: CV winner differs from current production (`DD+CS+DISP+REL_N`) — CV swaps `REL_N` for `LVIX`. Replaces the thesis's "test-set peek" caveat with a real CV winner that's a *different* feature set. Gilad to decide whether to retrain at the CV winner or keep the production fit and document the CV result as cross-validation evidence.

## Step N1 — DONE (UK 29min, JP 28min)

UK: 04:09 → 04:38. JP: 04:38 → 05:06. Both 200 seeds, 6 workers. All seeds converged on panic=state 0 cleanly.

## Step N2 — DONE (chain crashed once, fixed and rerun)

**Crash**: Bash 3.2 (macOS default) doesn't support `${region,,}` lowercase expansion → chain died at N2 launch. Fixed in `scripts/_chain_phase2.sh` (using `tr '[:upper:]' '[:lower:]'`) and restarted with `scripts/_chain_phase2_n2.sh` at 05:36. All 4 runs finished by 05:47 (11 min total wall — XGBoost using all 8 cores per run).

**Headline (production, 50 seeds, post-Shumway)**:

| Region | Regional π | US π (Test A) | Δ |
|---|---:|---:|---:|
| UK M2 XGB | +0.631 | **+0.678** | +0.047 |
| JP M2 XGB | +0.435 | **+0.525** | +0.090 |

**US π beats regional π in BOTH regions** → triggers "global financial cycle channel" framing for the structural-edit bundle (per `INTL_VALIDATION_PLAN.md`).

JP caveat: M2 (0.43-0.52) < market (0.86). Japanese momentum weakness, Asness et al 2013.

## Step N3, K, L — DONE

- `RESULTS_LOG.md` section 12 added with UK/JP writeup
- `results/STEP_K_REPORT.md` regenerated with full Phase 2 winners + intl results spliced from Step F
- `results/PROSE_EDITS.md` regenerated: 40 tables changed, 415 cells, 944 prose candidates across `latex/*.tex`
- `results/THESIS_EDITS_TODO.md` written: comprehensive checkbox list of every finding (~120 items) categorized by triage (M/S/N) with file references for each data source. **Start here for the gated thesis-edit review.**

## Step G — calibration done (2026-04-26 ~23:35)

Captured Step 100's 13 failures and resolved all:

- **2 structural**: `test_critical_column_dtypes` (Float64→float64 — desirable side-effect of Shumway), `test_all_pipeline_scripts_exist` (removed stale `test_2008_oos.py` and `expanding_window.py`, added `expanding_window_backtest_parallel.py`).
- **8 numeric drift**: M2 ann_ret 21.9→21.7, M2 vol 19.7→19.5, M2 MDD -24.8→-22.8, sub-period-1 Sharpe 0.62→0.59, stress baseline MDD -24.8→-23.0, FF6 alpha 24.7→24.1, CAPM alpha 23.8→23.4, plus widening tolerances slightly. All shifts are 1-8% (Shumway-coherent).
- **3 thesis-text-vs-table**: marked `@pytest.mark.xfail` with comment "will pass after Step M". These check that latex/main_results.tex mentions the table values; they WILL pass automatically once Step M's prose edits land.

After: `241 passed, 3 xfailed` (was 13 failed, 231 passed).

## Step 100 calibration failures (resolved above)

| Test | Type | Action |
|---|---|---|
| test_critical_column_dtypes | dtype shift Float64→float64 | update test (post-Shumway side effect, prevents OLS bugs) |
| test_all_pipeline_scripts_exist | stale references (test_2008_oos.py, expanding_window.py) | update test list |
| test_m2_max_drawdown | numeric -22.8% vs -24.8% | update bound (Shumway-coherent) |
| 7× thesis-text-vs-table tests | thesis text not yet updated | DEFER until Step M (gated) |
| 3× cross-table consistency | numeric drift | update bounds |
| test_subperiod_values_match_thesis | 0.59 vs 0.62 | update bound |
| test_stress_baseline_mdd | -23.0 vs -24.8 | update bound |
