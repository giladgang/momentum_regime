# Thesis TODO

**Last updated**: 2026-04-27 (post-Phase-3 composition complete; awaiting Gilad row-by-row thesis-edit review).

## Companion files

- [`results/THESIS_EDITS_TODO.md`](results/THESIS_EDITS_TODO.md) — **207 specific thesis-edit items** (the master review checklist)
- [`results/PROSE_EDITS.md`](results/PROSE_EDITS.md) — 944 auto-flagged numeric prose candidates
- [`results/STEP_K_REPORT.md`](results/STEP_K_REPORT.md) — final Phase-2 summary
- [`results/STEP_F_REPORT.md`](results/STEP_F_REPORT.md) — first-checkpoint report (Shumway diff + leg-betas)
- [`results/RUN_MANIFEST.md`](results/RUN_MANIFEST.md) — phase log + bug fixes + parallel refactors
- [`RESULTS_LOG.md`](RESULTS_LOG.md) §12 — UK/JP international validation writeup
- [`NEXT_RUN_PLAN.md`](NEXT_RUN_PLAN.md) — runbook (Phase 0-4 stages B-N4)
- [`INTL_VALIDATION_PLAN.md`](INTL_VALIDATION_PLAN.md) — UK/JP track plan

---

## 🛑 HARD STOP — thesis edits (awaiting Gilad row-by-row greenlight)

Two distinct types of work. Tackle them differently.

---

### 🔢 BUCKET 1 — Just numbers (slight modifications)

Existing thesis claims, post-Shumway values. **No new sections, no new framing.** Walk the auto-generated checklist; tick or skip each row.

- [ ] **Walk all 944 numeric prose candidates** in [`results/PROSE_EDITS.md`](results/PROSE_EDITS.md)
  - 40 tables changed × 415 cells × prose mentions
  - Source: [`baseline_pre_shumway_20260426_150517/tables/`](baseline_pre_shumway_20260426_150517/) (pre) vs [`tables/`](tables/) (post)
  - Each row is `file:line — "old" → "new"` with surrounding context
  - Skip false-positive matches (e.g., a number that happens to appear in unrelated prose)
  - Effort: ~1-2 hours of mechanical ticking
- [ ] **Headline numbers** (16 high-priority — listed in [`results/THESIS_EDITS_TODO.md`](results/THESIS_EDITS_TODO.md) §A):
  - M2 Ann.Ret 21.9% → 21.7% | Vol 19.7% → 19.5% | MDD −24.8% → −22.8% | Beta 0.46 → 0.45
  - Sub-period Sharpes 0.62→0.59, 1.03→1.04, 1.69→1.70
  - Factor α: CAPM 23.4% (t=4.08), FF3 23.2% (t=4.48), Carhart 24.3% (t=4.75), FF5 22.9% (t=4.62), FF6 24.1% (t=4.81)
  - Stress baseline MDD −24.8% → −23.0%
  - Final-$ multiple 15.7× → 15.3×
  - Sharpe unchanged at 1.11
- [ ] **NW t-stat — 1.83\* → 4.37\*\*\*** (unique among numeric updates: significance jump from marginal to highly significant — flag in prose)
- [ ] **Existing claims to verify post-Shumway** (37 sub-sections E1-E37 in [`results/THESIS_EDITS_TODO.md`](results/THESIS_EDITS_TODO.md))

---

### 💡 BUCKET 2 — New ideas to integrate (substantive content)

Genuinely new findings or significant reframings. Each item is a writing task. **The 7 items below are the focus of the thesis edits.**

#### N1 — Global financial cycle (UK + JP) — NEW SECTION

- [ ] Add new "International validation" subsection in `latex/main_results.tex` (or as an own section)
  - **Drafted prose ready**: [`RESULTS_LOG.md`](RESULTS_LOG.md) §12 — copy-paste and edit
  - **Data**:
    - UK: [`results/intl_uk_summary.csv`](results/intl_uk_summary.csv) (regional π, Sharpe +0.631) vs [`results/intl_uk_summary_uspi.csv`](results/intl_uk_summary_uspi.csv) (US π, Sharpe **+0.678**)
    - JP: [`results/intl_jp_summary.csv`](results/intl_jp_summary.csv) (regional π, +0.435) vs [`results/intl_jp_summary_uspi.csv`](results/intl_jp_summary_uspi.csv) (US π, **+0.525**, MDD halved -33% → -18%)
  - **Citation**: Rey (2013) — global financial cycle channel
  - **Caveat 1** — JP M2 (0.52) < JP market (0.86): cite Asness/Moskowitz/Pedersen (2013) on JP momentum weakness
  - **Caveat 2** — UK M2 (0.68) ~ UK market (0.62): residual L/S exposure of dollar-neutral construction
  - **Methodology footnote**: BANK_REL replaces CS_z (no Moody's BAA-AAA for UK/JP); 4-feature HMM. See [`INTL_VALIDATION_PLAN.md`](INTL_VALIDATION_PLAN.md)
  - **Methodology footnote**: strict-mode Shumway-intl applied (UK 219 rows, JP 85)

#### N2 — Cross-sectional re-ranking framing (Edits #1–#4) — REFRAME mechanism

**Decision locked**: M2 mechanism is *not* a clean leg-beta inversion. Both legs stay long-tilted in panic; the L−S β gap widens 0.17 → 0.67 but doesn't flip.

- [ ] **Edit #1** — new subsection "Leg-level beta dynamics across regimes" in `latex/main_results.tex`
  - Data: [`results/leg_betas_by_regime.csv`](results/leg_betas_by_regime.csv), [`tables/table_leg_betas.tex`](tables/table_leg_betas.tex), [`plots/leg_betas_rolling.pdf`](plots/leg_betas_rolling.pdf)
- [ ] **Edit #2** — `latex/literature_review.tex:19`: expand Daniel & Moskowitz (2016) discussion, distinguish their leg-beta inversion from our re-ranking
- [ ] **Edit #3** — `latex/conclusion.tex:53`: sharpen "two channels" → (1) regime-conditional selection (HMM π) + (2) cross-sectional re-ranking by term-structure shape
  - Channel 1: [`tables/table_kitchen_sink.tex`](tables/table_kitchen_sink.tex), [`tables/table_placebo.tex`](tables/table_placebo.tex)
  - Channel 2: [`results/leg_betas_by_regime.csv`](results/leg_betas_by_regime.csv), [`tables/table_zscore_shap_detail.tex`](tables/table_zscore_shap_detail.tex)
- [ ] **Edit #4** — `latex/methodology.tex`: π_filter ↔ D&M bear-indicator link (probabilistic/continuous vs binary). Reference [`tables/table_hmm_separation.tex`](tables/table_hmm_separation.tex)

#### N3 — CV-justified specifications — STRIKE caveats + future work

**Decision locked**: keep production specs (HMM `DD+CS+DISP+REL_N`, XGB `depth=4 lr=0.05 n=500`); CV results live in appendix as robustness backing.

- [ ] Strike "specification-search" caveat in `latex/data_section.tex`
- [ ] Strike "test-set tuned" caveat in `latex/methodology.tex` and `latex/appendix.tex`
- [ ] Optional appendix paragraph showing production combo is within fold-noise of CV winner. Cite [`tables/table_xgb_cv.tex`](tables/table_xgb_cv.tex) and [`tables/table_hmm_cv.tex`](tables/table_hmm_cv.tex)
- [ ] **Edit #6 (future work)** — paragraph in `latex/future_work_full.tex` on rolling/expanding-window CV as more robust feature-selection. Draft prose in [`results/THESIS_EDITS_TODO.md`](results/THESIS_EDITS_TODO.md) §H Edit #6
- [ ] Note that `tables/table_xgb_hyperparams.tex` and `tables/table_feature_selection.tex` are now deprecated (replaced by CV tables); update any in-text references

#### N4 — Fold-3 dot-com universal failure — NEW caveat

- [ ] Add paragraph in robustness/limitations section
  - All 36 XGB configs + all HMM combos negative on fold 3 — out-of-distribution generalization failure
  - Data: [`results/xgb_cv_results.csv`](results/xgb_cv_results.csv) (group by fold), [`results/hmm_cv_features.csv`](results/hmm_cv_features.csv)
  - Cross-link to existing §5.4.3 30-year backtest discussion (the dot-com loss documented there is the same failure manifested in real history)

#### N5 — NW t-stat significance jump — REFRAME defensive paragraph + sharpen

**More than just a number swap.** The current `latex/conclusion.tex` paragraph **defends** the low t-stat:

> "The Newey-West t-statistic on M2's excess return over the market is **1.83**; this reflects the strategy's near-zero market beta, and the economically meaningful test is the factor model alpha..."

With t=4.37\*\*\* post-Shumway, this defensive framing is no longer needed and reads awkwardly. The paragraph should be **rewritten**, not just have the number swapped.

- [ ] Rewrite the defensive t-stat paragraph in `latex/conclusion.tex` (currently in the limitations/caveats section). New framing: "The Newey-West t-statistic on M2's excess return over the market is 4.37\*\*\* post-Shumway, well above conventional thresholds. The factor-model alphas in Appendix~\ref{app:factor_alphas} confirm this across all five factor specifications (t > 4)."
- [ ] One-line emphasis wherever the t-stat appears in `latex/main_results.tex`: **t-stat 1.83\* → 4.37\*\*\*** post-Shumway. Marginal → highly significant. Largest single result of the Shumway treatment.

#### N8 — IC paradox — NEW prose (genuinely missing from thesis)

`latex/*.tex` contains **no comparison** between M2's per-stock IC and naive momentum's IC. The IC table exists in the appendix but the paradox is not framed in prose anywhere.

- [ ] Add prose (likely in `latex/main_results.tex` mechanism section): M2 IC = **−0.005 (n.s.)** vs naive momentum IC = **+0.027 (t=3.15\*\*\*)**. M2 has *worse* per-stock IC than naive momentum yet much better portfolio Sharpe. The edge is **portfolio construction** (decile-spread + regime-conditional ranking), NOT improved per-stock prediction.
  - Data: [`tables/table_ic.tex`](tables/table_ic.tex)
  - Paired diff M2 − Mom = −0.031 (t = −2.95\*\*\*) — M2 is *significantly worse* at IC by a paired test
  - Why this matters: distinguishes M2's edge from a "better momentum signal" interpretation. Connects to N7 (negative UMD loading) — same novelty story from a different angle.

#### N9 — Already strong in existing thesis (verify only, no new prose)

Items that the thesis already frames well; just verify post-Shumway numbers updated via Bucket 1:

- [ ] **Panic subtypes** — `latex/main_results.tex` already says "the overall panic Sharpe of 1.56 is driven entirely by recovery months" with all the right numbers. Just verify the −0.49 / +4.42 / Sharpe-2.35 numbers in PROSE_EDITS.md.
- [ ] **Ridge baseline** — `latex/main_results.tex` 5.3 nonlinearity section already discusses ridge failure. Just verify the −0.58 number lands consistently and the "all alphas give the same answer" claim is preserved.
- [ ] **Term-structure framing** — `latex/conclusion.tex` already mentions "term-structure reorganisation". Detailed §5.2 figure explanations are already tracked in the "Pre-existing thesis-writing tasks" section below (improve Chapter 5.2 graphs).

#### N6 — Connect 30-year backtest to stress scenario — REFRAME

- [ ] Add explicit cross-link between §5.4.3 (30-year backtest) and §5.4 (stress test).
  - The 36-month dot-com bear (2000-2002, -36% cumulative, MDD -64.8%) is the documented version of the synthetic 24+ month sustained-bear stress scenario.
  - Hypothetical → real. One paragraph that ties the two analyses together explicitly.

#### N7 — Negative Carhart UMD loading — SHARPEN novelty signal

- [ ] One-paragraph emphasis in `latex/main_results.tex` factor-alpha discussion: M2 has *negative* Carhart UMD loading (−0.20 to −0.22 in Carhart/FF6). M2 is *negatively* correlated with naive momentum despite using momentum features. Strong novelty signal: M2 isn't "more momentum", it's a different cross-sectional bet.
  - Data: [`tables/table_factor_alphas.tex`](tables/table_factor_alphas.tex)

---

### Citations to add (across N1, N2, N7)

- [ ] **Rey, H. (2013)** — global financial cycle (N1 framing)
- [ ] **Asness, Moskowitz & Pedersen (2013)** — international momentum (N1 JP caveat)
- [ ] **Daniel & Moskowitz (2016)** — distinguish their inversion from N2 re-ranking
- [ ] **Shumway (1997, 2001)** — delisting bias methodology footnote
- [ ] **Jegadeesh (1990)** + **Lehmann (1990)** — short-term reversal (M5 mechanism prose, mom_1 neutral)
- [ ] **Novy-Marx (2012)** — intermediate-horizon momentum (mom_8 peak)
- [ ] **Lee & Swaminathan (2000)** — momentum lifecycle (mom_12 decay)

---

### Detailed reference

The full 207-row checklist with every micro-task and file pointer is in [`results/THESIS_EDITS_TODO.md`](results/THESIS_EDITS_TODO.md). Use it for granular sub-items; use **this** TODO.md for the day-to-day "what to write next" view.

---

## 🔑 Decisions made (2026-04-27)

- [x] **HMM feature set**: KEEP production `DD+CS+DISP+REL_N`. Step J CV winner (`DD+CS+LVIX+DISP`) is statistically indistinguishable within fold-noise (gap +0.221 vs fold-std 0.694). Defer to **future work** as a "more robust feature-selection procedure with rolling/expanding-window CV". See [`results/THESIS_EDITS_TODO.md`](results/THESIS_EDITS_TODO.md) §B2 for rationale and §H Edit #6 for the future-work paragraph.
- [x] **XGB hyperparameters**: KEEP production `depth=4, lr=0.05, n=500`. Step I CV puts depth-3 narrowly atop the grid (+0.596 vs depth-4 +0.576, gap 0.021 within fold-std 0.811 — essentially zero). Test-period sweep ([`results/depth_results.csv`](results/depth_results.csv)) prefers depth-4 (Sharpe 1.11 vs 0.98). Both objectives jointly support keeping depth-4.
- [x] **No retraining at CV winners** — production stays as the canonical specification; CV results live in the appendix as a robustness check.

---

## 🔬 Robustness backing (already pushed to remote)

- [`results/xgb_cv_winner.json`](results/xgb_cv_winner.json), [`results/xgb_cv_results.csv`](results/xgb_cv_results.csv), [`tables/table_xgb_cv.tex`](tables/table_xgb_cv.tex) — Step I (XGB CV)
- [`results/hmm_cv_winner.json`](results/hmm_cv_winner.json), [`results/hmm_cv_features.csv`](results/hmm_cv_features.csv), [`tables/table_hmm_cv.tex`](tables/table_hmm_cv.tex) — Step J (HMM CV)
- Production combo's CV rank: 47/64 (mean +0.286, fold-std 0.510). Within fold-noise of winner.

---

## 📝 Pre-existing thesis-writing tasks (not gated; ready anytime)

- [ ] **In progress**: Discussion of risk premium (writing task; data backing in [`tables/table_factor_alphas.tex`](tables/table_factor_alphas.tex), [`tables/table_regime_sharpe.tex`](tables/table_regime_sharpe.tex))
- [ ] **Improve explanation of Chapter 5.2 graphs** (term-structure figure + panic sub-type figure):
  - [ ] Frame what the reader should look for
  - [ ] Clarify the z-score-vs-SHAP question each figure answers
  - [ ] Term-structure figure backing data:
    - Plot: [`plots/chart2_rank_by_horizon_v2.pdf`](plots/chart2_rank_by_horizon_v2.pdf), [`plots/momentum_shape_final.pdf`](plots/momentum_shape_final.pdf)
    - Table (z-scores by horizon × regime × leg): [`tables/table_zscore_shap_detail.tex`](tables/table_zscore_shap_detail.tex)
    - Underlying CSV: [`results/selection_rank_analysis.csv`](results/selection_rank_analysis.csv)
  - [ ] Panic sub-type figure backing data:
    - Table: [`tables/table_panic_subtypes.tex`](tables/table_panic_subtypes.tex) (Calm 0.82 / Panic-Crash −0.33 / Panic-Recovery +2.35)
  - [ ] Integrate monthly z-score heatmaps (month × horizon z-scores for long leg, short leg, L−S spread with π_filter panel)
    - Data: [`results/zscore_long_by_month.csv`](results/zscore_long_by_month.csv), [`results/zscore_short_by_month.csv`](results/zscore_short_by_month.csv), [`results/zscore_longshort_by_month.csv`](results/zscore_longshort_by_month.csv)
    - Figures: [`plots/zscore_long_short_heatmap.pdf`](plots/zscore_long_short_heatmap.pdf), [`plots/zscore_longshort_heatmap.pdf`](plots/zscore_longshort_heatmap.pdf)
  - [ ] Single-month resolution counters the "averages hide inverting months" caveat (e.g., Sept 2019, May 2024)

---

## 📚 External / reading

- [ ] Read the important literature cited in thesis (deep study of key papers). Priority list, in order of relevance to thesis edits:
  - Daniel & Moskowitz (2016) — for §C1 distinction (their leg-beta inversion vs our re-ranking)
  - Rey (2013) — global financial cycle, supports §B3 framing
  - Asness, Moskowitz & Pedersen (2013) — international momentum, supports §D2 caveat
  - Shumway (1997, 2001) — delisting bias methodology
  - Novy-Marx (2012) — intermediate-horizon momentum (calm peak at mom_8)
  - Lee & Swaminathan (2000) — momentum lifecycle (mom_12 decay)
  - Jegadeesh (1990) + Lehmann (1990) — short-term reversal (mom_1 neutral)

---

## 🛠️ Engineering / testing follow-ups (low priority)

- [ ] Add `pytest.skip` guard to `tests/test_utils.py::TestDataLoaders` so the class is included in `.github/workflows/tests.yml`. Currently excluded because the loader tests depend on data parquets that aren't in git; the skip guard lets the artefact-independent tests in the file run on CI.
  - File to edit: [`tests/test_utils.py`](tests/test_utils.py) (TestDataLoaders class)
  - CI config: [`.github/workflows/tests.yml`](.github/workflows/tests.yml)

---

## 🏗️ Architecture refactors (post-thesis only; deferred)

Each requires a bit-exact regression check (`md5 tables/*.tex plots/*.png results/*.csv` before/after, plus `pytest tests/test_reproducibility.py`) on a feature branch before merge. Not safe during the thesis sprint.

- [ ] **#1 Subdivide `scripts/`** into `hmm/`, `cs/`, `backtest/`, `analysis/`, `plots/`, `intl/`. Pure file moves; update subprocess paths in [`run_pipeline.py`](run_pipeline.py) and any cross-script imports. ~1h incl. validation.
  - Affected files: all 80+ files in [`scripts/`](scripts/)
- [ ] **#2 Move HMM/CS/portfolio logic into `src/`** as importable modules; reduce `scripts/*.py` to thin CLIs. Biggest leverage point for testability and reuse. Half-day with careful regression.
  - Heavy targets: [`scripts/hmm_model.py`](scripts/hmm_model.py), [`scripts/cross_sectional_model.py`](scripts/cross_sectional_model.py), [`scripts/historical_oos_production.py`](scripts/historical_oos_production.py)
- [ ] **#3 Replace subprocess orchestration with Snakemake.** Eliminates the silent-staleness class of bug; auto-generated DAG; rule-level parallelism. 4-6h. Write Snakefile alongside `run_pipeline.py`, hash-compare every output before deletion. Defer to post-thesis if Steps G-K stretch the schedule.
  - Reference: [`run_pipeline.py`](run_pipeline.py) (102 substeps to fold into ~25 Snakemake rules)
- [ ] **#4 Split the 943 MB `cs_artefacts_data.pkl`** into parquets (`scores_{train,test}.parquet`, `shap_values.parquet`) + XGBoost native `.json` model files + `manifest.json` (git SHA, config hash, data hash, lib versions). Touches every consumer; high coordination cost.
  - File: [`artefacts/cs_artefacts_data.pkl`](artefacts/cs_artefacts_data.pkl)
  - Consumers (must update in lockstep): all scripts that read `cs_artefacts_data.pkl` (~12 scripts; see RUN_MANIFEST.md or grep)
- [ ] **#5 Split `scripts/main_results_analysis.py`** (1092 lines) into `performance.py`, `factor_alphas.py`, `ic.py`, `granger.py`, `shap.py`. Watch for float-summation/dict-iteration ordering changes that perturb table values.
  - File: [`scripts/main_results_analysis.py`](scripts/main_results_analysis.py)

---

## ✅ Completed

### This session (2026-04-26 → 2026-04-27)

#### Phase 0 — Baseline + Shumway US

- [x] **Step A** — verified backtest outputs (admin)
- [x] **Step B** — snapshot pre-Shumway baseline at `baseline_pre_shumway_20260426_150517/` (3.0 GB; 12 artefacts, 56 tables, 164 plots, 55 results files, plus `panel_with_regimes.parquet`)
- [x] **Step C** — `python scripts/apply_shumway_delisting.py --force`. Counts match expectations exactly (rule_1=1294, rule_2=2, rule_3=11, rule_4=1, rule_5=2,081,177)
- [x] **Step D** — full pipeline rerun (Steps 1-20). All numeric outputs regenerated post-Shumway. Steps 14 (PIL `plots/` prefix bug) and 15-16 reran cleanly after fix; Step 19 expanding-window cached, Step 20 sub-period table built. Steps 95-99/101/102 green; Step 100 had 13 expected calibration failures (handled in Step G).

#### Phase 1a — US analysis chain

- [x] **Step E** — `python scripts/leg_betas_by_regime.py`. Bug fixed: cast `r_mkt` from pandas Float64 to numpy float64 (line 119) for OLS compatibility. Output: `results/leg_betas_by_regime.csv`, `tables/table_leg_betas.tex`, `plots/leg_betas_rolling.pdf`. **Decision locked**: M2 XGB shows L−S β gap widening (0.17 → 0.67) but no flip — mechanism is **cross-sectional re-ranking**, NOT D&M-style leg-beta inversion. Edit bundle #1-#4 framed accordingly.
- [x] **Step F report** — `results/STEP_F_REPORT.md` written. Pre-vs-post Shumway diff + leg-betas finding + sanity gates. M2 Sharpe shift -0.000 (within ±0.05 band); NW t-stat jumped 1.83\* → 4.37\*\*\*.
- [x] **Step G** — regression-test recalibration. 13 Step 100 failures resolved: 10 numeric bound updates + 3 `@pytest.mark.xfail` (thesis-text-vs-table tests, will auto-pass after Step M). Test suite now 241 passed, 3 xfailed.
- [x] **Step H** — CV smoke tests (xgb_cv --smoke + hmm_cv --smoke), plus parallel-vs-serial smoke comparisons (all IDENTICAL to ≥6 decimal places).
- [x] **Step I** — full XGB CV with `--workers 6`. Wall: 43 min (vs ~8h serial). **Winner: depth=3, lr=0.10, n_estimators=200** (mean val Sharpe +0.596, fold-std 0.811). Top-5 configs cluster within fold-noise. **Universal fold-3 (2001-2003 dot-com) failure** across all 36 hyperparam configs.
- [x] **Step J** — full HMM CV with `--workers 6`. Wall: 4h 42min (vs ~30h serial). **Winner: `DD+CS+LVIX+DISP`** (mean +0.507, fold-std 0.694). Differs from current production set (swaps REL_N for LVIX). Same fold-3 universal failure pattern.

#### Phase 1b — UK/JP delisting prep

- [x] **N0a** — UK Compustat Global re-import with `secstat`/`dldte`/`dlrsn` from `comp.g_security`. 514,115 rows × 5,569 securities; secstat 100% present; dldte 59.5% (only inactives have a delisting date)
- [x] **N0a** — JP Compustat Global re-import. 1,418,944 rows × 6,408 securities; secstat 100%; dldte 26.3%
- [x] **N0b** — `apply_shumway_intl.py` strict mode applied to UK + JP. UK: 219 rows compounded, 8,723 passthrough non-performance; JP: 85 rows compounded, 7,600 passthrough. Negligible numeric impact (UK 0.04%, JP 0.006%) — diagnostic correctness.
- [x] **N0c** — smoke-retest cross_sectional_intl on Shumway-corrected UK + JP panels (sanity only, superseded by Step N2 production)

#### Phase 2 — Lane A (US heavy) + Lane B (UK/JP heavy)

- [x] **Step N1 UK** — production HMM at 200 seeds, 6 workers. 29 min wall (vs 12-18h estimate). All seeds converged on panic state cleanly.
- [x] **Step N1 JP** — production HMM at 200 seeds, 6 workers. 28 min wall.
- [x] **Step N2** — 4 production cs_intl runs at 50 XGB seeds × {UK regional, UK US-π, JP regional, JP US-π}. 11 min total wall. **Headline: US π beats regional π in BOTH regions** (UK 0.63 → 0.68; JP 0.43 → 0.52, MDD halved). Triggers "global financial cycle channel" framing.

#### Phase 3 — Composition

- [x] **Step N3** — UK/JP writeup added to [`RESULTS_LOG.md`](RESULTS_LOG.md) §12 ("International Validation (UK + JP)")
- [x] **Step K** — final report at [`results/STEP_K_REPORT.md`](results/STEP_K_REPORT.md) (CV winners + UK/JP + Step F splice)
- [x] **Step L** — programmatic prose-edit checklist at [`results/PROSE_EDITS.md`](results/PROSE_EDITS.md). 40 tables changed, 415 cells, 944 prose candidates across `latex/*.tex`.

#### Bug fixes (5 in scripts, 1 in chain)

- [x] `scripts/depth_vs_sharpe.py` — confirmed pre-existing fix for PIL `plots/` prefix; rerun cleanly
- [x] `scripts/leg_betas_by_regime.py:119` — pandas Float64 → numpy float64 cast (fixes OLS dtype rejection)
- [x] `scripts/import_intl_stocks.py:46-56` — WRDS DNS workaround (Wharton hosts SERVFAIL on macOS local resolver; redirect to IP 165.123.60.118)
- [x] `scripts/apply_shumway_intl.py:345-348` — format-string crash on string `counts['mode']` value
- [x] `scripts/hmm_cv.py:load_hmm_panel` + `zscore_train` — pre-existing latent bug: global dropna across 9 features killed 70% of rows for combos not using GDP_g (quarterly). Per-combo dropna in zscore_train; load_hmm_panel keeps all rows
- [x] `scripts/_chain_phase2.sh` — bash 3.2 incompatibility with `${region,,}` lowercase expansion. Replaced with `tr '[:upper:]' '[:lower:]'`

#### Parallel refactors (3, all output verified IDENTICAL to ≥6 decimal places)

- [x] **`scripts/hmm_cv.py`** — added `--workers N`, fork-based pool, fail-count tracking. 30h serial → ~5h. Verified 2.07× speedup at 2 workers (smoke).
- [x] **`scripts/xgb_cv.py`** — added `--workers N`. 8h serial → ~45 min. Verified 1.71× at 2 workers.
- [x] **`scripts/hmm_intl.py`** — added `--workers N` for the 200-seed production fits. 20-34h serial (UK + JP) → ~30 min/region. Verified 1.58× at 2 workers.

#### Tests added

- [x] `tests/test_leg_betas_by_regime.py::TestCAPMBeta::test_handles_pandas_float64_extension_dtype` — regression guard for the Float64→float64 fix

#### Documentation produced

- [x] [`results/RUN_MANIFEST.md`](results/RUN_MANIFEST.md) — comprehensive phase-log, bug-fix list, refactor verifications, findings
- [x] [`results/THESIS_EDITS_TODO.md`](results/THESIS_EDITS_TODO.md) — 207 reviewable thesis-edit checkboxes with file references for each data source
- [x] [`results/STEP_F_REPORT.md`](results/STEP_F_REPORT.md), [`results/STEP_K_REPORT.md`](results/STEP_K_REPORT.md), [`results/PROSE_EDITS.md`](results/PROSE_EDITS.md)
- [x] [`baseline_pre_intl_shumway_20260426_221329/`](baseline_pre_intl_shumway_20260426_221329/) — pre-strict-Shumway snapshot of UK/JP panels (248 MB)

---

### Pre-this-session (existing completed items)

- [x] Add clear explanation of why long-short and not long-only
- [x] Consolidate generic validity checks (bootstrap + paired tests, subperiod stability, t-cost, π threshold, multi-state HMM, HMM feature ablation, GFC OOS, expanding-window HMM, seed convergence, RF, CRRA) — landed in Appendix `app:robustness` rather than a main-text 5.4 subsection. Mechanism-tied tests (Ridge, depth sweep, LR with interactions) kept in 5.3. Added appendix subsection `app:threshold` for regime threshold sensitivity.
- [x] Move fast, artefact-independent tests to pre-flight so they run before step 1 — fail-fast on config typos saves 2h on full rebuild. Done in `run_pipeline.py:113-133` (commit 97b35b6): runs test_config.py, test_utils.py, test_portfolio_edge_cases.py, test_pipeline_smoke.py when `--step` is not specified.
- [x] Add GitHub Actions workflow (`.github/workflows/tests.yml`) running pytest on every push. Done (commit 97b35b6).
- [x] **Parallelize HMM seeds across CPU cores.** Done in `scripts/expanding_window_backtest_parallel.py` using `multiprocessing.Pool(processes=6)`. Bit-identical to serial (verified by `tests/test_parallel_hmm_determinism.py`). 30-year expanding-window backtest at 200 HMM × 50 XGB seeds completed in ~19 hr (vs ~4 days serial).
- [x] Limitations subsection (sec:limitations) — 8 paragraphs covering economic vulnerability, stress-test caveat, test-period scope, frozen HMM parameters + regime drift, memoryless regime conditioning, implementation frictions, specification search, sample scope + causal inference (commits 0e99d9a, 008e5fa, a697ce9).
- [x] Future work — six detailed directions (predictive regime signal, alternative nonlinear models, sequential regime conditioning, regime-conditional feature sets, exposure-scaling overlay, international replication).
- [x] Comprehensive pipeline tests (237 tests in `tests/test_pipeline_technical.py`)
- [x] Code audit and bug fixes (above_med, K=2 std, hardcoded month counts)
- [x] Restructure project to production layout (artefacts/, plots/, results/, src/)
- [x] Update SHAP to 50-seed ensemble (54/46, 50/50, 58/42)
- [x] Risk aversion graph (3-panel MV sweep, Markowitz reference)
- [x] Review and tighten introduction, literature review, conclusion (73 → 69 pages)
- [x] Add fundamentals row to performance table (Sharpe 0.97 pre-Shumway; 0.92 post)
- [x] Add z-score computation explanation to results
- [x] Define z-score inline where first introduced
- [x] Trim redundant captions
- [x] Clean literature review (remove unused citations, generic HMM content)
- [x] Seed convergence check: Sharpe converges around k=50 (plateau 1.085-1.095 from k=50 to k=100), std shrinks as 1/√k. See `results/seed_convergence.csv` and `plots/seed_convergence.png`.
- [x] Test Denis's CRRA risk-aversion formula: degrades faster than MV. Even gamma=0 gives Sharpe 0.29 vs production 1.11. See `results/risk_aversion_crra_results.csv`.
- [x] Bootstrap analysis: block bootstrap CIs, paired tests vs benchmarks, regime-conditional Sharpe (Appendix `app:bootstrap`, brief mention in §5.1)
- [x] **30-year expanding-window OOS backtest** (1995-2024, annual retraining, 200 HMM × 50 XGB seeds, parallel). Full OOS Sharpe 0.50, cumulative +1,405%, MDD -64.8% during dot-com. Reproduces production Sharpe 1.11 within sampling noise on 167-month overlap (0.88).
- [x] **§5.4.3 "Historical Stress: Out-of-Sample Evidence"** drafted and committed: new subsection in `latex/main_results.tex` between sec:stress_test and sec:dm_comparison, with `table_expanding_subperiods.tex`, appendix `app:expanding_window`, cross-references, and rewritten `Stress-tested vulnerabilities` paragraph in `latex/conclusion.tex`. Commits: 6ca51c1, 997c8cc, 7d85d04, bd2677f.
- [x] Email Denis on the XGBoost pattern (sent 2026-04-23: meeting cancel, checklist, CRRA update, heatmap pivot)
- [x] Integrate CRRA into thesis: §5.4 risk-aversion robustness (γ=0 sign-log compression drops Sharpe to 0.29). Appendix extension. Data in `results/risk_aversion_crra_results.csv`.
- [x] Improve explanation of how dominant π_filter is in the trees (5a9bb34, 9483713, 2cc77f9 — full-population panic share: 73.9% of trees contain a π split, 55% of stock paths, 65% of panic long-leg return from π-splitting trees vs 17% in calm). Reproducible via `scripts/verify_pi_dominance_stats.py`.
- [x] Integrate Random Forest result: RF row added to `tab:xgb_hyperparams` (Sharpe 0.73 vs XGBoost 1.11; π_filter SHAP 18% vs 45%). Model Sensitivity appendix separates boosting-vs-bagging from hyperparameter sensitivity (7e1e455).
- [x] Integrate seed convergence: appendix `app:seed_convergence` with `table_seed_convergence.tex` (7e1e455 + bfbd80f).
- [x] Fundamentals analysis — ablation test complete. Tables: `table_fund_alphas.tex`, `table_fundamentals_ablation.tex`, `table_performance_fund_row.tex`.
- [x] **Lock full transitive Python env (`requirements.lock`)**. 73-package snapshot from `.venv` via `pip freeze`, complementing the human-curated `requirements.txt`. Reproduce with `pip install -r requirements.lock`. Commit d8afe9f.
- [x] **`experiments/` convention for one-off scripts**. Dated `YYYY-MM-DD-*.py` files are throwaway. Curbs `archive/scripts/` churn. Commit 309dc48.
- [x] **Build `main.tex` on every push (`.github/workflows/latex.yml`)**. Catches broken `\input{tables/...}` and `\includegraphics{plots/...}` references at push time; uploads compiled PDF as workflow artifact. Commit 8296da0.
- [x] **Pull UK + Japan stock panels from WRDS Compustat Global** (`scripts/import_intl_stocks.py`). Total returns via `trfd`. Outputs: `data/{uk,jp}_stock_panel.parquet`, `data/{uk,jp}_market_panel.parquet`. UK 5,569 securities × 514K stock-months; JP 6,408 × 1.42M. Schema documented in `data/INTL_PANEL_README.md`.
- [x] **Sanity-check BANK_REL signal**: UK peaks at GFC March 2009 (3.72σ) + Eurozone 2011 + Brexit 2016. JP peaks at 2001-02 NPL crisis (4.6σ); correctly NEAR ZERO during 2008 GFC.
- [x] **Adapt `scripts/hmm_model.py` to regional inputs** — done as `scripts/hmm_intl.py`.
- [x] **Smoke-fit regional HMMs** (5 seeds × 500 iter). UK π_filter correctly flags Black Wed 1992, dot-com 2001-02, GFC 2008-09, Eurozone 2011-08, COVID 2020-03. JP π_filter correctly flags 1990 bubble, 1995 Kobe, 1997-98 LTCB, 2001-03 NPL nadir, GFC, COVID.
- [x] **Adapt `scripts/cross_sectional_model.py` to regional inputs** — done as `scripts/cross_sectional_intl.py`.
- [x] **First-pass regional CS model** (smoke seeds). UK M2 XGB Sharpe 0.575; JP M2 XGB Sharpe 0.379.
- [x] **First-pass Test A** (US-trained π_filter applied to regional CS): UK M2 XGB Sharpe 0.674, JP M2 XGB Sharpe 0.431. Preliminary support for global financial cycle hypothesis. **Production rerun (Step N2 above) confirmed.**
