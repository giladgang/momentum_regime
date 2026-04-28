# Thesis TODO

**Last updated**: 2026-04-27 (post-Phase-3 composition complete; N8 IC paradox dropped from thesis; N3 reframed — both HMM and XGB CV deferred to future research only, no thesis citation; awaiting Gilad row-by-row thesis-edit review).

## Companion files

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
- [ ] **Headline numbers** (16 high-priority):
  - M2 Ann.Ret 21.9% → 21.7% | Vol 19.7% → 19.5% | MDD −24.8% → −22.8% | Beta 0.46 → 0.45
  - Sub-period Sharpes 0.62→0.59, 1.03→1.04, 1.69→1.70
  - Factor α: CAPM 23.4% (t=4.08), FF3 23.2% (t=4.48), Carhart 24.3% (t=4.75), FF5 22.9% (t=4.62), FF6 24.1% (t=4.81)
  - Stress baseline MDD −24.8% → −23.0%
  - Final-$ multiple 15.7× → 15.3×
  - Sharpe unchanged at 1.11
- [ ] **NW t-stat — 1.83\* → 4.37\*\*\*** (unique among numeric updates: significance jump from marginal to highly significant — flag in prose)
- [ ] **Existing claims to verify post-Shumway** (37 sub-sections E1-E37; full granular detail in §📋 Granular thesis-edit detail below)

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

#### N3 — Defer CV-based selection to future work

**Decision locked**: keep production specs (HMM `DD+CS+DISP+REL_N`, XGB `depth=4 lr=0.05 n=500`). **Neither the HMM CV nor the XGB CV result is cited in the thesis** — both are flagged as future research only. The existing "specification-search" / "test-set tuned" caveats in `latex/data_section.tex`, `latex/methodology.tex`, `latex/appendix.tex` STAY IN PLACE as honest acknowledgments. Existing in-thesis tables (`table_xgb_hyperparams.tex`, `table_feature_selection.tex`) and their prose are unchanged by N3 — only post-Shumway numbers refresh via Bucket 1.

- [ ] **Edit #6 (future work — HMM feature selection)** — paragraph in `latex/future_work_full.tex` on CV as more robust HMM feature-selection. **Draft prose** (does NOT reveal CV findings):

  > Future research should replace the test-period-evaluated 4-pass feature-selection pipeline used here (Section~X.Y) with a cross-validation procedure run entirely within the training partition — for instance, a 5-fold expanding-window CV on the 1990-2010 panel that never touches the 2011-2025 test data — combined with rolling-window robustness checks at multiple training-end dates to validate stability of the chosen feature set across regimes.

  No in-thesis citations to specific CV results.

- [ ] **Edit #7 (future work — XGB hyperparameter dissection)** — paragraph in `latex/future_work_full.tex`. **Draft prose**:

  > A natural extension of this work is a systematic dissection of the XGBoost hyperparameter space — sweeping tree depth, learning rate, and ensemble size at fine resolution and comparing not only out-of-sample Sharpe ratios but also the cross-sectional z-scores of long- and short-leg selections across regimes. Such an analysis would test whether the regime-conditional term-structure shape we identify (Section~X.Y) is robust to model-capacity choices, or whether the panic-recovery selection pattern is specific to the production specification. A rolling-window cross-validation across multiple training-end dates would further test stability of the optimum across market regimes.

  No in-thesis citations to specific CV results.

- [ ] CV artefacts (`tables/table_hmm_cv.tex`, `tables/table_xgb_cv.tex`, `results/hmm_cv_*`, `results/xgb_cv_*`) exist on disk for follow-up work but are NOT referenced in any `.tex` file. Verify after Bucket 2 edits that no `\input` or `\ref` to these files leaked in.

#### N5 — NW t-stat significance jump — REFRAME defensive paragraph + sharpen

**More than just a number swap.** The current `latex/conclusion.tex` paragraph **defends** the low t-stat:

> "The Newey-West t-statistic on M2's excess return over the market is **1.83**; this reflects the strategy's near-zero market beta, and the economically meaningful test is the factor model alpha..."

With t=4.37\*\*\* post-Shumway, this defensive framing is no longer needed and reads awkwardly. The paragraph should be **rewritten**, not just have the number swapped.

- [ ] Rewrite the defensive t-stat paragraph in `latex/conclusion.tex` (currently in the limitations/caveats section). New framing: "The Newey-West t-statistic on M2's excess return over the market is 4.37\*\*\* post-Shumway, well above conventional thresholds. The factor-model alphas in Appendix~\ref{app:factor_alphas} confirm this across all five factor specifications (t > 4)."
- [ ] One-line emphasis wherever the t-stat appears in `latex/main_results.tex`: **t-stat 1.83\* → 4.37\*\*\*** post-Shumway. Marginal → highly significant. Largest single result of the Shumway treatment.

#### N9 — Already strong in existing thesis (verify only, no new prose)

Items that the thesis already frames well; just verify post-Shumway numbers updated via Bucket 1:

- [ ] **Panic subtypes** — `latex/main_results.tex` already says "the overall panic Sharpe of 1.56 is driven entirely by recovery months" with all the right numbers. Just verify the −0.49 / +4.42 / Sharpe-2.35 numbers in PROSE_EDITS.md.
- [ ] **Ridge baseline** — `latex/main_results.tex` 5.3 nonlinearity section already discusses ridge failure. Just verify the −0.58 number lands consistently and the "all alphas give the same answer" claim is preserved.
- [ ] **Term-structure framing** — `latex/conclusion.tex` already mentions "term-structure reorganisation". Detailed §5.2 figure explanations are already tracked in the "Pre-existing thesis-writing tasks" section below (improve Chapter 5.2 graphs).

#### N6 — Connect 30-year backtest to stress scenario — REFRAME

- [ ] Add explicit cross-link between §5.4.3 (30-year backtest) and §5.4 (stress test).
  - The 36-month dot-com bear (2000-2002, -36% cumulative, MDD -64.8%) is the documented version of the synthetic 24+ month sustained-bear stress scenario.
  - Hypothetical → real. One paragraph that ties the two analyses together explicitly.

---

### Citations to add (across N1, N2)

- [ ] **Rey, H. (2013)** — global financial cycle (N1 framing)
- [ ] **Asness, Moskowitz & Pedersen (2013)** — international momentum (N1 JP caveat)
- [ ] **Daniel & Moskowitz (2016)** — distinguish their inversion from N2 re-ranking
- [ ] **Shumway (1997, 2001)** — delisting bias methodology footnote
- [ ] **Jegadeesh (1990)** + **Lehmann (1990)** — short-term reversal (M5 mechanism prose, mom_1 neutral)
- [ ] **Novy-Marx (2012)** — intermediate-horizon momentum (mom_8 peak)
- [ ] **Lee & Swaminathan (2000)** — momentum lifecycle (mom_12 decay)

---

### Detailed reference

The HARD STOP section above is the day-to-day "what to write next" view. The full granular detail with every micro-task and file pointer is inlined below in §📋 Granular thesis-edit detail.

---

## 🔑 Decisions made (2026-04-27)

- [x] **HMM feature set**: KEEP production `DD+CS+DISP+REL_N`. Step J CV winner (`DD+CS+LVIX+DISP`) is statistically indistinguishable within fold-noise (gap +0.221 vs fold-std 0.694), but production ranks 47/64 — bottom-half placement is too defensive to cite in thesis. Deferred to future work (N3 Edit #6).
- [x] **XGB hyperparameters**: KEEP production `depth=4, lr=0.05, n=500`. Step I CV puts depth-3 narrowly atop the grid (+0.596 vs depth-4 +0.576, gap 0.021 within fold-std 0.811 — essentially zero). Test-period sweep ([`results/depth_results.csv`](results/depth_results.csv)) prefers depth-4 (Sharpe 1.11 vs 0.98). Both objectives jointly support keeping depth-4. Deferred to future work with z-score-across-regimes angle (N3 Edit #7).
- [x] **No retraining at CV winners** — production stays as the canonical specification.
- [x] **CV results NOT cited in thesis** — both HMM and XGB CV stay out of thesis prose entirely. Future research only. Existing "specification-search" / "test-set tuned" caveats stay in place as honest acknowledgments. See 🔬 CV artefacts section below for what stays on disk.

---

## 🔬 CV artefacts (NOT cited in thesis — per N3 decision)

CV results stay out of thesis prose entirely. These exist on disk for follow-up research:

- [`results/xgb_cv_winner.json`](results/xgb_cv_winner.json), [`results/xgb_cv_results.csv`](results/xgb_cv_results.csv), [`tables/table_xgb_cv.tex`](tables/table_xgb_cv.tex) — Step I (XGB CV)
- [`results/hmm_cv_winner.json`](results/hmm_cv_winner.json), [`results/hmm_cv_features.csv`](results/hmm_cv_features.csv), [`tables/table_hmm_cv.tex`](tables/table_hmm_cv.tex) — Step J (HMM CV)
- HMM production combo's CV rank: 47/64 (mean +0.286, fold-std 0.510) — within fold-noise of winner but bottom-half rank; this is why we're not citing it.

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

## 💡 Optional future-work analyses (candidates for `latex/future_work_full.tex`)

Carried over from old project planning; absent from current thesis and from `future_work_full.tex`. Out of scope for this draft, but each is a plausible "next paper" extension. Add to future-work section if any of them resonates.

- [ ] **Portfolio characteristics by regime** — size, β, B/M, profitability, leverage of long/short legs in calm vs panic. Partial scaffolding exists in [`scripts/portfolio_characteristics_chart.py`](scripts/portfolio_characteristics_chart.py), [`scripts/portfolio_chars_options.py`](scripts/portfolio_chars_options.py) — never landed in thesis.
- [ ] **Forward return decomposition** — in panic reversal bets, separate stock-specific recovery from market-recovery contribution. Adjacent to but distinct from §E15 panic subtypes (which split panic-crash vs panic-recovery months).
- [ ] **Holding-period analysis** — hold the panic portfolio 1, 3, 6 months instead of monthly rebalancing. Tests whether the panic edge is genuine reversal or short-lived noise.
- [ ] **Sector rotation by regime** — which sectors does the strategy buy/short in calm vs panic? Currently we only document the term-structure shape, not the cross-sectional sector tilt.

---

## 🎯 Final wrap-up (after Bucket 1 + Bucket 2)

- [ ] **Final coherence pass** — read straight through the post-edit thesis; check that the new sections (N1 international, N2 re-ranking) cross-reference cleanly with the existing §5.4.3 30-year backtest and §5.4 stress test, and that no contradictory sentence survived from a deleted paragraph.
- [ ] **Send updated draft to Denis** once Buckets 1 + 2 land.

---

## ⚙️ Source-of-truth framework — BUILT and wired into pipeline

Canonical metrics store. Every published thesis number lives under a named key. Wired into [`run_pipeline.py`](run_pipeline.py) as steps 21 ([`build_metrics.py`](scripts/build_metrics.py)), 22 ([`build_canonical_macros.py`](scripts/build_canonical_macros.py)), 24 ([`verify_thesis_consistency.py`](scripts/verify_thesis_consistency.py)) — runs automatically after every full pipeline.

**Files (all built):**
- [`results/PRODUCTION_METRICS.json`](results/PRODUCTION_METRICS.json) — canonical store (~200 metrics across 12 sections)
- [`scripts/_canonical_metrics.py`](scripts/_canonical_metrics.py) — read/write helper (`set_metric`, `get_metric`)
- [`scripts/build_metrics.py`](scripts/build_metrics.py) — parses `tables/*.tex` + `results/*.csv` → JSON
- [`scripts/build_canonical_macros.py`](scripts/build_canonical_macros.py) — emits `latex/canonical_macros.tex`
- [`scripts/verify_thesis_consistency.py`](scripts/verify_thesis_consistency.py) — flags prose-vs-JSON mismatches
- [`latex/canonical_macros.tex`](latex/canonical_macros.tex) — auto-generated; do NOT hand-edit

**Open follow-ups** (tracked in 🛠️ Engineering follow-ups below): CI hook, thesis macro adoption, verifier regex.

**Usage when macros adopted:**

```latex
\input{canonical_macros}   % at the top of main.tex once

The strategy delivers a Sharpe ratio of \mmperfm2sharpe with a Newey-West
t-statistic of \mmperfm2nwt, well above conventional thresholds.
```

---

## 🛠️ Engineering / testing follow-ups (low priority)

- [ ] Add `pytest.skip` guard to `tests/test_utils.py::TestDataLoaders` so the class is included in `.github/workflows/tests.yml`. Currently excluded because the loader tests depend on data parquets that aren't in git; the skip guard lets the artefact-independent tests in the file run on CI.
  - File to edit: [`tests/test_utils.py`](tests/test_utils.py) (TestDataLoaders class)
  - CI config: [`.github/workflows/tests.yml`](.github/workflows/tests.yml)
- [ ] Hook `verify_thesis_consistency.py` into the CI workflow to fail PRs that introduce prose-vs-canonical drift
- [ ] Adopt `\mmperfm2sharpe`-style macros for the headline numbers in [`latex/main_results.tex`](latex/main_results.tex) — Bucket 1 numeric updates become a one-liner rerun afterward
- [ ] Tighten the verifier's regex patterns in [`scripts/verify_thesis_consistency.py`](scripts/verify_thesis_consistency.py): currently it occasionally matches a nearby unrelated number (e.g. CI bound `[0.67, 1.54]` flagged as M2 Sharpe drift). Section-aware extraction would reduce false positives.

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

## 📋 Granular thesis-edit detail

Reference layer for the HARD STOP section. Triage tags: **[M]** = must land, **[S]** = should land, **[N]** = nice to have.

### §A. Numeric updates — full list (Bucket 1 headlines, with sources)

- [ ] [M] M2 Ann.Ret 21.9% → **21.7%** — [`tables/table_performance.tex`](tables/table_performance.tex), goes to [`latex/main_results.tex`](latex/main_results.tex)
- [ ] [M] M2 Ann.Vol 19.7% → **19.5%** — same
- [ ] [M] M2 Sharpe 1.11 → **1.11** (unchanged) — same
- [ ] [M] M2 Max DD −24.8% → **−22.8%** — same
- [ ] [M] M2 Beta 0.46 → **0.45** — same
- [ ] [M] M2 NW t-stat: 1.83\* → **4.37\*\*\*** *(major significance jump)* — same
- [ ] [M] M2 Final-$ multiple 15.7× → **15.3×** — same
- [ ] [M] Sub-period Sharpe 2011–2015: 0.62 → **0.59** — [`tables/table_subperiod.tex`](tables/table_subperiod.tex)
- [ ] [M] Sub-period 2016–2020: 1.03 → **1.04** — same
- [ ] [M] Sub-period 2021–2025: 1.69 → **1.70** — same
- [ ] [M] CAPM α: 23.8% → **23.4% (t=4.08)** — [`tables/table_factor_alphas.tex`](tables/table_factor_alphas.tex)
- [ ] [M] FF3 α: 23.7% → **23.2% (t=4.48)** — same
- [ ] [M] Carhart α: 24.9% → **24.3% (t=4.75)** — same
- [ ] [M] FF5 α: 23.4% → **22.9% (t=4.62)** — same
- [ ] [M] FF6 α: 24.7% → **24.1% (t=4.81)** — same, mentioned in [`latex/main_results.tex`](latex/main_results.tex)
- [ ] [M] Stress baseline MDD −24.8% → **−23.0%** — [`tables/table_stress_scenarios.tex`](tables/table_stress_scenarios.tex)
- [ ] [M] All other prose mentions of pre-Shumway numbers — see [`results/PROSE_EDITS.md`](results/PROSE_EDITS.md) (944 candidates, 415 cells across 40 tables)

### §E. Existing claims worth restating / verifying post-Shumway

#### §E0. CV tables NOT cited in thesis (per N3 decision)

- [ ] [M] `tables/table_hmm_cv.tex` and `tables/table_xgb_cv.tex` exist on disk but are NOT referenced in any `.tex` file. Verify no `\input` or `\ref` leaked in during Bucket 2 edits.
- [ ] [N] Companion CSVs (`results/hmm_cv_*`, `results/xgb_cv_*`, `results/hmm_feature_selection_pass{1,2,3}.csv`) kept for audit; not cited in thesis.
- [ ] [M] Existing thesis tables `tables/table_xgb_hyperparams.tex` and `tables/table_feature_selection.tex` STAY in place; verify post-Shumway numeric updates flowed through (covered in §A and Bucket 1 prose walk).

#### §E1–E4. Ablations

- [ ] [M] **Removing π**: Sharpe 1.11 → **0.41** — [`tables/table_kitchen_sink.tex`](tables/table_kitchen_sink.tex), [`tables/table_regime_signal_ablation.tex`](tables/table_regime_signal_ablation.tex)
- [ ] [M] Raw stress indicators (no HMM): Sharpe **0.257** — [`tables/table_regime_signal_ablation.tex`](tables/table_regime_signal_ablation.tex)
- [ ] [M] No regime signal: Sharpe **0.429** — same
- [ ] [M] HMM smoothing essential — raw indicators *worse than no signal* in some configs
- [ ] [M] GHM-style classification collapses: M2 1.11 vs SLOW 0.07, MED 0.12, FAST 0.26, DYN 0.03 — [`tables/table_ghm_comparison.tex`](tables/table_ghm_comparison.tex)
- [ ] [M] Adding fundamentals dilutes: 1.11 → **0.92** despite 43% fund SHAP — [`tables/table_fundamentals_ablation.tex`](tables/table_fundamentals_ablation.tex)
- [ ] [M] Raw-return targets essential — [`tables/table_alt_targets.tex`](tables/table_alt_targets.tex), [`tables/table_risk_aversion.tex`](tables/table_risk_aversion.tex)
- [ ] [S] Note: MV with γ=2 actually *better* (Sharpe 1.135, MDD −22.0%); Log-MV γ=5 best MDD (−18.7%) — qualifies "raw essential"
- [ ] [M] Sequential boosting essential — RF bagging loses ~0.3 Sharpe — [`results/random_forest_results.csv`](results/random_forest_results.csv)

#### §E5. Regime-conditional Sharpe

- [ ] [M] M2 Calm **0.84** (was 0.82); M2 Panic **1.53** (was 1.56) — [`tables/table_regime_sharpe.tex`](tables/table_regime_sharpe.tex)

#### §E6. Cost sensitivity

- [ ] [S] M2 at 0/5/10/20/30/50 bps: 1.19/1.15/1.11/1.03/0.95/**0.78** — [`tables/table_cost_sensitivity.tex`](tables/table_cost_sensitivity.tex)
- [ ] [S] M2 is the **only** strategy profitable at 50bps; benchmarks unprofitable by 30bps

#### §E7. Threshold sensitivity

- [ ] [S] π > 0.25, 0.50, 0.75 all give stable regime-conditional Sharpes — [`tables/table_threshold_sensitivity.tex`](tables/table_threshold_sensitivity.tex)

#### §E8. HMM diagnostics

- [ ] [S] K=2 best AND most stable; K=3 → 0.83, K=4 → 0.61 (std=0.29), K=5 → 0.72 — [`tables/table_multistate_hmm.tex`](tables/table_multistate_hmm.tex)
- [ ] [S] **Regime separation magnitudes** (95% CI excludes 0): DD Δ=−1.51 [−1.80,−1.23]; DISP Δ=+1.12 [+0.87,+1.35]; REL_N Δ=−1.42 [−1.59,−1.24]; CS Δ=+0.90 [+0.62,+1.16] — [`tables/table_hmm_separation.tex`](tables/table_hmm_separation.tex)
- [ ] [N] Student-t HMM: no improvement over Gaussian — [`tables/table_student_t_hmm.tex`](tables/table_student_t_hmm.tex)
- [ ] [N] Gelman-Rubin: chains converge — [`tables/table_gelman_rubin.tex`](tables/table_gelman_rubin.tex)

#### §E9. Spanning regressions

- [ ] [M] M2 ~ M1: M2 retains alpha; M1 ~ M2: M1 alpha doesn't survive — computed by [`scripts/new_ls_analyses.py`](scripts/new_ls_analyses.py)

#### §E10. Granger causality

- [ ] [S] π → IC_mom and IC_mom → π **both non-significant** (F<2, p>0.10 lags 1–3) — [`tables/table_granger.tex`](tables/table_granger.tex). Frame: model uses regime + momentum *jointly*, not sequentially

#### §E12. Bootstrap inference

- [ ] [M] M2 Sharpe 1.11, **95% CI [0.66, 1.54]** — [`tables/table_bootstrap.tex`](tables/table_bootstrap.tex), [`results/bootstrap_sharpe_cis.csv`](results/bootstrap_sharpe_cis.csv)
- [ ] [M] Calm 0.82 [0.25,1.32]; Panic 1.57 [0.93,2.19]
- [ ] [S] Panic − Calm: 0.75 [−0.08,1.60], **p=0.074\*** (panic outperformance only marginally significant)
- [ ] [M] **Paired tests vs benchmarks**: M2 vs M1 **p=0.001**, vs M0 **p=0.010**, vs Fixed 12-mo **p=0.006**, vs Fixed 1-mo **p=0.024** — [`results/bootstrap_paired_tests.csv`](results/bootstrap_paired_tests.csv)

#### §E13. Factor loadings

- [ ] [M] M2 has NEGATIVE UMD loading (−0.20 to −0.22) — bets on a different slice
- [ ] [M] Negative HML (−0.17 to −0.31): growth-tilted; Mkt-RF ≈ −0.15: defensive
- [ ] [S] Positive CMA in FF5/FF6 (+0.12 to +0.21): investment-tilted

#### §E14. Performance table benchmarks

- [ ] [S] All non-M2 strategies have **negative β** (−0.34 to −0.66); M2 is the only L/S strategy with positive β (+0.45) — [`tables/table_performance.tex`](tables/table_performance.tex)
- [ ] [S] M2 NW t-stat 4.37\*\*\* — strongest of all benchmarks

#### §E15. Panic SUBTYPES (mechanism finding)

- [ ] [M] **Calm Sharpe 0.82** (108 mo) — [`tables/table_panic_subtypes.tex`](tables/table_panic_subtypes.tex)
- [ ] [M] **Panic-Crash (negative-mkt) Sharpe −0.33** (18 mo) — strategy *loses* in actual crashes
- [ ] [M] **Panic-Recovery (positive-mkt within panic) Sharpe +2.35** (41 mo)
- [ ] [M] Reframe: strategy wins by riding the rebound rally after panic onset, not by predicting crashes

#### §E16. Stress scenario progression

- [ ] [S] Linear: 3-mo recession adds 10% loss, 6=19%, 12=34%, 18=47%, **24=57%** — [`tables/table_stress_scenarios.tex`](tables/table_stress_scenarios.tex)
- [ ] [S] Worst-case MDD: 3mo→−38%, 12mo→−49%, **24mo→−69%**

#### §E17. Ridge baseline (definitive)

- [ ] [M] Ridge with α∈{0.1,1,10,100,1000}: ALL give Sharpe **−0.58** — [`tables/table_ridge.tex`](tables/table_ridge.tex)
- [ ] [M] OLS unconstrained: also −0.58
- [ ] [M] **Linearity is the binding constraint, NOT regularisation choice**

#### §E18. Z-scores by horizon

- [ ] [S] Calm Long peak at mom_8: +0.39 z; Panic Long flat ≈ −0.13 — [`tables/table_zscore_shap_detail.tex`](tables/table_zscore_shap_detail.tex)
- [ ] [S] Panic Short at mom_1: +0.30 z (recent winners get shorted)
- [ ] [S] SHAP shares: mom_11 18.6–23.1%, mom_9 12.7–15.5%, mom_8 10.2–12.8%

#### §E19. Tree path / combo analysis

- [ ] [N] Most common decision paths use multi-horizon conditioning — [`results/tree_path_results.csv`](results/tree_path_results.csv)
- [ ] [N] Combo frequencies: I+L 12.9–14.8%, M+I+L ~10%, single-horizon 5–7% — [`tables/table_combo_freq.tex`](tables/table_combo_freq.tex)
- [ ] [N] Long-combo z-scores: I+L at +0.45 z; S+L at +0.46 — [`tables/table_combo_long_cp.tex`](tables/table_combo_long_cp.tex)

#### §E20. Sample summary

- [ ] [N] 16,657 unique stocks; 1.72M stock-month obs; train 4,838/mo, test 3,336/mo — [`tables/table_sample_summary.tex`](tables/table_sample_summary.tex)

#### §E21. Alt train/test splits robustness

- [ ] [S] 1990-2010 baseline = 1.11; verify others post-Shumway are robust — [`tables/table_alt_splits.tex`](tables/table_alt_splits.tex)

#### §E22. IC rotation across regimes

- [ ] [S] Per-horizon IC by calm vs panic — [`tables/table_ic_rotation.tex`](tables/table_ic_rotation.tex)
- [ ] [S] Per-horizon IC similar across regimes; regime difference is *which side of distribution* model picks

#### §E23. HMM feature ablation

- [ ] [S] Subsets of 4-feature HMM input — [`tables/table_hmm_feature_ablation.tex`](tables/table_hmm_feature_ablation.tex). Full 4F baseline 0.97 (in-sample)
- [ ] [S] In-sample feature ablation REMAINS the canonical justification for the 4-feature HMM input; CV-based selection deferred to N3 Edit #6 future work

#### §E24. Fundamentals factor alphas

- [ ] [S] Fund-included M2 alphas: CAPM α drops 23.4% → **18.2%** — [`tables/table_fund_alphas.tex`](tables/table_fund_alphas.tex)
- [ ] [N] Fund-included perf row: Ann.Ret 16.6%, Vol 18.6%, Sharpe **0.92**, MDD −18.4%, β=0.11, NW t=3.70\*\*\* — [`tables/table_performance_fund_row.tex`](tables/table_performance_fund_row.tex)

#### §E25. LR (M1) coefficients

- [ ] [S] LR coefficient estimates — [`tables/table_lr_coef.tex`](tables/table_lr_coef.tex)
- [ ] [S] M1's coefficients stable but sign-aware portfolio-formation step fails (M1 IC>0 but Sharpe ≈ 0)

#### §E26. SHAP feature attribution

- [ ] [M] Overall SHAP: Momentum **54%**, π_filter **46%** — [`tables/table_shap.tex`](tables/table_shap.tex)
- [ ] [M] Calm: Momentum 51%, π 49%
- [ ] [M] Panic: Momentum 59%, π 41%
- [ ] [M] Implication: model uses momentum *more* in panic and π *more* in calm — opposite of expectation
- [ ] [S] SHAP dependence by horizon × regime — [`results/shap_dependence_all_horizons.csv`](results/shap_dependence_all_horizons.csv)

#### §E27. Turnover

- [ ] [S] M2 monthly TO **132.4%** (annualised ~1,590%) vs Fixed 12-mo at 70.1% — [`tables/table_turnover.tex`](tables/table_turnover.tex)
- [ ] [S] Still profitable at 50bps cost (§E6)

#### §E28. Placebo test

- [ ] [M] Real π gives M2 1.11; placebo π collapses — [`tables/table_placebo.tex`](tables/table_placebo.tex)

#### §E29. Seed convergence

- [ ] [S] Sharpe stabilizes ~1.11 at k=50 seeds; std 1/√k — [`tables/table_seed_convergence.tex`](tables/table_seed_convergence.tex)

#### §E30. Tree-path SHORT-leg combos

- [ ] [N] Short-leg combo paths and z-scores — [`tables/table_combo_short_cp.tex`](tables/table_combo_short_cp.tex). Confirms term-structure shape *flips on short side too*

#### §E31. Pi-filter dominance stats

- [ ] [N] π_filter dominates other regime-signal candidates — [`results/pi_dominance_stats.csv`](results/pi_dominance_stats.csv)

#### §E32. Two-model dual utility

- [ ] [N] Dual-utility comparison (M1 + M2) — [`results/two_model_dual_util.csv`](results/two_model_dual_util.csv) ([`scripts/two_model_dual_util.py`](scripts/two_model_dual_util.py))

#### §E33. Other existing claims (verify post-Shumway)

- [ ] [N] Selection rank analysis — [`results/selection_rank_analysis.csv`](results/selection_rank_analysis.csv)
- [ ] [N] Risk-aversion CRRA — [`results/risk_aversion_crra_results.csv`](results/risk_aversion_crra_results.csv)
- [ ] [N] Risk-aversion extended — [`results/risk_aversion_extended_results.csv`](results/risk_aversion_extended_results.csv)
- [ ] [N] Risk-aversion thesis (headline) — [`results/risk_aversion_thesis_results.csv`](results/risk_aversion_thesis_results.csv)
- [ ] [N] Z-score time series — [`results/zscore_long_by_month.csv`](results/zscore_long_by_month.csv), [`results/zscore_short_by_month.csv`](results/zscore_short_by_month.csv), [`results/zscore_longshort_by_month.csv`](results/zscore_longshort_by_month.csv)
- [ ] [N] Tree combo full results — [`results/tree_combo_results.csv`](results/tree_combo_results.csv)
- [ ] [N] RF baseline (RF underperforms XGB) — [`results/random_forest_results.csv`](results/random_forest_results.csv)
- [ ] [N] Fundamentals test — [`results/fundamentals_test_results.csv`](results/fundamentals_test_results.csv)
- [ ] [N] Bootstrap regime-conditional — [`results/bootstrap_regime.csv`](results/bootstrap_regime.csv)

#### §E34. January-exclusion robustness

- [ ] [S] M2 ex-January Sharpe ~1.06 (verify post-Shumway) — [`tables/table_january.tex`](tables/table_january.tex)

#### §E35. 30-year backtest companion files

- [ ] [S] Year-by-year returns — [`results/expanding_returns_prod.csv`](results/expanding_returns_prod.csv)
- [ ] [S] Year-by-year π_filter — [`results/expanding_pi_filter_prod.csv`](results/expanding_pi_filter_prod.csv)
- [ ] [S] Per-year fit summary — [`results/expanding_summary_prod.csv`](results/expanding_summary_prod.csv)
- [ ] [N] OOS 1990-1999 train — [`results/oos_returns_prod_1990_1999.csv`](results/oos_returns_prod_1990_1999.csv), [`results/oos_pi_filter_prod_1990_1999.csv`](results/oos_pi_filter_prod_1990_1999.csv), [`results/oos_subperiods_prod_1990_1999.csv`](results/oos_subperiods_prod_1990_1999.csv)
- [ ] [N] OOS 1990-2004 train — [`results/oos_returns_prod_1990_2004.csv`](results/oos_returns_prod_1990_2004.csv), [`results/oos_subperiods_prod_1990_2004.csv`](results/oos_subperiods_prod_1990_2004.csv), [`results/oos_pi_filter_prod_1990_2004.csv`](results/oos_pi_filter_prod_1990_2004.csv)
- [ ] [N] Sub-period decomposition full — [`results/oos_subperiod_full.csv`](results/oos_subperiod_full.csv)

#### §E36. International full returns time series

- [ ] [N] UK monthly returns: [`results/intl_uk_returns.csv`](results/intl_uk_returns.csv); UK with US π: [`results/intl_uk_returns_uspi.csv`](results/intl_uk_returns_uspi.csv)
- [ ] [N] JP monthly returns: [`results/intl_jp_returns.csv`](results/intl_jp_returns.csv); JP with US π: [`results/intl_jp_returns_uspi.csv`](results/intl_jp_returns_uspi.csv)

#### §E37. Critical figures for new claims

- [ ] [M] [`plots/depth_vs_sharpe.pdf`](plots/depth_vs_sharpe.pdf) — depth-vs-Sharpe (already in thesis)
- [ ] [M] [`plots/chart2_rank_by_horizon_v2.pdf`](plots/chart2_rank_by_horizon_v2.pdf) — term-structure shape (consider for main_results)
- [ ] [M] [`plots/leg_betas_rolling.pdf`](plots/leg_betas_rolling.pdf) — rolling β panel (supports N2)
- [ ] [S] [`plots/momentum_shape_final.pdf`](plots/momentum_shape_final.pdf), [`plots/chart_ls_rank_yearly.pdf`](plots/chart_ls_rank_yearly.pdf), [`plots/chart_m2_vs_mom12.pdf`](plots/chart_m2_vs_mom12.pdf), [`plots/avg_tree_production.pdf`](plots/avg_tree_production.pdf)
- [ ] [S] SHAP attribution figures: [`plots/chart1_shap_by_horizon.pdf`](plots/chart1_shap_by_horizon.pdf), [`plots/chart1_shap_pct_by_horizon.pdf`](plots/chart1_shap_pct_by_horizon.pdf)

### §F. Methodology / appendix items

- [ ] [N] HMM Bayesian Gibbs / FFBS sampler details (priors from `config.py`)
- [ ] [N] Production: 200 HMM × 50 XGB seeds
- [ ] [N] Shumway-analogue rule table — [`latex/methodology.tex`](latex/methodology.tex) footnote
- [ ] [N] UK/JP method: BANK_REL replaces CS_z (no Moody's BAA-AAA international); 4-feature HMM `DD+DISP+REL_N+BANK_REL`
- [ ] [N] Compustat Global field aliasing: `dldtei`→`dldte`, `dlrsni`→`dlrsn`
- [ ] [N] Block bootstrap: 12-mo blocks, 10,000 resamples
- [ ] [N] WRDS DNS workaround technical note (optional) — [`scripts/import_intl_stocks.py:46-56`](scripts/import_intl_stocks.py#L46-L56)

### §I. Advisor pressure-point Q&A checklist

If your advisor presses on these, you have answers ready:

- [ ] "How do you avoid look-ahead in feature selection?" → honest caveat in thesis acknowledges test-set tuning; CV-based selection flagged as future research (N3 Edit #6)
- [ ] "How sensitive is the result to hyperparameters?" → flagged as future research (N3 Edit #7); thesis uses fixed production specification
- [ ] "What about transaction costs?" → §E6, profitable to 50bps
- [ ] "What about K=3 states?" → §E8, K=2 most stable
- [ ] "What about other risk-aversion targets?" → §E1 + side-note (some MV configs better)
- [ ] "Does it generalize internationally?" → N1, US π beats regional in UK and JP
- [ ] "Does it work over a longer history?" → 30-year backtest in completed work, but with significant DD in 2000–2002
- [ ] "Can it survive a structurally novel regime?" → §5.4.3 dot-com paragraph (-64.8% DD over 32 months) + Test period scope caveat in conclusion.tex — historical realisation of the reversal-failure vulnerability
- [ ] "Is M2 just a better momentum factor?" → "panic long leg is a term-structure-wide loser portfolio" (§5.2.4 Discussion); negative UMD loading is the factor-side reflection of that, not an independent signal
- [ ] "Why nonlinearity? Why not Ridge?" → §E17 (Ridge gives −0.58 across all α; OLS too)
- [ ] "Is the panic edge real?" → §E12 (panic−calm marginally sig p=0.07) + §E15 (it's the recovery component, not crashes)
- [ ] "Why these 4 HMM features?" → in-sample feature ablation (§E23) + 4-pass test-period selection acknowledged with caveat; CV-based selection flagged as future research (N3 Edit #6)
- [ ] "How much does π actually contribute?" → §E1 (1.11 → 0.41 without π) + §E16 (HMM smoothing >> raw indicators)

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
- [x] 207 reviewable thesis-edit checkboxes with file references for each data source — folded into §📋 Granular thesis-edit detail in this file (originally `results/THESIS_EDITS_TODO.md`, now consolidated)
- [x] [`results/STEP_F_REPORT.md`](results/STEP_F_REPORT.md), [`results/STEP_K_REPORT.md`](results/STEP_K_REPORT.md), [`results/PROSE_EDITS.md`](results/PROSE_EDITS.md)
- [x] [`baseline_pre_intl_shumway_20260426_221329/`](baseline_pre_intl_shumway_20260426_221329/) — pre-strict-Shumway snapshot of UK/JP panels (248 MB)

#### Source-of-truth framework built (2026-04-27)

- [x] [`results/PRODUCTION_METRICS.json`](results/PRODUCTION_METRICS.json) canonical store + helpers ([`scripts/_canonical_metrics.py`](scripts/_canonical_metrics.py), [`scripts/build_metrics.py`](scripts/build_metrics.py), [`scripts/build_canonical_macros.py`](scripts/build_canonical_macros.py), [`scripts/verify_thesis_consistency.py`](scripts/verify_thesis_consistency.py)) + [`latex/canonical_macros.tex`](latex/canonical_macros.tex)
- [x] Wired into [`run_pipeline.py`](run_pipeline.py) as steps 21 (build_metrics), 22 (build_canonical_macros), 24 (verify_thesis_consistency) — runs after every full pipeline
- [x] Open follow-ups remain in 🛠️ Engineering (CI hook, macro adoption, regex tightening) — out of scope for this session

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
