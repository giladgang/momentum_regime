# Thesis TODO

## In progress
- [ ] Discussion of risk premium

## Post-Shumway rerun + leg-betas finding
**Runbook:** `NEXT_RUN_PLAN.md` (self-contained, resumable; gates B-K and the thesis-edit bundle #1-#4 are spelled out there).

Triggered by audit finding that the existing `data/crsp_msf_raw.parquet` lacked Shumway (2001) delisting return treatment. 1,276 rows out of 2,082,485 received corrected `ret_adj` values (rule 1 dominates: 1,294 rows compounded with `dlret`; rule 4 imputes -30% for performance delistings with both `ret` and `dlret` missing). Backup at `data/crsp_msf_raw.parquet.bak_before_shumway` (Apr 24 mtime).

- [x] **Step B** — Snapshot pre-Shumway state to `baseline_pre_shumway_20260426_150517/` (3.0 GB; 12 artefacts, 56 tables, 164 plots, 55 results files, plus `panel_with_regimes.parquet`). Rollback target if post-Shumway numbers shift anomalously.
- [x] **Step C** — `python scripts/apply_shumway_delisting.py --force` applied. Counts match expectations exactly (rule_1=1294, rule_2=2, rule_3=11, rule_4=1, rule_5=2,081,177).
- [ ] **Step D** — `python run_pipeline.py` running in background (PID 91531, started 15:14 on 2026-04-26). Pre-flight tests cleared (caught one too-strict path-with-args check in `test_step_scripts_exist`, fixed in commit `d3b2b60`). Estimated ~3h. Skips Step 1 (HMM cached) and Step 19 (expanding-window backtest results exist). Early diff at Step 4: M2 XGB Sharpe shifts from 1.11 to 1.107 (-0.003), well inside the <0.05 expected band. Log: `logs/pipeline_post_shumway.log`.
- [ ] **Step E** — `python scripts/leg_betas_by_regime.py`. Computes calm vs panic leg betas per strategy. Outputs `results/leg_betas_by_regime.csv`, `tables/table_leg_betas.tex`, `plots/leg_betas_rolling.pdf`. Decides whether the thesis-edit bundle #1-#4 lands as the "clean inversion" framing or the "cross-sectional re-ranking" reframe.
- [ ] **Step F report** — `results/STEP_F_REPORT.md` (skeleton already drafted). Compares pre-vs-post Shumway: headline Sharpe / SHAP shares / sub-period Sharpes / factor alphas, plus the leg-betas finding. STOP for Gilad review. NO `latex/*.tex` edits before greenlight (memory: `feedback_thesis_edits_await_data`).
- [ ] **Steps G-J** — regression-test recalibration, CV smoke, then the full XGB CV (~2-5h) and HMM CV (~15-30h) — see "Methodology validation (CV retrofits)" below.
- [ ] **Step K final report** — once CV winners land, compose final report listing every thesis line that needs updating (numerical consistency + the #1-#4 bundle). STOP. Gilad explicitly greenlights each edit.
- [ ] **Thesis-edit bundle (gated on Step F + Step K greenlight)**: Edit #1 new subsection "Leg-level beta dynamics across regimes" in `latex/main_results.tex` (or reframe if no clean inversion); Edit #2 expand D&M beta discussion in `latex/literature_review.tex:19`; Edit #3 sharpen "two channels" framing in `latex/conclusion.tex:53`; Edit #4 pi_filter ↔ D&M bear-indicator link in `latex/methodology.tex` (standalone, lands regardless of #1 outcome).

## International validation (UK + Japan)
**Runbook:** `INTL_VALIDATION_PLAN.md` (self-contained, resumable across sessions). Goal: test whether the regime+momentum architecture transfers to UK and Japan. 4-feature regional HMM uses `[DD_z, DISP_z, REL_N_z, BANK_REL_z]` where `BANK_REL_z` substitutes for `CS_z` (no Moody's BAA-AAA equivalent for UK/JP). Cross-sectional model is momentum-only (no fundamentals).

- [x] Pull UK + Japan stock panels from WRDS Compustat Global (`scripts/import_intl_stocks.py`). Total returns via `trfd`. Outputs: `data/{uk,jp}_stock_panel.parquet`, `data/{uk,jp}_market_panel.parquet`. UK 5,569 securities × 514K stock-months; JP 6,408 × 1.42M. Schema documented in `data/INTL_PANEL_README.md`.
- [x] Sanity-check BANK_REL signal: UK peaks at GFC March 2009 (3.72σ) + Eurozone 2011 + Brexit 2016. JP peaks at 2001-02 NPL crisis (4.6σ); correctly NEAR ZERO during 2008 GFC (Japanese banks didn't underperform — economically right).
- [x] Adapt `scripts/hmm_model.py` to regional inputs — done as standalone `scripts/hmm_intl.py` wrapper (self-contained, doesn't import hmm_model.py to avoid triggering its US fit at import time).
- [x] Smoke-fit regional HMMs (5 seeds × 500 iter). UK pi_filter correctly flags Black Wed 1992, dot-com 2001-02, GFC 2008-09, Eurozone 2011-08, COVID 2020-03. JP pi_filter correctly flags 1990 bubble, 1995 Kobe, 1997-98 LTCB, 2001-03 NPL nadir, GFC, COVID. JP BANK_REL_z sign came out as −1 (driven by 1990 bubble crash where banks outperformed); HMM still correctly identifies 2002 NPL via the joint multivariate likelihood.
- [ ] Production-fit regional HMMs (200 seeds × 2 regions). Wait for Step D pipeline / robustness checks to free CPU. Reuse `scripts/hmm_intl.py` (drop `--smoke`).
- [x] Adapt `scripts/cross_sectional_model.py` to regional inputs — done as `scripts/cross_sectional_intl.py` wrapper. No fundamentals (momentum + pi_filter only). Unconditional decile breakpoints. Optional `--use-us-pi` flag for Test A.
- [ ] Production-run regional cross-sectional model (50 XGB seeds × 4 variants: UK regional, UK Test A, JP regional, JP Test A). Smoke-test results so far: UK Method 0 Sharpe 0.61 (regional pi), Method 2 XGB 0.58, MDD -41% vs fixed-mom -60%. JP fixed momentum dead (Sharpe 0.08 — well-known no-momentum market) but XGB still extracts Sharpe 0.38.
- [ ] Test A (global financial cycle): apply the *US-trained* `pi_filter` to the regional cross-sectional model. UK Test A smoke running now; JP Test A smoke queued next. If this works it's the strongest publishable result.
- [ ] After Gilad reviews numbers: append "International validation" section to `RESULTS_LOG.md`. DO NOT edit `latex/*.tex` until then (memory: `feedback_thesis_edits_await_data`).

## Editing thesis
- [x] Add clear explanation of why long-short and not long-only
- [x] Consolidate generic validity checks (bootstrap + paired tests, subperiod stability, t-cost, pi threshold, multi-state HMM, HMM feature ablation, GFC OOS, expanding-window HMM, seed convergence, RF, CRRA) — landed in Appendix `app:robustness` rather than a main-text 5.4 subsection. Mechanism-tied tests (Ridge, depth sweep, LR with interactions) kept in 5.3. Added appendix subsection `app:threshold` for regime threshold sensitivity.

## New research
- [ ] Improve explanation of the graphs in Chapter 5.2 (term-structure figure and panic sub-type figure): frame what the reader should look for, clarify z-score vs SHAP question they each answer. Includes integrating monthly z-score heatmaps (month × horizon z-scores for long leg, short leg, L-S spread with pi_filter panel; figures `plots/zscore_long_short_heatmap.pdf` and `plots/zscore_longshort_heatmap.pdf`; data in `results/zscore_long_by_month.csv`, `results/zscore_short_by_month.csv`, `results/zscore_longshort_by_month.csv`) — single-month resolution counters the "averages hide inverting months" caveat seen in Sept 2019 / May 2024.

## External / reading
- [ ] Read the important literature cited in thesis (key papers to study deeply)

## Engineering / testing
- [x] Move fast, artefact-independent tests to pre-flight so they run before step 1 — fail-fast on config typos saves 2h on full rebuild. Done in `run_pipeline.py:113-133` (commit 97b35b6): runs test_config.py, test_utils.py, test_portfolio_edge_cases.py, test_pipeline_smoke.py when `--step` is not specified.
- [x] Add GitHub Actions workflow (`.github/workflows/tests.yml`) running pytest on every push. Done (commit 97b35b6). Runs test_config.py, test_portfolio_edge_cases.py, test_pipeline_smoke.py. test_utils.py excluded because its TestDataLoaders class depends on data files not in git (see open follow-up below).
- [x] **Parallelize HMM seeds across CPU cores.** Done in `scripts/expanding_window_backtest_parallel.py` using `multiprocessing.Pool(processes=6)`. Bit-identical to serial (verified by `tests/test_parallel_hmm_determinism.py`). 30-year expanding-window backtest at 200 HMM × 50 XGB seeds completed in ~19 hr (vs ~4 days serial). Reusable pattern for any future HMM-heavy sweep.
- [ ] Add `pytest.skip` guard to `tests/test_utils.py::TestDataLoaders` so the class is included in `.github/workflows/tests.yml`. Currently excluded because the loader tests depend on data parquets that aren't in git; the skip guard lets the artefact-independent tests in the file run on CI without forcing the loader tests to fail.

## Methodology validation (CV retrofits)
Sequenced after Step D pipeline rerun + Step E leg-betas finish (see "Post-Shumway rerun + leg-betas finding" above). The autonomous run will chain through Steps H-J before the final Step K report.

- [ ] Smoke-test `scripts/xgb_cv.py` (Step H part 1): `python scripts/xgb_cv.py --smoke` (~30s). Output: `results/xgb_cv_smoke.csv`.
- [ ] Smoke-test `scripts/hmm_cv.py` (Step H part 2): `python scripts/hmm_cv.py --smoke` (~5-10 min). Output: `results/hmm_cv_smoke.csv`.
- [ ] Full XGB CV (Step I): `python scripts/xgb_cv.py` (~2-5h). Output: `results/xgb_cv_results.csv` + `results/xgb_cv_winner.json` + `tables/table_xgb_cv.tex`. Thesis payoff: replaces the false "cross-validation" claim in `latex/appendix.tex` with real CV selection over depth × lr × n_estimators grid.
- [ ] Full HMM CV (Step J): `python scripts/hmm_cv.py` (~15-30h). Output: `results/hmm_cv_features.csv` + `results/hmm_cv_winner.json` + `tables/table_hmm_cv.tex`. Thesis payoff: removes the specification-search caveat paragraph in `latex/data_section.tex`. 93 DD-inclusive combinations × 5 folds × 3 HMM × 10 XGB seeds.
- [ ] After both CV runs finish AND Gilad reviews results: update `latex/methodology.tex` and `latex/appendix.tex` with CV-based framing; remove existing test-set-peek language. DO NOT edit any `.tex` before Gilad sees the numbers — the right framing depends on what the CV winners actually are.

## Architecture refactors (post-thesis)
Deferred until after submission. Each requires a bit-exact regression check (`md5 tables/*.tex plots/*.png results/*.csv` before/after, plus `pytest tests/test_reproducibility.py`) on a feature branch before merge. Not safe during the thesis sprint.

- [ ] **#1 Subdivide `scripts/`** into `hmm/`, `cs/`, `backtest/`, `analysis/`, `plots/`, `intl/`. Pure file moves; update subprocess paths in `run_pipeline.py` and any cross-script imports. ~1h incl. validation. Lowest-risk of the refactors; could be done immediately after pipeline finishes if desired.
- [ ] **#5 Split `scripts/main_results_analysis.py`** (1092 lines) into `performance.py`, `factor_alphas.py`, `ic.py`, `granger.py`, `shap.py`. Watch for float-summation/dict-iteration ordering changes that perturb table values.
- [ ] **#2 Move HMM/CS/portfolio logic into `src/`** as importable modules; reduce `scripts/*.py` to thin CLIs. Biggest leverage point for testability and reuse, but largest refactor surface. Half-day with careful regression.
- [ ] **#3 Replace subprocess orchestration with DVC (or Snakemake).** Gains: hash-keyed incremental rebuilds, free parallelism, audit trail. Cost: full pipeline rewrite + re-validation. Worth it for the next project, not this one.
- [ ] **#4 Split the 943 MB `cs_artefacts_data.pkl`** into parquets (`scores_{train,test}.parquet`, `shap_values.parquet`) + XGBoost native `.json` model files + `manifest.json` (git SHA, config hash, data hash, lib versions). Touches every consumer of the artefact; high coordination cost but yields version-stable artefacts and a reproducibility receipt.

## Completed
- [x] Work on limitations of the method: Limitations subsection (sec:limitations) now 8 paragraphs covering economic vulnerability, stress-test simulation caveat, test-period scope, frozen HMM parameters + regime drift, memoryless regime conditioning, implementation frictions, specification search, sample scope + causal inference (0e99d9a, 008e5fa, a697ce9).
- [x] Future work: expanded to six detailed directions (predictive regime signal, alternative nonlinear models, sequential regime conditioning, regime-conditional feature sets, exposure-scaling overlay, international replication). Memoryless-regime limitation trimmed to point at sequential-conditioning paragraph.
- [x] Create comprehensive pipeline tests (237 tests in `tests/test_pipeline_technical.py`)
- [x] Code audit and bug fixes (above_med, K=2 std, hardcoded month counts)
- [x] Restructure project to production layout (artefacts/, plots/, results/, src/)
- [x] Update SHAP to 50-seed ensemble (54/46, 50/50, 58/42)
- [x] Risk aversion graph (3-panel MV sweep, Markowitz reference)
- [x] Review and tighten introduction, literature review, and conclusion (73 -> 69 pages)
- [x] Add fundamentals row to performance table (Sharpe 0.97)
- [x] Add z-score computation explanation to results
- [x] Define z-score inline where first introduced
- [x] Trim redundant captions
- [x] Clean literature review (remove unused citations, generic HMM content)
- [x] Seed convergence check: Sharpe converges around k=50 (plateau 1.085-1.095 from k=50 to k=100), std shrinks as 1/sqrt(k) as expected. No runaway climb, production number is legitimate. See `results/seed_convergence.csv` and `plots/seed_convergence.png`.
- [x] Test Denis's CRRA risk aversion formula. Degrades faster than MV: even gamma=0 (no vol penalty, just sign(r)*log(1+|r|/eps) target) gives Sharpe 0.29 vs production 1.11. The log-compression of returns removes the magnitude info XGBoost needs for ranking. gamma>=0.5 is negative. Pi_filter SHAP share collapses (54% -> 7%) as momentum SHAP explodes from vol-penalty variance dominance. See `results/risk_aversion_crra_results.csv`, `plots/risk_aversion_crra.png`.
- [x] Bootstrap analysis: block bootstrap CIs, paired tests vs benchmarks, regime-conditional Sharpe (full table in Appendix app:bootstrap, brief mention in Section 5.1)
- [x] **30-year expanding-window OOS backtest** (1995-2024, annual retraining, 200 HMM × 50 XGB seeds, parallel). Full OOS Sharpe 0.50, cumulative +1,405%, MDD -64.8% during dot-com. Reproduces production Sharpe 1.11 within sampling noise on 167-month overlap (0.88). Pipeline integration: step 19 in `run_pipeline.py` with skip-if-results-exist guard; table generated by step 20 (`scripts/build_table_expanding_subperiods.py`). Outputs: `results/expanding_*_prod.csv`, `tables/table_expanding_subperiods.tex`. Five exploratory scripts (B, C, D&M-OOS, panic-subtypes, pi-filter-diagnostic) archived to `archive/scripts/` after analysis dropped from thesis scope.
- [x] **Section 5.4.3 "Historical Stress: Out-of-Sample Evidence"** drafted and committed: new subsection in `latex/main_results.tex` between sec:stress_test and sec:dm_comparison, with `table_expanding_subperiods.tex`, appendix subsection `app:expanding_window`, cross-reference in 5.1, forward-reference in 5.4.2, and rewritten `Stress-tested vulnerabilities` paragraph in `latex/conclusion.tex` (using observed dot-com numbers: -64.8% MDD, -43.8% cumulative). Commits: `6ca51c1`, `997c8cc`, `7d85d04`, `bd2677f`.
- [x] Write Denis an email on the XGBoost pattern (sent 2026-04-23: meeting cancel, checklist, CRRA update, heatmap pivot)
- [x] Integrate CRRA (Denis's formula) into the thesis: adds robustness to Section 5.4 risk aversion — even at gamma=0, the sign-log compression of returns drops Sharpe to 0.29. Appendix extension of risk aversion discussion. Data in `results/risk_aversion_crra_results.csv`.
- [x] Improve explanation of how dominant pi_filter is in the trees (rewritten with economic framing in 5a9bb34; numbers verified against full 50-seed ensemble in 9483713 and refined in 2cc77f9 to full-population panic share: 73.9% of trees contain a pi split, 55% of stock paths, 65% of panic long-leg return from pi-splitting trees vs 17% in calm). Reproducible via `scripts/verify_pi_dominance_stats.py`.
- [x] Integrate Random Forest result into the thesis: RF row added to `tab:xgb_hyperparams` (Sharpe 0.73 vs XGBoost 1.11; pi_filter SHAP 18% vs 45%); Model Sensitivity appendix separates boosting-vs-bagging from hyperparameter sensitivity (7e1e455).
- [x] Integrate seed convergence result into the thesis: appendix subsection `app:seed_convergence` with `table_seed_convergence.tex` (7e1e455 + bfbd80f). Data in `results/seed_convergence.csv`.
- [x] Fundamental analysis — ablation test complete. Tables: `table_fund_alphas.tex`, `table_fundamentals_ablation.tex`, `table_performance_fund_row.tex`. Data in `results/fundamentals_test_results.csv`. (Cash flow WRDS pull was separate and not required for this ablation.)
- [x] **Lock full transitive Python env (`requirements.lock`).** 73-package snapshot from `.venv` via `pip freeze`, complementing the human-curated `requirements.txt` direct-deps list. Reproduce the thesis env with `pip install -r requirements.lock`. Commit `d8afe9f`.
- [x] **`experiments/` convention for one-off scripts.** Dated `YYYY-MM-DD-*.py` files are throwaway (no tests, no maintenance); promotion to `scripts/` is deliberate. Curbs the `archive/scripts/` churn. Commit `309dc48`.
- [x] **Build `main.tex` on every push (`.github/workflows/latex.yml`).** Catches broken `\input{tables/...}` and `\includegraphics{plots/...}` references at push time; uploads the compiled PDF as a workflow artifact. Commit `8296da0`.
