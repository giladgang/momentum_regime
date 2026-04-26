# Thesis TODO

## In progress
- [ ] Draft Section 5.4.3 "Historical Stress: Out-of-Sample Evidence" — write the new subsection in `latex/main_results.tex` between sec:stress_test and sec:dm_comparison; insert `tables/table_expanding_subperiods.tex`; add appendix subsection app:expanding_window; add cross-reference sentence in 5.1; rewrite the `Stress-test evidence is simulation-based` paragraph in `latex/conclusion.tex` to use the new historical numbers (-64.8% MDD, -43.8% cumulative dot-com).
- [ ] Discussion of risk premium

## International validation (UK + Japan)
**Runbook:** `INTL_VALIDATION_PLAN.md` (self-contained, resumable across sessions). Goal: test whether the regime+momentum architecture transfers to UK and Japan. 4-feature regional HMM uses `[DD_z, DISP_z, REL_N_z, BANK_REL_z]` where `BANK_REL_z` substitutes for `CS_z` (no Moody's BAA-AAA equivalent for UK/JP). Cross-sectional model is momentum-only (no fundamentals).

- [x] Pull UK + Japan stock panels from WRDS Compustat Global (`scripts/import_intl_stocks.py`). Total returns via `trfd`. Outputs: `data/{uk,jp}_stock_panel.parquet`, `data/{uk,jp}_market_panel.parquet`. UK 5,569 securities × 514K stock-months; JP 6,408 × 1.42M. Schema documented in `data/INTL_PANEL_README.md`.
- [x] Sanity-check BANK_REL signal: UK peaks at GFC March 2009 (3.72σ) + Eurozone 2011 + Brexit 2016. JP peaks at 2001-02 NPL crisis (4.6σ); correctly NEAR ZERO during 2008 GFC (Japanese banks didn't underperform — economically right).
- [ ] Adapt `scripts/hmm_model.py` to regional inputs (write `scripts/hmm_intl.py` wrapper). Validation: `pi_filter` bounded in [0,1]; panic state spikes at known crisis months.
- [ ] Smoke-fit regional HMMs (`--smoke` ≈ 5 seeds, ~2 min each). Validate before committing to production-scale.
- [ ] Production-fit regional HMMs (200 seeds × 2 regions). CPU budget: limit to 2 workers if `expanding_window_backtest_parallel.py` is still running.
- [ ] Adapt `scripts/cross_sectional_model.py` to regional inputs without fundamentals (write `scripts/cross_sectional_intl.py`).
- [ ] Run regional cross-sectional model: report UK/JP Sharpes, factor alphas, subperiod stability vs benchmarks (regional market, fixed 12-mo mom, fixed 1-mo mom).
- [ ] Test A (global financial cycle): apply the *US-trained* `pi_filter` (from `data/panel_with_regimes.parquet`) to the regional cross-sectional model. If this works it's the strongest publishable result.
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
- [x] Add GitHub Actions workflow (`.github/workflows/tests.yml`) running pytest on every push. Done (commit 97b35b6). Runs test_config.py, test_portfolio_edge_cases.py, test_pipeline_smoke.py. test_utils.py excluded because its TestDataLoaders class depends on data files not in git; a pytest.skip guard on that class would let it be included (minor follow-up).
- [x] **Parallelize HMM seeds across CPU cores.** Done in `scripts/expanding_window_backtest_parallel.py` using `multiprocessing.Pool(processes=6)`. Bit-identical to serial (verified by `tests/test_parallel_hmm_determinism.py`). 30-year expanding-window backtest at 200 HMM × 50 XGB seeds completed in ~19 hr (vs ~4 days serial). Reusable pattern for any future HMM-heavy sweep.

## Methodology validation (CV retrofits)
- [ ] Smoke-test `scripts/xgb_cv.py` after expanding-window backtest finishes: `python scripts/xgb_cv.py --smoke` (~30s). Then kick off full run (~2-5h). Output: `results/xgb_cv_results.csv`. Thesis payoff: replaces false "cross-validation" claim in `latex/appendix.tex` with real CV selection over depth × lr × n_estimators grid.
- [ ] Smoke-test `scripts/hmm_cv.py` after backtest finishes: `python scripts/hmm_cv.py --smoke` (~5-10 min). Then kick off full run (~15-30h). Output: `results/hmm_cv_features.csv`. Thesis payoff: removes specification-search caveat paragraph in `latex/data_section.tex`. 93 DD-inclusive combinations × 5 folds × 3 HMM × 10 XGB seeds.
- [ ] After both CV runs finish AND Gilad reviews results: update `latex/methodology.tex` and `latex/appendix.tex` with CV-based framing; remove existing test-set-peek language. DO NOT edit any `.tex` before Gilad sees the numbers — the right framing depends on what the CV winners actually are.

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
- [x] Write Denis an email on the XGBoost pattern (sent 2026-04-23: meeting cancel, checklist, CRRA update, heatmap pivot)
- [x] Integrate CRRA (Denis's formula) into the thesis: adds robustness to Section 5.4 risk aversion — even at gamma=0, the sign-log compression of returns drops Sharpe to 0.29. Appendix extension of risk aversion discussion. Data in `results/risk_aversion_crra_results.csv`.
- [x] Improve explanation of how dominant pi_filter is in the trees (rewritten with economic framing in 5a9bb34; numbers verified against full 50-seed ensemble in 9483713 and refined in 2cc77f9 to full-population panic share: 73.9% of trees contain a pi split, 55% of stock paths, 65% of panic long-leg return from pi-splitting trees vs 17% in calm). Reproducible via `scripts/verify_pi_dominance_stats.py`.
- [x] Integrate Random Forest result into the thesis: RF row added to `tab:xgb_hyperparams` (Sharpe 0.73 vs XGBoost 1.11; pi_filter SHAP 18% vs 45%); Model Sensitivity appendix separates boosting-vs-bagging from hyperparameter sensitivity (7e1e455).
- [x] Integrate seed convergence result into the thesis: appendix subsection `app:seed_convergence` with `table_seed_convergence.tex` (7e1e455 + bfbd80f). Data in `results/seed_convergence.csv`.
- [x] Fundamental analysis — ablation test complete. Tables: `table_fund_alphas.tex`, `table_fundamentals_ablation.tex`, `table_performance_fund_row.tex`. Data in `results/fundamentals_test_results.csv`. (Cash flow WRDS pull was separate and not required for this ablation.)
