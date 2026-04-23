# Thesis TODO

## In progress
- [ ] Work on limitations of the method
- [ ] Polish the stress-test section

## Editing thesis
- [x] Add clear explanation of why long-short and not long-only
- [x] Add a Section 5.4 "Robustness / Method Validity" subsection consolidating generic validity checks (bootstrap + paired tests, subperiod stability, t-cost, pi threshold, multi-state HMM, HMM feature ablation, GFC OOS, expanding-window HMM, seed convergence, RF, CRRA). Mechanism-tied tests (Ridge, depth sweep, LR with interactions) kept in 5.3. Placed as new 5.4 between Nonlinearity (stays 5.3) and Risk Aversion (pushed to 5.5). Added appendix subsection `app:threshold` for regime threshold sensitivity.

## New research
- [ ] Improve explanation of the graphs in Chapter 5.2 (term-structure figure and panic sub-type figure): frame what the reader should look for, clarify z-score vs SHAP question they each answer. Includes integrating monthly z-score heatmaps (month × horizon z-scores for long leg, short leg, L-S spread with pi_filter panel; figures `plots/zscore_long_short_heatmap.pdf` and `plots/zscore_longshort_heatmap.pdf`; data in `results/zscore_long_by_month.csv`, `results/zscore_short_by_month.csv`, `results/zscore_longshort_by_month.csv`) — single-month resolution counters the "averages hide inverting months" caveat seen in Sept 2019 / May 2024.
- [ ] Fundamental analysis — ablation test including cash flow. Baseline in `results/fundamentals_test_results.csv` (script: `scripts/fundamentals_test.py`). Cash flow piece **blocked on expired WRDS password** — `~/.pgpass` is correctly formatted but PAM auth fails (verified 2026-04-23). Fix: log in to wrds-www.wharton.upenn.edu, reset password, update `.pgpass` line `wrds-pgdata.wharton.upenn.edu:9737:wrds:giladgang:NEW_PW`. Pull script `scripts/pull_cashflow.py` is ready (reads creds from .pgpass) — pulls oancfy/capxy/ibq/atq from comp.fundq, computes cfo_a, fcf_a, accruals (Sloan). Once auth works: run pull (~5 min), then write companion ablation script (baseline + add-one for each of 10 fundamentals + LOO from full + fund-only).

## External / reading
- [ ] Read the important literature cited in thesis (key papers to study deeply)

## Completed
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
- [x] Write Denis an email on the XGBoost pattern (sent 2026-04-23: meeting cancel, checklist, CRRA update, heatmap pivot)
- [x] Integrate CRRA (Denis's formula) into the thesis: adds robustness to Section 5.4 risk aversion — even at gamma=0, the sign-log compression of returns drops Sharpe to 0.29. Appendix extension of risk aversion discussion. Data in `results/risk_aversion_crra_results.csv`.
- [x] Improve explanation of how dominant pi_filter is in the trees (rewritten with economic framing in 5a9bb34; numbers verified against full 50-seed ensemble in 9483713: 73.9% of trees contain a pi split, 55% of stock paths, 67% of panic long-leg return from pi-splitting trees vs 17% in calm). Reproducible via `scripts/verify_pi_dominance_stats.py`.
- [x] Integrate Random Forest result into the thesis: RF row added to `tab:xgb_hyperparams` (Sharpe 0.73 vs XGBoost 1.11; pi_filter SHAP 18% vs 45%); Model Sensitivity appendix separates boosting-vs-bagging from hyperparameter sensitivity (7e1e455).
- [x] Integrate seed convergence result into the thesis: appendix subsection `app:seed_convergence` with `table_seed_convergence.tex` (7e1e455 + bfbd80f). Data in `results/seed_convergence.csv`.
