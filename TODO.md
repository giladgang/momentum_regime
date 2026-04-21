# Thesis TODO

## Pending
- [ ] Work on fundamental analysis (results in `results/fundamentals_test_results.csv`, script: `scripts/fundamentals_test.py`)
- [ ] Future work: add alternative nonlinear method (not just XGBoost)
- [ ] Work on limitations of the method
- [ ] Add clear explanation of why long-short and not long-only
- [ ] Ablation test on fundamentals: check specifically for cash flow. Blocked on WRDS auth (interactive login fails). Pull script is ready at `scripts/pull_cashflow.py` — pulls oancfy/capxy/ibq/atq from comp.fundq, computes cfo_a, fcf_a, accruals (Sloan). Once auth fixed: run pull, then write companion ablation script (baseline + add-one for each of 10 fundamentals + LOO from full + fund-only).
- [ ] Write Denis an email on the XGBoost pattern
- [ ] Read the important literature cited in thesis (key papers to study deeply)
- [ ] Improve explanation of the graphs in Chapter 5.2 (term-structure figure and panic sub-type figure): frame what the reader should look for, clarify z-score vs SHAP question they each answer
- [ ] Improve explanation of how dominant pi_filter is in the trees (75.8% of trees contain a pi split, 54% of stock paths, 67% of panic long-leg return comes from pi-splitting trees vs only 20% in calm)
- [ ] Integrate Random Forest result into the thesis: RF depth-matched gives Sharpe 0.73 vs XGBoost 1.11, with pi_filter SHAP share dropping from 46% to 18%. Shows the finding is not about "any tree ensemble" but specifically requires sequential boosting. Best placement: add to Section 5.3 (Nonlinearity) as further robustness, or mention briefly in future work section. Data in `results/random_forest_results.csv`.
- [ ] Integrate seed convergence result into the thesis: Sharpe drifts up from 1.05 (k=1) to 1.10 (k=100), IQR collapses. Production 50-seed captures most of the noise reduction. Worth mentioning as a methodology caveat (either footnote in Section 3 or appendix). Data in `results/seed_convergence.csv`.
- [ ] Integrate CRRA (Denis's formula) into the thesis: adds robustness to Section 5.4 risk aversion -- even at gamma=0, the sign-log compression of returns drops Sharpe to 0.29. Confirms that any magnitude-compressing target breaks the mechanism, not just MV. Best placement: appendix extension of the risk aversion discussion. Data in `results/risk_aversion_crra_results.csv`.
- [ ] Integrate monthly z-score heatmaps into the thesis: month x horizon z-scores for long leg, short leg, and L-S spread, with pi_filter panel alongside. Shows the regime-conditional selection pattern at single-month resolution rather than regime averages (useful counter to the "averages hide months that invert the pattern" caveat seen in Sept 2019 / May 2024). Figures `plots/zscore_long_short_heatmap.pdf` and `plots/zscore_longshort_heatmap.pdf`; data in `results/zscore_long_by_month.csv`, `results/zscore_short_by_month.csv`, `results/zscore_longshort_by_month.csv`.
- [ ] Add a Section 5.1 "Robustness / Method Validity" subsection: consolidate generic validity checks (bootstrap CIs + paired tests, subperiod stability, transaction cost sensitivity, pi threshold sensitivity, multi-state HMM, HMM feature ablation, GFC OOS, expanding window HMM, seed convergence, CRRA, Random Forest) in one place. Keep mechanism-tied tests (Ridge, depth sweep, LR with interactions) in 5.3 where they earn their narrative role. Gives the RF / seed-convergence / CRRA integration tasks a natural home and lets the reader see the robustness story at a glance.

## Completed
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
