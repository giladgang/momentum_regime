# Thesis TODO

## Pending
- [ ] Work on fundamental analysis (results in `results/fundamentals_test_results.csv`, script: `scripts/fundamentals_test.py`)
- [ ] Future work: add alternative nonlinear method (not just XGBoost)
- [ ] Work on limitations of the method
- [ ] Add clear explanation of why long-short and not long-only
- [ ] Ablation test on fundamentals: check specifically for cash flow. Blocked on WRDS auth (interactive login fails). Pull script is ready at `scripts/pull_cashflow.py` — pulls oancfy/capxy/ibq/atq from comp.fundq, computes cfo_a, fcf_a, accruals (Sloan). Once auth fixed: run pull, then write companion ablation script (baseline + add-one for each of 10 fundamentals + LOO from full + fund-only).
- [ ] Write Denis an email on the XGBoost pattern
- [ ] Test Denis's risk aversion formula: constant-elasticity utility `sign(x) * log(1 + |x|/eps) - gamma * log(sigma)` (CRRA-style: fixed proportional trade-off between return and vol, vs CARA/MV which has fixed absolute trade-off for variance)
- [ ] Read the important literature cited in thesis (key papers to study deeply)
- [ ] Improve explanation of the graphs in Chapter 5.2 (term-structure figure and panic sub-type figure): frame what the reader should look for, clarify z-score vs SHAP question they each answer
- [ ] Improve explanation of how dominant pi_filter is in the trees (75.8% of trees contain a pi split, 54% of stock paths, 67% of panic long-leg return comes from pi-splitting trees vs only 20% in calm)
- [ ] Check if increasing the number of XGB seeds systematically increases the Sharpe ratio. If it does, this is a problem: Sharpe should converge as seed count grows (averaging reduces noise around a stable mean), not keep rising. A monotonic upward trend would suggest the ensemble is exploiting seed diversification rather than a real signal.

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
- [x] Bootstrap analysis: block bootstrap CIs, paired tests vs benchmarks, regime-conditional Sharpe (full table in Appendix app:bootstrap, brief mention in Section 5.1)
