# Thesis TODO

## Pending
- [ ] Work on fundamental analysis (results in `results/fundamentals_test_results.csv`, script: `scripts/fundamentals_test.py`)
- [ ] Future work: add alternative nonlinear method (not just XGBoost)
- [ ] Work on limitations of the method
- [ ] Add clear explanation of why long-short and not long-only
- [ ] Ablation test on fundamentals: check specifically for cash flow
- [ ] Write Denis an email on the XGBoost pattern
- [ ] Test Denis's risk aversion formula: constant-elasticity utility `sign(x) * log(1 + |x|/eps) - gamma * log(sigma)` (CRRA-style: fixed proportional trade-off between return and vol, vs CARA/MV which has fixed absolute trade-off for variance)
- [ ] Bootstrap analysis: block bootstrap CIs for M2 Sharpe, paired bootstrap M2 vs benchmarks, regime-conditional Sharpe significance (code exists in `main_results_analysis.py` but not prominently reported)
- [ ] Read the important literature cited in thesis (key papers to study deeply)

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
