# Thesis Checklist: Momentum Regime Shifts (Long-Short)

## Phase 1: Finalize HMM Features
- [ ] Wait for 1-3 feature search to complete
- [ ] Compare best 1, 2, 3 feature combos vs 4-feature (DD+DISP+REL_N+CS)
- [ ] Pick final features and update config.py

## Phase 2: Rerun Full Pipeline
- [ ] Run hmm_model.py (generate pi_filter, save to panel)
- [ ] Run cross_sectional_model.py (LR + XGB ensemble, build L/S portfolios)
- [ ] Get final L/S results for all strategies (M0, M1, M2, fixed mom, GHM, D&M)

## Phase 3: Analyses (once numbers are final)
- [ ] L/S performance table (Ann.Ret, Vol, Sharpe, Beta, Alpha, NW t-stat, bootstrap CI)
- [ ] Regime-conditional Sharpe table (calm vs panic for all strategies)
- [ ] IC rotation table: IC by horizon (mom_1-12) in calm vs panic with t-stats
- [ ] IC rotation by sub-period (2011-15, 2016-20, 2021-25) stability check
- [ ] SHAP overall importance table
- [ ] SHAP by long leg vs short leg (signed and absolute)
- [ ] SHAP by regime x leg (calm/panic x long/short)
- [ ] PDP of pi_filter (non-monotonic staircase)
- [ ] Ablation: M2 full features vs no fundamentals vs mom only
- [ ] Ablation: M1 same variants (to show nothing helps LR)
- [ ] Spanning test: regress M2 on M1 (and reverse)
- [ ] D&M bear indicator comparison (replace pi_filter, add alongside)
- [ ] HMM feature ablation (each feature alone, leave-one-out)
- [ ] Risk aversion analysis (gamma 0-10, SHAP importance by gamma)
- [ ] Cumulative wealth plot (long-short)
- [ ] Sub-period Sharpe table (three 5-year windows)
- [ ] Factor model alphas (CAPM through FF5+Mom)
- [ ] M1 polynomial/interaction tests in L/S
- [ ] Portfolio characteristics by regime (size, beta, B/M, profitability, leverage of long/short legs in calm vs panic)
- [ ] Forward return decomposition (stock recovery vs market recovery in panic reversal bets)
- [ ] Holding period analysis (hold panic portfolio 1, 3, 6 months -- genuine reversal or noise?)
- [ ] Sector rotation analysis (which sectors does the strategy buy/short in calm vs panic?)

## Phase 4: Robustness & Stress Test
- [ ] Robustness: sub-period, turnover, cost sensitivity, XGB hyperparams
- [ ] Robustness: threshold sensitivity, K=2,3,4,5 HMM, skip-month
- [ ] Robustness: alt splits, placebo, kitchen sink, expanding window
- [ ] HMM diagnostics: Student-t emissions, Gelman-Rubin, separation table
- [ ] Stress test: vulnerability diagnosis (Parts A1-A7)
- [ ] Stress test: formal tests (VaR, CVaR, drawdown conditioning, scenarios)
- [ ] Stress test: prolonged bear simulation (B7) and recovery analysis (B8)

## Phase 5: Write Thesis
- [ ] Section 5.1: Portfolio Performance
- [ ] Section 5.2: Regime-Conditional Performance
- [ ] Section 5.3: The Momentum-Regime Interaction (IC Analysis)
- [ ] Section 5.4: SHAP Decomposition
- [ ] Section 5.5: Ablation
- [ ] Section 5.6: Risk Aversion Analysis
- [ ] Section 5.7: Economic Mechanism (why momentum behaves differently in calm vs panic)
- [ ] Update data_section.tex (feature selection methodology)
- [ ] Update methodology.tex (L/S construction, new HMM features, new citations: Gu et al., Chen et al., Sander & Lubenau, Bagnara)
- [ ] Update literature review (Wiedemann & Beckmeyer, Giner & Zakamulin, Dierkes & Krupski, Butt et al., Cakici et al., Cooper et al.)
- [ ] Add new papers to references.bib
- [ ] Update introduction (L/S framing)
- [ ] Update conclusion (L/S findings, D&M comparison)
- [ ] Update abstract
- [ ] Update appendix (portfolio definition, robustness prose, all numbers)
- [ ] Write future work: credit-spread circuit breaker for prolonged bears

## Phase 6: Final Checks
- [ ] Coherence review across all sections
- [ ] Check all numbers in text match tables
- [ ] Search for stale references (long-only, VOL_z, old feature names)
- [ ] Push to thesis_git
- [ ] Send updated draft to Denis
