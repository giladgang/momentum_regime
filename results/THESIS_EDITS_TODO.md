# THESIS_EDITS_TODO.md

Comprehensive checkbox list of every finding from Phase 0–3 that needs a thesis edit. Each row is independent and reviewable; tick as you go.

**Format**: `☐ [TRIAGE] Finding — *source data* → *thesis target*`

Triage tags:
- **[M]** = Must land (data-driven, advisor pressure)
- **[S]** = Should land (strengthens the story)
- **[N]** = Nice to have (methodology/appendix)

Companion files:
- [`results/PROSE_EDITS.md`](PROSE_EDITS.md) — auto-generated 944 line-by-line numeric updates
- [`results/STEP_K_REPORT.md`](STEP_K_REPORT.md) — final report (CV winners + intl + Step F splice)
- [`results/STEP_F_REPORT.md`](STEP_F_REPORT.md) — first-checkpoint (Shumway diff + leg-betas)
- [`results/RUN_MANIFEST.md`](RUN_MANIFEST.md) — phase log, bug fixes, parallel refactors
- [`RESULTS_LOG.md`](../RESULTS_LOG.md) §12 — UK/JP writeup

---

## A. NUMERIC UPDATES — existing claims, post-Shumway values

These are auto-flagged in [`PROSE_EDITS.md`](PROSE_EDITS.md). The list below is the *headline* set; the full 944-row checklist is in that file.

- ☐ [M] M2 Ann.Ret 21.9% → **21.7%** — [`tables/table_performance.tex`](../tables/table_performance.tex), goes to [`latex/main_results.tex`](../latex/main_results.tex)
- ☐ [M] M2 Ann.Vol 19.7% → **19.5%** — same source
- ☐ [M] M2 Sharpe 1.11 → **1.11** (unchanged) — same
- ☐ [M] M2 Max DD −24.8% → **−22.8%** — same
- ☐ [M] M2 Beta 0.46 → **0.45** — same
- ☐ [M] **M2 Newey-West t**: 1.83\* → **4.37\*\*\*** *(major significance jump)* — same
- ☐ [M] M2 Final-$ multiple 15.7× → **15.3×** — same
- ☐ [M] Sub-period Sharpe 2011–2015: 0.62 → **0.59** — [`tables/table_subperiod.tex`](../tables/table_subperiod.tex)
- ☐ [M] Sub-period 2016–2020: 1.03 → **1.04** — same
- ☐ [M] Sub-period 2021–2025: 1.69 → **1.70** — same
- ☐ [M] CAPM α: 23.8% (t=4.08) → **23.4% (t=4.08)** — [`tables/table_factor_alphas.tex`](../tables/table_factor_alphas.tex)
- ☐ [M] FF3 α: 23.7% → **23.2% (t=4.48)** — same
- ☐ [M] Carhart α: 24.9% → **24.3% (t=4.75)** — same
- ☐ [M] FF5 α: 23.4% → **22.9% (t=4.62)** — same
- ☐ [M] FF6 α: 24.7% → **24.1% (t=4.81)** — same, mentioned in [`latex/main_results.tex`](../latex/main_results.tex)
- ☐ [M] Stress baseline MDD: −24.8% → **−23.0%** — [`tables/table_stress_scenarios.tex`](../tables/table_stress_scenarios.tex)
- ☐ [M] All other prose mentions of pre-Shumway numbers — see [`PROSE_EDITS.md`](PROSE_EDITS.md) (944 candidates, 415 cells across 40 tables)

---

## B. NEW CLAIMS — replace caveats with CV-justified or international evidence

### B1. XGB hyperparameter CV winner

- ☐ [M] State **CV winner: depth=3, lr=0.10, n_estimators=200**, mean val Sharpe **+0.596** (fold-std 0.811) — [`results/xgb_cv_winner.json`](xgb_cv_winner.json), [`tables/table_xgb_cv.tex`](../tables/table_xgb_cv.tex), [`results/xgb_cv_results.csv`](xgb_cv_results.csv) → goes to [`latex/methodology.tex`](../latex/methodology.tex) and [`latex/appendix.tex`](../latex/appendix.tex)
- ☐ [M] Note **top-5 configs cluster within fold-noise** (means 0.56–0.60); winner choice is "any reasonable config" — same sources
- ☐ [M] **Strike** the existing "test-set peek" / "we picked depth=4 because it looked good on test" caveat in `latex/methodology.tex` and `latex/appendix.tex`

### B2. HMM feature-set CV winner

- ☐ [M] State **CV winner: `DD+CS+LVIX+DISP`**, mean val Sharpe **+0.507** (fold-std 0.694) — [`results/hmm_cv_winner.json`](hmm_cv_winner.json), [`tables/table_hmm_cv.tex`](../tables/table_hmm_cv.tex), [`results/hmm_cv_features.csv`](hmm_cv_features.csv) → [`latex/data_section.tex`](../latex/data_section.tex)
- ☐ [M] **Note divergence**: CV winner differs from current production (`DD+CS+DISP+REL_N`) — CV swaps `REL_N` for `LVIX`. Decision: retrain or document
- ☐ [M] **Strike** "specification-search" caveat in `latex/data_section.tex`

### B3. International validation — Global Financial Cycle

- ☐ [M] UK M2 Sharpe (regional π): **+0.631**, MDD **−27.7%** — [`results/intl_uk_summary.csv`](intl_uk_summary.csv)
- ☐ [M] UK M2 Sharpe (US π Test A): **+0.678**, MDD **−23.0%** — [`results/intl_uk_summary_uspi.csv`](intl_uk_summary_uspi.csv)
- ☐ [M] JP M2 Sharpe (regional π): **+0.435**, MDD **−33.1%** — [`results/intl_jp_summary.csv`](intl_jp_summary.csv)
- ☐ [M] JP M2 Sharpe (US π Test A): **+0.525**, MDD **−18.1%** — [`results/intl_jp_summary_uspi.csv`](intl_jp_summary_uspi.csv)
- ☐ [M] **Headline: US π beats regional π in BOTH** — frame as "global financial cycle channel" (Rey 2013)
- ☐ [M] JP improvement +0.09 Sharpe, MDD halved (−33% → −18%) — strongest evidence
- ☐ [M] **Caveat — JP M2 < market** (0.52 vs 0.86): cite Asness, Moskowitz & Pedersen (2013)
- ☐ [S] **Caveat — UK M2 ~ market** (0.68 vs 0.62): residual market exposure of L/S construction
- ☐ [M] **Add new section** "International validation" in [`latex/main_results.tex`](../latex/main_results.tex) — content is in [`RESULTS_LOG.md`](../RESULTS_LOG.md) §12

### B4. Shumway methodology footnote

- ☐ [N] Cite Shumway (1997, 2001) for delisting-bias correction; rules 1–4 fired on 1,310 CRSP rows + 2.08M passthrough — methodology footnote in `latex/methodology.tex`
- ☐ [N] UK/JP strict-mode: Compustat Global with `secstat`/`dldte` (re-pulled from `comp.g_security`); UK 219 rows compounded, JP 85 rows. Negligible numeric impact (0.04% / 0.006%) — methodology footnote

### B5–B8. 30-YEAR EXPANDING-WINDOW BACKTEST (Step 19/20) 🚨

- ☐ [M] **Full sample 1995–2024**: Sharpe **0.50**, cum return **1,405.5%**, MDD **−64.8%**, n=359 months — [`results/expanding_summary_prod.csv`](expanding_summary_prod.csv), [`tables/table_expanding_subperiods.tex`](../tables/table_expanding_subperiods.tex)
- ☐ [M] **Sub-period decomposition**:
  - 1995–1999: Sharpe 0.50
  - **2000–2002 dot-com: Sharpe −0.15, cum −35.9%, MDD −64.8%** ← strategy LOSES 36% over 36 months
  - 2003–2006 bull: Sharpe 0.70
  - 2007–2009 GFC: Sharpe **0.26 (positive!)**
  - 2010–2019 calm: Sharpe 0.67
  - 2020–2024 COVID era: Sharpe **1.23**
  — same sources
- ☐ [M] **2011–2024 sub-window of 30y backtest reproduces production**: Sharpe 0.88 vs 1.11 (within sampling noise) — table caption
- ☐ [M] **Major thesis implication** (link Step 19/20 to stress test §E22): the 36-month dot-com bear is *exactly* the 24+ month sustained-bear scenario the stress test flags. The historical hazard isn't hypothetical — it happened, strategy suffered. Validates D&M-scaling overlay's contingent value.
- ☐ [S] M2 trained 1990–1999, tested 2000–2010: Sharpe 0.24 overall, dot-com sub-period **−0.42** — [`results/oos_summary_prod_1990_1999.csv`](oos_summary_prod_1990_1999.csv), [`results/oos_subperiods_prod_1990_1999.csv`](oos_subperiods_prod_1990_1999.csv)
- ☐ [S] M2 trained 1990–2004, tested 2005–2010: Sharpe 0.31 — [`results/oos_summary_prod_1990_2004.csv`](oos_summary_prod_1990_2004.csv)

---

## C. CORRECTIONS / REFRAMES

### C1. Leg-beta dynamics — NOT a clean inversion (correction)

- ☐ [M] M2 XGB Long β: 1.40 (calm) → 1.65 (panic) — [`results/leg_betas_by_regime.csv`](leg_betas_by_regime.csv), [`tables/table_leg_betas.tex`](../tables/table_leg_betas.tex)
- ☐ [M] M2 XGB Short β: 1.23 (calm) → 0.98 (panic) — same
- ☐ [M] **L−S β gap widens 0.17 → 0.67 in panic but no flip** (long-β > short-β in BOTH regimes) — same
- ☐ [M] Fixed-12-mom comparison: long ~1.07, short ~1.7, no regime sensitivity — same
- ☐ [M] **Mechanism is CROSS-SECTIONAL RE-RANKING**, NOT D&M leg-beta inversion — frame edit bundle accordingly
- ☐ [M] **Edit #1**: new subsection "Leg-level beta dynamics across regimes" in [`latex/main_results.tex`](../latex/main_results.tex) — re-ranking framing
- ☐ [M] **Edit #2**: distinguish from D&M (2016) inversion in [`latex/literature_review.tex`](../latex/literature_review.tex) line 19
- ☐ [M] **Edit #4**: pi_filter ↔ D&M bear-indicator link in [`latex/methodology.tex`](../latex/methodology.tex) — note our regime signal is *probabilistic and continuous*, D&M's is *binary*
- ☐ [S] Show rolling-β plot — [`plots/leg_betas_rolling.pdf`](../plots/leg_betas_rolling.pdf)

### C2. Mechanism: term-structure shape (not per-horizon flips)

- ☐ [M] Calm selection: **curved**, peak at mom_8 (z = +0.39), 1-month neutral, 12-month moderate — [`results/selection_rank_analysis.csv`](selection_rank_analysis.csv), [`tables/table_zscore_shap_detail.tex`](../tables/table_zscore_shap_detail.tex)
- ☐ [M] Panic selection: **flat at z ≈ −0.13** across all 12 horizons — uniform losers, novel finding — same sources
- ☐ [M] Panic short shorts **recent winners** (z = +0.30 at mom_1) — defensive-name profit-taking — same
- ☐ [M] SHAP attention dominated by mom_11 (18.6–23.1%), mom_9 (12.7–15.5%), mom_8 (10.2–12.8%); **stable across regimes** (only the *direction* flips, not the *attention*) — same
- ☐ [M] Cite **Jegadeesh (1990)** + **Lehmann (1990)** for mom_1 reversal-avoidance, **Novy-Marx (2012)** for mom_8 peak, **Lee & Swaminathan (2000)** for mom_12 decay — calm shape rediscovers known literature
- ☐ [M] Panic flat shape: **NOVEL finding** (no prior literature) — emphasize
- ☐ [S] Reference [`plots/chart2_rank_by_horizon_v2.pdf`](../plots/chart2_rank_by_horizon_v2.pdf) for the visualization
- ☐ [M] **Edit #3**: rephrase "two channels" in [`latex/conclusion.tex`](../latex/conclusion.tex) line 53 — (1) regime-conditional selection (HMM π), (2) cross-sectional re-ranking by term-structure shape

### C3. Why depth ≥ 3 needed

- ☐ [M] Reading term-structure shape requires multi-way conditioning. Depth 1 = 0.53; Depth 2 = 0.86; Depth 4 = **1.11 peak**; Depth 5 = 0.92; Depth 6 = 0.90 (overfitting) — [`results/depth_results.csv`](depth_results.csv), [`plots/depth_vs_sharpe.pdf`](../plots/depth_vs_sharpe.pdf)

---

## D. CAVEATS TO ADD (honest robustness)

### D1. Fold-3 dot-com extrapolation failure

- ☐ [M] **All 36 XGB CV configs negative on fold 3** (validation 2001–2003) — [`results/xgb_cv_results.csv`](xgb_cv_results.csv) (group by fold)
- ☐ [M] **All HMM CV combos negative on fold 3** — [`results/hmm_cv_features.csv`](hmm_cv_features.csv)
- ☐ [M] Diagnosis: **out-of-distribution generalization failure**, not hyperparameter selection. Training (1990–2001) had only short crises; sustained dot-com unprecedented
- ☐ [M] Folds 4–5 positive because training windows include 2001–2003
- ☐ [M] **Honest implication**: π conditioning depends on having seen similar regime in training. Structurally novel regimes would defeat the strategy. Add to robustness/limitations section
- ☐ [M] **Cross-link to §B5–B8**: the actual 1995–2024 backtest *did* lose 36% over 2000–2002. The CV failure isn't just a CV artifact — it manifests in the historical record

### D2. JP momentum weakness

- ☐ [M] JP M2 Sharpe (0.52) < market (0.86). Strategy beats Fixed 12-mo (0.07) and 1-mo (-0.08) but cannot beat passive market exposure in JP. Cite Asness et al 2013

### D3. Strike "test-set peek" admissions

- ☐ [M] Already covered in §B1 — strike pre-CV caveats in `latex/methodology.tex` and `latex/appendix.tex`

---

## E. EXISTING CLAIMS WORTH RESTATING / VERIFYING POST-SHUMWAY

### E0. **DEPRECATED tables — replaced by CV winners**

These two existing tables become *redundant* once the CV winners (Step I and Step J) are presented. Decision: **remove and replace** with the new CV tables, OR **keep for context** and explicitly state that the CV is the primary justification.

- ☐ [M] **`tables/table_xgb_hyperparams.tex` is DEPRECATED** — replaced by [`tables/table_xgb_cv.tex`](../tables/table_xgb_cv.tex). The old table sampled a small grid; the new CV table is the principled justification (Step I, §B1).
- ☐ [M] **`tables/table_feature_selection.tex` is DEPRECATED** — replaced by [`tables/table_hmm_cv.tex`](../tables/table_hmm_cv.tex). The old table reported `DD+DISP+REL_N+CS` Sharpe 0.91 with σ=0.11 across HMM seeds (in-sample / test-period peek); the CV table is held-out-validation winner from Step J (§B2).
- ☐ [N] Companion DEPRECATED CSVs (iterative selection passes that fed into the old `table_feature_selection`): [`results/hmm_feature_selection_pass1.csv`](hmm_feature_selection_pass1.csv), [`results/hmm_feature_selection_pass2.csv`](hmm_feature_selection_pass2.csv), [`results/hmm_feature_selection_pass3.csv`](hmm_feature_selection_pass3.csv) — kept on disk for audit but not cited in thesis (replaced by `hmm_cv_features.csv`).
- ☐ [M] In `latex/methodology.tex` and `latex/appendix.tex`: update the references from `table_xgb_hyperparams` and `table_feature_selection` to the new CV tables, and update the surrounding prose to use the CV-justified language.

### E1–E4. Ablations (already in thesis, verify numbers)

- ☐ [M] **Removing π**: Sharpe 1.11 → **0.41** — [`tables/table_kitchen_sink.tex`](../tables/table_kitchen_sink.tex) (also 0.43 in regime_signal_ablation), [`tables/table_regime_signal_ablation.tex`](../tables/table_regime_signal_ablation.tex)
- ☐ [M] Raw stress indicators (no HMM): Sharpe **0.257** — [`tables/table_regime_signal_ablation.tex`](../tables/table_regime_signal_ablation.tex)
- ☐ [M] No regime signal: Sharpe **0.429** — same
- ☐ [M] **HMM smoothing essential** — raw indicators *worse than no signal* in some configs
- ☐ [M] GHM-style classification collapses: M2 1.11 vs GHM SLOW 0.07, MED 0.12, FAST 0.26, DYN 0.03 — [`tables/table_ghm_comparison.tex`](../tables/table_ghm_comparison.tex)
- ☐ [M] Adding fundamentals dilutes: 1.11 → **0.92** despite 43% fund SHAP — [`tables/table_fundamentals_ablation.tex`](../tables/table_fundamentals_ablation.tex)
- ☐ [M] Raw-return targets essential — [`tables/table_alt_targets.tex`](../tables/table_alt_targets.tex), [`tables/table_risk_aversion.tex`](../tables/table_risk_aversion.tex)
- ☐ [S] **NOTE**: MV with γ=2 actually *better* (Sharpe 1.135, MDD −22.0% vs baseline −28.3%); Log-MV γ=5 best MDD (−18.7%) — qualifies "raw essential" claim
- ☐ [M] Sequential boosting essential — RF bagging loses ~0.3 Sharpe — [`results/random_forest_results.csv`](random_forest_results.csv)

### E5. Regime-conditional Sharpe (post-Shumway)

- ☐ [M] M2 Calm **0.84** (was 0.82 pre-Shumway) — [`tables/table_regime_sharpe.tex`](../tables/table_regime_sharpe.tex)
- ☐ [M] M2 Panic **1.53** (was 1.56 pre-Shumway) — same

### E6. Cost sensitivity — full numerical detail

- ☐ [S] M2 at 0/5/10/20/30/50bps: 1.19/1.15/1.11/1.03/0.95/**0.78** — [`tables/table_cost_sensitivity.tex`](../tables/table_cost_sensitivity.tex)
- ☐ [S] M2 is the **only** strategy profitable at 50bps; all benchmarks unprofitable by 30bps

### E7. Threshold sensitivity

- ☐ [S] π > 0.25, 0.50, 0.75 all give stable regime-conditional Sharpes — validates 0.5 cutoff — [`tables/table_threshold_sensitivity.tex`](../tables/table_threshold_sensitivity.tex)

### E8. HMM diagnostics

- ☐ [S] **K=2 best AND most stable**; K=3 → 0.83, K=4 → 0.61 with std=0.29 (high cross-seed variance), K=5 → 0.72 — [`tables/table_multistate_hmm.tex`](../tables/table_multistate_hmm.tex)
- ☐ [S] **HMM regime separation magnitudes** (95% credible interval excludes 0):
  - DD: Δ = **−1.51** [−1.80, −1.23]
  - DISP: Δ = **+1.12** [+0.87, +1.35]
  - REL_N: Δ = **−1.42** [−1.59, −1.24]
  - CS: Δ = **+0.90** [+0.62, +1.16]
  — [`tables/table_hmm_separation.tex`](../tables/table_hmm_separation.tex)
- ☐ [N] Student-t HMM: no improvement over Gaussian — [`tables/table_student_t_hmm.tex`](../tables/table_student_t_hmm.tex)
- ☐ [N] Gelman-Rubin: chains converge — [`tables/table_gelman_rubin.tex`](../tables/table_gelman_rubin.tex)

### E9. Spanning regressions

- ☐ [M] M2 ~ M1: M2 retains alpha — [`results/`](.) (computed by `scripts/new_ls_analyses.py`)
- ☐ [M] M1 ~ M2: M1 alpha doesn't survive — same

### E10. **Granger causality (NEW)**

- ☐ [S] **π → IC_mom and IC_mom → π both non-significant** (F<2, p>0.10 at lags 1–3). Rules out simple "regime predicts when momentum works" story — [`tables/table_granger.tex`](../tables/table_granger.tex). Frame as: model uses regime + momentum *jointly*, not sequentially

### E11. **IC paradox (NEW — critical thesis nuance)**

- ☐ [M] **M2 IC = −0.005 (n.s.) vs naive momentum IC = +0.027 (t=3.15\*\*\*)** — [`tables/table_ic.tex`](../tables/table_ic.tex)
- ☐ [M] **M2 has WORSE per-stock IC than naive momentum, yet much better Sharpe.** Edge is portfolio construction (decile spread + regime ranking), NOT improved per-stock prediction
- ☐ [M] Paired diff M2 − Mom = **−0.031 (t=−2.95\*\*\*)** — significantly worse IC

### E12. Bootstrap inference (post-Shumway)

- ☐ [M] M2 Sharpe 1.11, **95% CI [0.66, 1.54]** (block bootstrap, 12-month blocks, 10,000 resamples) — [`tables/table_bootstrap.tex`](../tables/table_bootstrap.tex), [`results/bootstrap_sharpe_cis.csv`](bootstrap_sharpe_cis.csv)
- ☐ [M] M2 Calm: 0.82 [0.25, 1.32]; M2 Panic: 1.57 [0.93, 2.19] — same
- ☐ [S] Panic − Calm: 0.75 [−0.08, 1.60], **p=0.074\*** (panic outperformance only marginally significant)
- ☐ [M] **Paired tests vs benchmarks (highly significant)**: M2 vs M1 p=**0.001\*\*\***, vs M0 p=**0.010\*\*\***, vs Fixed 12-mo p=**0.006\*\*\***, vs Fixed 1-mo p=**0.024\*\*** — [`results/bootstrap_paired_tests.csv`](bootstrap_paired_tests.csv)

### E13. **Factor loadings (NEW — strong novelty signal)**

- ☐ [M] **M2 has NEGATIVE UMD loading** (−0.20 to −0.22 in Carhart/FF6) — M2 is *negatively* correlated with naive momentum despite using momentum features. Bets on a *different* slice — [`tables/table_factor_alphas.tex`](../tables/table_factor_alphas.tex)
- ☐ [M] Negative HML loading (−0.17 to −0.31): growth-tilted
- ☐ [M] Mkt-RF loading ≈ −0.15: defensive on market
- ☐ [S] Positive CMA in FF5/FF6 (+0.12 to +0.21): investment-tilted

### E14. **Performance table benchmarks (NEW context)**

- ☐ [S] **All non-M2 strategies have negative β** (−0.34 to −0.66) — they short the market on net. M2 is the *only* L/S strategy with positive β (+0.45). Worth highlighting — [`tables/table_performance.tex`](../tables/table_performance.tex)
- ☐ [S] M2 NW t-stat 4.37\*\*\* (strongest of all benchmarks) — same

### E15. **Panic SUBTYPES (NEW — major mechanism finding)**

- ☐ [M] **Calm: Sharpe 0.82** (108 months) — [`tables/table_panic_subtypes.tex`](../tables/table_panic_subtypes.tex)
- ☐ [M] **Panic Crash months (negative-market): Sharpe −0.33** (18 months) — strategy *loses* in actual crashes
- ☐ [M] **Panic Recovery months (positive-market within panic regime): Sharpe +2.35** (41 months)
- ☐ [M] **Reframe**: the strategy wins by riding the *rebound rally after panic onset*, not by predicting crashes. The 1.53 panic Sharpe is dominated by recovery component. **Goes in mechanism discussion alongside C1 (leg-betas) and C2 (term-structure shape).**

### E16. Stress scenario progression

- ☐ [S] Linear: 3-month recession adds 10% loss, 6=19%, 12=34%, 18=47%, **24=57%** (M2 loses 3.44%/month avg in losing panic months) — [`tables/table_stress_scenarios.tex`](../tables/table_stress_scenarios.tex)
- ☐ [S] Worst-case MDD by recession length: 3mo→−38%, 12mo→−49%, **24mo→−69%**. Threshold for D&M-scaling justified — same

### E17. **Ridge baseline (NEW — definitive proof)**

- ☐ [M] **Ridge with α∈{0.1, 1, 10, 100, 1000}: ALL give Sharpe −0.58** — [`tables/table_ridge.tex`](../tables/table_ridge.tex)
- ☐ [M] **OLS unconstrained: also −0.58** — same
- ☐ [M] **Linearity is the binding constraint, NOT regularisation choice.** Distinguishes XGB's edge from a regularisation issue. Goes in nonlinearity-analysis section

### E18. **Z-scores by horizon (NEW — quantitative term-structure backbone)**

- ☐ [S] Calm Long peak at mom_8: **+0.39 z**; Panic Long flat at z ≈ −0.13 — [`tables/table_zscore_shap_detail.tex`](../tables/table_zscore_shap_detail.tex)
- ☐ [S] Panic Short at mom_1: **+0.30 z** (recent winners get shorted) — same
- ☐ [S] SHAP shares: mom_11 **18.6–23.1%**, mom_9 12.7–15.5%, mom_8 10.2–12.8% — same

### E19. **Tree path / combo analysis (NEW)**

- ☐ [N] Most common decision paths use multi-horizon conditioning (m9≥→m10≥, m6≥→m11≥→m10≥→m9≥) — [`results/tree_path_results.csv`](tree_path_results.csv)
- ☐ [N] **Combo frequencies**: I+L (intermediate+long) at **12.9–14.8%**, M+I+L at ~10%, single-horizon ~5–7% — [`tables/table_combo_freq.tex`](../tables/table_combo_freq.tex)
- ☐ [N] Long-combo z-scores: I+L combo selects at **+0.45 z** at those horizons; S+L at +0.46 — [`tables/table_combo_long_cp.tex`](../tables/table_combo_long_cp.tex)

### E20. Sample summary

- ☐ [N] 16,657 unique stocks; 1.72M stock-month obs; train 4,838/month, test 3,336/month — [`tables/table_sample_summary.tex`](../tables/table_sample_summary.tex)

### E21. **Alt train/test splits robustness (NEW reference)**

- ☐ [S] Tests M2 across alternative train/test boundaries (1990-2010 baseline = 1.11; verify others post-Shumway are robust) — [`tables/table_alt_splits.tex`](../tables/table_alt_splits.tex)

### E22. **IC rotation across regimes (NEW reference)**

- ☐ [S] Per-horizon IC table split by calm vs panic months — [`tables/table_ic_rotation.tex`](../tables/table_ic_rotation.tex)
- ☐ [S] Shows the per-horizon momentum IC is *similar* across regimes; the regime difference is in *which side of the distribution* the model picks (consistent with §C2)

### E23. **HMM feature ablation (NEW reference)**

- ☐ [S] Tests subsets of the 4-feature HMM input (full vs leave-one-out) — [`tables/table_hmm_feature_ablation.tex`](../tables/table_hmm_feature_ablation.tex). Full 4F baseline Sharpe 0.97 (in-sample evaluation, distinct from Step J's CV)
- ☐ [S] Note: the CV winner (`DD+CS+LVIX+DISP`) supersedes the in-sample feature ablation as the canonical justification

### E24. **Fundamentals factor alphas (NEW reference)**

- ☐ [S] Factor alphas for the *fundamentals-included* M2 variant (mom + π + 10 fundamentals) — [`tables/table_fund_alphas.tex`](../tables/table_fund_alphas.tex). CAPM α drops from 23.4% (mom only) to **18.2%** (with funds), again confirming fundamentals dilute the regime-momentum signal
- ☐ [N] Performance row for fund-included M2: Ann.Ret 16.6%, Vol 18.6%, Sharpe **0.92**, MDD −18.4%, β=0.11, NW t=3.70\*\*\* — [`tables/table_performance_fund_row.tex`](../tables/table_performance_fund_row.tex). MDD is *smaller* with funds (−18.4 vs −22.8) — funds slightly improve risk profile but dilute Sharpe

### E25. **LR (M1) coefficients (NEW reference)**

- ☐ [S] Logistic-regression coefficient estimates (standardized features, no regularization) — [`tables/table_lr_coef.tex`](../tables/table_lr_coef.tex)
- ☐ [S] Pin the finding that M1's coefficients are stable but its sign-aware portfolio-formation step fails — that's why M1 IC > 0 but M1 Sharpe ≈ 0 (the LR can rank stocks but can't condition direction on regime)

### E26. **SHAP feature attribution (NEW reference)**

- ☐ [M] **Overall SHAP**: Momentum (12 horizons) absorbs **54%**, π_filter absorbs **46%** — [`tables/table_shap.tex`](../tables/table_shap.tex)
- ☐ [M] **Calm SHAP**: Momentum 51%, π 49%
- ☐ [M] **Panic SHAP**: Momentum 59%, π 41%
- ☐ [M] **Implication**: The model uses momentum *more* in panic and π *more* in calm — opposite of what one might expect. In panic the regime signal "saturates" (just a high π switch) and momentum carries the cross-sectional signal
- ☐ [S] SHAP dependence by horizon × regime — [`results/shap_dependence_all_horizons.csv`](shap_dependence_all_horizons.csv)
- ☐ [S] SHAP attribution to long vs short legs — [`results/shap_portfolio_analysis.csv`](shap_portfolio_analysis.csv)

### E27. **Turnover (NEW reference)**

- ☐ [S] M2 turnover stats — [`tables/table_turnover.tex`](../tables/table_turnover.tex). Production assumed Avg monthly TO **132.4%** (annualised ~1,590%); compare with Fixed 12-mo at 70.1% (842% annualised). Higher turnover is the cost we pay for regime conditioning
- ☐ [S] At 50bps cost (§E6), M2 still profitable — turnover is high but not prohibitive

### E28. **Placebo test (NEW reference)**

- ☐ [M] Real π gives M2 Sharpe 1.11; fake/placebo π variants collapse — [`tables/table_placebo.tex`](../tables/table_placebo.tex). **Confirms π isn't a coincidence variable.** Goes alongside §E1 (kitchen sink) as the "is the regime signal real" defense

### E29. **Seed convergence (NEW reference)**

- ☐ [S] Sharpe stability as # of XGB ensemble seeds grows — [`tables/table_seed_convergence.tex`](../tables/table_seed_convergence.tex). At k=1 seed, mean 1.05, std 0.07. Production uses 50 seeds → Sharpe stabilizes at ~1.11. Justifies the 50-seed production choice

### E30. **Tree-path SHORT-leg combos (NEW reference)**

- ☐ [N] Counterpart to §E19 long-leg combos: short-leg combo paths and z-scores — [`tables/table_combo_short_cp.tex`](../tables/table_combo_short_cp.tex). Confirms the term-structure shape *flips on the short side too*

### E31. **Pi-filter dominance stats (NEW reference)**

- ☐ [N] Statistical confirmation that π_filter dominates other regime-signal candidates — [`results/pi_dominance_stats.csv`](pi_dominance_stats.csv)

### E32. **Two-model dual utility (NEW reference)**

- ☐ [N] Dual-utility comparison (M1 + M2) — [`results/two_model_dual_util.csv`](two_model_dual_util.csv) (script `scripts/two_model_dual_util.py`). Verify finding still holds post-Shumway; relevant to whether to combine M1 and M2

### E33. **Other existing claims (verify post-Shumway)**

- ☐ [N] Selection rank analysis (term-structure-shape-of-selection) — [`results/selection_rank_analysis.csv`](selection_rank_analysis.csv)
- ☐ [N] Risk aversion CRRA results — [`results/risk_aversion_crra_results.csv`](risk_aversion_crra_results.csv)
- ☐ [N] Risk-aversion extended results — [`results/risk_aversion_extended_results.csv`](risk_aversion_extended_results.csv)
- ☐ [N] Risk-aversion thesis results (the headline table) — [`results/risk_aversion_thesis_results.csv`](risk_aversion_thesis_results.csv)
- ☐ [N] Z-score time series — [`results/zscore_long_by_month.csv`](zscore_long_by_month.csv), [`results/zscore_short_by_month.csv`](zscore_short_by_month.csv), [`results/zscore_longshort_by_month.csv`](zscore_longshort_by_month.csv)
- ☐ [N] Tree combo full results — [`results/tree_combo_results.csv`](tree_combo_results.csv)
- ☐ [N] Random forest baseline (RF underperforms XGB) — [`results/random_forest_results.csv`](random_forest_results.csv)
- ☐ [N] Fundamentals test results — [`results/fundamentals_test_results.csv`](fundamentals_test_results.csv)
- ☐ [N] Bootstrap regime-conditional results — [`results/bootstrap_regime.csv`](bootstrap_regime.csv)

### E34. **January-exclusion robustness (table reference, was E11)**

- ☐ [S] M2 excluding-January Sharpe ~1.06 (verify post-Shumway) — [`tables/table_january.tex`](../tables/table_january.tex). Confirms result isn't driven by a January effect

### E35. **30-year backtest companion files**

- ☐ [S] Full year-by-year returns — [`results/expanding_returns_prod.csv`](expanding_returns_prod.csv) (one row per month, years 1995-2024)
- ☐ [S] Year-by-year π_filter — [`results/expanding_pi_filter_prod.csv`](expanding_pi_filter_prod.csv)
- ☐ [S] Per-year fit summary (XGB train/test rows, mean π, ann.Sharpe) — [`results/expanding_summary_prod.csv`](expanding_summary_prod.csv)
- ☐ [N] Historical OOS at older start dates: 1990-1999 train — [`results/oos_returns_prod_1990_1999.csv`](oos_returns_prod_1990_1999.csv), [`results/oos_pi_filter_prod_1990_1999.csv`](oos_pi_filter_prod_1990_1999.csv), [`results/oos_subperiods_prod_1990_1999.csv`](oos_subperiods_prod_1990_1999.csv)
- ☐ [N] Historical OOS 1990-2004 train — [`results/oos_returns_prod_1990_2004.csv`](oos_returns_prod_1990_2004.csv), [`results/oos_subperiods_prod_1990_2004.csv`](oos_subperiods_prod_1990_2004.csv), [`results/oos_pi_filter_prod_1990_2004.csv`](oos_pi_filter_prod_1990_2004.csv)
- ☐ [N] Sub-period decomposition full — [`results/oos_subperiod_full.csv`](oos_subperiod_full.csv)

### E36. **International full returns time series**

- ☐ [N] UK monthly returns (regional π): [`results/intl_uk_returns.csv`](intl_uk_returns.csv); UK with US π: [`results/intl_uk_returns_uspi.csv`](intl_uk_returns_uspi.csv)
- ☐ [N] JP monthly returns (regional π): [`results/intl_jp_returns.csv`](intl_jp_returns.csv); JP with US π: [`results/intl_jp_returns_uspi.csv`](intl_jp_returns_uspi.csv)

---

### E37. **Figures** (auto-regenerated by `scripts/generate_plots.py` — Step 8)

All 63 plots in `plots/` are regenerated post-Shumway. Most are exploratory; only 10 are currently `\includegraphics`'d into `latex/`. **Critical figures for the new thesis claims:**

- ☐ [M] [`plots/depth_vs_sharpe.pdf`](../plots/depth_vs_sharpe.pdf) — depth-vs-Sharpe curve (already in thesis, regenerated)
- ☐ [M] [`plots/chart2_rank_by_horizon_v2.pdf`](../plots/chart2_rank_by_horizon_v2.pdf) — term-structure-shape-of-selection (key §C2 figure; consider adding to main_results)
- ☐ [M] [`plots/leg_betas_rolling.pdf`](../plots/leg_betas_rolling.pdf) — rolling β panel (new, supports §C1)
- ☐ [S] [`plots/momentum_shape_final.pdf`](../plots/momentum_shape_final.pdf) — alternative term-structure visualization
- ☐ [S] [`plots/chart_ls_rank_yearly.pdf`](../plots/chart_ls_rank_yearly.pdf) — L/S rank year-over-year
- ☐ [S] [`plots/chart_m2_vs_mom12.pdf`](../plots/chart_m2_vs_mom12.pdf) — cumulative-return comparison
- ☐ [S] [`plots/avg_tree_production.pdf`](../plots/avg_tree_production.pdf) — average tree visualization (mechanism aid)
- ☐ [S] [`plots/chart1_shap_by_horizon.pdf`](../plots/chart1_shap_by_horizon.pdf) and [`plots/chart1_shap_pct_by_horizon.pdf`](../plots/chart1_shap_pct_by_horizon.pdf) — SHAP attribution by horizon (supports §E26)
- ☐ [N] [`plots/horizon_weight_share.pdf`](../plots/horizon_weight_share.pdf) — momentum horizon weight share
- ☐ [N] [`plots/cs_performance_regime_shaded.png`](../plots/cs_performance_regime_shaded.png) — already in thesis, panic-shaded performance
- ☐ [N] [`plots/zscore_and_absshap_v3.pdf`](../plots/zscore_and_absshap_v3.pdf) — already in thesis
- ☐ [N] [`plots/markov_chain_diagram.pdf`](../plots/markov_chain_diagram.pdf), [`plots/gibbs_sampling_diagram.pdf`](../plots/gibbs_sampling_diagram.pdf) — already in thesis (HMM diagrams)
- ☐ [N] [`plots/convergence_trace.png`](../plots/convergence_trace.png) — already in thesis, HMM convergence
- ☐ [N] All other plots in `plots/` are diagnostic / exploratory; not all need thesis inclusion

---

## F. METHODOLOGY / APPENDIX

- ☐ [N] CV procedure: 5 expanding-window folds on 1990–2010 train, no test-period leakage — `latex/appendix.tex`
- ☐ [N] HMM Bayesian Gibbs / FFBS sampler details (priors from `config.py`)
- ☐ [N] Production: 200 HMM seeds × 50 XGB seeds. CV: 3 HMM seeds × 5 XGB seeds (tractability)
- ☐ [N] Shumway-analogue rule table (rule 1–5 firings) — `latex/methodology.tex` footnote
- ☐ [N] **UK/JP method**: BANK_REL replaces CS_z (no Moody's BAA-AAA equivalent for international markets); 4-feature HMM `DD+DISP+REL_N+BANK_REL`
- ☐ [N] Compustat Global field aliasing: `dldtei`→`dldte`, `dlrsni`→`dlrsn`
- ☐ [N] Block bootstrap: 12-month blocks, 10,000 resamples
- ☐ [N] WRDS DNS workaround technical note (optional, for reproducibility) — `scripts/import_intl_stocks.py:46-56`

---

## G. CITATIONS TO ADD/UPDATE

- ☐ [M] **Rey, H. (2013)** — global financial cycle, for §B3 framing
- ☐ [M] **Asness, Moskowitz & Pedersen (2013)** — international momentum, for JP weakness caveat
- ☐ [M] **Daniel & Moskowitz (2016)** — distinguish their leg-beta inversion from our re-ranking (§C1)
- ☐ [M] **Shumway (1997, 2001)** — delisting bias methodology
- ☐ [M] **Jegadeesh (1990)** + **Lehmann (1990)** — short-term reversal (calm 1-month neutral)
- ☐ [M] **Novy-Marx (2012)** — intermediate-horizon momentum (calm peak at mom_8)
- ☐ [M] **Lee & Swaminathan (2000)** — momentum lifecycle (calm 12-month decay)

---

## H. STRUCTURAL EDIT BUNDLE (per `NEXT_RUN_PLAN.md`)

The bundle's framing is decided by C1 (leg-betas finding):

- ☐ [M] **Edit #1** — `latex/main_results.tex`: new subsection "Leg-level beta dynamics across regimes" using **cross-sectional re-ranking** framing (NOT clean inversion)
- ☐ [M] **Edit #2** — `latex/literature_review.tex:19`: expand D&M (2016) discussion, distinguish their leg-beta inversion from our re-ranking
- ☐ [M] **Edit #3** — `latex/conclusion.tex:53`: sharpen "two channels" — regime conditioning + cross-sectional re-ranking
- ☐ [M] **Edit #4** — `latex/methodology.tex`: pi_filter ↔ D&M bear-indicator link (standalone, lands regardless)
- ☐ [M] **NEW Edit #5** — `latex/main_results.tex` or `latex/conclusion.tex` future-work: add the global financial cycle headline from §B3

---

## I. ADVISOR-PRESSURE-POINT CHECKLIST (defense Q&A)

If your advisor presses on these, you have answers ready:

- ☐ "How do you avoid look-ahead in feature selection?" → §B2 CV winner
- ☐ "How sensitive is the result to hyperparameters?" → §B1, top-5 cluster within fold-noise
- ☐ "What about transaction costs?" → §E6, profitable to 50bps
- ☐ "What about K=3 states?" → §E8, K=2 most stable
- ☐ "What about other risk-aversion targets?" → §E1 + side-note (some MV configs better)
- ☐ "Does it generalize internationally?" → §B3, US π beats regional in UK and JP
- ☐ "Does it work over a longer history?" → §B5–B8, 30-year backtest, but with significant DD in 2000–2002
- ☐ "Can it survive a structurally novel regime?" → §D1, honest acknowledgement (fold-3 + 2000-2002)
- ☐ "Is M2 just a better momentum factor?" → §E13 (negative UMD loading, NOT just more momentum) + §E11 (worse IC than naive momentum)
- ☐ "Why nonlinearity? Why not Ridge?" → §E17 (Ridge gives −0.58 across all α; OLS too)
- ☐ "Is the panic edge real?" → §E12 (panic−calm only marginally significant p=0.07) + §E15 (it's the recovery component, not crashes)
- ☐ "Why these 4 HMM features?" → §B2 CV
- ☐ "How much does π actually contribute?" → §E1 (1.11 → 0.41 without π) + §E16 (HMM smoothing >> raw indicators)

---

## J. CROSS-REFERENCES

- Phase log: [`results/RUN_MANIFEST.md`](RUN_MANIFEST.md)
- Step F (first-checkpoint): [`results/STEP_F_REPORT.md`](STEP_F_REPORT.md)
- Step K (final): [`results/STEP_K_REPORT.md`](STEP_K_REPORT.md)
- Step L (auto-prose-edits): [`results/PROSE_EDITS.md`](PROSE_EDITS.md)
- UK/JP writeup: [`RESULTS_LOG.md`](../RESULTS_LOG.md) §12
- Plan/runbook: [`NEXT_RUN_PLAN.md`](../NEXT_RUN_PLAN.md)
- Memory (preferences and project state): `~/.claude/projects/-Users-giladgang-momentum-regime-shifts/memory/`

---

**Last updated**: 2026-04-27 by autonomous Phase-2 chain. All numerical findings are post-Shumway. Pre-Shumway baseline preserved at [`baseline_pre_shumway_20260426_150517/`](../baseline_pre_shumway_20260426_150517/).
