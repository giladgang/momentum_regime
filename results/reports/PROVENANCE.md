# Thesis provenance map

The chain that connects every thesis result to its source script + config.

```
config.py knobs
   ↓
analysis scripts (scripts/*.py)
   ↓
data outputs (tables/*.tex, plots/*.png|pdf, results/*.csv)
   ↓
\input{} / \includegraphics{} in latex (latex/*.tex, main.tex)
   ↓
canonical metrics (results/PRODUCTION_METRICS.json — 200+ named values)
   ↓
canonical_macros.tex  (auto-generated, optional in prose)
   ↓
drift detection (verify_thesis_consistency.py + PROSE_DRIFT_REPORT.md)
```

After every pipeline rerun, steps 80–83 of `run_pipeline.py` re-walk the chain
automatically; `results/METRICS_DIFF.md` shows what changed and which configs
likely caused it.

---

## Thesis assets by chapter

### Chapter 1 — Title page

| Asset | Type | Source |
|---|---|---|
| `plots/tilburg_logo.pdf` | logo (static) | Tilburg University asset; not regenerated |

### Chapter 2 — Introduction (`latex/introduction.tex`)

No `\input{tables/...}` or `\includegraphics{}`. Numbers cited in prose only;
verified against canonical via `verify_thesis_consistency.py`.

### Chapter 3 — Literature review (`latex/literature_review.tex`)

No data assets.

### Chapter 4 — Methodology (`latex/methodology.tex`)

| Asset | Type | Source script |
|---|---|---|
| `plots/markov_chain_diagram.png` | static schematic | `archive/scripts/generate_markov_diagram.py` (do not regen) |
| `plots/gibbs_sampling_diagram.pdf` | static schematic | `archive/scripts/generate_gibbs_diagram.py` (do not regen) |

### Chapter 5 — Data (`latex/data_section.tex`)

| Asset | Type | Source script | Config knob |
|---|---|---|---|
| `plots/features_hmm.png` | figure | `scripts/plot_features_hmm.py` (Step 47) | HMM_FEATURES |
| `plots/regime_probabilities.png` | figure | `scripts/hmm_model.py` | HMM_FEATURES, HMM_K_STATES |
| `tables/table_feature_selection.tex` | table — **frozen at Apr 14 by decision** | Still `\input`-ed in `data_section.tex:73`. NOT regenerated post-Shumway: the 4-pass HMM feature-selection pipeline that produced the table picked DD+CS+DISP+REL_N as the winning combo, which is the production HMM feature set. Rerunning would shift every reported sub-Sharpe but not the qualitative selection. Per the 2026-04-27 decision, prose stays as-is. | — (frozen) |
| `tables/table_hmm_feature_ablation.tex` | table — **frozen at Apr 12 by decision** | Still `\input`-ed in `data_section.tex:77`. NOT regenerated post-Shumway. Per the 2026-04-27 decision: keep old numbers, don't mention CV in the main body. The qualitative pattern (which features matter most) is preserved. | HMM_FEATURES (frozen) |
| `tables/table_hmm_separation.tex` | table | `scripts/hmm_diagnostics.py` | HMM_FEATURES |

### Chapter 6 — Main results (`latex/main_results.tex`)

| Asset | Source script | Config knob |
|---|---|---|
| `tables/table_performance.tex` | `scripts/main_results_analysis.py` | MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, XGB_SEEDS, MOM_FEATURES |
| `tables/table_subperiod.tex` | `scripts/main_results_analysis.py` | (same) |
| `tables/table_regime_sharpe.tex` | `scripts/main_results_analysis.py` | (same) |
| `tables/table_factor_alphas.tex` | `scripts/new_ls_analyses.py` | (same) |
| `tables/table_alt_targets.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_alt_splits.tex` | `scripts/main_results_analysis.py` | TRAIN_END |
| `tables/table_shap.tex` | `scripts/main_results_analysis.py` | MOM_FEATURES, FUND_FEATURES |
| `tables/table_zscore_shap_detail.tex` | `scripts/selection_rank_analysis.py` | — |
| `tables/table_regime_signal_ablation.tex` | `scripts/main_results_analysis.py` | HMM_FEATURES |
| `tables/table_xgb_hyperparams.tex` | `scripts/depth_vs_sharpe.py` | MAX_DEPTH |
| `tables/table_cost_sensitivity.tex` | `scripts/main_results_analysis.py` | TRADING_FEE |
| `tables/table_expanding_subperiods.tex` | `scripts/build_table_expanding_subperiods.py` | TRAIN_END (rolling) |
| `tables/table_leg_betas.tex` | `scripts/leg_betas_by_regime.py` | — |
| `tables/table_combo_freq.tex` | `scripts/tree_combo_grouped.py` | — |
| `tables/table_fundamentals_ablation.tex` | `scripts/fundamentals_test.py` | FUND_FEATURES |
| `tables/table_turnover.tex` | `scripts/robustness_checks.py` (Step 4) | — |
| `tables/table_international_results.tex` | `scripts/build_thesis_tables.py` (Step 82) | — |
| `tables/table_cluster_k4_descriptors.tex` | `scripts/build_table_cluster_k4_descriptors.py` (Step 48) — depends on Steps 40 + 41 | — |
| `tables/table_cluster_k4_features_appendix.tex` | `scripts/build_table_cluster_k4_features_appendix.py` (Step 49) — depends on Step 40 | — |
| `tables/table_cluster_k4_leg_betas.tex` | `scripts/build_table_cluster_k4_leg_betas.py` (Step 50) — depends on Step 42 | — |
| `tables/table_cluster_k4_shap_shares.tex` | `scripts/build_table_cluster_k4_shap_shares.py` (Step 51) — depends on Step 43 | — |
| `tables/table_cluster_k4_short_leg.tex` | `scripts/build_table_cluster_k4_short_leg.py` (Step 52) — depends on Steps 40 + 41 | — |
| `plots/cs_performance_regime_shaded.png` | `scripts/generate_plots.py` | — |
| `plots/shap_per_horizon_by_leg.png` | `scripts/plot_shap_per_horizon_by_leg.py` (Step 44) — depends on Step 38 | — |
| `plots/zscore_long_heatmap.png` | `scripts/plot_zscore_long_heatmap.py` (Step 45) — depends on Steps 18 + 39 | — |
| `plots/zscore_l2_k4_panel_c{0..3}.png` | `scripts/plot_cluster_k4_individual_panels.py` (Step 46) — depends on Steps 2 + 18 + 39 | — |

### Available but not cited in the body

These assets are produced by the pipeline but not currently `\input{}`-ed or `\includegraphics`-ed by the thesis. They are kept available as exploratory outputs / candidates for future revisions; removing them does not affect the compiled PDF.

| Asset | Source script |
|---|---|
| `tables/table_features_summary.tex` | `scripts/hmm_model.py` |
| `tables/table_kitchen_sink.tex` | `scripts/main_results_analysis.py` |
| `tables/table_ic.tex` | `scripts/main_results_analysis.py` |
| `tables/table_granger.tex` | `scripts/main_results_analysis.py` |
| `tables/table_lr_coef.tex` | `scripts/main_results_analysis.py` |
| `tables/table_placebo.tex` | `scripts/main_results_analysis.py` |
| `tables/table_panic_subtypes.tex` | `scripts/panic_subtype_analysis.py` |
| `tables/table_ghm_comparison.tex` | `scripts/main_results_analysis.py` |
| `tables/table_xgb_cv.tex` | `scripts/xgb_cv.py` |
| `tables/table_hmm_cv.tex` | `scripts/hmm_cv.py` |
| `tables/table_threshold_sensitivity.tex` | `scripts/main_results_analysis.py` |
| `tables/table_stress_scenarios.tex` | `scripts/stress_test.py` |
| `tables/table_risk_aversion.tex` | `scripts/risk_aversion_thesis_table.py` |
| `tables/table_ridge.tex` | `scripts/ridge_baseline_test.py` |
| `tables/table_performance_fund_row.tex` | `scripts/fundamentals_test.py` |
| `plots/depth_vs_sharpe.png` | `scripts/depth_vs_sharpe.py` |
| `plots/cs_avg_tree.png` | `scripts/cross_sectional_model.py` |
| `plots/zscore_and_absshap_v3.png` | `scripts/selection_rank_analysis.py` |
| `plots/zscore_panic_subtypes.png` | `scripts/panic_subtype_analysis.py` |

### Chapter 7 — Conclusion (`latex/conclusion.tex`)

No `\input` / `\includegraphics`. Numbers cited in prose only.

### Appendix (`latex/appendix.tex`)

| Asset | Source script |
|---|---|
| `tables/table_multistate_hmm.tex` | `scripts/hmm_diagnostics.py` |
| `tables/table_gelman_rubin.tex` | `scripts/hmm_diagnostics.py` |
| `tables/table_student_t_hmm.tex` | `scripts/hmm_diagnostics.py` |
| `tables/table_bootstrap.tex` | `scripts/bootstrap_analysis.py` |
| `tables/table_january.tex` | `scripts/january_exclusion.py` |
| `tables/table_seed_convergence.tex` | `scripts/seed_convergence.py` |
| `tables/table_sample_summary.tex` | `scripts/cross_sectional_model.py` |
| `tables/table_fund_alphas.tex` | `scripts/fundamentals_test.py` |
| `tables/table_combo_long_cp.tex` | `scripts/tree_combo_grouped.py` |
| `tables/table_combo_short_cp.tex` | `scripts/tree_combo_grouped.py` |
| `plots/convergence_trace.png` | `scripts/hmm_model.py` |
| `plots/risk_aversion_dual_util.pdf` | `scripts/plot_crra_dual_util.py` |

---

## Reverse map: changing a config

Want to know what changes if you flip a knob? Look here:

| Config knob | Affected tables | Affected figures | Cost |
|---|---|---|---|
| `HMM_FEATURES` | regime_sharpe, hmm_separation, multistate_hmm, **all M2 tables** | regime_probabilities, features_hmm, convergence_trace | full pipeline rerun (~hours) |
| `HMM_K_STATES` | multistate_hmm, hmm_separation | (same) | full pipeline rerun |
| `MAX_DEPTH` | performance, factor_alphas, regime_sharpe, subperiod, kitchen_sink, xgb_hyperparams | depth_vs_sharpe, cs_avg_tree | partial rerun (XGB + downstream) |
| `LEARNING_RATE` | performance, factor_alphas, regime_sharpe, subperiod | (same) | partial rerun |
| `N_ESTIMATORS` | performance, factor_alphas, seed_convergence | (same) | partial rerun |
| `XGB_SEEDS` (size of ensemble) | performance, factor_alphas, seed_convergence | seed_convergence | partial rerun |
| `FUND_FEATURES` | **fund_alphas, fundamentals_ablation, performance_fund_row** | (none) | only `scripts/fundamentals_test.py` |
| `TRADING_FEE` | performance, cost_sensitivity, subperiod | (none) | partial rerun (post-portfolio) |
| `TRAIN_END` | **everything** (different boundary changes the universe) | (most) | full pipeline rerun |

Source: [`scripts/_config_deps.py`](../scripts/_config_deps.py) — extend by adding new entries.

---

## Verifying the chain

After any rerun:

```bash
python scripts/build_metrics.py              # canonical store ← tables/+CSVs           (step 80)
python scripts/build_canonical_macros.py     # latex/canonical_macros.tex ← canonical   (step 81)
python scripts/build_thesis_tables.py        # headline tables ← canonical              (step 82)
python scripts/verify_thesis_consistency.py  # PROSE_DRIFT_REPORT.md ← latex vs canonical (step 83)
```

Or run the wired pipeline (steps 80–83 do this automatically):

```bash
python run_pipeline.py
```

What the diff system tells you:
- `results/METRICS_DIFF.md` — what numbers changed since last run, cross-referenced
  against the dependency map (so you see "FUND_FEATURES changed → these
  fund_alphas.* metrics drifted, which is expected").
- `results/PROSE_DRIFT_REPORT.md` — which lines in `latex/*.tex` still cite stale
  numbers. Walk this file row-by-row to update prose.

If a metric drifts but **no** config knob shows as changed, the diff flags it
under "Unexpected drifts" (could be stochastic seed variance — small Δ — or a
config you forgot to track in `_config_deps.py`).
