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

After every pipeline rerun, steps 21/22/23 of `run_pipeline.py` re-walk the chain
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
| `plots/features_hmm.png` | figure | `archive/scripts/regenerate_thesis_plots.py` | HMM_FEATURES |
| `plots/regime_probabilities.png` | figure | `scripts/hmm_model.py` | HMM_FEATURES, HMM_K_STATES |
| `tables/table_features_summary.tex` | table | `scripts/hmm_model.py` | HMM_FEATURES |
| `tables/table_feature_selection.tex` | table — DEPRECATED | (not regenerated; replaced by table_hmm_cv per N3 decision) | — |
| `tables/table_hmm_feature_ablation.tex` | table | (no current generator; per user keep old numbers) | HMM_FEATURES |
| `tables/table_hmm_separation.tex` | table | `scripts/hmm_diagnostics.py` | HMM_FEATURES |

### Chapter 6 — Main results (`latex/main_results.tex`)

| Asset | Source script | Config knob |
|---|---|---|
| `tables/table_performance.tex` | `scripts/main_results_analysis.py` | MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS, XGB_SEEDS, MOM_FEATURES |
| `tables/table_subperiod.tex` | `scripts/main_results_analysis.py` | (same) |
| `tables/table_regime_sharpe.tex` | `scripts/main_results_analysis.py` | (same) |
| `tables/table_factor_alphas.tex` | `scripts/new_ls_analyses.py` | (same) |
| `tables/table_kitchen_sink.tex` | `scripts/main_results_analysis.py` | MOM_FEATURES |
| `tables/table_alt_targets.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_alt_splits.tex` | `scripts/main_results_analysis.py` | TRAIN_END |
| `tables/table_ic.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_granger.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_lr_coef.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_shap.tex` | `scripts/main_results_analysis.py` | MOM_FEATURES, FUND_FEATURES |
| `tables/table_placebo.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_panic_subtypes.tex` | `scripts/panic_subtype_analysis.py` | — |
| `tables/table_zscore_shap_detail.tex` | `scripts/selection_rank_analysis.py` | — |
| `tables/table_ghm_comparison.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_regime_signal_ablation.tex` | `scripts/main_results_analysis.py` | HMM_FEATURES |
| `tables/table_xgb_hyperparams.tex` | `scripts/depth_vs_sharpe.py` | MAX_DEPTH |
| `tables/table_cost_sensitivity.tex` | `scripts/main_results_analysis.py` | TRADING_FEE |
| `tables/table_threshold_sensitivity.tex` | `scripts/main_results_analysis.py` | — |
| `tables/table_stress_scenarios.tex` | `scripts/stress_test.py` | — |
| `tables/table_xgb_cv.tex` | `scripts/xgb_cv.py` | XGB hyperparam grid |
| `tables/table_hmm_cv.tex` | `scripts/hmm_cv.py` | HMM feature subsets |
| `tables/table_expanding_subperiods.tex` | `scripts/build_table_expanding_subperiods.py` | TRAIN_END (rolling) |
| `tables/table_leg_betas.tex` | `scripts/leg_betas_by_regime.py` | — |
| `tables/table_combo_freq.tex` | `scripts/tree_combo_grouped.py` | — |
| `tables/table_fundamentals_ablation.tex` | `scripts/fundamentals_test.py` | FUND_FEATURES |
| `plots/cs_performance_regime_shaded.png` | `scripts/generate_plots.py` | — |
| `plots/depth_vs_sharpe.png` | `scripts/depth_vs_sharpe.py` | MAX_DEPTH |
| `plots/zscore_and_absshap_v3.png` | `scripts/selection_rank_analysis.py` | — |
| `plots/zscore_panic_subtypes.png` | `scripts/panic_subtype_analysis.py` | — |

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
| `tables/table_performance_fund_row.tex` | `scripts/fundamentals_test.py` |
| `tables/table_combo_long_cp.tex` | `scripts/tree_combo_grouped.py` |
| `tables/table_combo_short_cp.tex` | `scripts/tree_combo_grouped.py` |
| `tables/table_risk_aversion.tex` | `scripts/risk_aversion_thesis_table.py` |
| `tables/table_ridge.tex` | `scripts/ridge_baseline_test.py` |
| `plots/cs_avg_tree.png` | `scripts/cross_sectional_model.py` |
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
python scripts/build_metrics.py            # canonical store ← tables/+CSVs
python scripts/build_canonical_macros.py   # latex/canonical_macros.tex ← canonical store
python scripts/verify_thesis_consistency.py  # PROSE_DRIFT_REPORT.md ← latex prose vs canonical
```

Or run the wired pipeline (steps 21/22/23 do this automatically):

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
