# Project organization plan (post-Shumway, 2026-04-27)

## New directory structure

```
results/
  thesis/                    # ALL CSVs/JSONs whose values appear in thesis tables/prose
    bootstrap_*.csv
    expanding_*_prod.csv
    fundamentals_test_results.csv
    fundamentals_returns.pkl
    hmm_separation.csv
    intl_*.csv (8 files)
    leg_betas_by_regime.csv
    pi_dominance_stats.csv
    random_forest_results.csv
    risk_aversion_*.csv
    seed_convergence.csv
    selection_rank_analysis.csv
    shap_dependence_all_horizons.csv
    shap_*.csv (existing in results/)
    tree_combo_results.csv
    two_model_dual_util.csv
    zscore_*_by_month.csv
    depth_results.csv
    fundamentals_test_results.csv
    bootstrap_regime.csv

  cv/                        # cross-validation grids + winners (mostly appendix tables)
    hmm_cv_features.csv
    hmm_cv_winner.json
    xgb_cv_results.csv
    xgb_cv_winner.json
    xgb_cv_smoke.csv
    hmm_feature_selection_pass1.csv  (deprecated, keep for audit)
    hmm_feature_selection_pass2.csv
    hmm_feature_selection_pass3.csv

  oos_historical/            # NOT in thesis but kept (older train-end-date OOS)
    oos_*_prod_1990_1999.csv (3 files)
    oos_*_prod_1990_2004.csv (3 files)

  reports/                   # markdown + per-step reports
    METRICS_DIFF.md
    PROSE_DRIFT_REPORT.md
    PROSE_EDITS.md
    PROVENANCE.md
    RUN_MANIFEST.md
    STEP_F_REPORT.md
    STEP_K_REPORT.md
    ORGANIZATION_PLAN.md     ← this file

  PRODUCTION_METRICS.json    # canonical store stays at root for path stability

plots/
  thesis/                    # the 11 \includegraphics targets
    convergence_trace.png
    cs_performance_regime_shaded.png
    depth_vs_sharpe.png
    features_hmm.png
    gibbs_sampling_diagram.pdf
    markov_chain_diagram.png
    regime_probabilities.png
    risk_aversion_dual_util.pdf
    tilburg_logo.pdf
    zscore_and_absshap_v3.png
    zscore_panic_subtypes.png

  diagnostic/                # everything else still in plots/ (~39 files)
    cs_avg_tree.png
    cs_feature_importance.png
    cs_lr_weights.png
    cs_pdp_pi_filter.png
    cs_performance.png
    seed_convergence.png
    leg_betas_rolling.{pdf,png}
    risk_aversion_crra.png
    chart2_rank_by_horizon_v2.{png,pdf}
    chart_ls_rank_yearly.{png,pdf}
    momentum_shape_*  (when revived)
    zscore_long_short_heatmap.{pdf,png} (currently broken — plotly)
    zscore_longshort_heatmap.{pdf,png}   (currently broken — plotly)
    zscore_and_absshap.{pdf,png} (older variants)
    zscore_and_shap*
    + ~20 more

tables/
  (FLAT)                     # all tables are thesis-grade \input{} targets
    table_*.tex (production)
    table_*.canonical.tex (auto-rendered from PRODUCTION_METRICS.json — staging)
```

## Migration cost

| Update | What changes |
|---|---|
| **File moves** | ~120 files repathed |
| **Script `to_csv`/`savefig` paths** | ~30 scripts: each `'results/foo.csv'` → `'results/thesis/foo.csv'` etc. |
| **Latex `\input{tables/...}`** | unchanged (tables/ stays flat) |
| **Latex `\includegraphics{plots/...}`** | 11 references: `plots/foo.png` → `plots/thesis/foo.png` |
| **Framework scripts (build_metrics, build_thesis_tables)** | path constants updated |
| **Tests** | any test that hardcodes `results/...` or `plots/...` |

## Migration sequence (atomic)

1. **WAIT** for in-flight scripts to finish (don't disrupt their writes)
2. Run `scripts/migrate_to_organized.py` (move files; bulk-update script paths via sed)
3. Update latex `\includegraphics` paths via sed
4. Run `python run_pipeline.py --step 21` (build_metrics) to verify nothing broke
5. Run `pdflatex main.tex` to verify thesis still compiles
6. Commit + push as a single atomic refactor commit

## What I won't move

- `tables/` — stays flat (every .tex there is thesis-grade)
- `archive/` — already segregated; don't disturb
- `unused_plots/` — already segregated
- `baseline_pre_*` — snapshots; preserve as-is
- `tests/` (Python pytest) — separate from data outputs
- `experiments/` — one-off scripts, not data
- `data/` — raw/derived input data, not outputs
- `artefacts/` — pickles/intermediate data; can stay flat
- `latex/` — thesis prose
- `scripts/` — code

## After migration

- `python scripts/build_metrics.py` rebuilds canonical from new paths
- `python scripts/verify_thesis_consistency.py` still flags drift
- `latex/canonical_macros.tex` still emits from JSON
- The thesis PDF compiles unchanged
- Future reruns of `run_pipeline.py` write to new locations

## Rollback plan

If anything breaks: `mv results/thesis/* results/ && mv plots/thesis/* plots/ && ...` reverses the move. The `baseline_pre_full_rerun_*` snapshot also captures the pre-migration state.
