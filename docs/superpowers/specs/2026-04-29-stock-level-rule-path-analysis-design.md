# Stock-Level Rule-Path Analysis — Design Spec

Date: 2026-04-29
Status: approved by user, awaiting spec review
Scope: thesis-chapter-grade (c), 2–3 weeks

## Goal

Research how the XGBoost ensemble decides which stocks to pick, mechanistically — clearly enough that the explanation also reveals how momentum and the regime signal interact. Use the findings to revise §5.2 of the thesis. Not pre-committed to any specific narrative outcome; the analysis is load-bearing, the narrative follows.

## Motivation

The current §5.2 frames the regime-momentum interaction at the *output* level — long-leg z-profile shape per regime — and reads "calm produces humped continuation, panic produces reversal." Session work surfaced two problems with that framing:

1. **The model produces both shapes in both regimes.** Output-first per-regime k=2 clustering finds ~40% of calm months get the "defensive/flat" shape and ~63% of panic months get the "humped continuation" shape. The clean calm-vs-panic dichotomy is data-incompatible.
2. **Aggregate inputs (mom tertiles + π) explain only ~48% (calm) / ~62% (panic) of output shape variance.** Increasing aggregate granularity does not raise this ceiling. The remaining variance is at the stock level, invisible to cross-sectional summaries.

The fix is to introspect the model directly: cluster long-leg stock-months by which trees fire and which leaves they land in. The discovered rule structure is the right unit for describing the regime-momentum interaction.

## Approach

**Primary:** leaf-membership clustering of long-leg stock-months across the 50-seed × 500-tree ensemble (~25,000 leaves total). Each long-leg stock-month is represented by its leaf-index vector. Cluster on this representation. Each cluster = a "rule path" the ensemble actually used.

**Complementary:** within-month rule-path heterogeneity. Per month, measure dispersion of rule-paths firing for the long-leg picks. Tests whether the model uses one rule per month or many.

**Appendix:** SHAP-based stock-month clustering as alternative interpretive view; comparison to leaf-membership findings. Pre-computed SHAP values are already in `artefacts/cs_artefacts_data.pkl`.

## Chapter structure (revised §5.2)

Build-up arc: aggregate → limitation → stock-level → synthesis.

| section | content | source |
|---|---|---|
| §5.2.1 Aggregate output structure | Long-leg z-profile per regime (k=2 output clusters, bootstrap CIs). Replaces existing 5.2.1+5.2.2. | session work, `output_first_by_regime.py` |
| §5.2.2 Aggregate inputs are insufficient | kNN R² ceiling at 0.48–0.62. Input → output ARI 0.10–0.34. Motivates stock-level introspection. | session work, `landscape_feature_ablation.py` |
| §5.2.3 Rule-path analysis | Primary: leaf-membership clustering. Discovered rules + Sharpes + bootstrap CIs. Replaces existing 5.2.3. | new |
| §5.2.4 Within-month rule heterogeneity | Per-month dispersion of rule-path firings. | new |
| §5.2.5 Momentum × regime interaction | Cross-tabulation of rules × regime × cross-section. Replaces "router with state-dependent influence" framing. | new |
| Appendix | SHAP-based clustering as alternative view. | new |

The existing 5.2.3 mild/transition/deep panic decomposition will appear as a sub-finding within 5.2.5 if the rule-path analysis confirms it, otherwise dropped or noted as superseded.

## Analytical pipeline

Five phases, each producing verifiable artefacts. Total ~2–3 weeks.

### Phase 1: Data extraction (~2 days)

Load the 50-seed × 500-tree ensemble (artefact: `pi_verify_trees_seeds50.pkl` + `cs_artefacts_data.pkl::all_trees`). For each long-leg stock-month, extract leaf indices via `model.apply()`. Output: `(n_long_stockmonths, ~25000)` integer matrix saved to `artefacts/leaf_signatures.npz`.

Sanity checks:
- Total long-leg stock-months ≈ 167 months × ~330/month ≈ 55,200.
- Tree-path counts (e.g. 73.9% of trees containing a π split) reproduce thesis 5.3.1 numbers.

### Phase 2: Rule discovery (~3–4 days)

Methodological choice deferred — try multiple clustering approaches and pick what gives stable, interpretable rules:

- KMeans on PCA-reduced one-hot leaf encoding
- Hierarchical on pairwise Hamming distance
- Graph community detection on stock-month co-occurrence graph (edges weighted by shared leaves)

Sweep k (or resolution) over a range. Pick natural k by silhouette + stability across ensemble seeds.

Output: rule label per long-leg stock-month, saved to `results/thesis/rule_path_labels.csv`.

### Phase 3: Rule characterisation (~3 days)

For each discovered rule:

- **Stock signature**: typical mom profile (12-d), distribution of input feature values
- **Regime signature**: π distribution, calm/panic share
- **Performance**: realised monthly return, annualised Sharpe, bootstrap 95% CI (using existing `bootstrap_helpers.block_bootstrap_sharpe`)
- **Mechanism**: dominant tree splits along the rule path (which features at which thresholds)

Output: per-rule centroid CSV + plots.

### Phase 4: Robustness (~3 days)

- **Seed stability**: refit clustering at 6 seeds (mirrors existing thesis 5.2.3 practice). Sorted cluster sizes per seed, pairwise ARI. Reuses `bootstrap_helpers.seed_stability`.
- **Disjoint-ensemble validation**: re-extract leaf signatures from a disjoint 50-seed retraining (already exists in `pi_verify_trees_seeds50.pkl`). Cluster independently. ARI between the two label vectors.
- **Bootstrap CIs**: per-rule Sharpes, mirroring the calm/panic robustness already done.
- **Subsample stability**: hold out 20% of stock-months, refit clustering on the rest, predict held-out labels (kNN), report agreement.

### Phase 5: Synthesis and writing (~3–5 days)

- Cross-tabulate rules × regime, rules × cross-sectional landscape.
- Test rule-crossing: do rules appear in both regimes, or are they regime-specific?
- Draft LaTeX section: revised 5.2.1–5.2.5.
- Generate thesis-style figures (mirror existing 5.2.3 dispersion-plot format).

## Deliverables

- **Code (~5 new scripts):**
  - `scripts/leaf_signatures.py` — phase 1 extraction
  - `scripts/rule_path_clustering.py` — phase 2
  - `scripts/rule_path_characterisation.py` — phase 3
  - `scripts/rule_path_robustness.py` — phase 4
  - `scripts/rule_path_writeup.py` — phase 5 figure/table generation

- **Helpers (TDD-tested, mirroring `bootstrap_helpers.py`):**
  - `scripts/leaf_clustering_helpers.py` — methodology-specific primitives (Hamming distance, leaf one-hot, etc.)
  - `scripts/test_leaf_clustering_helpers.py`

- **Results (~6 CSVs):**
  - `results/thesis/rule_path_labels.csv`
  - `results/thesis/rule_path_centroids.csv`
  - `results/thesis/rule_path_robustness.csv`
  - `results/thesis/rule_path_sharpe_ci.csv`
  - `results/thesis/rule_path_regime_crosstab.csv`
  - `results/thesis/rule_path_landscape_crosstab.csv`

- **Plots (~4 thesis-grade figures):**
  - `plots/thesis/rule_path_dispersion.{png,pdf}` — per-rule member-month dispersion
  - `plots/thesis/rule_path_crosstabs.{png,pdf}` — rule × regime, rule × landscape heatmaps
  - `plots/thesis/rule_path_centroid_features.{png,pdf}` — typical-stock signature per rule
  - `plots/thesis/rule_path_within_month_homogeneity.{png,pdf}` — phase 4 diagnostic

- **LaTeX:** revised `latex/main_results.tex` §5.2 + new subsections; appendix entry for SHAP comparison.

## Validation strategy

Mirrors the rigor pattern already established in this codebase:

- TDD for any new statistical primitive (Hamming distance computation, leaf-similarity, etc.)
- Foundation audit before claiming numbers (re-derive from raw data, compare to artefact-stored values)
- Seed-stability + bootstrap CIs on every per-rule statistic before reporting in LaTeX
- Honest reporting of overlapping CIs / null results

## Open methodological choices

Two things deferred to in-flight decisions, not committed upfront:

1. **Phase 2 clustering algorithm**: KMeans/Hierarchical/Community detection — pick the one that gives stable, interpretable rules. Likely a brief comparison in the spec'd phase.
2. **Number of rules (k)**: data-driven via silhouette + stability. Could end up at k=2 (matching output-side), k=3 (matching mild/transition/deep), or something different.

Both are flagged in the design as open; the implementation plan will codify them as branch points after preliminary results.

## Out of scope

- Retraining the XGBoost ensemble (existing 50-seed ensemble is reused).
- Extending to international samples (UK/JP) — that's its own initiative per `INTL_VALIDATION_PLAN.md`.
- Comparing to alternative model classes (Random Forest, Linear) beyond what's already in 5.1.

## Success criteria

The chapter is "done" when:

1. Phase 1–5 artefacts all exist and verify (TDD passing, foundation audit clean).
2. Rule-path clustering is seed-stable (ARI ≥ 0.95 across 6 seeds, mirroring 5.2.3).
3. Per-rule Sharpe CIs are reported (point estimates and 95% bands).
4. Cross-tabulation of rules × regime answers the headline question: are rules regime-specific or do they cross regimes?
5. LaTeX §5.2 is rewritten consistent with what the data shows; figures regenerated.
6. End-to-end script reproduces all numbers in §5.2 from clean cache.
