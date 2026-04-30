# Picked-Stock Cluster Feature Search — Design Spec (v2)

Date: 2026-04-30 (revision 2 — supersedes the v1 "output-shape" design above)
Status: approved approach, awaiting spec review
Scope: ~1 day of focused analysis to identify what month-level features predict the model's stock-selection patterns.

## Goal

Cluster months by the model's **stock-selection identity** — the raw cross-sectional momentum profile of long-leg picks (level + structure, NOT z-scored shape). Then identify which month-level features (regime, momentum landscape, history) best predict cluster membership.

The winning feature(s) are the natural "group-by" variables for §5.2 — they describe what regime context drives the model toward each kind of basket.

## Why this differs from prior approaches in this project

- **Prior "shape" clustering** (hierarchical_labels): clustered on the z-scored 12-d output curve. Z-score normalisation per-month removes absolute level — restricted us to "what shape did the model produce" rather than "what stocks did it pick".
- **Leaf-signature (trees-fired) clustering**: already done — gave 2 architectural rules (calm-rule / panic-rule). Answers "which tree paths fired", not "what does the basket look like in feature space".
- **This spec**: answers "what does the picked basket look like in observable feature space, and what regime context predicts it?"

## Clustering target: picked-stock fingerprint

For each month with long-leg picks, compute a 15-d fingerprint:

- **12-d mom level/shape**: mean raw momentum of long-leg picks at each horizon h=1..12.
- **3-d dispersion**: std of long-leg picks at S/M/L tertiles (within-month dispersion of picks).

This captures all three of: where on the momentum curve the picks sit (level), the curve's shape, and how concentrated vs spread the basket is.

**Not in the clustering basis:** π_filter — it describes the regime, not the basket. π enters as a *predictor* (Group A below).

## Note on π_filter

π is binary-ish in practice (very near 1 in panic months, very near 0 in calm months — almost no in-between). So:

- All π-derived features are computed as binary indicators or fractions of months in panic, NOT as continuous π values.
- In the decision tree, expect a clean split near π = 0.5.
- In L1 logistic regression, the π coefficient is interpretable as a panic-vs-calm effect, not a continuous slope.

## Method

### Step 1: Cluster

- Standardise all 15 fingerprint dimensions (per-feature z-score across months).
- KMeans for K ∈ {2, 3, 4, 5}, n_init=20.
- Stability: 6-seed run, pairwise mean ARI ≥ 0.90 to accept.
- Silhouette per K.
- Pick K: the smallest K that is stable AND has interpretable centroids. Report the full sweep.

### Step 2: Predictor feature library (~14 features)

**Group A — context (regime):**
1. `pi_panic` — binary indicator (π > 0.5).
2. Cross-section mom_short, mom_mid, mom_long tertile means (3 features) — landscape structure.
3. `mom_overall` — mean of all 12 horizons of cross-section mean ("how high is momentum overall this month").
4. Cross-section dispersion at S/M/L tertiles (3 features).

**Group B — time-series:**
5. `pi_panic_freq_6mo` — fraction of past 6 months with π > 0.5.
6. `pi_panic_freq_12mo` — fraction of past 12 months with π > 0.5.
7. `past_sharpe_12mo` — past 12-month annualised Sharpe of M2 (the strategy itself).

**Group C — higher cross-section moments:**
8. Cross-section skewness at S/M/L (3 features).

Total: 14 features.

### Step 3: Univariate ANOVA ranking
Per-feature F-statistic against cluster labels. Saves a ranked CSV.

### Step 4: Multinomial L1 logistic regression with LOO-CV
- Standardise features.
- Grid: C ∈ {0.05, 0.1, 0.3, 1.0, 3.0}.
- Pick best by leave-one-out CV accuracy.
- Report sparse coefficient matrix + best CV accuracy + class-imbalance baseline (max class prevalence).

### Step 5: Depth-3 decision tree
- `DecisionTreeClassifier(max_depth=3)`.
- Export as indented text rule.

### Step 6: Per-cluster centroid summary
For interpretation: per-cluster mean ± std of every fingerprint dimension AND every predictor feature. CSV.

## Outputs (in `results/thesis/`)

- `picked_stock_cluster_sweep.csv` — K vs silhouette + ARI.
- `picked_stock_cluster_labels.csv` — month-level cluster labels at chosen K.
- `picked_stock_cluster_centroids.csv` — fingerprint centroid per cluster.
- `picked_stock_feature_panel.csv` — per-month predictor feature panel (14 features).
- `picked_stock_feature_ranking.csv` — univariate F-stat per feature.
- `picked_stock_logistic_coefficients.csv` — L1 multinomial coefficients per cluster × feature.
- `picked_stock_decision_tree.txt` — depth-3 tree readout.
- `picked_stock_cluster_summary.csv` — per-cluster mean ± std for fingerprint + predictor features.
- Console headline: chosen K, LOO-CV accuracy, top features, success/weak/negative verdict.

## Success criteria

- **Strong success:** ≥ 70% LOO-CV accuracy. Features cleanly recover the cluster structure → identifies the natural group-by variables for §5.2.
- **Soft success:** depth-3 tree gives a 2- or 3-leaf rule that explains most clusters cleanly but mixes one pair.
- **Negative result:** < 60% accuracy. Picks are determined by something below the monthly aggregate (stock-level) — itself a thesis-relevant boundary for §5.2.

## Validation

- 6-seed stability sweep on the K choice (ARI ≥ 0.90).
- Leave-one-out CV for logistic regression.
- Decision tree: report both training accuracy AND `cross_val_predict` LOO accuracy (training is upper bound; LOO is honest).

## Out of scope

- Stock-level explanations (intra-month, within-cluster).
- Macro indicators outside the feature panel (VIX, yield curve, etc.).

## Estimated time

~4–5 hours of focused work:
- Compute picked-stock fingerprint + cluster sweep: 1 hour
- Predictor feature panel: 30 min
- Univariate + multinomial + tree: 30 min
- Per-cluster summary + writeup: 30 min
- Buffer / debugging: 1.5–2 hours
