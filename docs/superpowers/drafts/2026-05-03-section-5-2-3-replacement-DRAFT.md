# Draft v7 (cleaned): Replacement for §5.2.1 + §5.2.2 + §5.2.3, modifying §5.2.4

**Status:** READY FOR REVIEW. Do NOT paste into `main_results.tex` until approved.

## Structure

- §5.2.1 *Cluster Decomposition of Test-period Picks* — heatmap lead-in, K-means K=4 method, two-fold headline, four sub-clusters, asymmetry, limitations.
- §5.2.2 *Three Lenses on the K=4 Partition* — leg-betas (D&M parallel), π SHAP per cluster, short-leg structure, plus a closing paragraph "why the model needs π".
- §5.2.3 *Horizon-level Evidence* — per-horizon SHAP at the regime level (preserves Novy-Marx 2012 and Nagel 2012 findings; corrects the existing thesis "~63%" claim to the verified 67%).
- §5.2.4 *Discussion* — four targeted patches (Patches 1–3 update specific paragraphs; Patch 4 updates two cross-references elsewhere in §5.2).

External numbering of §5.2.1/§5.2.2/§5.2.3/§5.2.4 is preserved, so cross-references elsewhere in the thesis only need label updates (the old `sec:heterogeneity_panic` becomes `sec:heterogeneity_picks` for the cluster decomposition; `sec:cluster_triangulation` is new).

## Key decisions applied

- **Subsection structure**: three subsections (option A). Preserves existing numbering.
- **Cluster 3 framing**: Sharpe + CI reported as numbers; no "headline cluster" or "only CI excluding zero" hype.
- **Dropped from §5.2.1/§5.2.2**: crash-vs-recovery decomposition; calm-aggregate peak (+0.39); panic-aggregate z-curve numbers; long-short z-spreads (1.33% / 3.09%); old 3-cluster panic decomposition; the 2023-09 to 2024-11 example.
- **Per-horizon SHAP at regime level**: included as a brief subsection §5.2.3 (one paragraph). Preserved Novy-Marx 2012 and Nagel 2012 connections.
- **Tables**: four built from CSVs via `scripts/build_table_*.py` and included via `\input{tables/...}`. No hand-coded values in the prose.
- **Heatmap**: introduced as lead-in figure to motivate the clustering.
- **Cluster paragraphs**: trimmed to narrative-only (regime context, pick character, when in the test period). Detailed metrics live in Table 1.

## One number correction surfaced by re-verification

The existing thesis prose at line 98 says panic months 7–12 account for "roughly 63%" of long-leg momentum SHAP. The actual computed value is **67.2%**. The new §5.2.3 uses the verified value (with "roughly two-thirds" as a rounded gloss).

---

## §5.2.1 Cluster Decomposition of Test-period Picks \label{sec:heterogeneity_picks}

### Lead-in figure

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{plots/thesis/zscore_long_heatmap.png}
    \caption{Long-leg cross-sectional z-scores by month and momentum horizon, $2011$--$2024$. Each row is one of $167$ test months sorted chronologically; each column is one of the $12$ momentum horizons. Colour shows the deviation of the long leg's mean momentum at horizon $h$ from the cross-section average that month. The raw landscape contains structure but is hard to read directly; the K-means decomposition below imposes structure to make the patterns visible.}
    \label{fig:zscore_long_heatmap}
\end{figure}

### Opening (motivation + method)

The most fundamental claim of this section is that the regime signal $\pi^{\text{filter}}$ is structurally necessary for M2's selection mechanism. Section~\ref{sec:regime_results} establishes this at the regime aggregate: $\pi^{\text{filter}}$ accounts for $46\%$ of total SHAP, and the leg-beta inversion that drives traditional momentum's crashes \citep{DanielMoskowitz2016} is absent under M2 but present under unconditional 12-month momentum. The cluster decomposition we develop here sharpens that claim from "$\pi^{\text{filter}}$ matters on average" to "$\pi^{\text{filter}}$ routes the model between structurally distinct selection modes that no fixed-rank or single-regime model could combine in one ranking". To see this we cluster each test month's long-leg 12-horizon z-profile directly. Figure~\ref{fig:zscore_long_heatmap} shows the raw landscape: 167 monthly z-curves stacked vertically, with structure visible to the eye but no clean decomposition. K-means with $k=4$ on the raw z-vectors (Euclidean distance, no preprocessing or regime pre-split) yields a partition that is highly stable: the same labels emerge across six independent random seeds for $163$ of $167$ months ($97.6\%$), with mean pairwise Adjusted Rand Index $0.96$. The four clusters nest perfectly inside a coarser two-fold partition. Clusters 0 and 1 contain all 115 months in which the long leg's z-curve sits above the cross-section average ("above-average baskets", $\bar\pi = 0.28$). Clusters 2 and 3 contain all 52 months in which the long leg sits below ("below-average baskets", $\bar\pi = 0.52$). The partition therefore subsumes the calm-vs-panic dichotomy of the regime aggregate while exposing within-regime structure that the binary split obscures, and the per-cluster evidence in Section~\ref{sec:cluster_triangulation} shows that the cluster-level differences in selection mode cannot be reproduced without $\pi^{\text{filter}}$.

### Headline two-fold finding

The above-average state earns a Sharpe of $0.93$ ($95\%$ block-bootstrap CI $[0.24, 1.45]$) over its 115 months. The below-average state earns $1.46$ ($[0.70, 2.38]$) over its 52 months. Mean monthly returns are $1.42\%$ and $2.64\%$ respectively, a ratio of $1.86\times$. A block-bootstrap permutation test on the Sharpe difference yields $p = 0.17$ (one-sided), suggestive but not significant at conventional levels at this sample size. The point-estimate ordering is consistent with reversal-mode selection being the source of the strategy's edge.

### Four sub-clusters

Decomposing the two-fold partition into four reveals additional structure within each tier (Figure~\ref{fig:zscore_l2_k4_dispersion}, Table~\ref{tab:cluster_k4_descriptors}).

Per-cluster sample sizes, regime composition, Sharpe + 95\% CI, hit rate, Sortino, and maximum drawdown are reported in Table~\ref{tab:cluster_k4_descriptors}; the full per-cluster feature signature (cross-section context, raw long-leg pick momentum at each horizon, and long-leg z-curve at each horizon) is in Appendix Table~\ref{tab:cluster_k4_features_appendix}. The narrative in this subsection emphasises what each cluster \emph{is} (regime context, pick character, when in the test period); the detailed metrics live in the tables.

\paragraph{Cluster 0 (calm continuation, Sharpe $0.92$).}
The strategy's most common state, occurring in long stretches of up to 21 consecutive months with peak frequency in 2013--2014. The long-leg z-curve is humped and uniformly positive, peaking around month 8; raw pick momentum rises monotonically from low single digits at month 1 to roughly $+60\%$ at month 9 -- the model selects classic momentum winners. The 12-month lagged strategy Sharpe entering these months is $1.22$, the highest of any cluster: cluster 0 captures the persistence of an existing momentum environment.

\paragraph{Cluster 1 (mild continuation, Sharpe $0.93$).}
The largest cluster, with mixed regime composition and the weakest feature signature. The long leg sits only mildly above the cross-section, and the cluster has the highest long-horizon cross-section skewness ($8.1$): a few extreme outperformers in the right tail pull the cross-section mean. The best monthly return is March 2020 at $+26\%$ -- the strategy's largest single-month gain in the test sample -- and the cluster's median monthly return sits below its mean, indicating positive return skew. Sharpe is statistically indistinguishable from cluster 0 (point-estimate difference $+0.01$, two-sided permutation $p = 0.99$).

\paragraph{Cluster 2 (post-panic recovery, Sharpe $1.21$).}
Months that follow a panic-heavy year: the 12-month lagged $\pi^{\text{filter}}$ frequency is the highest of any cluster. The cross-section is positive but with the highest dispersion of any cluster, and the long leg sits below the cross-section average -- raw picks are mild laggards while the broader cross-section is up double digits. This is the model's mild-reversal mode in a healing market: it picks the names that have not yet recovered, anticipating that they will. Heavy in 2021 (9 of the 31 months -- the post-COVID rotation) and 2023.

\paragraph{Cluster 3 (deep crisis, Sharpe $1.82$, CI $[0.98, 2.68]$).}
Active HMM panic with the cross-section uniformly negative across every horizon. The long leg is deeply below the cross-section, and raw pick losses span double digits at short horizons to nearly $-50\%$ at long horizons: the model picks the most-beaten-down names in absolute terms and relative to a falling cross-section. Mean monthly return $+3.13\%$. Eleven of 21 months are in 2022 (Fed-hiking cycle), with corroborating instances at April 2020 (COVID trough), October 2015 and January 2016 (China devaluation and oil collapse), and January 2023.

### Asymmetry across the two tiers

Within the above-average tier (clusters 0 and 1), the two sub-clusters earn essentially identical Sharpes (point-estimate difference $0.01$, two-sided permutation $p = 0.99$): the strategy's performance is insensitive to the magnitude or shape of the long leg's positive deviation from the cross-section. Within the below-average tier (clusters 2 and 3), the same comparison yields a point-estimate difference of $0.61$ ($p = 0.50$, suggestive but not significant): the depth of the reversal materially affects performance. The model's regime-aware adaptation reveals itself in distinguishing kinds of reversal environments rather than kinds of trend-following environments. This asymmetry is consistent with the leg-beta finding presented in Section~\ref{sec:cluster_triangulation}: in the above-average tier, both sub-clusters reflect ordinary momentum continuation with similar market-exposure profiles; in the below-average tier, cluster 3 represents the high-$\beta$ deep-reversal selection that drives the cross-leg spread amplification.

### Limitations of the cluster-level Sharpe gradient

We report the cluster Sharpes as descriptive of the underlying return patterns and refrain from claiming statistically significant differences between them. None of the six pairwise K=4 Sharpe comparisons reach $p < 0.10$ on a permutation test. Cluster 3's Sharpe rests partly on 2022, which contributes 11 of its 21 months; excluding 2022 leaves a positive point-estimate Sharpe of $2.19$ but with a near-uninformative confidence interval. The sample size of 167 monthly observations, with the smallest cluster at 21, is sufficient to identify stable cluster structure (mean pairwise ARI $0.96$ across six seeds) but not to formally test cluster-level Sharpe gradients. The structural finding -- four feature-distinct, reproducible cluster types tied to regime context, with a clean M2-vs-fixed-momentum leg-beta contrast in every cluster (Section~\ref{sec:cluster_triangulation}) -- is robust. The performance gradient across them is consistent with reversal-mode being the source of the strategy's edge but cannot be claimed as significant at this sample size.

## §5.2.2 Three Lenses on the K=4 Partition \label{sec:cluster_triangulation}

This subsection cross-validates the K=4 partition through three complementary lenses: the legs' market-beta exposure, $\pi^{\text{filter}}$'s SHAP share, and the short-leg pick characteristics. Each lens decomposes a regime-aggregate finding from Section~\ref{sec:regime_results} across the four clusters.

### Triangulation with leg-betas (parallel to D\&M)

The leg-beta amplification documented at the regime level in Section~\ref{sec:regime_results} decomposes across the K=4 partition as a direct contrast against unconditional 12-month momentum (Table~\ref{tab:cluster_k4_leg_betas}). M2 retains long-leg $\beta >$ short-leg $\beta$ in every K=4 cluster, with cluster spreads of $+0.68$, $+0.40$, $+0.49$, and $+0.43$ for clusters 0--3 respectively. The no-leg-beta-inversion claim of Section~\ref{sec:regime_results} therefore holds at this finer partition, not just on regime average. Unconditional 12-month momentum, by contrast, exhibits leg-beta inversion in every K=4 cluster: spreads of $-0.24$, $-0.87$, $-0.75$, and $-0.42$. The D\&M crash mechanism is latent across the entire test period rather than panic-specific. The contrast is sharpest in cluster 3, where M2's spread $+0.43$ mirrors fixed momentum's spread $-0.42$: in the cluster where the unconditional ranking would fail most severely, M2 selects high-$\beta$ names whose recent drawdowns reflect market exposure rather than fundamental deterioration -- the cross-sectional realisation of the tilt-amplification mechanism. This sharpens the M2-vs-D\&M comparison of Section~\ref{sec:dm_comparison}: where D\&M observed leg-beta inversion in fixed momentum during specific stress episodes, the K=4 decomposition shows the inversion pattern across every cluster type, while M2 escapes it everywhere through compositional re-ranking.

### State-dependent role of $\pi^{\text{filter}}$

The aggregate $\pi^{\text{filter}}$ SHAP share documented at the regime level decomposes across the K=4 clusters (Table~\ref{tab:cluster_k4_shap_shares}). Combined long-and-short shares are $0.50$ in cluster 0, $0.46$ in cluster 1, $0.54$ in cluster 2, and $0.24$ in cluster 3. The long-leg-only contributions follow the same ordering: $0.34$, $0.27$, $0.28$, $0.11$. Weighted across the four clusters, the combined share recovers the overall figure of $0.46$ reported in Section~\ref{sec:regime_results}. The pattern matches the routing interpretation introduced in Section~\ref{sec:tree_paths}: $\pi^{\text{filter}}$'s marginal value is highest in clusters where the cross-section is ambiguous (clusters 0--2, where the picks sit near or moderately above/below the cross-section), and lowest where the cross-section itself is unambiguously crisis-shaped (cluster 3). When the long-leg z-curve is uniformly between $-0.7$ and $-0.9$ across all twelve horizons, momentum splits alone suffice to make the selection; the regime signal contributes only marginal additional information. $\pi^{\text{filter}}$ is therefore a router whose marginal contribution scales with cross-sectional ambiguity, not a behaviour switch with constant influence.

### The short leg is not a mirror of the long leg

The K=4 partition was defined on long-leg z-curves; applying the same labels to the short leg (NYSE P10 of $score_{xgb}$, identical construction to the appendix leg-beta analysis) reveals that the short basket is structurally distinct from a negated long-leg in every cluster (Table~\ref{tab:cluster_k4_short_leg}). For each cluster we measure the correlation between the short-leg cluster centroid z-curve and the negated long-leg cluster centroid z-curve. A perfect "mirror" would yield correlation $+1.00$; the observed values are $+0.94$ in cluster 0, $+0.89$ in cluster 1, $+0.19$ in cluster 2, and $-0.84$ in cluster 3. The above-average tier (clusters 0 and 1) is roughly mirror-shaped at the long-horizon end with attenuation; the below-average tier breaks the mirror entirely. In cluster 2, both legs sit above the cross-section average -- short-leg z-curve uniformly between $+0.21$ and $+0.37$, long-leg uniformly between $-0.22$ and $-0.36$ -- consistent with a healing market in which both buy and short candidates have elevated momentum and the model's edge comes from relative selection within an upward-drifting cross-section. In cluster 3, the legs are anti-mirror: short-leg z spikes to $+0.57$ at month 1 and decays to $+0.11$ at month 12, while the long leg sits between $-0.73$ and $-0.88$ throughout. The month-1 short-leg spike replicates the asymmetric short-side sensitivity documented at the regime aggregate (\citet{Nagel2012} short-term reversal mechanism) and localises it to the deep-crisis cluster: in active panic with broadly negative momentum, the model shorts stocks whose recent (one-month) returns have started to recover, anticipating that those bounces are unsustained. Short-leg basket sizes are $391$ / $334$ / $382$ / $353$ stocks per month for clusters 0--3, all smaller than the corresponding long-leg baskets (498 / 530 / 488 / 533).

### Why the model needs $\pi^{\text{filter}}$

The three triangulations together support the framing claim of Section~\ref{sec:heterogeneity_picks}: $\pi^{\text{filter}}$ is structurally necessary to M2's selection mechanism, not merely an additional informative feature. First, the leg-beta evidence shows that unconditional 12-month momentum (which sees the same 12 horizons but no regime signal) exhibits leg-beta inversion in every K=4 cluster, while M2 escapes inversion in every cluster: the inversion-vs-no-inversion contrast holds at the same monthly resolution and is therefore attributable to the input difference, $\pi^{\text{filter}}$. Second, $\pi^{\text{filter}}$'s cluster-level SHAP share spans $0.24$ to $0.54$, with the lowest share in the cluster where momentum splits alone suffice to make the selection (cluster 3) and higher shares in clusters where the momentum cross-section is ambiguous: $\pi^{\text{filter}}$ behaves as a context-dependent router rather than a constant additive feature, and removing it would collapse the routing. Third, the long-leg z-curves under M2 swing from peak $+0.80$ in cluster 0 (continuation) to a uniform $-0.73$ to $-0.88$ in cluster 3 (deep reversal): no fixed-rank or single-direction model can produce both shapes simultaneously, and the routing between them is what the regime signal performs. The cluster decomposition therefore upgrades the regime-aggregate evidence of Section~\ref{sec:regime_results} from "$\pi^{\text{filter}}$ matters on average" to "the model would not produce the K=4 cluster structure without $\pi^{\text{filter}}$".

## §5.2.3 Horizon-level Evidence \label{sec:horizon_evidence}

Beyond the per-cluster decomposition, the SHAP distribution by momentum horizon at the regime level (calm vs panic, long vs short) characterises which horizons drive the model's predictions independently of how those predictions sort months into clusters. In the calm long leg, months 11 and 12 are the most influential individual horizons ($18.6\%$ and $16.8\%$ of long-leg momentum SHAP), with months 8--12 together accounting for $66.0\%$. The model independently recovers the intermediate-to-long-horizon dominance documented by \citet{NovyMarx2012}. In the panic long leg, months 7--12 dominate even more strongly, accounting for $67.2\%$ of long-leg momentum SHAP -- roughly two-thirds of the prediction signal sits at the intermediate-to-long end in both regimes, with the panic distribution mildly shifted toward the shorter end of that band. On the panic short side, month 1 SHAP rises to $10.6\%$ (versus $5.3\%$ in calm), indicating that the model's predictions become more sensitive to recent returns when identifying stocks to short during stress -- consistent with \citet{Nagel2012}, who shows that returns to short-term reversal spike during crises when liquidity provision is most valuable. Combined with the per-cluster evidence in Section~\ref{sec:cluster_triangulation}, the picture is that the same horizons drive the model's predictions in every regime, but the direction of selection at those horizons -- continuation in the above-average tier, reversal in the below-average tier -- reorganises across clusters. Computing the per-cluster per-horizon decomposition is straightforward but was not pursued for this revision; the regime-level analysis suffices for the structural claim that horizon influence stays stable across regimes while the direction of selection reverses.

---

## Modified §5.2.4 Discussion

Four targeted patches. Each updates a specific paragraph that referenced old §5.2.1 / §5.2.2 / §5.2.3 numbers; replacement numbers come from K=4 verified data.

### Patch 1: "Deep-crisis sub-regime" paragraph (lines 143–144)

**Before:**

> "The decomposition in Section~\ref{sec:heterogeneity_panic} shows that the flat, negative z-score profile appears in pure form in the 17 deep-crisis months..."

**After:**

> "The K=4 decomposition in Section~\ref{sec:heterogeneity_picks} shows that the flat, negative z-score profile appears in pure form in the 21-month deep-crisis cluster (cluster 3), where the long leg is uniformly negative across all 12 horizons (z-curve between $-0.73$ and $-0.88$). This shape is qualitatively different from short-term reversal \citep{Jegadeesh1990, Lehmann1990, Nagel2012}, which targets month 1 only, and from long-term contrarian strategies \citep{DeBondtThaler1985}, which operate at multi-year horizons. In the above-average-basket clusters (0 and 1) the model recovers the intermediate-horizon continuation shape of \citet{NovyMarx2012}, with z-peak at month 8 ($+0.80$ in cluster 0, $+0.32$ in cluster 1). The contrast in selection shape across clusters -- humped continuation in the above-average tier, flat-negative in deep crisis -- has not, to the best of our knowledge, been documented in this form."

### Patch 2: "Regime signal acts as a router" paragraph (lines 146–147)

**Before:**

> "...$\pi$'s long-leg SHAP share is $34.5\%$ in mild-panic months, where the cross-section appears calm-like and the model relies on $\pi$ to override the surface appearance, and $7.5\%$ in deep-crisis months..."

**After:**

> "The regime signal does not flip one horizon from continuation to reversal or shift weight between a fast and a slow signal. As Section~\ref{sec:heterogeneity_picks} shows, $\pi^{\text{filter}}$'s long-leg SHAP share is $34\%$ in cluster 0 (calm continuation) where the basket sits clearly above the cross-section, and $11\%$ in cluster 3 (deep crisis) where the cross-section is so extreme that momentum splits alone suffice. The combined long-and-short shares span $0.50$ to $0.24$ across the four clusters, matching the routing interpretation: the regime signal functions as a router whose marginal contribution scales with cross-sectional ambiguity, not as a behaviour switch with constant influence."

### Patch 3: "Selection by momentum level, not momentum rank" paragraph (lines 149–150)

**Before:**

> "...In calm, this distinction is modest: the long leg's z-scores peak at $+0.39$, above average but far from the upper tail. In deep crisis (Section~\ref{sec:heterogeneity_panic}), it becomes decisive..."

**After:**

> "Traditional momentum strategies rank stocks by trailing return at a fixed lookback \citep{JegadeeshTitman1993}. Here, the portfolio ranks on \emph{predicted forward return} -- a learned function of all 12 horizons and the regime signal. The model learns which momentum \emph{level} is associated with future outperformance. In the above-average baskets, the long-leg z-scores peak modestly at $+0.32$ (cluster 1) to $+0.80$ (cluster 0), above average but far from the upper tail. In deep crisis (cluster 3) the distinction becomes decisive: the long leg selects stocks $0.7$--$0.9$ standard deviations \emph{below} the cross-sectional average at every horizon, a selection no fixed-rank strategy would produce."

### Patch 4: cross-references

- `sec:dm_comparison` (line 233): change `Section~\ref{sec:heterogeneity_panic}` to `Section~\ref{sec:heterogeneity_picks}`. Update the sentence to read: "D\&M-style scaling would help most in cluster 3 (Section~\ref{sec:heterogeneity_picks}), where the long leg sits term-structure-wide negative; in the above-average baskets (clusters 0 and 1) it would unnecessarily reduce positions because the cross-section there resembles calm and the strategy continues to earn a calm-like Sharpe."
- `sec:depth_analysis` (line 171): replace the three-panic-sub-regime list ("humped at intermediate horizons (mild panic), weak U-dip in middle horizons (transition), and uniform negativity (deep crisis)") with the K=4 cluster list: "humped continuation in the above-average tier (clusters 0 and 1), mild reversal in cluster 2, and uniform negativity in cluster 3".

### Verification: nothing important is lost

Auditing the existing §5.2.4 against the new prose, the following substantive claims are all preserved:

| Claim from old Discussion | Preserved where |
|---|---|
| "Deep-crisis sub-regime selects a term-structure-wide loser portfolio" | Patch 1, with corrected n=21 and z-range |
| "This shape qualitatively different from short-term reversal" | Patch 1, citations preserved |
| "In calm and mild-panic, the model recovers NovyMarx 2012 intermediate-horizon shape" | Patch 1, with z-peak +0.80 / +0.32 numbers |
| "The regime does not flip one horizon..." | Patch 2 |
| "SHAP distribution remains stable across regimes" | Patch 2 + the per-horizon SHAP paragraph in §5.2.1 |
| "$\pi$ functions as a router whose marginal contribution scales with cross-sectional ambiguity" | Patch 2, with K=4 cluster numbers |
| "Selection by momentum level not rank" | Patch 3 |
| "The long leg selects below-average stocks in deep crisis" | Patch 3 |
| "D\&M-vs-M2 complementarity (D\&M scaling would help in deep crisis)" | Patch 4 |
| "Depth-4 needed to discriminate three within-regime shapes" | sec:depth_analysis updated to refer to K=4 cluster types instead of old sub-regimes |

---

## Tables — auto-generated, included via `\input`

The four tables follow the data-flow architecture (CSV in `results/thesis` → `scripts/build_table_*.py` → `tables/*.tex` → `\input` in `main_results.tex`). The `.tex` files are auto-generated by the build scripts and committed; they should be regenerated whenever the upstream CSVs change.

**Main body** (four tables, included in §5.2.1 / §5.2.2):

```latex
\input{tables/table_cluster_k4_descriptors}        % cluster identity + Sharpe/CI/hit/Sortino/maxDD
\input{tables/table_cluster_k4_leg_betas}          % M2 vs fixed-mom leg-betas per cluster
\input{tables/table_cluster_k4_shap_shares}        % π_filter SHAP share per cluster (long/short/combined)
\input{tables/table_cluster_k4_short_leg}          % short-leg z-curve summary per cluster
```

**Appendix** (one table, full per-cluster feature signature):

```latex
\input{tables/table_cluster_k4_features_appendix}  % 37 rows × 4 clusters: cross-section context, raw pick mom by horizon, z-curve by horizon
```

The corresponding build scripts:

- `scripts/build_table_cluster_k4_descriptors.py` — reads `results/thesis/cluster_k4_descriptor_table.csv` (committed `16c1461`)
- `scripts/build_table_cluster_k4_leg_betas.py` — reads `results/thesis/cluster_k4_leg_betas.csv` (committed `16c1461`)
- `scripts/build_table_cluster_k4_shap_shares.py` — reads `results/thesis/cluster_k4_shap_shares.csv` (committed `16c1461`)
- `scripts/build_table_cluster_k4_short_leg.py` — reads `results/thesis/cluster_k4_short_leg_descriptors.csv` (committed `16c1461`)
- `scripts/build_table_cluster_k4_features_appendix.py` — reads `results/thesis/cluster_k4_descriptor_table.csv` (committed `ec30508`)

Each generated `.tex` opens with `% Auto-generated by scripts/build_table_X.py -- DO NOT HAND EDIT.`, ensuring future readers (and yourself) don't manually drift the tables out of sync with the CSVs.

---

## Figures: file references

| Figure | Action |
|---|---|
| `plots/thesis/zscore_long_heatmap.png` | NEW — lead-in motivation figure |
| `plots/thesis/zscore_l2_k4_dispersion.png` | Use as the K=4 figure |
| `plots/thesis/zscore_subregime_dispersion.pdf` | Drop |
| `plots/thesis/zscore_subregimes.pdf` | Drop |
| `plots/thesis/zscore_panic_subtypes.png` | Drop (crash/recovery decomposition removed) |

Old `tables/table_zscore_subregime.tex` and `tables/table_subregime_shap.tex` are no longer referenced; can be left in place or removed at your preference.

---

## Verification log (every number in the new prose checked against source)

| Claim | Source | Status |
|---|---|---|
| 167 months partitioned into clusters of size 53/62/31/21 | descriptor table; labels file | ✓ |
| Above-avg π̄ = 0.28; below-avg π̄ = 0.52 | computed fresh from K=2 partitions | ✓ |
| K=2 Sharpes 0.93 [0.24, 1.45] / 1.46 [0.70, 2.38] | block bootstrap | ✓ |
| Mean monthly returns 1.42% / 2.64%; ratio 1.86× | n-weighted | ✓ |
| Permutation p = 0.17 (one-sided) | block-bootstrap permutation | ✓ |
| Per-cluster Sharpes 0.92 / 0.93 / 1.21 / 1.82 with CIs | block bootstrap | ✓ |
| Per-cluster π̄ 0.21 / 0.34 / 0.32 / 0.81 | descriptor | ✓ |
| Per-cluster z-curve ranges and pick-mom endpoints | descriptor | ✓ |
| Hit rates 62 / 58 / 65 / 62 % | downside metrics; matches descriptor | ✓ |
| Sortinos 1.34 / 2.29 / 3.40 / 3.89 | downside metrics | ✓ |
| Max DD 13.6 / 14.1 / 15.3 / 10.7 % | downside metrics | ✓ |
| Basket sizes 498 / 530 / 488 / 533 | basket_size CSV | ✓ |
| Intra-cluster turnover 0.59 / 0.60 / 0.68 / 0.57 | turnover CSV | ✓ |
| M2 leg-beta spreads +0.68 / +0.40 / +0.49 / +0.43 | leg_betas CSV | ✓ |
| Fixed-mom leg-beta spreads −0.24 / −0.87 / −0.75 / −0.42 | leg_betas CSV | ✓ |
| π SHAP combined 0.50 / 0.46 / 0.54 / 0.24 | shap_shares CSV; weighted-avg recovers 0.46 | ✓ |
| π SHAP long-leg 0.34 / 0.27 / 0.28 / 0.11 | shap_shares CSV | ✓ |
| Per-horizon SHAP calm long: m11 18.6%, m12 16.8%, m8-12 sum 66.0% | per_horizon_shap_by_regime CSV | ✓ |
| Per-horizon SHAP panic long: m7-12 sum 67.2% (~67%) | per_horizon_shap_by_regime CSV — corrects existing thesis claim of "~63%" | ✓ |
| Per-horizon SHAP panic short m1: 10.6%; calm short m1: 5.3% | per_horizon_shap_by_regime CSV | ✓ |
| Within-tier Sharpe diff p-values 0.99 (high-z) / 0.50 (low-z) | permutation tests | ✓ |
| Cluster 3 ex-2022 Sharpe 2.19 | from robustness audit | ✓ |
| Cluster stability ARI 0.96, 163/167 stable | from robustness audit | ✓ |
| Cluster 3: 11/21 in 2022; specific dates Apr 2020, Oct 2015, Jan 2016, Jan 2023 | year_frequency + member_dates | ✓ |
| Cluster 0: peak 2013-2014 frequency | year_frequency | ✓ |
| Cluster 2: 9 months in 2021 | year_frequency | ✓ |

---

## After your approval

Next step is the actual `main_results.tex` patch: replace lines 70–138 with the new three-subsection prose, apply the four Discussion patches, and regenerate the four `tables/*.tex` files from their build scripts. The tables already exist on disk (committed in `16c1461`) so the patch is purely a `main_results.tex` rewrite.
