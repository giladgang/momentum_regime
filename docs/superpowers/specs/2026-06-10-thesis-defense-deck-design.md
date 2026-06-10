# Thesis Defense Presentation — Design Spec

**Date:** 2026-06-10
**Author:** Gilad Gang
**Thesis:** *Momentum Across Market Regimes — A Machine Learning Approach* (MSc QFAS, Tilburg)
**Supervisor:** Denis Kojevnikov

## 1. Goal

Produce the slide deck for the thesis oral defense (open session), built to the Tilburg
*Master Thesis Guidelines (2019)*. The deck presents the thesis at an intuition level for a
mixed faculty/student audience, with a deeper backup deck held in reserve for the closed Q&A.

## 2. Constraints (from the Tilburg guidelines)

- Open session ~10–15 min, then a closed ~30 min Q&A.
- Audience has only a general background → avoid detail, prefer figures/diagrams over tables.
- Max ~15 lines per slide; title slide + a sense of outline; the closing slides summarize the
  most significant results plus conclusions/recommendations.
- "Optional" slides may be set aside to answer questions → realized here as the backup deck.
- Present tense; precise, plain language; APT (Audience / Purpose / Timing).

## 3. Locked decisions

| Decision | Choice |
|---|---|
| Output format | **PowerPoint (`.pptx`)**, generated from a fresh Python (`python-pptx`) build script |
| Story emphasis | **Mechanism-forward** (matches the thesis's stated aim: understand the anomaly) |
| Main deck length | **11 slides** (~13–14 min) |
| Backup deck | **12 reference slides** (B1–B12), shown only if asked |
| Speaker notes | **Yes** — a notes block on every main slide (Presenter view) |
| Visual style | Clean-minimal; Tilburg logo on the title slide |
| Gadi files | **Ignored entirely** (`build_deck.py`, `momentum_across_regimes.pptx` at repo root) |

## 4. Deliverable & tooling

- New directory **`presentation/`** at repo root, containing:
  - `build_defense_deck.py` — standalone `python-pptx` builder (no dependency on the Gadi script).
  - `momentum_defense.pptx` — the generated 16:9 deck (main + backup slides).
- Figures are pulled from `plots/thesis/`. **PDF-only assets must be rasterized to PNG** for
  embedding (PowerPoint cannot embed PDF): at minimum `tilburg_logo.pdf`; for backup slides also
  `gibbs_sampling_diagram.pdf`, `risk_aversion_dual_util.pdf`, `factor_residual_acf_ff6.pdf`.
  Rasterize to a `presentation/assets/` folder; do not overwrite the originals.
- Tables shown in the deck are **rebuilt as native pptx tables** from canonical numbers (we cannot
  `\input` LaTeX). Every number is sourced in §8.
- **Scope note:** this is a one-off presentation artifact, not part of the analysis pipeline. It is
  NOT thesis-worthy in the `run_pipeline.py` / `RESULTS_LOG.md` sense and does not need a pipeline
  step or pytest. It only consumes already-published numbers and figures.

## 5. Visual style

- 16:9 widescreen (13.333 × 7.5 in). Sans-serif (Arial/Calibri). Generous whitespace.
- Restrained palette: near-black text, one accent (Tilburg navy), green/red only to signal
  good/bad outcomes (e.g., XGB vs fixed momentum). No gradients or clipart.
- Consistent slide furniture: short title top-left; one dominant figure or one compact table per
  slide; a single takeaway line at the bottom. Honor the ~15-line cap.

## 6. Main deck — slide by slide

Each slide lists: **purpose · on-slide elements · figure/asset · key numbers (source in §8) ·
speaker-note gist.**

1. **Title**
   - Title, author, supervisor, "MSc Quantitative Finance and Actuarial Science", Tilburg, date.
   - Asset: `tilburg_logo.pdf` → PNG. No numbers.
   - Notes: one-sentence framing of the talk.

2. **Motivation & research question**
   - Momentum is the most profitable anomaly *and* the most fragile (crashes >50% at regime turns).
   - The question: *how does the cross-section of momentum reorganize between calm and panic, and
     what selection rule drives it?*
   - Asset: text + 2–3 big stat callouts (no heavy figure). Numbers: fixed-momentum MDD −72.2%.
   - Notes: set up the puzzle; state aim is understanding, not beating the market.

3. **Literature & the gap**
   - Three buckets → the gap. (1) Momentum & crashes (JT93; D&M16). (2) Crash management adjusts
     *exposure/horizon* (Barroso–Santa-Clara; GHM23). (3) Regime-dependence noted but coarse
     (Cooper04) + ML cross-section (Gu20; Beckmeyer–Wiedemann25).
   - **Gap:** no one studies how the *full momentum term structure* reorganizes across regimes, or
     whether capturing it requires nonlinear regime-conditioned selection.
   - Asset: a simple 3-column "prior work → gap" layout (pptx shapes). No numbers.
   - Notes: position the contribution against each bucket.

4. **Methodology I — Stage 1: the HMM regime classifier** *(Chapter 3)*
   - 2-state Bayesian HMM on four monthly stress features (DD, DISP, REL_N, CS) → a continuous
     panic probability π that updates each month. Estimated by Gibbs sampling (kept light here).
   - Asset: small architecture schematic (`markov_chain_diagram.png`) + `regime_probabilities.png`
     (π spikes at dot-com / GFC / COVID / 2022). Numbers: train 1990–2010, test 2011–2024.
   - Notes: π is a *context variable*, not a return predictor; Gibbs detail in backup B1.

5. **Methodology II — Stage 2: cross-sectional selection** *(Chapter 3)*
   - Inputs = 12 momentum horizons + π. Rank stocks on **predicted forward return**, not raw
     momentum → the long leg need not hold high-momentum names. Top/bottom decile, value-weighted,
     long–short. Three methods: DET (formula), LR (linear), **XGB** (nonlinear; π *routes* the term
     structure). A small pipeline strip (HMM → π → XGB → long–short) anchors the two method slides.
   - Asset: pipeline schematic (pptx shapes). Numbers: optional preview of DET 0.11 / LR −0.01 / XGB 1.11.
   - Notes: why nonlinear; XGB specifics (depth-4, 50-seed ensemble) live in backup B12.

6. **Headline results**
   - XGB Sharpe **1.11**, six-factor α **24.1%** (t = **4.81**). Fixed 12-mo momentum Sharpe 0.06,
     MDD −72.2%. Linear baseline collapses; removing π cuts the Sharpe by >60% (→ 0.43).
   - Asset: `cs_performance_regime_shaded.png` (cumulative wealth, panic shading) + a 3-row compact
     table (Market / Fixed-12mo / XGB).
   - Notes: this earns credibility before the mechanism.

7. **Mechanism I — the direction flip**
   - In calm, the long-leg picks sit **above** the cross-sectional mean (winners). In panic, they
     sit **below** at *every* horizon (the beaten-down). "Buy the dip" emerges only after
     conditioning on π.
   - Asset: `zscore_long_heatmap.png`. Numbers: z-mean calm ≈ +0.3..+0.8 vs panic −0.7..−0.9.
   - Notes: read the heatmap top-to-bottom; foreshadow the four modes.

8. **Mechanism II — four modes = one market cycle**
   - The heatmap clusters into four selection modes tracing a single cycle: calm continuation →
     transition → post-panic recovery → deep crisis, with per-cluster Sharpes.
   - Asset: 2×2 of `zscore_l2_k4_panel_c0..c3.png` + cluster Sharpes (0.92 / 0.93 / 1.21 / 1.82).
   - Notes: "momentum where it works, reversal where it crashes."

9. **Why it's novel**
   - π is a **routing variable** (46% of SHAP, ~6× its 1/13 share). Nonlinearity is the binding
     constraint (DET 0.11, LR −0.01, XGB 1.11). **Composition, not exposure**: long-leg β > short-leg
     β in *every* regime — unlike D&M/Barroso exposure scaling.
   - Asset: compact leg-β table (long 1.40→1.65, short 1.22→0.98) + the SHAP 46% callout.
   - Notes: contrast explicitly with the three literature buckets.

10. **Robustness**
    - Survives the 2009 momentum crash: **+20.5%** vs **−62.6%** for unconditional momentum.
      Replicates in UK (Sharpe 0.72 vs 0.35) and Japan (0.57 vs −0.01, where baseline momentum is
      absent). Alpha survives all five factor models.
    - Asset: compact international table + a 2009 callout.
    - Notes: generalization beyond the single sample/market.

11. **Conclusions & future work**
    - Two contributions (novel filter+model combination; concrete stock-level mechanism).
      Limitation: reversal-failure in sustained bears (dot-com −43.8% cumulative, −64.8% MDD).
      Future: richer regime signal (forecast/trajectory), neural nets, a D&M exposure overlay.
    - Asset: text, three tight columns. Notes: end on the contribution + one future direction.

## 7. Backup deck (B1–B12, for the closed Q&A)

B1 HMM spec + Gibbs/FFBS (`gibbs_sampling_diagram.pdf`→PNG) · B2 convergence (ESS, R̂=1.001) ·
B3 four-pass feature selection (DD+DISP+REL_N+CS) · B4 factor-alpha table (CAPM→FF6, Newey–West,
Ljung–Box; `factor_residual_acf_ff6.pdf`→PNG) · B5 risk-aversion sweep (`risk_aversion_dual_util.pdf`→PNG) ·
B6 25-yr expanding window + reversal-failure / D&M complement · B7 SHAP per-horizon by leg
(`shap_per_horizon_by_leg.png`) · B8 leg-betas by regime & per-cluster · B9 international feature
selection · B10 K=2 vs 3/4/5 + Student-t emissions · B11 transaction costs / turnover ·
B12 XGB hyperparameters (depth-4, lr 0.05, 500 trees, 50-seed ensemble).

## 8. Data & figure sourcing + verification

Every number in the deck must trace to a canonical source; no inventing figures. Primary sources:
`results/PRODUCTION_METRICS.json`, and the LaTeX tables under `tables/` (which the thesis `\input`s):

- Performance (Sharpe 1.11; market 0.84/11.9%; fixed-12mo 0.06/−72.2%; DET 0.11; LR −0.01) →
  `tables/table_performance.tex`.
- Regime Sharpe (panic 1.53, calm 0.84) → `tables/table_regime_sharpe.tex`.
- Six-factor α 24.1%, t 4.81; CAPM α 23.4% → `tables/table_factor_alphas.tex` / Appendix.
- Cluster Sharpes (0.92/0.93/1.21/1.82) + π means → `tables/table_cluster_k4_descriptors.tex`.
- Leg betas (long 1.40→1.65, short 1.22→0.98) → `tables/table_leg_betas.tex`.
- International (UK 0.72 vs 0.35; JP 0.57 vs −0.01; MDDs) → `tables/table_international_results.tex`.
- SHAP π share 46% → `tables/table_shap.tex`.
- 2009 / dot-com / GFC numbers → `main_results.tex` §Historical Stress.

**Verification gate (before "done"):** cross-check every deck number against these sources; the
deck must use the thesis's *canonical* numbers (the Gadi deck's slightly older cut — e.g. α 24.7%,
panic Sharpe 1.56 — must NOT be reused). Confirm each embedded figure renders and is the intended one.

## 9. Out of scope

- No edits to the thesis `.tex`, the analysis, or any results.
- No re-running of models; the deck only consumes published outputs.
- No pipeline registration / pytest (not a production artifact).

## 10. Open assumptions (flag if wrong)

- Branding: clean-minimal + Tilburg logo (no mandated faculty template).
- Talk length ≈ 13–14 min for 11 slides is acceptable within the 10–15 min open session.
- Speaker notes delivered inside the `.pptx` notes pane (Presenter view), not a separate document.
