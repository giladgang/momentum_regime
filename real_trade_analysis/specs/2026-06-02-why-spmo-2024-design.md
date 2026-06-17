# Research design — Why did SPMO outperform the market after 2024?

*Exploratory (real_trade_analysis). Production untouched. Date: 2026-06-02.*

## 1. Research question

Why did the SPMO replica's excess return over the cap-weighted S&P 500 switch from
approximately zero over 2011-2023 (+0.3%/yr average) to large over 2024-2025
(+13.7%/yr average)? Decompose the mechanism, assess in-sample durability, and frame
the result for the thesis.

Three-stage sequence: **(1) mechanism decomposition -> (2) durability -> (3) thesis framing.**

## 2. Hypotheses (non-exclusive, each falsifiable)

- **H1 Factor.** The post-2024 excess is the momentum factor (UMD) reasserting.
  *Prediction:* regression alpha does NOT jump post-2024; excess tracks UMD; the
  ladder's momentum-selection rung captures most of the excess.
- **H2 Concentration / weighting.** The excess comes from SPMO's float-cap x momentum-
  score weighting overweighting a few soaring mega-caps relative to the index.
  *Prediction:* the cap-x-score WEIGHTING rung of the ladder adds the most post-2024;
  equal-weighted momentum selection captures far less; name attribution is concentrated
  in NVDA / AVGO / META.
- **H3 Regime.** The outperformance is conditional on narrow mega-cap leadership /
  high cross-sectional dispersion. *Prediction:* excess concentrates in the high-
  leadership / high-dispersion months; near zero elsewhere.

These are not mutually exclusive; the decomposition quantifies each one's contribution.

## 3. Data and definitions

- **Universe:** Top-500 by market cap (S&P 500 proxy), eligible common stock
  (shrcd 10/11 or CIZ common; major exchange; price > $1), monthly, 2011 - Dec-2025.
- **Data source:** real CRSP, legacy panel spliced with CRSP v2 (msf_v2, CIZ) through
  Dec-2025. Already built: `real_trade_analysis/data/_ext_crsp_real.parquet`.
- **SPMO replica:** top-quintile risk-adjusted 12-1 momentum, float-cap x momentum-
  score weighting, 9% cap, semi-annual (Mar/Sep) rebalance with drift. Validated
  against the real SPMO ETF (2015-2024) to within ~1 Sharpe pt.
- **Benchmark ("market"):** cap-weighted Top-500.
- **Excess:** SPMO return - market return, monthly.
- **Factors:** Fama-French 6 + UMD (FF parquet, decimal units, runs to Feb-2026).
- **Breadth / leadership proxy:** cap-weighted minus equal-weighted Top-500 return
  (positive = a few big stocks led).
- **Split:** pre-period 2011-2023, post-period 2024-2025.

## 4. Stage 1 — Mechanism decomposition (hybrid, three converging lenses)

### 1a. Counterfactual portfolio ladder (constructive)
Four rungs, each re-run and split pre/post-2024; report each rung's annualized excess
over the market and the MARGINAL excess added by each step:
1. Cap-weighted market (baseline, excess = 0 by construction).
2. + momentum SELECTION: top-quintile by risk-adj momentum, equal-weighted.
3. + cap x score WEIGHTING: float-cap x momentum-score (no cap yet).
4. + MECHANICS: 9% cap, semi-annual rebalance (= full SPMO replica).

The marginal post-2024 jump at each rung isolates selection vs weighting vs mechanics.
Co-linearity caveat: also run the reverse order (weighting-first) as a robustness check
and report both Shapley-style orderings if they disagree materially.

### 1b. Factor + idiosyncratic regression
Regress monthly SPMO excess on [Mkt-RF, SMB, HML, UMD, breadth], NW(6) errors:
- full sample, pre-period, post-period (separately);
- 36-month rolling alpha and UMD loading (time series);
- formal test of an alpha jump: pooled regression with a post-2024 dummy interacted
  with the constant (and optionally with UMD), report the dummy coefficient + t.

H1 is supported if the post-2024 excess is absorbed by loadings (UMD) with no alpha
jump; H2/idiosyncratic is supported if the ALPHA jumps post-2024.

### 1c. Name-level attribution (Brinson, exact)
Since SPMO and market weights both sum to 1, `sum_p (w_SPMO,p - w_mkt,p) * ret_p`
equals the monthly excess exactly. Aggregate per name per year; report:
- top +/- contributors per year (esp. 2024, 2025; contrast lagging 2016, 2021);
- concentration of the excess: top-3 / top-5 share and HHI of contributions, pre vs post.

### Convergence check
State explicitly whether (a) the ladder, (b) the regression, and (c) the attribution
agree on the same story. Flag any disagreement rather than papering over it.

## 5. Stage 2 — Durability (in-sample only; honest about limits)

- **Rolling stability:** 36-month rolling excess and Sharpe; is the post-2024 window a
  statistical outlier vs the prior distribution?
- **Regime conditioning:** sort months into terciles by ex-ante (lagged) dispersion,
  breadth, and SPMO concentration; report mean excess by tercile. Tests whether the
  edge lives in the narrow-leadership / high-dispersion state (H3).
- **Episode honesty:** bootstrap (block-bootstrap, ~12-mo blocks) CI on the post-2024
  monthly excess; count how many independent episodes drive it (effectively 2024).
  Explicit conclusion: 1-2 episodes CANNOT establish a durable premium; report this as
  a limitation, not a positive durability claim.

## 6. Stage 3 — Thesis framing and deliverables

- Synthesis: one defensible paragraph stating the quantified verdict, connected to the
  existing thesis finding that the edge is dispersion/regime-dependent and time-concentrated.
- Charts (pdf + png): (i) ladder marginal-excess bars pre vs post; (ii) 36m rolling
  alpha + UMD loading; (iii) name attribution of the 2024 excess.
- Markdown writeup foldable into `real_trade_analysis/RESULTS_SECTION_DRAFT.md`.

## 7. Success criteria

- Quantified contribution of each driver to the post-2024 excess, e.g.
  "X pp from weighting, Y pp from selection, Z pp factor; NVDA+AVGO+META = N pp".
- A statistically-backed factor-vs-idiosyncratic verdict (rolling alpha + dummy test).
- An honest in-sample durability statement with bootstrap CI and the episode-count caveat.

## 8. What would falsify the leading (H2 concentration) hypothesis

- If equal-weighted momentum selection (ladder rung 2) already captures most of the
  post-2024 excess -> it is SELECTION/factor, not the weighting scheme.
- If the regression alpha does NOT jump post-2024 (excess fully explained by the UMD
  loading) -> it is the momentum FACTOR (H1), not idiosyncratic concentration.

## 9. Out of scope (explicit)

- No pre-2011 historical analogs / synthetic-index build (durability is in-sample only).
- No conditioning/predictive model of forward excess.
- No extension of the XGB strategy through 2025 (needs the 2025 HMM regime panel; tracked separately).
- Production pipeline untouched; all outputs under `real_trade_analysis/`.

## 10. Deliverables / file plan

- `real_trade_analysis/scripts/2026-06-02-spmo-mechanism-ladder.py` (Stage 1a)
- `real_trade_analysis/scripts/2026-06-02-spmo-mechanism-regression.py` (Stage 1b)
- extend `2026-06-02-why-spmo-2024.py` for the attribution/regime pieces (Stage 1c, 2)
- charts in `real_trade_analysis/plots/`
- writeup section appended to `RESULTS_SECTION_DRAFT.md`
