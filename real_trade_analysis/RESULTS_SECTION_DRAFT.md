# Results section draft — Regime-momentum XGB on the S&P 500 (train Top-1000, trade S&P 500)

*Draft for the thesis results section, restricted to the "real-tradeable" configuration: XGB
regime-momentum model trained on the liquid Top-1000 universe, traded long-only on the S&P 500
(Top-500 proxy), top-decile NYSE-P90, value-weighted, monthly, net of 10bps. Experiment-grade
(±~0.03 Sharpe regime-vintage slack; single 2011-2024 test period). Production untouched.*

## 1. Headline: a risk-adjusted tie with SPMO and the market

Full test period 2011-2024 (167 months):

| Strategy | Ann | Vol | Sharpe | MDD | Calm Sh | Panic Sh |
|---|---:|---:|---:|---:|---:|---:|
| S&P 500 (cap-wt) | 14.1% | 14.4% | 0.99 | -24.3% | 0.76 | 1.34 |
| SPMO replica | 15.1% | 15.4% | 1.00 | -26.0% | 0.70 | 1.44 |
| **XGB long-only** | **19.6%** | 20.0% | **1.00** | -27.9% | 0.73 | 1.39 |

Risk-adjusted, the three are indistinguishable (Sharpe ~1.0, within sampling noise, SE~0.25).
The XGB earns markedly higher **raw return** (19.6% vs 15.1%/14.1%) but at higher volatility and a
deeper drawdown, so the Sharpe lands on top of the others. **It does not beat SPMO or the market
risk-adjusted in the S&P 500 long-only space.**

Validation: our SPMO replica (15.1%/1.00) matches the real SPMO ETF over its 2015-2024 life
(16.9%/1.04), and the full-universe long-short reproduces the thesis headline (Sharpe ~1.08).

## 2. The edge is time-concentrated, not a steady premium

| | 2011-15 | 2016-20 | 2021-24 |
|---|---:|---:|---:|
| XGB long-only Sharpe | **0.41** | 1.38 | 1.19 |
| S&P 500 Sharpe | 0.92 | 1.15 | 0.88 |
| Cross-sectional dispersion (DISP z) | **-0.91** | -0.08 | +0.63 |

The strategy **lagged the market in 2011-2015**, then outperformed 2016-2024, with the cumulative
dollar advantage concentrated in 2023-2024. The full-period "tie" averages a weak early period and
a strong recent one.

## 3. Why 2011-2015 was weak: low cross-sectional dispersion

2011-2015 had a normal share of panic months (30%), so the weakness is **not** a regime-frequency
issue. Instead the strategy was weak in both regimes (calm 0.49, panic 0.37 vs 2.61 in 2016-20)
because **cross-sectional dispersion was low** (DISP z -0.91, vs -0.08 / +0.63 later) — the QE-era
"everything rises together" market gave the cross-sectional momentum/reversal signal little to
differentiate on. **The edge is dispersion-dependent.**

## 4. The regime signal is accurate (and load-bearing)

The HMM panic probability is a genuine, persistent stress/volatility classifier:
- Crisis alignment: flags dot-com (avg pi 0.98), GFC (0.86), 2011 (1.00), 2015-16 (0.97), COVID (1.00);
  correctly calm in calm windows (~0.00). **Misses** the sharp Q4-2018 selloff (0.02) and only partly
  catches the 2022 bear (0.67) -- it is tuned to *sustained high-vol* regimes, not quick corrections.
- Concordance: corr(pi, drawdown) = -0.61, corr(pi, volatility) = +0.62, but only +0.12 with negative
  return -- it detects the **high-volatility regime** (crashes *and* violent recoveries), not merely
  down months. Appropriate, since the strategy profits in recovery.
- Persistence: avg regime run 19.4 months, 10 panic episodes over 34 years -- persistent, not flickering.

## 5. Mechanism: the strategy flips momentum exposure by regime

Value-weighted momentum of holdings, calm vs panic:

| | mom_1 | mom_12 |
|---|---:|---:|
| XGB calm | +0.028 | 0.538 |
| **XGB panic** | **-0.016** | **0.308** |
| SPMO (any regime) | +0.015..0.018 | ~0.44-0.54 |

In **calm** the XGB chases winners (like SPMO); in **panic** it tilts toward recently-fallen names
(negative recent momentum) -- a contrarian/reversal posture that captures the panic-*recovery* rebound
which static momentum (SPMO) is structurally exposed to (the "momentum crash"). SPMO, by construction
(top-quintile, always-positive momentum score), can only ever buy winners. This is the strategy's
distinctive behavior -- though in the large-cap long-only space it keeps pace with rather than beats
SPMO in panic (1.39 vs 1.44); the panic edge fully expresses only in the full-universe long-short
(panic Sharpe 1.56), which can short crashing winners and reach higher-reversal small/mid-caps.

## 6. vs SPMO: factor vs concentration

FF6 regressions (NW(6)):

| | alpha | t(a) | Mkt-RF | UMD |
|---|---:|---:|---:|---:|
| XGB long-only | +4.9%/yr | 1.66 | +1.16 | -0.04 |
| SPMO | -0.3%/yr | -0.17 | +1.06 | +0.36 |

SPMO has ~zero FF6 alpha and a large momentum loading -- it *is* the momentum factor packaged. The
XGB has a modest (not yet significant) positive alpha and **~zero UMD loading** -- its return is not
mechanically the momentum factor. Separately, deep research established that SPMO's 2024-2026 surge
was driven by **idiosyncratic concentration in mega-cap AI/semiconductor winners** (NVIDIA, Broadcom,
etc.) amplified by the index's cap x score weighting -- *not* the diversified momentum factor (UMD was
flat-to-negative in 2025). The replica's 2024 top holdings confirm this (NVIDIA ~10%, Amazon, Meta,
Eli Lilly, Broadcom, Microsoft, Alphabet, Walmart, GE: 9 of 10 among the 11 largest stocks).

## 7. Costs

Fee stress (Sharpe; one-way turnover bps, ~2x = per-side):

| fee bps | XGB LO | SPMO |
|---|---:|---:|
| 10 (prod) | 1.00 | 1.00 |
| 20 (~10/side) | 0.96 | 0.99 |
| 50 | 0.84 | 0.96 |

XGB turns over ~65%/month vs SPMO ~12% (it needs the monthly refresh -- quarterly rebalancing cost
~0.11 Sharpe), so it is **more fee-sensitive**: at realistic per-side costs SPMO overtakes it.

## 8. Bottom line

A regime-conditioned momentum strategy traded long-only on the S&P 500 is **market-like on a
risk-adjusted basis over most of the sample** and outperforms primarily in the **recent
high-dispersion regime**; it delivers higher raw return at higher risk. Its edge is **dispersion-
dependent** (failed in low-dispersion 2011-2015) and **fee-sensitive**. The HMM regime signal is
genuinely accurate, and the strategy's distinctive panic-reversal mechanism is real but only fully
expresses with shorting and breadth. **Whether the recent edge is durable regime-timing alpha or
exposure to a favorable high-dispersion/AI cycle is the central open question** -- evidence for
durability is the ~zero UMD loading and the accurate regime signal; evidence against is the
time-concentration and dispersion-dependence.

## 9. Why did the SPMO replica beat the S&P 500 after 2024?

### 9.1 The headline puzzle

Over 2011-2023, the SPMO replica's annual excess return versus the S&P 500 averaged approximately +0.3%/yr (compounded), statistically indistinguishable from zero. Then in 2024 it jumped to +23.0% (compounded) / +17.9% (linear), and +4.3% in 2025, for a post-2024 mean of +11.6%/yr. The question is what mechanism drove this step-change, and whether it reflects a structural edge or a favorable realized draw.

### 9.2 Mechanism: the ladder decomposition

To isolate the source of the post-2024 excess, we ran a ladder ("rung-by-rung") decomposition separating five structural contributors: momentum selection, cap x score weighting, the 9% cap, rebalancing/drift (the passive hold between semi-annual rebalances), and fees. Results (annualized, compounded within each period):

| Component | PRE <=2023 | POST >=2024 |
|---|---:|---:|
| Selection | -1.5% | -0.6% |
| Weighting | +1.4% | -2.5% |
| Cap (9%) | -0.2% | +0.6% |
| Rebal/drift | +0.4% | **+16.2%** |
| Fee | -0.1% | -0.1% |
| **Total** | **-0.1%** | **+13.5%** |

The verdict is unambiguous: virtually the entire post-2024 excess comes from **rebalancing inertia** -- the low-turnover semi-annual hold that allows a few large winners to run between rebalance dates. Momentum selection contributed -0.6% (detracted), cap x score weighting contributed -2.5% (detracted), and the 9% cap was roughly neutral (+0.6%). This revises the earlier "cap x score concentration" narrative from prior deep-research work: the driver is not how positions are weighted or which names are selected, but the fact that the index holds them for six months and lets them compound. Validation: the ladder's full rung reproduces the real SPMO replica exactly (max abs difference: 0.0).

See `real_trade_analysis/plots/2026-06-02-spmo-mechanism.{pdf,png}` for the visual decomposition.

### 9.3 Factor vs idiosyncratic: regression evidence

A rolling 36-month FF5+breadth regression with a POST-2024 alpha-jump dummy (NW(6) standard errors) yields: post-2024 alpha increment **+12.3%/yr** (t = +2.43, p = 0.015). The rolling UMD loading was +0.38 pre-2024 and shifts to +0.29 post-2024 -- it does not absorb the excess return jump. The tail values of the rolling 36m series (2025-12) show alpha_ann = +0.034 and umd = +0.471; across 2024, the alpha rose from approximately -0.005 (2023-12) to positive territory, while UMD remained elevated and did not spike to match. The post-2024 excess is therefore **idiosyncratic to this product's structure**, not explained by loading on the diversified momentum factor.

### 9.4 Names: attribution of the 2024 excess

Per-name attribution for 2024 (linear monthly excess = +17.9%):

| Rank | Ticker | Contribution |
|---|---|---:|
| 1 | NVDA | +3.9% |
| 2 | AVGO | +3.3% |
| 3 | META | +3.0% |
| 4 | WMT | +1.7% |
| 5 | JPM | +1.5% |
| 6 | GE | +1.4% |
| 7 | AMZN | +1.2% |
| 8 | GOOG | +1.2% |
| 9 | LLY | +1.2% |
| 10 | APP | +0.6% |

The top-3 names (NVDA, AVGO, META) together contributed +10.2%, representing 56.7% of the linear 2024 excess (top3_share_of_excess = 0.567 from the attribution CSV). These three names nearly account for the entire year's outperformance.

One important caveat on the compounding wedge: per-name contributions sum to the linear monthly excess of approximately +17.9%, not the compounded annual figure of +23.0%. The gap reflects within-year compounding; the name-level attribution is arithmetic and should be compared to the linear figure only.

Note on the PRE top3_share_of_excess metric: in years when excess return is near zero, this ratio has a near-zero denominator and becomes extremely noisy (e.g., -8.4 in 2019, +41.6 in 2022 per the CSV). These pre-2024 values are not interpretable and are excluded; the 2024 POST figure (0.567) and 2025 figure (1.455) are reported as the meaningful observations.

### 9.5 Durability: an honest assessment

Several signals point toward a real post-2024 structural shift. The pre/post excess comparison yields z = +2.28, and the regression alpha dummy is significant (t = +2.43, p = 0.015). However, the durability evidence weakens substantially under further scrutiny:

- **Bootstrap CI spans zero:** The 95% block-bootstrap confidence interval on post-2024 monthly excess is [-2.4%, +18.9%], based on n = 23 months. The interval is wide and includes zero.
- **Not cleanly regime-conditional:** Conditioning excess returns on lagged dispersion tercile gives: low dispersion +3.7%, mid -2.3%, high +2.9% -- a U-shaped pattern, not the monotonic relationship expected if the edge were a clean structural premium tied to a specific regime.
- **Breadth tercile conditioning:** low breadth +0.8%, mid breadth +5.0%, high breadth -1.2% -- also non-monotonic.
- **Episode count:** Approximately 1-2 post-2024 episodes provide the entire signal base.

Taken together: we **cannot claim a durable premium in-sample**. The evidence is consistent with a favorable realized draw from a large-cap AI/semiconductor concentration event (principally NVDA, AVGO, META holding and compounding through the 6-month rebalance window) rather than a stable structural edge. The regime-conditioning analysis does not identify a discriminating condition under which the excess is reliably positive.

### 9.6 Connection to the broader thesis finding

This finding reinforces the thesis's central result from Section 2: momentum-related edges in this universe are **time-concentrated and regime/luck dependent** rather than steady premia; the SPMO replica's post-2024 outperformance appears to be a specific episode in which the product's rebalancing inertia happened to hold the right winners at the right moment, consistent with an idiosyncratic favorable draw rather than a generalizable structural advantage.
