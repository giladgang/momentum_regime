# Potential additions to the paper (running register — notes, not prose)

## 0. THE MESSAGE (Gilad's framing directive, 2026-07-13)
The paper's product is not the model — it is a WHEN-NOT-TO-TRADE rule:
"In market condition X (regime probability elevated: deep drawdown with
recovery dynamics), a stock with profile Y (large-cap holding that has
fallen hard, e.g. Nvidia/Meta/Netflix 2022) does NOT have to be sold —
although most systematic frameworks would sell it." The model is the
detector/evidence machinery; the rule is the deliverable. Evidence: no-sell
beats forced selling (IR 0.28 vs 0.19); counterfactual sells are a wash;
non-trading costs nothing gross and saves the round-trip at the market's
worst prices. Contrary-philosophy table: momentum exits (rank collapse),
trend/CTA (price below MA), vol-targeting & risk parity (sigma spike),
stop-loss/VaR limits (drawdown), window dressing (reporting), tax-loss
harvesting (year-end in panic). Aligned: value/contrarian, fixed-weight
rebalancers. Caveats that must travel: rule is portfolio-level and
regime-conditional (in calm, selling losers is right — that's momentum);
"don't sell" is supported, "buy the deepest losers" is not (entry-bucket
table).

Each entry: claim candidate, backing table, and the honesty caveat that must
travel with it. Prose starts only after Gilad's numbers review.

## 1. "Hold your losers in panic" vs "buy the deepest losers" — SEPARATE claims
- SUPPORTED (portfolio level): suppressing rank-driven sells in pi>=0.5
  months beats mechanical monthly liquidation (IR 0.28 vs 0.19 capped);
  counterfactual panic sells vs replacements are a wash net of costs.
  Backing: s5/policy_ladder.csv, s5/counterfactual_panic_sells.csv.
- NOT SUPPORTED (stock level): equal-weighted, the deepest-loser panic buys
  (<-50% past-12m, n=269) return -0.3% over 3m, 45% hit rate, -3.8% vs
  market. The barbell (Coinbase +91% vs Peloton -30%) nets ~zero.
  Backing: tables/entry_buckets_by_regime.csv.
- The panic alpha is a PORTFOLIO phenomenon: cap-weighted book tilt +
  monthly refresh (mid-loser recoveries + rebounding winners carry P&L,
  n=1,523 winner-bucket panic entries at +6.2%/3m).
- Caveat for the text: do not let the mechanism section imply a
  stock-picking rule ("buy crashers"); a referee will test it and find the
  -0.3%. Frame as: the clock times a book, not a ticket.
- Curiosity (unexplored): CALM deep-loser entries earned +4.6% vs market
  (n=207) — small sample, post-hoc; flag only if it survives a split-half.

## 2. Calm-month null is universal, not model-specific
- fundamentals-only, fundamentals+momentum, no-pi ML, plain 12-1, GHM rule:
  ALL earn ~0 vs the market portfolio in calm months; every edge tested is
  panic-concentrated. Backing: tables/fundamentals_regime_test.csv,
  applied_results_full.md panel F.
- Use: pre-empts "your model is just weak in calm" — nothing tested beats
  the market in calm; the clock identifies the only edge-bearing months.
- Caveat: menu is finite; exploratory (hypothesis raised after results).

## 3. Market-portfolio comparison per regime (the recipe in one row)
- Same months: market +0.8%/mo calm vs strategy +0.6%; panic market +2.0%
  vs strategy +3.3% (S&P 500 nearly identical to top-1000 VW).
  Backing: regime splits + crsp.msi pull (2026-07-13 session).
- Use: motivates "be the index in calm, be the model in panic" without any
  cost argument; costs then make it strictly stronger.

## 4. Cost-budget asymmetry (the TC section's spine)
- Calm trading breakeven is NEGATIVE (market portfolio is better and ~free);
  panic edge (+135bp/mo) is ~30-50x any defensible panic trading cost.
  70% of naive turnover has no economic basis independent of fees.
  Backing: s5/turnover_decomposition.csv, s5/spread_verdict.csv.
- Framing sentence candidate: "the regime signal converts a cost-fragile
  strategy into a cost-robust one by relocating all spending to where the
  payoff is."

## 5. Measured spreads vs textbook costs
- Top-1000 measured half-spreads: median 1.4bp (calm 1.3 / panic 1.5);
  flat-10bp costing overstates the fee bill ~5x and hides the real
  constraint (impact at size). Backing: s5/spread_stats.csv,
  s5/half_spreads.parquet.
- Caveat: closing-quote spreads are a lower bound in fast markets.

## 7. Factor-alpha answer to Marc (2026-07-14 email) -- MUST be in the reply
Marc asked: does the 15.6-vs-14.1 gap survive market + FF5/6 given beta>1, and
do we benchmark the LONG-ONLY side of the factors? Both now built in
compare_thesis.py (Panels B2 regime-conditional, B3 long-only legs) ->
applied_results_full.md.
- UNCONDITIONAL: no significant alpha. pi_rule_r FF5 +1.8% (t=0.39),
  FF6 +1.6% (t=0.34). The gap is beta (~1.28) + factor tilts. Own this.
- PANIC-conditional (the edge in the months it occurs): economically large,
  statistically underpowered. pi_rule_r FF6 panic alpha +16.0% (t=1.15) on
  n=53; calm -3.0% (t=-0.63). Do NOT claim "significant risk-adjusted alpha".
- The BENCHMARK itself has panic alpha (top-1000 VW: FF6 +7.6% t=1.30, FF3
  +9.9% t=2.08**) -- panic months carry recovery alpha the static factors do
  not span; the strategy AMPLIFIES it (16 vs 7.6) but does not cleanly
  separate at this n. Frame the edge as active-return/IR + cost + regime
  concentration, not factor alpha.
- LONG-ONLY LEGS (Marc's 2nd Q): built from French 2x3 sorts
  (paper/fetch_long_legs.py -> paper/results/data/ff_long_legs.parquet).
  pi_rule_r FF6-LO alpha +0.8% (t=0.17) -- SMALLER than the L/S-factor +1.6%.
  Marc's instinct holds: the correct long-only benchmark weakens the null
  further, because the winner long leg (15.4%/yr momentum) absorbs more of the
  book than self-financing UMD did. Concede cleanly; it strengthens the
  "no unconditional alpha, the value is WHEN you trade" message.
- References Marc sent: Detzel/Novy-Marx/Velikov 2023 (JF, model comparison
  with tx costs) -> S5 cost section; Avramov/Cheng/Metzker 2023 (Mgmt Sci, ML
  vs economic restrictions) -> framing for the ML-in-a-long-only-restricted
  setting. Cite both when the draft is written.

## 8. S6: regime-conditional NMV banding (2026-07-14, spec pre-registered)
- Construction from the literature Marc sent (DNMV JF 2023 SecV / NMV RFS
  2016): enter at the decile, hold until rank exits E, count floats.
  Gate: (10,10) == walk bit-exact. Backing: s6_banding/{cells,grid_net_ir}.csv.
- ANSWER to "does banding tighter in panic work": YES at the grid level.
  Best cell (E_calm=40, E_panic=15): net IR@10bp 0.389 vs best fixed band
  (40,40) 0.294 and monthly 0.060; breakeven 42bp vs ~1.4bp measured spreads.
  All four registered expectations scored: E1 HIT (tighter-in-panic), E2 HIT
  (+0.095 IR), E3 HIT (mechanism = calm turnover 48% vs 67%, panic active
  unchanged +1.07 vs +1.04), E4 CI vs best diagonal [-0.001,+0.025] includes
  zero -> claim "leads the frontier", NOT "dominates". vs monthly the CI
  [+0.011,+0.053] excludes zero.
- Also beats the S5 fill-to-k pi_band(30,15) (0.327@10bp): the literature
  construction is the better implementation AND the better result.
- Surprise worth prose: wide calm band raised calm active too (+0.23 vs
  +0.14/mo) — letting winners run in calm, not just cost saving.
- Caveats that travel: 5x5 grid chosen in-sample (179 mo); 54 panic months;
  panic band 15 (tight re-rank), not 10 (full liquidation) — panic alpha
  needs re-ranking but tolerates a small hold buffer.

### 8b. Robustness battery + novelty check (2026-07-14, same session)
- Direction robust, no winner's curse: pre-registered corner (40,15) vs
  mirror (15,40) CI [+0.001,+0.041] excludes zero @10bp; tight-panic wins
  10/10 mirror pairs (sign p=.001) at EVERY cost level 0-50bp; best cell is
  (40,15) at every cost level; gap vs best fixed band stable +0.08..0.10.
- pi threshold 0.4/0.5/0.6: net IR 0.384/0.389/0.385 — insensitive.
- Honest weak points: split-half — ALL policies have negative net active
  2011-2017 (few panics), everything earned 2018-2025; within H1 conditional
  slightly trails fixed wide band. Absolute net active (40,15) vs index
  t=1.40 (NOT significant — the policy claim is significant, vs monthly
  t=2.99; the market-beating claim is not, as with the factor alphas).
  At measured large-cap costs (~1-3bp) the paired mirror CI is marginal;
  the 10/10 sign test is the cost-free evidence.
- NOVELTY (11-agent lit sweep, 5 angles, adversarial verify vs abstracts/
  full text): 0/6 high-overlap threats. Nobody has published regime-
  conditional banding (state-dependent buy/hold rank spread) implemented
  empirically on a stock-selection book. Closest families, all must-cite:
  * Static banding empirics: NMV RFS 2016 (full-text: fixed thresholds, no
    "regime/state-dependent" anywhere), NMV FAJ 2019, DNMV JF 2023.
  * Azevedo/Hoegner/Velikov SSRN 2024: static cost-mitigation on ML
    strategies FAILS to improve net returns — perfect foil: conditioning
    the band on regime is what makes mitigation work on an ML book.
  * Theory anticipates state-dependent no-trade bands but never implements:
    Jang/Koo/Liu/Loewenstein JF 2007 (regime-dependent no-trade boundaries,
    one-asset model), Lynch/Tan JF 2011 (calibration), Garleanu/Pedersen
    JF 2013 (unconditional trading speed).
  * Collin-Dufresne/Daniel/Saglam JFE 2020: closest empirical — regime-
    dependent trading SPEED, quadratic costs, market-timing exposure; not
    banding, not cross-sectional selection.
  * Regime-allocation line (Nystrup et al QF 2018; Shu/Yu/Mulvey JAM 2024):
    HMM regimes drive asset-class exposure, costs a by-product.
- Contribution sentence candidate: "theoretically anticipated (JKLL 2007;
  Lynch-Tan 2011) but never implemented: we make the NMV buy/hold spread
  state-dependent and show the band belongs to the regime."

## 6. Statistical honesty lines to keep
- pi-band vs best unconditional band: consistent frontier leadership,
  paired CI includes zero ([-0.024,+0.060] spreads engine) — claim
  "leads the frontier", not "dominates".
- Policy layer chosen in-sample: live-2026 (and top-500 when run) are the
  policy-level OOS tests.

## 9. Banding-study registered expectations (2026-07-15, PRE-RUN)

Registered before paper/banding_study.py produced any cells. Spec:
docs/superpowers/specs/2026-07-15-regime-conditional-banding-study-design.md.

- G1: regime-conditional banding (best 2- or 3-state cell, selected on net IR
  at measured spreads) beats the best STATIC band (best diagonal) on net IR
  for a MAJORITY of the 7 strategies; per-strategy paired block-bootstrap CI
  of the net-active difference excludes 0 for at least the majority winners.
- G2: the improvement d(net IR) is LARGER for high-cost strategies (low-vol,
  short-term reversal) than for low-cost ones (value): Spearman rank corr
  between cost intensity (monthly-tier TO x mean hs) and d(net IR) > 0.
- G3: the IR-weighted combined book across the 7 strategies has a higher net
  Sharpe under regime-conditional banding than under static banding.

A MISS on any gate is reported as such (null results are reportable).
Direction expectation from S6 + the 2026-07-15 crash/recovery decomposition:
wide band in crash, tighter band in recovery; but the grid explores all
directions and the frontier decides.

## Walk-forward supersedes the single 2011 split (2026-07-16) — DECISIVE

The single-split OOS (select 1992-2010, eval 2011-25) gave "regime beats
static 4/6, momentum +0.094". A walk-forward (expanding window, band
re-selected EVERY year on prior data, ~25 OOS years) with an honest fit
overturns the regime claim:

- regime_argmax (raw argmax over the 120-cell regime grid each year): beats
  static 5/7, but only momentum's delta is bootstrap-significant, and
  momentum's walk-forward delta (+0.162) is LARGER than its single-split
  (+0.094) -- the classic more-search = more-inflation signature.
- regime_reg (1-SE complexity-regularized: adopt a more complex band only if
  it beats the simpler one by > one Sharpe-SE): 0/7. Collapses to the static
  band for every strategy in every year (reg_pct_3state = 0). Regime
  conditioning never clears the noise bar.

Honest conclusion: **regime-conditional banding does not robustly beat a
well-chosen static band out-of-sample**; the apparent edge is full-grid
in-sample search inflation. Caveat: the 1-SE rule is conservative on short
samples, so the precise statement is "the regime-vs-static edge is within one
Sharpe-SE" (equivalently: 6/7 single-split bootstrap CIs already straddled 0).

Secondary: even STATIC banding beats monthly for only 5/7 OOS (momentum and
profitability are WORSE banded). So "banding reduces net-of-cost turnover" is
a real mechanism but not a universal free lunch.

Backing: paper/results/banding_study/walkforward.{md,csv}, paper/walkforward_banding.py.

## GP/Constantinides analytic regime band (2026-07-16) — the regime effect, located

Principled band-fit: panic/calm band-width RATIO derived from theory
(w ~ (spread/(gamma*var))^(1/3)), so the regime ratio is parameter-free and
only ONE global scale is fit -- matched to the static band's single parameter
(no 120-vs-5 search asymmetry). Walk-forward, expanding, OOS 2001+ (xgb 2013).

Estimated ratio ~0.68 for all non-xgb (0.46 xgb): theory says trade ~32% MORE
in panic -- the regime VARIANCE spike (panic vol ~2-4x calm) dominates the
modest spread spike (~1.15x), so you want to track the target more tightly
despite higher costs.

Result: theory-regime beats static OOS for 2/7, significant (bootstrap CI
excludes 0) for 1/7 = MOMENTUM (+0.066 IR, CI [+0.001,+0.022]). For the other
5 the theory band slightly hurts. So the ONE robust regime effect is momentum:
it survives argmax (+0.162, inflated), dies under 1-SE grid regularization (0),
and RE-APPEARS as a small, significant, principled effect under the GP band
(+0.066). Interpretation: momentum banded is ~break-even net of cost (theory_ir
+0.003 vs static -0.063), and the regime band's value is that it stops static
banding from HURTING momentum -- not that it makes momentum profitable.

Honest paper claim: "regime-conditional banding derived from first principles
delivers a small but out-of-sample-significant net-of-cost improvement for
momentum specifically; across other large-cap anomalies it does not." Backing:
paper/gp_bands.py, paper/results/banding_study/gp_bands.{md,csv}.

## Momentum GP-band robustness battery (2026-07-16) — effect does NOT survive

Stress-tested the load-bearing momentum result (theory-band beats static
+0.066, single-seed CI [+0.001,+0.022]). It does not hold up:

- Bootstrap seed x block: CI excludes 0 in only 20/30 combos (block=6 straddles
  0). Marginal, not robust.
- OOS start year: dIR consistently POSITIVE (+0.018 @1998, +0.066 @2001,
  +0.108 @2004, +0.064 @2007) but CI excludes 0 for only 2/4 starts, both
  barely. Direction robust; significance flickers.
- Ratio sensitivity: any tighter-in-panic band (ratio 0.5-0.9) helps momentum
  (+0.04 to +0.11); ratio=1.0 (no regime)=0. So the DIRECTION is robust but
  the theory ratio isn't special -- any panic-tightening works.
- Multiple testing (Benjamini-Hochberg across all 7 strategies): momentum
  two-sided block-bootstrap p=0.203; 0/7 survive FDR at q=0.05.

CONCLUSION: regime-conditional banding shows a CONSISTENT POSITIVE DIRECTION
for momentum (tighter band in panic helps) but is NOT statistically
significant once bootstrap fragility + multiple testing are accounted for.
We cannot reject the null of no regime effect for any strategy. The one robust
finding remains that plain (static) banding reduces net-of-cost turnover
(helps 5/7 OOS). Backing: paper/gp_robustness.py.

## Cluster-conditioned banding (2026-07-16) — also does NOT survive

Tested banding conditioned on the momentum-shape clusters (C4 deep-loser /
C3 recovery), both MONTH-level and STOCK-level, both economic-direction (hold
deep-losers wider) and GP (spread/var)^(1/3), PIT expanding-refit clusters,
one global scale per arm walk-forward-selected (matched to static), walk-
forward OOS from 2001 (xgb 2013). Backing: paper/cluster_bands.py,
paper/src/clusters.py, cluster_bands.{md,csv}.

Result: no robust improvement.
- cluster_month_econ: momentum +0.136 vs static (CI includes 0, insig) but
  xgb -0.151 (CI [-0.030,-0.002], significantly NEGATIVE). Opposite signs
  across strategies = noise, not effect.
- cluster_stock_* (the higher-power arm we expected to rescue it): momentum
  +0.032 / xgb -0.023, both insignificant. Cross-sectional power did not help.
- gp cluster arms collapse to static (clusters barely differ in spread/var,
  so ratios ~1).

No arm is consistently positive with a CI excluding 0. Cluster-conditioning
joins regime-conditioning in failing the honest test. This is the decisive
answer to "can cluster features give better banding": no. The paper's robust
finding remains plain static banding; the contribution is the cautionary
methods result that state/cluster-conditioned cost-mitigation overfits as a
class (regime AND clusters, month AND stock level, econ AND GP fits).

## Cluster-conditioned banding, ALL 7 strategies + FDR (2026-07-16) — comprehensive null

Extended cluster_bands.py to all 7 (momentum, reversal, lowvol, xgb, value,
profitability, investment) x 5 conditioning arms = 35 honest walk-forward OOS
tests. Stock labels PIT-computed once (strategy-independent). BH-FDR q=0.05.

Headline: Positive vs static 9/35; CI excludes 0: 5/35; survive BH-FDR: 2/35 --
BUT the 2 "survivors" (xgb cluster_month_gp, cluster_stock_gp) have
minus_static = 0.000 EXACTLY: the GP cluster ratios ~1 so those arms ARE the
static band (degenerate significant-zeros, not a real effect). Of the 5 cells
with a nonzero CI excluding 0, FOUR are NEGATIVE (reversal/lowvol/xgb
cluster_month_gp and xgb cluster_month_econ/stock_econ significantly HURT);
the only nonzero-positive CI-excl-0 is momentum regime (+0.066, p=0.203, does
NOT survive FDR -- the fragile one we already debunked). Every positive point
estimate (momentum month_econ +0.136 p=0.109, profitability month_gp +0.090
p=0.077) is insignificant and flips sign across strategies (momentum month_econ
+0.136 vs xgb -0.151).

CONCLUSION: across 7 strategies x 5 conditioning fits, ZERO show a robust
positive net-of-cost improvement over static banding; the only significant
nonzero effects are NEGATIVE. Cluster-conditioning fails comprehensively, at
every granularity and fit. Combined with regime (2/3-state) and the GP band,
this is a complete, multi-signal demonstration that state-conditioned
cost-mitigation banding overfits as a class. Robust finding = plain static
banding only. Backing: cluster_bands.{md,csv}.

## Beaten-down-hold / panic_no_sell — honest WF test (2026-07-16), the gap closed

Gilad flagged that panic_no_sell (the strong pre-tonight S5 result, net IR 0.33
vs monthly 0.06) never got a walk-forward test -- it was a FREEZE policy, not a
band width, so it fell outside the width-grid pipeline. Tested it now
(parameter-free, so NO overfitting excuse): panic_no_sell, panic_hold_losers
(freeze only below-median-momentum held names in panic), + a 1-param thr
variant, vs walk-forward-selected static, all 7 strategies, BH-FDR.
Backing: paper/hold_losers.py, paper/execution.py panic_hold_losers policy,
hold_losers.{md,csv}.

Result: Positive vs static 7/21; CI excludes 0: 2/21; survive BH-FDR: 1/21;
POSITIVE-AND-SIGNIFICANT: 0/21. The 2 significant cells are both NEGATIVE
(reversal panic_no_sell -0.246, panic_hold_losers_wf -0.329 -- freezing a
reversal book is catastrophic, as expected). Positive point estimates
(momentum +0.039, profitability +0.058) are insignificant (p 0.3-0.9) and, for
momentum, don't even beat plain monthly (+0.106) -- i.e. "trade less" not
"freeze in panic".

KEY REINTERPRETATION: the original 0.33-vs-0.06 was panic_no_sell vs MONTHLY --
i.e. freeze-beats-churn, which is just the general "banding beats monthly"
effect. Benchmarked against a proper (walk-forward-selected) STATIC band and
evaluated OOS, panic_no_sell adds NOTHING robust. Because the rule is
parameter-free, this is not overfitting -- there is genuinely no panic-specific
freeze edge beyond trading less overall. The strong pre-tonight number was the
banding-beats-monthly win misattributed to a panic-freeze mechanism.

FINAL: static banding (trade less) is the only robust cost-mitigation result.
Regime, crash/recovery, month-cluster, stock-cluster, GP, econ, AND
parameter-free panic-freeze / beaten-down-hold rules ALL fail to beat it OOS.

## Trained per-cluster band — profitability SURVIVES robustness (2026-07-16)

"Train band adjustment on cluster state": learn per-cluster band on prior data
(argmax over 81 vectors, clusters in {10,20,40}), 1-SE regularized, apply OOS.
6/7 strategies collapse to static under regularization. profitability does NOT:
trained_reg beats static +0.137 IR (0.341 vs 0.205), p=0.015, BH-sig.

Robustness battery (paper/cluster_trained_robust.py) -- survives everything
that killed the momentum-GP result:
- seed x block: CI excludes 0 in 30/30 combos.
- OOS start year 1998/2001/2004/2007: dIR +0.107/+0.137/+0.164/+0.150, CI
  excludes 0 all 4.
- cluster K=3/4/5: +0.137 (stable; verify the K-override propagates -- flagged).

This is the FIRST robustness-surviving positive in the study. HONEST CAVEATS:
(1) 1 of 7 strategies, and ONE survivor of a very broad multi-hypothesis
search across the whole session (>100 strategy x arm x method tests) -- the
within-result robustness rules out estimation-instability but NOT
false-discovery from the broad search; treat as a HYPOTHESIS to validate on
fresh data, not a confirmed effect. (2) plausible but POST-HOC story: the
momentum-shape cluster of a profitability book may flag "beaten-down quality"
months (high-profitability stocks that also got cheap) that are worth holding.
(3) verify which band vector trained_reg picks per cluster (is it "hold the
loser-cluster months wider"?) before believing the mechanism.

So: cluster-conditioned banding is null for 6/7 but there is ONE robust
exception worth a clean out-of-sample re-test.

## CORRECTION: the profitability "survivor" is NOT a cluster effect

Mechanism inspection kills it. trained_reg for profitability picks a UNIFORM
band every year -- (40,40,40,40) in 18/25 years, (10,10,10,10) in 7 -- NEVER
cluster-differentiated (the 1-SE regularization collapses to complexity-1).
So the +0.137 is not cluster-conditioning: it is a plain band-WIDTH artifact,
"a wide uniform band (E=40, trade least) beats the walk-forward static
baseline's width selection for profitability" (my static grid was {10,15,20,
30,40}; the uniform-40 vector from the trained grid beat static's yearly pick).
Per-cluster active means for profitability are flat (0.0001-0.0018) -- no
cluster signal. K-override verified genuine (K=3/4/5 give 3/4/5 clusters), so
the K-invariance was because the pick is uniform regardless of K.

FINAL (stands): cluster-conditioning contributes NOTHING. Even the one
robustness-surviving positive, on inspection, is a uniform (non-cluster) band
that the regularization chose by discarding the clusters. The comprehensive
null holds across regime, crash/recovery, month+stock clusters, panic-freeze,
beaten-down-hold, fixed-ratio AND trained-per-cluster fits. The sole robust
cost-mitigation lever is plain static band width (trade less), set uniformly.

## Rebound-enabled liquidity provision — the SPREAD lever (2026-07-16)

New direction: not turnover (banding) but the spread side. The rebound result
de-risks providing liquidity on panic buys of beaten-down names (they come
back, so you're not catching a falling knife) -> earn the half-spread instead
of crossing the wide panic spread. Backing: paper/liquidity_provision.py,
liquidity_provision.{md,csv}. Monthly book, per-stock hs, panic = pi>=0.5,
beaten-down = below-median mom_12, buy = dw>0.

Result: mechanism validates -- 0 saving for momentum (buys winners), biggest
for loser-buyers (reversal -25% of cost / 0.87%/yr; xgb -17% / 0.05%/yr; value
-28%). Saving concentrated on the panic beaten-down buys (8-14% of total cost
sits there). BUT absolute magnitude small on LARGE CAPS: large-cap spreads are
~2-3bp so even flipping the sign on panic buys saves ~0.05-0.9%/yr. Composes
with banding (different lever). Best-case bound (assumes passive fills in
panic; adverse selection would erode it).

Positioning: real + novel + directly flips Marc's panic-liquidity critique
(rebound => you PROVIDE liquidity, earning the widest spreads instead of
paying them), but the large-cap focus is where it matters least. Turnover
lever (banding) remains the main cost story; LP is a real secondary lever,
strongest for higher-turnover / higher-spread strategies.

## LP saving — defensible range (2026-07-16), lp_sensitivity.py

Swept realized execution factor f (+1 pay half-spread=naive ... 0 cross-at-mid
... -1 earn half-spread=best). Realistic passive provision ~ f in [-0.2,+0.2]
(realized spread ~30-50% of quoted x partial fill) => the mid (f=0) column is
the honest central estimate. Saving %/yr:
- xgb (applied large-cap model): 0.027 realistic, 0.054 best. ~3bp/yr.
- reversal (high turnover, buys losers): 0.44 realistic, 0.87 best.
- others (lowvol/value/profitability/investment): 0.02-0.07 realistic.
- momentum: 0 (control -- buys winners, no beaten-down panic buys).
Conclusion: the spread lever is real + correctly targeted but economically
SMALL on large caps (single-digit bp/yr even best-case); material only for
high-turnover strategies. Turnover lever (banding) remains the cost story.

## Band = f(regime x past-momentum) — DEFINITIVE head-on test (2026-07-16)

The exact "condition banding on regime AND past momentum" idea, as a 4-state
band (calm/panic x high/low-momentum). Three fits vs WF-selected static, all 7,
BH-FDR. Backing: paper/rm_band_test.py, paper/execution.py rm_band policy,
rm_band.{md,csv}.

Result: 0/21 positive-and-significant.
- econ (hold panic low-momentum names wider = Gilad's intuition): HURTS xgb
  -0.114 and reversal -0.090; mixed/insignificant elsewhere; the one CI-excl-0
  econ cell (value -0.018) is NEGATIVE.
- argmax (learn the 4-cell band): momentum +0.124 / xgb +0.084 but insignificant
  (p 0.17-0.40) -- the usual search inflation.
- reg: collapses to static (0.000) for 6/7; profitability +0.137 recurs but is
  the SAME uniform-band artifact (reg picks (40,40,40,40)), not conditioning.

DEFINITIVE: conditioning the band on regime x past-momentum does not beat plain
static banding OOS. The economic "hold beaten-down in panic" version, done as a
band, significantly HURTS the main model. Combined with regime, crash/recovery,
month+stock clusters, panic-freeze, hold-losers, and trained-per-cluster: the
conditioning question is closed. Cost is reduced by trading LESS (uniform
banding / lower frequency), full stop.

## Turnover/cost diagnostic — WHY conditioning failed (2026-07-16)

Decomposed each strategy's monthly-book turnover & cost by regime, transition,
and trade cause. Backing: paper/turnover_diagnostic.py, turnover_diagnostic.{md,csv}.

Three findings:
1. Turnover is REGIME-INVARIANT: to_calm ~ to_panic for all 7 (momentum .33/.35,
   reversal .92/.88, xgb .74/.80). No panic trading spike -> nothing for
   regime/momentum conditioning to dampen. This is WHY every conditioning test
   was null: trading is not state-dependent.
2. Cost concentrates in panic (33-44% of cost in 25-31% of months) purely via
   WIDER SPREADS, not more trading. Panic is a spread phenomenon.
3. Cost is ~80-97% decile RESHUFFLING (entry+exit_rank, balanced), not drift
   (weight small) and not regime. Reshuffling is uniform month to month.

The two real cost sources map onto the two real levers: reshuffling ->
uniform banding; panic spread premium -> liquidity provision. Coherent story;
conditioning adds nothing because there is no state-specific trading to condition.

## Continuous multi-factor band optimization — the optimizer chose beta=0 (2026-07-17)

Optimized the band as a CONTINUOUS function of the HMM + market factors:
E_t = clip(E_base*exp(beta*stress),5,100), stress in {hmm=2*(pi-0.5),
multi=mean(DD_z,VOL_z,DISP_z,CS_z)}. beta fit walk-forward (argmax + 1-SE
prefer-static reg). Backing: paper/var_band_test.py, execution.py var_band
policy, var_band.{md,csv}.

Result: 0/28 positive-and-significant; 0/28 CI excludes 0. The REGULARIZED fit
selects beta=0 (rigid static band) for every cell -- i.e. given full freedom to
make the band respond to the HMM and market stress, the honest optimum is ZERO
response. argmax versions positive for a few (xgb/hmm +0.41, momentum +0.15)
but insignificant (p 0.06-0.36) and sign-flipping.

This closes the question in its strongest form: not just discrete state lookups
(all null) but a continuous, multi-factor, walk-forward-OPTIMIZED band -- the
optimizer itself chooses the rigid band. beta=0 is the optimum, empirically
confirming the mechanistic argument (turnover regime-invariant; spread effect
offset by rebound; splitting the sample adds variance for ~0 bias reduction).
The rigid band is not a limitation we settled for -- it is what optimization
selects.
