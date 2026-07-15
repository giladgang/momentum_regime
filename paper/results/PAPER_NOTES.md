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
