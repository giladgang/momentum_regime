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

## 6. Statistical honesty lines to keep
- pi-band vs best unconditional band: consistent frontier leadership,
  paired CI includes zero ([-0.024,+0.060] spreads engine) — claim
  "leads the frontier", not "dominates".
- Policy layer chosen in-sample: live-2026 (and top-500 when run) are the
  policy-level OOS tests.
