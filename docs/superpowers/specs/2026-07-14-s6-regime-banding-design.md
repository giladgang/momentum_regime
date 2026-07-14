# S6: Regime-conditional banding (NMV construction) — design + pre-registration

Date: 2026-07-14. Branch: paper/applied-study. Status: registered BEFORE results.

## Question

Does making the banding buy/hold spread REGIME-CONDITIONAL (tighter in panic,
wider in calm) add net-of-cost value over the best FIXED band, when banding is
implemented in the literature-standard form?

Origin: Gilad's email to Marc Stam (2026-07-14, "regime-based banding");
Marc's reply attached Detzel, Novy-Marx & Velikov (JF 2023), whose Section V
banding (from Novy-Marx & Velikov, RFS 2016) is the canonical cost-mitigation
construction: stricter threshold to ENTER a position than to EXIT it.

## Construction (NMV-faithful, floating count)

Book: main model only (pi_rule_r walk xsecs, score_pi ranking, top-1000 PIT,
2011-01..2025-11). Each month t:

- ENTER: names ranked in the top decile (10%) of score_pi.
- HOLD:  previously-held names still in the universe with rank_pct <= E/100.
- Members = enter UNION hold. Value-weighted by me. Holding count FLOATS
  (no fill-to-k; this differs from the S5 'band' policy and matches
  NMV 2016 / DNMV 2023 fn.11).
- E switches on the regime signal known at formation:
  E = E_panic if pi_t >= 0.5 else E_calm. No new fitting; pi is the
  existing HMM(DD) walk output.

## Grid (pre-committed, full cross)

E_calm x E_panic in {10, 15, 20, 30, 40}^2 (25 cells).
- Diagonal = unconditional bands (the null family).
- (10,10) = monthly top-decile baseline; GATE: must reproduce
  walk_returns.csv gross returns exactly (S5-style bit gate).
- Off-diagonal upper vs lower triangle = wider-in-calm/tighter-in-panic vs
  the mirror (built-in falsification).

## Evaluation (S5 conventions, unchanged)

- Drift-adjusted turnover; net = gross - traded*bps for bps in {0,5,10,20,50};
  benchmark pays its own reconstitution turnover; breakeven bp vs benchmark.
- Headline: net IR @10bp per cell (5x5 table), mean holding count per cell.
- Paired moving-block bootstrap (block 12, 10k) of net-active @10bp:
  best off-diagonal cell vs (a) best diagonal cell, (b) monthly.
- Panic/calm mean active split for the best cell vs best diagonal.

## Registered expectations (score HIT/MISS after the run)

- E1 (direction): the best off-diagonal cell has E_panic < E_calm
  (tighter in panic). Prior: S5 pi_band(30,15) led the ladder; product
  comparison showed alpha needs intra-panic re-ranking.
- E2 (value-add): best conditional net IR @10bp exceeds best diagonal by
  >= +0.03 (prior: S5 saw +0.05 with the fill-to-k construction).
- E3 (mechanism): the conditional gain comes from CALM turnover reduction,
  not higher panic active return — at matched panic band, calm turnover
  falls while panic-month mean active stays within noise.
- E4 (honesty): the paired CI of best-conditional minus best-diagonal may
  include zero (S5 analog did). Report it either way; if it includes zero
  the claim is "leads the frontier", not "dominates".

## Honesty caveats that must travel with results

- The 5x5 policy layer is evaluated in-sample over 179 months (like the S5
  ladder). OOS confirmations available later: top-500 tree, live-2026.
- Only 53-54 panic months drive the panic side.
- Flat-bps costing; measured half-spreads (s5/spreads) say flat 10bp
  overstates large-cap spread cost ~5x, so net numbers are conservative.

## Out of scope (phase 2, separate spec)

Momentum-conditioned hold rule (protect held deep losers in panic) and the
learned-band ladder. Runs only after this literature-anchored check.

## Artifacts

- Code: paper/execution.py (new 'nmv_band' policy branch), paper/banding.py
  (S6 runner). Test: paper/tests/test_banding.py.
- Results: paper/results/s6_banding/{grid_net_ir.csv, cells.csv, report.md}.
