# S5 Execution/Cost Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans.
> Design basis: Gilad-approved 4-layer proposal (2026-07-13) + adjustments
> (benchmark-in-calm as labeled endpoint; pre-committed policy grid).

**Goal:** Net-of-cost execution frontier for the applied study: does the
π-clock's value survive costs, and does π-timed trading beat static banding?

**Data:** `paper/results/xsec/xsec_<year>_<combo>.parquet` (rule_r path),
179 months, per-stock-month {permno, me, pi, score_pi, score_nopi, mom_12,
ret_fwd}. Live-2026 EXCLUDED from all cost claims.

## Accounting semantics (exact, pre-committed)

- State: holdings dict w_t (weights, sum=1) after trading at formation
  month t's close. Return earned over t+1: sum_i w_t,i * ret_fwd_t,i.
- Drift: pre-trade weights at t are w~_t,i = w_{t-1,i}(1+r_t,i)/norm, where
  r_t,i = ret_fwd_{t-1,i} (the month-t realized return of the held name).
- A held name absent from month-t's cross-section = forced liquidation
  (cause `delist/exit_univ`), proceeds redistributed by the policy.
- One-way turnover TO_t = 0.5 * sum_i |w_t,i - w~_t,i|. Traded volume
  = sum_i |dw_i| = 2*TO_t. Cost_t = sum_i |dw_i| * c_i,t with c = ONE-SIDE
  cost. Net return over t+1 = gross_{t+1} - Cost_t.
- Weight targets for members: me-proportional (index-reconstitution
  semantics) every month, for every policy. Since me drifts ~ position
  drift, weight-turnover is near zero and turnover is membership-driven —
  clean decomposition. Capped variant: 5% iterative cap, same as
  tables/capped_weight_variant.csv.

## Policy ladder (grid REGISTERED before any net result is read)

| policy | definition | grid |
|---|---|---|
| monthly | target = top decile by score, every month | baseline |
| band(b) | buy top 10%; hold while score-rank <= b% | b in {15, 20, 30, 40} |
| panic_no_sell | monthly baseline, but in pi>=0.5 months no rank-exit sells (forced exits only) | one variant |
| flip_freeze | full rebalance only in months where pi crosses 0.5 (either way); frozen otherwise (forced exits excepted) | one variant |
| pi_band(bc, bp) | band width by regime: b=bc calm, b=bp panic | (bc,bp) in {(40,15),(40,20),(30,15)} |
| calm_bench | benchmark book when pi<0.5, strategy decile when pi>=0.5 | labeled ENDPOINT, not headline |
| benchmark | top-1000 me-VW (its own reconstitution turnover, netted in comparisons) | reference |

Applied to: score_pi (headline), score_nopi and mom_12 (comparators).
Full grid reported; no tuned-point selection.

## Cost engines

1. `flat`: c in {0, 5, 10, 20, 50} bps one-side. Breakeven bps per policy
   = mean gross active / mean traded volume.
2. `spread` (CORE; awaits Gilad's WRDS go): stock-month half-spread via
   Corwin-Schultz on daily high/low (pull: dsf_v2 dailies 2010-12..2025-12,
   top-1000 permnos, columns verified at pull time incl. bid/ask if
   licensed). Winsor [1, 200] bps; missing -> cap-decile month median.
3. `impact` (optional, capacity note): sqrt-law k*sigma_i*sqrt(q/ADV),
   k=0.1, at $100M / $1B AUM. Needs daily volume (same pull).

## Layers / tasks

1. **Engine + gate** (`paper/execution.py`): simulator + trade ledger
   {month, permno, dw, cause: entry|exit_rank|exit_univ|delist|weight,
   type: loser|winner (mom_12 vs book median)}.
   GATE: policy `monthly` gross series must equal walk_returns.csv
   strat_ret to 1e-12; weights sum to 1 every month; benchmark policy
   one-way turnover ~ 1-3%/mo.
2. **Layer-1 tables**: turnover by regime x cause x type, paired with the
   active return earned in the same buckets (the motivating table).
3. **Layer-3 frontier**: all policies x flat grid: turnover, gross/net IR,
   breakeven bps; net-IR-vs-turnover frontier figure; paired moving-block
   bootstrap (block=12, thesis convention; 10k draws) CIs on policy
   DIFFERENCES vs monthly baseline.
4. **Layer-4 exhibits**: (a) regime cost-alpha cross per policy;
   (b) counterfactual panic loser-sells: each rank-exit sell the monthly
   baseline makes in pi>=0.5 months — sold name's next-3/6-month return vs
   its replacement's, net of round-trip at the applicable engine.
5. **Spread engine** (after WRDS go): pull + CS estimator + re-price
   layers 3-4; the flat-vs-spread deltas are themselves reported (how much
   flat costing flatters each policy).
6. **Honesty box** in the output doc: costs post-hoc (thesis App. C.2
   stance); capped-variant frontier alongside uncapped; benchmark
   reconstitution netted; live-2026 excluded; grid fully reported.

## Verification

- Engine gate (task 1) before any table.
- Turnover sanity: monthly momentum decile one-way 20-40%/mo expected;
  banding should cut it 2-4x (literature range).
- Ledger identity: sum |dw| per month == 2*TO_t (1e-12).
- Bootstrap: pairing check — CI of (policy minus itself) must be exactly 0.

Outputs: `paper/results/s5/` (tables CSVs, `s5_summary.md`, frontier fig
`plots/applied_s5_frontier.{png,pdf}`).
