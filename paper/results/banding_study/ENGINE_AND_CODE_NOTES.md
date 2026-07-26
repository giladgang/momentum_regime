# Engine & Code Notes — Banding Initiative (for Marc)

Companion to `OVERVIEW_FOR_MARC.md`. That document is findings-oriented; this one is
code/methodology-oriented — how the engine works, and what every related script in `paper/` does
and found. Two engines exist: the **production engine** (`execution.py`, used by the earlier
multi-strategy campaign) and the **clean-room engine** (`tv_band.py`, the independent
re-implementation built for this study). They agree to machine precision.

---

## 1. The clean-room engine — `paper/tv_band.py` (the methodology, in detail)

**Why a second engine.** The earlier campaign concluded that regime-conditional banding is null.
To rule out that this was a bug in the production engine, `tv_band.py` re-implements the entire
backtest from scratch, importing **nothing** from `execution.py`/`banding_study.py` — only
numpy/pandas and the raw data. If the null were an artifact of the engine, independent code on the
same inputs would disagree. It doesn't.

**The trust gate (the load-bearing check).** Before any band result is computed, the no-band
`monthly` policy must reproduce the production `walk_returns.csv` (main model, rule_r/DD) to
≤ 1e-6 per month. It reproduces it to **1.7e-16** — machine precision. So the clean-room engine is
provably computing the same returns/weights/universe as production, and any band number it produces
is trustworthy.

**Backtest mechanics (`simulate`).** For each month:
- Rank the ~1000-stock universe by the XGB score; the book is the **top decile** (`k = 10%`),
  **long-only, value-weighted** by market cap. The benchmark is the value-weighted full universe.
- Turnover is **drift-adjusted**: last month's weights first grow by their realized return and are
  renormalized, so turnover counts only *active* rebalancing, not passive drift.
- The output is the **active return** (book − benchmark), the one-way **turnover**, and a
  **per-trade ledger** (each `Δweight`, used for costing).
- **Costs** are measured **half-spreads** (`half_spreads.parquet`, PIT), `cost = Σ|Δw|·spread`; a
  flat-10bp alternative and a panic-stress multiplier (×2/×5 on high-π months) are available.

**The band mechanic (`policy_band`).** A percentile no-trade band / hysteresis: hold a name while
its score rank% ≤ `E_exit`, add a new name while rank% ≤ `E_enter` (with `E_enter ≤ E_exit`). A wider
`E_exit` = hold longer = trade less. The static baseline is one `(10, E*)` pair; the conditioned
arms let `E_exit` vary by state.

**The conditioning features (`month_features`).** Per month: the **12-horizon long-leg z-curve**
(the picks' momentum, cross-sectionally z-scored, per horizon — the "shape" the thesis mechanism
turns on), the **12-horizon cross-sectional term structure** (universe-mean momentum per horizon),
and **π** (the HMM stress probability). Plus a compressed 4-feature version (level/slope/curvature
of the z-curve + π). All PIT (month-t cross-section only).

**Gate 0 — the ceiling (`run_gate0`, `oracle_bin_ir`).** With perfect hindsight, assign each
feature-bin its net-IR-optimal band width by **coordinate ascent seeded at the best static band**.
Seeding at static and only accepting improvements makes the result a *valid upper bound* (guaranteed
≥ static) — the most any feature function could achieve in-sample. **g0_pass = True**: +18.5% over
static (measured), robust to ×2/×5 panic-cost stress. `run_gate0_impl` then checks *direction*: the
oracle earns it by **trading more in panic** (turnover 0.68 vs 0.57) — flagged as less implementable.

**Gate 1 — the realizable test (`walkforward`, `run_gate1`).** For each OOS year Y ≥ 2013, every arm
is re-fit on data **before** Y only, applied to year Y, and the realized year-Y active returns are
stitched into one OOS series. This is leakage-clean (an independent opus-model review traced every
arm and confirmed it). The nine arms: `monthly`, `static`, `cluster_band` (free per-cluster width),
`trainer_A` (linear-exp policy fit by Nelder-Mead net-IR), `trainer_B` (**GBM**), `trainer_C`
(ridge), and `rebound_c3/c4/both` (widen in the below-average bins). Supervised trainers (B/C) fit to
a **trailing-window ex-post-optimal band target** (a tree can't ingest net IR directly).

**The honesty gate (`_delta_ci`, `_bh_reject`).** Per arm: a paired **block-bootstrap** CI (block 12,
10k reps) on the (arm − static) net-IR difference, plus a one-sided **BH-FDR** correction across the
7 learned arms (testing many arms inflates single-arm significance — the correction the prior
campaign also used). `g1_win` requires surviving FDR. **Result: g1_win = False.**

**Tests (`tests/test_tv_band.py`, 24 tests).** Trust gate, band monotonicity, feature construction
(z-curve matches the thesis clustering definition), the oracle-≥-static guarantee, walk-forward
no-leakage, and the FDR logic. All pass.

---

## 2. The production engine (what `tv_band.py` re-implements)

- **`execution.py`** — the "S5" execution/cost engine used by the multi-strategy campaign. Its
  `simulate(x, policy)` is the original of the loop above, with a library of band policies
  (`nmv_band`, `pi_band`, `regime3_band`, `panic_no_sell`, `panic_hold_losers`, `flip_freeze`,
  `freq`, ...) and `simulate_blend` for holdings-level **exposure** overlays (`pi_scale`, `bsc`
  cash-blend). The clean-room `tv_band.py` reproduces its `monthly` and band returns exactly.
- **`spreads.py`** — the "S5" spread engine: builds the stock-month half-spread panel and the
  spread-priced verdict. Source of `half_spreads.parquet`.
- **`pipeline.py`, `walk_report.py`, `config.py`, `build_pi_history.py`** — orchestration, the S3
  yearly-walk / S4 report, all paths/knobs, and the monthly DD-only regime-π history (1991-2025).
- **Data pulls** — `pull_daily_history.py` (daily CRSP backfill for IVOL + pre-2011 spreads),
  `pull_fundamentals.py` (Compustat→CRSP link), `fetch_ff_daily.py`, `fetch_long_legs.py` (Ken
  French factor legs for the long-only benchmark).

---

## 3. The banding sweep and the learned-band campaign (the "other files")

**The sweep.** `banding_study.py` — the master **regime-conditional banding sweep**: 7 strategies
(the XGB main model + momentum, reversal, low-vol, value, profitability, investment) × band policies
× cost levels, with pre-registered gates. `banding.py` registers the S6 expectations. This is the
breadth backdrop; `tv_band.py` is the depth on the main model.

**The learned / conditioned band attempts — all null under honest testing:**

| Script | What it conditions the band on | Finding |
|---|---|---|
| `walkforward_banding.py` | best static vs best regime band, annual re-selection | regularized regime **0/7** beat static (argmax 5/7 — the gap is overfitting) |
| `gp_bands.py` | Garleanu-Pedersen / Constantinides theory ratio (no grid) | 2/7 positive, **1/7** CI excludes 0 (momentum, tiny) — fails multiplicity |
| `cluster_bands.py` | thesis K=4 cluster (month & stock, econ & GP) | 9/35 positive, 2/35 survive BH-FDR — both artifacts (Δ=0.000) |
| `cluster_optimal_band.py` | jointly-optimal per-cluster width (brute force, hindsight) | even the perfect-hindsight per-cluster band ≈ uniform at matched turnover |
| `cluster_trained.py` / `cluster_trained_robust.py` | band trained as f(cluster), walk-forward | a "+0.137, p=0.015" survivor that **robustness retracted** as a uniform-band-width artifact, not clusters |
| `ml_band.py` | learned weight vector over the full market-factor set | 5/14 positive, **0/14** positive-and-BH-significant |
| `train_band.py` | per-stock band = f(stock momentum z, π) | regularized **0/7**; the fit collapses to static |
| `mom_pi_band.py` / `mom_pi_band_oos.py` | continuous per-stock band shaped by momentum + π | null; the momentum-spread signal doesn't cut cost at matched turnover |
| `learned_band.py` | the band as a LEARNED XGBoost decision surface | OOS AUC ≈ 0.50 — a definitive null |
| `var_band_test.py` | continuous band = f(HMM + market factors) | optimizer selects ≈ rigid (state-independent) band |
| `rm_band_test.py` | band = f(regime × past momentum) | 0/21 positive-and-significant — definitive null |
| `hold_losers.py` | "don't sell the beaten-down names in panic" rule | null vs static |

The consistent verdict across all of these — different conditioning variables, different model
classes, different granularities (month vs stock), honest walk-forward with FDR — is that **no
regime/feature-conditioned band beats a well-chosen static band out of sample.** `tv_band.py`
re-confirms this independently, in depth, for the main model.

**Cost-focused cluster studies** (asking "how much cost does widening save?", separate from "does
it beat static on net IR"):
- `cluster_cost_study.py`, `c3_cost_study.py` — direct cost saved by widening the band in each thesis
  cluster (and cluster 3 specifically). Cost falls when you widen, of course; the question `tv_band`
  answers is whether that helps *net of the return given up* — it doesn't.
- `cost_frontier.py` — unconditional turnover reducers (frequency, staggering) as a cost/return
  frontier; the "no conditioning" baseline against which the conditioned bands failed to improve.

---

## 4. The alternative levers (not banding)

- `liquidity_provision.py`, `lp_sensitivity.py` — the **spread lever** (post as a liquidity provider
  / earn the spread on the rebound trade) rather than the turnover lever. This one is **real but
  modest** on large caps (~3-5 bp/yr for the XGB model at realistic execution; larger, ~40 bp/yr,
  for the high-turnover reversal strategy). It is the one non-null cost-side result, and it is a
  different mechanism from banding.
- (The **exposure lever** — scaling position size by π — is the D&M result; see
  `experiments/2026-07-26-exposure-overlay-diagnostic.py` and the OVERVIEW.)

---

## 5. Diagnostics and validation

- `crash_recovery.py` — the calm / panic-crash / panic-recovery decomposition. **The economic core:**
  the edge is the 29 panic-recovery months (+1.65%/mo, t=2.1); crash months drag; calm is flat. Its
  docstring is explicit that the split is *contemporaneous/descriptive, not tradeable* (you can't
  tell crash from recovery ex ante) — which is precisely why regime-conditioning adds only a modest,
  non-robust increment and the lesson is patience.
- `turnover_diagnostic.py` — where each strategy trades and where cost concentrates by regime. Trading
  is ~regime-invariant; cost concentrates in panic only via wider spreads (which a band can't fix).
- `compare_thesis.py`, `replicate_results.py` — reconcile the applied (expanding-window, long-only,
  top-1000) config against the thesis numbers side by side.
- `ablation_walk.py` — single-feature HMM regime-signal ablation (which stress indicators matter).
- `live2026.py` — the live-2026 out-of-sample continuation on the Compustat splice.

---

## 6. One-line map

- **Engine:** `tv_band.py` (clean-room, this study) ⇄ `execution.py` + `spreads.py` (production).
- **Breadth:** `banding_study.py` (7 strategies) — the campaign backdrop.
- **The null, many ways:** `walkforward_banding`, `gp_bands`, `cluster_bands`, `cluster_optimal_band`,
  `cluster_trained(_robust)`, `ml_band`, `train_band`, `mom_pi_band(_oos)`, `learned_band`,
  `var_band_test`, `rm_band_test`, `hold_losers`.
- **Cost side:** `cluster_cost_study`, `c3_cost_study`, `cost_frontier` (banding cost); `liquidity_
  provision`, `lp_sensitivity` (the one modest non-null, a different lever).
- **Diagnostics:** `crash_recovery`, `turnover_diagnostic`.
- **Infra/data/validation:** `pipeline`, `walk_report`, `config`, `build_pi_history`, the pulls,
  `compare_thesis`, `replicate_results`, `ablation_walk`, `live2026`.
- **This study's follow-ups (`experiments/`):** `2026-07-26-narrow-band-arms`,
  `-exposure-overlay-diagnostic`, `-factor-spanning-test`.
