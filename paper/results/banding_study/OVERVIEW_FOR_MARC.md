# Time-Varying Banding Study — Code & Findings Overview (for Marc)

**Date:** 2026-07-26 · **Branch:** `paper/applied-study` · **Underlying strategy:** the applied
regime-aware XGB selector (long-only, value-weighted, top-decile of a ~1000-stock large-cap US
universe; expanding window; measured half-spread costs). Performance is measured as **active
return vs the value-weighted universe**, so "net IR" = annualized information ratio of the
net-of-cost active return.

---

## 1. The question and the headline

**Question.** Can we improve the strategy's net-of-cost performance by *conditioning the
rebalancing band* on the market state — i.e. widen the no-trade band (trade less) or narrow it
(trade more) as a function of momentum, the momentum term structure, and the regime probability
π — rather than using a single static band?

**Headline finding.** No. A well-chosen **static band already sits at the optimum**, and no
regime-conditioned band beats it out of sample. The signal *exists in-sample* (a perfect-hindsight
"oracle" band beats static by +18.5%), but it **does not survive out of sample** — the classic
overfitting gap. Every direction we pushed — widen or narrow, in panic / recovery / calm, fit by
XGB / ridge / direct optimization — is null or negative once tested honestly (walk-forward, with a
multiple-testing correction). The one economically robust lesson is a *negative* one that vindicates
your original implementability point: the profit is concentrated in the panic-recovery months, where
the right action is to **trade *more*** (establish the rebound) — which is exactly the expensive,
hard-to-execute thing. Banding, which can only make you trade *less*, has no regime where it helps.

---

## 2. Design (two staged gates + follow-ups)

The study was built as a **clean-room re-implementation** — a fresh module that imports *none* of
the existing backtest engine — precisely so that if the earlier null were a bug, independent code
on the same inputs would disagree. It doesn't: the new engine reproduces the production return
series to **1.7e-16** (machine precision) before any band result is computed (the "trust gate").

- **Gate 0** — the *ceiling*: with perfect hindsight, how much better is a flexible (state-varying)
  band than a strict (static) one? This is the "does the signal even exist" test. It also runs an
  **implementability check** (does the oracle trade more or less in panic?) and a **panic-cost
  stress test** (×2/×5 wider panic spreads).
- **Gate 1** — the *realizable* test: can a band *learned only from past data* (walk-forward, re-fit
  annually) beat static out of sample? Nine arms, an honesty gate (BH-FDR across the learned arms).
- **Follow-ups** (from our discussion): the exposure lever (D&M), a factor spanning test, and your
  specific narrow-in-calm / narrow-in-cluster-3 ideas.

---

## 3. The code, file by file (what it does + what it found)

### Core engine and study — `paper/tv_band.py` (778 lines, the whole thing)

A single self-contained module. Imports only numpy/pandas + the raw data artifacts. Key pieces:

- `load_panel`, `load_spreads` — build the monthly panel (per-stock XGB score, π, market cap,
  12-horizon momentum, forward return) and the measured half-spread cost panel.
- `simulate(panel, policy)` — the from-scratch backtest: long-only value-weighted top-decile book,
  drift-adjusted turnover, active return vs the VW universe, per-trade ledger. **Trust gate:** its
  no-band `monthly` policy reproduces the production `walk_returns.csv` to 1.7e-16.
- `policy_band(E_enter, E_exit)` — the band mechanic: hold a name while its score rank% ≤ `E_exit`,
  add a new name while rank% ≤ `E_enter` (a percentile no-trade band / hysteresis).
- `month_features` — the conditioning features: the **12-horizon long-leg z-curve** (the picks'
  momentum shape), the **12-horizon cross-sectional term structure** (universe momentum), and **π**.
- `run_gate0` + `oracle_bin_ir` — the perfect-hindsight ceiling. Coordinate-ascent per feature-bin
  band, seeded at the best static band (so it is a *valid upper bound*, guaranteed ≥ static).
  **Result: g0_pass = True**, oracle beats static +18.5% (measured cost), +0.063 IR (flat 10bp).
- `run_gate0_impl` + `band_regime_turnover` — the implementability lens. **Result: the oracle earns
  its edge by trading *more* in panic** (cluster-oracle panic turnover 0.68 vs static 0.57 — flagged),
  and the edge survives ×2/×5 panic-spread stress (so it is not merely a cheap-execution artifact).
- `regime_turnover` — independent re-derivation of the "trading is regime-invariant" diagnostic.
  **Result: calm turnover 0.74 vs panic 0.80** — nearly flat (matches the pre-existing diagnostic).
- Gate 1 arms — `fit_static`, `fit_cluster_band`, `fit_trainer_A` (linear-exp policy, Nelder-Mead
  net-IR), `fit_trainer_B` (**GBM — the "XGB" band learner**), `fit_trainer_C` (ridge), and the
  rebound arms `fit_rebound_c3/c4/both`; `walkforward` (annual re-fit, stitched OOS 2013-2025);
  `_delta_ci` + `_bh_reject` (paired block-bootstrap CI + BH-FDR honesty gate); `run_gate1`.
  **Result: g1_win = False** — no learned arm beats static under the multiple-testing correction.

### Tests — `tests/test_tv_band.py` (289 lines, 24 tests)

Verifies the engine end to end: the trust gate, the band mechanic monotonicity, feature
construction (z-curve matches the thesis clustering definition), the oracle-≥-static guarantee, the
walk-forward harness (no look-ahead), the BH-FDR logic. All 24 pass. A separate opus-model code
review confirmed the walk-forward is **leakage-clean** (every fitted parameter for OOS year Y uses
only data before Y).

### Supporting diagnostics

- `paper/crash_recovery.py` → `crash_recovery.csv` — decomposes the active return into calm /
  panic-crash / panic-recovery. **This is the economic core of the whole story.** The strategy's
  entire edge is the 29 **panic-recovery** months (+1.65%/mo, t=2.1); the 16 **panic-crash** months
  *drag* (−0.96%/mo); calm is flat (+0.04%/mo, t=0.1). *Crucially, its own docstring flags that this
  split is **contemporaneous / descriptive, not tradeable*** — you cannot know at decision time
  whether a panic month will be crash or recovery.
- `paper/turnover_diagnostic.py` → `turnover_diagnostic.{md,csv}` — where trading and cost
  concentrate by regime. Shows trading is ~regime-invariant; cost concentrates in panic only because
  *spreads* are wider (which a band cannot fix — it can only skip trades).

### Follow-up experiments (`experiments/`, exploratory)

- `2026-07-26-narrow-band-arms.py` — **your two ideas.** Narrow the band (trade *more*) in calm and
  in cluster 3, walk-forward vs static. **Result: narrow-in-calm −0.004 (nothing — calm has no active
  edge to trade for); narrow-in-cluster-3 −0.023 (hurts — the data preferred *widening* there).**
- `2026-07-26-exposure-overlay-diagnostic.py` — the *exposure* lever (position size), not the band.
  **Result: scaling exposure *up* in panic (λ=π) raises IR from 0.21 to 0.40 — but that is the
  Daniel & Moskowitz (2016) dynamic-momentum result, applied to the mirror side; not novel, and its
  low drawdown is a no-sustained-bear-in-2011-2025 artifact.** De-risking in panic (λ=1−π) *destroys*
  the edge (IR → −0.04), since the edge lives in panic.
- `2026-07-26-factor-spanning-test.py` — is the model's alpha a repackaging of momentum / reversal /
  D&M? **Result: the active return is *orthogonal* to those factors (R²≈0.01, near-zero loadings) —
  so it is genuinely distinct — but the alpha is only ~2.7%/yr at t≈0.8, i.e. not statistically
  significant on this 15-year long-only sample.** The significance lives in the thesis long-short
  strategy, not the applied long-only active return.

### Design / plan documents (for reproducibility & method)

- `docs/superpowers/specs/2026-07-20-tv-band-design.md` — the full design and pre-registered gates
  (G0/G1/G2/G3), including the honesty conventions.
- `docs/superpowers/plans/2026-07-20-tv-band-gate0.md`, `...-gate1.md` — the implementation plans.

### Banding literature review — `notes_banding_research.json`

A fact-checked survey of the banding literature (Novy-Marx & Velikov "buy/hold spread" / sS rule,
their FAJ cost-mitigation comparison, Frazzini-Israel-Moskowitz trading costs, Garleanu-Pedersen,
Korajczyk-Sadka), with the canonical parameterizations and turnover/cost-reduction magnitudes.

---

## 4. Results at a glance

**Gate 0 (ceiling, in-sample):** oracle beats static **+18.5%** IR (measured cost); survives ×2/×5
panic-cost stress; but the oracle **trades more in panic** than static (0.68 vs 0.57).

**Gate 1 (walk-forward OOS, 2013-2025, vs static net IR 0.414):**

| Arm | OOS net IR | Δ vs static | p | survives BH-FDR |
|---|---|---|---|---|
| no band (monthly) | 0.318 | −0.095 | — | — |
| cluster band | 0.429 | +0.015 | 0.76 | no |
| Trainer A (linear-exp) | 0.391 | −0.023 | 0.51 | no |
| Trainer B (GBM/"XGB") | 0.427 | +0.013 | 0.93 | no |
| Trainer C (ridge) | 0.512 | +0.098 | 0.28 | no |
| rebound_c4 (widen in crisis) | 0.380 | −0.034 | 0.21 | no |
| rebound_c3 (widen in recovery) | 0.470 | +0.056 | 0.048 | **no** (fails FDR) |
| rebound_both | 0.506 | +0.092 | 0.28 | no |
| **g1_win** | | | | **False** |

**Band probed in every direction × regime (walk-forward Δ vs static):**

| | Panic / crisis | Recovery (cluster 3) | Calm |
|---|---|---|---|
| **Widen** (trade less) | −0.034 (hurts) | +0.056 (marginal, fails FDR) | ~0 |
| **Narrow** (trade more) | — | −0.023 (hurts) | −0.004 (nothing) |

Plus GBM / ridge / direct-net-IR learned bands — all null.

---

## 5. The interpretation (the paper-ready story)

1. **A static band works** (net IR 0.414 vs 0.318 for rebalancing every month) — banding *is* a real
   cost-mitigation tool.
2. **Conditioning the band on the regime does not add tradable value.** The reason is structural: a
   band can only make you trade *less*, but the regime's genuine signal is about when to trade
   *more* — establish the rebound at the panic peak, where all the profit lives (crash/recovery
   decomposition). That direction is outside a band's range of actions, and forcing the band to
   widen there *loses* money (rebound_c4). Everywhere else the effect is below the noise floor, so
   the in-sample ceiling (+18.5%) is unlearnable and vanishes out of sample.
3. **This is exactly your (Marc's) implementability point, made quantitative:** the profitable move
   requires trading precisely when trading is hardest/costliest, so the honest lesson is *patience*
   (hold through), not faster/smarter reaction.
4. **The exposure lever is D&M** (known), and **the model's alpha is distinct but underpowered** on
   the applied long-only sample.

**Net for a practitioner paper:** the defensible, novel-enough claim is not "a smart adaptive band
beats static" (false) but *"a well-chosen static band mitigates cost and survives out of sample;
conditioning it on the regime does not add tradable value — because the strategy's edge lives in
the panic-recovery episodes where the profitable action is to trade more, not less."* That is a
clean characterization result with the mechanism explained, and it directly answers the
implementability critique.

---

## 6. Complete file manifest

**Code (committed):**
- `paper/tv_band.py` — clean-room engine + Gate 0/1 (the main file)
- `tests/test_tv_band.py` — 24 tests
- `experiments/2026-07-26-narrow-band-arms.py` — narrow-in-calm / narrow-in-cluster-3
- `experiments/2026-07-26-exposure-overlay-diagnostic.py` — the D&M exposure lever
- `experiments/2026-07-26-factor-spanning-test.py` — alpha vs momentum/reversal/D&M
- `paper/crash_recovery.py` — calm/crash/recovery decomposition (economic core)
- `paper/turnover_diagnostic.py` — regime turnover/cost concentration

**Results / artifacts:**
- `paper/results/banding_study/tv_band_gate0.{md,csv}` — ceiling (g0_pass=True)
- `paper/results/banding_study/tv_band_gate0_impl.{md,csv}` — implementability + panic-cost stress
- `paper/results/banding_study/tv_band_gate1.{md,csv}` — walk-forward OOS, 9 arms (g1_win=False)
- `paper/results/banding_study/tv_band_gate1_attributes.csv` — what market states get which band width
- `paper/results/banding_study/crash_recovery.csv` — calm/crash/recovery active-return split
- `paper/results/banding_study/turnover_diagnostic.{md,csv}` — regime turnover/cost

**Design / method / literature:**
- `docs/superpowers/specs/2026-07-20-tv-band-design.md`
- `docs/superpowers/plans/2026-07-20-tv-band-gate0.md`, `...-gate1.md`
- `notes_banding_research.json` — banding literature review

**Reproduce headline numbers:** `.venv/bin/python -m paper.tv_band --stage gate0` (ceiling +
implementability + regime turnover) and `--stage gate1` (the 9-arm walk-forward sweep).

**Corroborating prior campaign** (broader null across 7 strategies, earlier engine): the
`paper/results/banding_study/` folder also holds `cluster_bands.md`, `gp_bands.md`, `walkforward.md`,
`ml_band.md`, `train_band.md`, etc. — regime-conditional and learned bands across all strategies, all
null under FDR. The new clean-room study confirms that result independently for the main model.
