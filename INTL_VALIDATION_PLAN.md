# International Validation Plan (UK + Japan)

Self-contained runbook for the international out-of-sample validation of the
US-trained HMM regime classifier and momentum cross-sectional model. Any future
Claude Code session (or Gilad himself) can resume from here after a session
disconnect. All gates, commands, and decision branches are spelled out
explicitly so nothing depends on prior conversation context.

Written: 2026-04-25 (session with Claude Opus 4.7 / 1M context).

---

## Goal

Test whether the regime + momentum architecture validated on US data (CRSP +
Compustat NA) generalizes to UK and Japanese stock markets. The cleanest
publishable claim is: **"The same regime+momentum architecture, refit on
regional market features, predicts cross-sectional momentum returns out-of-
sample in UK and Japan."** A stronger secondary claim, if it holds, is that
the *US-trained* `pi_filter` directly predicts regional momentum crashes
(global financial cycle hypothesis).

---

## Design decisions (locked)

1. **Markets:** United Kingdom (GBR/GBP) and Japan (JPN/JPY) — both common
   equity, traded in local currency, from Compustat Global.
2. **Sample:** 1990-01-31 → 2026-04-30, monthly. Train 1990-2010, test 2011-2025.
3. **HMM features:** `['DD_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z']` (4 features).
   - First three identical to US production (config.py:14: `['DD_z', 'DISP_z', 'REL_N_z', 'CS_z']`).
   - `BANK_REL_z` substitutes for `CS_z` because Moody's BAA-AAA has no regional
     equivalent. `BANK_REL` = 12-month rolling mean of (bank value-weighted
     return − market value-weighted return), sign-flipped (positive = banking-
     sector underperformance = stress). Banks defined as SIC 6000-6199. This
     captures the same financial-system-stress channel as CS, derived purely
     from the regional stock panel (no external macro pull).
4. **Cross-sectional model:** momentum-only (mom_1..mom_12) + `pi_filter`. NO
   fundamentals (book/equity, ROE, earnings growth, etc.). International
   fundamentals coverage is uneven and the validation question is whether the
   regime+momentum architecture transfers, not whether fundamentals do.
5. **Returns:** Total returns including dividends + splits via Compustat
   Global's `trfd` (Daily Total Return Factor). `adj_close = prccd / ajexdi *
   trfd`; `ret_t = adj_close_t / adj_close_{t-1} - 1`.
6. **Z-scoring:** features z-scored on training period (1990-2010) μ/σ — same
   convention as US production.

---

## Current state (last updated 2026-04-26)

### Done

- ✅ `scripts/import_intl_stocks.py` — production import script (parameterized
  by `--region UK|JP`, supports `--from-checkpoint` resume). Uses server-side
  monthly aggregation via PostgreSQL window functions to keep local memory
  under ~200 MB during pull.
- ✅ `data/uk_stock_panel.parquet` — 514,290 rows, 5,569 securities, range
  1990-02-28 to 2026-04-30. Columns: secid, gvkey, iid, date, year_month,
  conm, sic, is_bank, prc_close, adj_close, me, log_me, ret, ret_fwd,
  mom_1..mom_12.
- ✅ `data/uk_market_panel.parquet` — 434 monthly rows. Columns: date,
  mkt_ret, DD, VOL, DISP, REL_N, BANK_REL, DD_z, VOL_z, DISP_z, REL_N_z,
  BANK_REL_z, n_stocks, n_banks. avg banks/month = 28. BANK_REL non-null in
  429/434 months.
- ✅ `data/jp_stock_panel.parquet` — 1,418,985 rows, 6,408 securities.
- ✅ `data/jp_market_panel.parquet` — 434 rows, avg banks/month = 124. BANK_REL
  non-null 429/434.
- ✅ `data/INTL_PANEL_README.md` — schema, build commands, known limitations.
- ✅ Sanity checks: train-period z-scores have mean 0 / std 1 in both regions.
  UK BANK_REL_z peaks at 3.72σ in March 2009 (Lehman aftermath), 1.66σ at
  Brexit June 2016, 4.0–4.9σ across April–July 2011 (Eurozone). JP BANK_REL_z
  peaks at 4.6σ in 2001-02 (NPL crisis nadir) and is correctly NEAR ZERO during
  2008 GFC (Japanese banks didn't underperform in 2008 because of low US
  subprime exposure). Both are economically defensible.
- ✅ `scripts/hmm_intl.py` written (2026-04-26 15:25). Standalone wrapper that
  does NOT import `scripts/hmm_model.py` so the regional path can run while a
  US fit is in flight without triggering the US script's import-time fit.
- ✅ `scripts/cross_sectional_intl.py` written (2026-04-26 15:38). No
  fundamentals (mom_1..mom_12 + pi_filter only); unconditional decile
  breakpoints (no NYSE-equivalent). Optional `--use-us-pi` flag for Test A.
- ✅ HMM smoke fit done for UK and JP (5 seeds × 500 iter). Outputs:
  `data/uk_panel_with_regimes.parquet`, `data/jp_panel_with_regimes.parquet`,
  `data/{uk,jp}_mcmc_draws.npz`. UK pi_filter correctly flags Black Wed 1992,
  dot-com 2001-02, GFC 2008-09, Eurozone 2011-08, COVID 2020-03. JP pi_filter
  correctly flags 1990 bubble, 1995 Kobe, 1997-98 LTCB, 2001-03 NPL nadir,
  GFC, COVID. JP BANK_REL_z sign came out as −1 (driven by 1990 bubble crash
  where banks outperformed); HMM still correctly identifies 2002 NPL via the
  joint multivariate likelihood.
- ✅ First-pass regional cross-sectional model run (UK + JP). Outputs:
  `results/intl_{uk,jp}_returns.csv`, `results/intl_{uk,jp}_summary.csv`.
  Headline numbers (smoke seeds — see "Pending" below for production):
    - UK Method 2 XGB: Sharpe 0.575, ann ret 11.0%, MDD -41% (vs fixed 12-mo
      Sharpe 0.480, market Sharpe 0.624).
    - JP Method 2 XGB: Sharpe 0.379, ann ret 4.3%, MDD -31% (vs fixed 12-mo
      Sharpe 0.076 — JP momentum well-known dead — and market Sharpe 0.857).
- ✅ First-pass Test A run (UK + JP under US-trained pi_filter):
  `results/intl_{uk,jp}_returns_uspi.csv`,
  `results/intl_{uk,jp}_summary_uspi.csv`. Headline:
    - UK M2 XGB with US pi: **Sharpe 0.674**, ann ret 11.7%, MDD **-23.2%**
      (vs UK regional pi: 0.575, MDD -41%). US pi BEATS regional pi.
    - JP M2 XGB with US pi: Sharpe 0.431, ann ret 5.3%, MDD -22.6%
      (vs JP regional pi: 0.379). US pi marginally better.
  Both regions show the US-trained pi_filter generalizes — preliminary support
  for the global financial cycle hypothesis (Rey 2015). Subject to production
  rerun before any thesis claim.

### Known minor limitations (documented in INTL_PANEL_README.md)

- 0.005% (JP) to 0.04% (UK) of rows have a slightly inflated `ret` value
  inherited from the original pull's extreme-return follow-up logic.
  Re-pulling from WRDS would eliminate it (~30 min total). Aggregate impact on
  HMM features and momentum signals is well below noise. Fix in current script
  for any future pulls.
- Survivorship bias inherent to Compustat Global (Asness et al. 2013, Daniel-
  Moskowitz 2016 note the same caveat).
- `is_bank` uses current SIC, not historical (industry-standard caveat).
- **Delisting return treatment** — `apply_shumway_delisting.py` is CRSP-specific
  (uses `dlret`/`dlstcd` columns and codes 500-599). Compustat Global has
  different delisting fields and conventions; an analogous treatment for
  Compustat Global is a separate methodology audit, deferred until the
  US-side Shumway story is settled and reviewed. Track as a follow-up.

### Pending

- ❌ **Production HMM fit** (200 seeds × 2 regions, ~12-18h on 6 workers).
  Smoke fits used 5 seeds; production seed count required before quoting any
  number in the thesis. Must run AFTER US Step D completes (CPU contention).
- ❌ **Production cross-sectional rerun** (50 XGB seeds × 4 variants: UK
  regional, UK Test A, JP regional, JP Test A). Smoke runs above used reduced
  XGB seeding; production seed count gives the seed-stable Sharpe and the
  factor-model alpha t-stats.
- ❌ **Factor-alpha tables** for UK/JP (CAPM with regional market index;
  Asness 2013 international FF if available).
- ❌ **Compustat Global delisting audit** (analogue to Shumway for CRSP).
- ❌ Compose UK/JP results section in `RESULTS_LOG.md` AFTER Gilad reviews
  numbers (`feedback_thesis_edits_await_data`).

---

## Timeline integration with the main post-Shumway chain

The autonomous queue running on the US side is B → C → D → E → F → G → H →
I → J → K → L → (M, gated). UK/JP runs as a parallel track that slots in
once US Step D releases CPU.

| When | UK/JP task | US chain in flight | CPU plan |
|---|---|---|---|
| US Step D running | (wait — both tracks need CPU) | D | UK/JP idle |
| US Step D done | **UK/JP-N1**: kick off production HMM fits (200 seeds × 2 regions, ~12-18h) | E + F (US leg-betas + first report, ~20 min, low CPU) → G + H (~30 min, low CPU) | Run UK/JP HMM in parallel — different parquets, no collision |
| Continuing | **UK/JP-N1** still running | I (US XGB CV, 2-5h) | Both compete for cores; Step I limited to 50% if UK/JP-N1 active |
| UK HMM done | **UK/JP-N2 UK**: production CS model + factor alphas | J (US HMM CV, ~30h) | Quick (~30 min); minimal CPU contention |
| JP HMM done | **UK/JP-N2 JP**: production CS model + factor alphas | J still running | Quick (~30 min) |
| All done | **UK/JP-N3**: append UK/JP section to `RESULTS_LOG.md` (no `.tex` edit) | K + L (US final report + prose-edit list) | Pure file work |
| Gated | **UK/JP-N4**: thesis edits for UK/JP findings — bundled with US Step M after Gilad greenlight | M | — |

### Quick-launch commands (run in order after US Step D completes)

```bash
# UK/JP-N1: production HMM fits (parallel-friendly with US Step E/F/G/H)
nohup python scripts/hmm_intl.py --region UK --seeds 200 \
    > logs/hmm_intl_uk_prod.log 2>&1 &
nohup python scripts/hmm_intl.py --region JP --seeds 200 \
    > logs/hmm_intl_jp_prod.log 2>&1 &

# UK/JP-N2: production CS + Test A (after each region's HMM finishes)
python scripts/cross_sectional_intl.py --region UK \
    > logs/cs_intl_uk_prod.log 2>&1
python scripts/cross_sectional_intl.py --region UK --use-us-pi \
    > logs/cs_intl_uk_uspi_prod.log 2>&1
python scripts/cross_sectional_intl.py --region JP \
    > logs/cs_intl_jp_prod.log 2>&1
python scripts/cross_sectional_intl.py --region JP --use-us-pi \
    > logs/cs_intl_jp_uspi_prod.log 2>&1
```

### Decision branches after production rerun

After production HMM + CS fits land, two scenarios for the thesis:
1. **US-pi beats regional pi in BOTH regions** (current smoke pattern) →
   strongest publishable claim: "global financial cycle channel". Goes into
   the same Step F/K/L review process as the US story.
2. **US-pi beats regional pi in only ONE region or neither** → weaker but
   still publishable: "regional regime+momentum architecture transfers" with a
   regional-specific caveat. Frame in `latex/conclusion.tex` future-work
   section instead of as headline.

In either case, no `latex/*.tex` edit until Gilad reviews production numbers
(memory: `feedback_thesis_edits_await_data`).

---

## Detailed steps (legacy reference — keep for traceability)

### Step 1 — Adapt `scripts/hmm_model.py` for regional inputs ✅ DONE

The production `hmm_model.py` reads `data/panel.parquet` and uses
`HMM_FEATURES` from `config.py`. For UK/JP we need it to read
`data/{region}_market_panel.parquet` and use the regional 4-feature set
(`DD_z`, `DISP_z`, `REL_N_z`, `BANK_REL_z`).

**Two adaptation options:**

(a) **Add `--region` CLI flag to `scripts/hmm_model.py`.** When set:
  - Load `data/{region}_market_panel.parquet` instead of `data/panel.parquet`
  - Override `HMM_FEATURES` to `['DD_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z']`
  - Write outputs to `data/{region}_panel_with_regimes.parquet` and
    `data/{region}_mcmc_draws.npz`

(b) **Write a thin wrapper `scripts/hmm_intl.py`** that imports core fitting
  functions from `hmm_model.py` and applies them with regional config. Keeps
  the production script untouched.

**Recommendation: (b)** — `hmm_model.py` is referenced from many places
(`expanding_window_backtest_parallel.py`, `historical_oos_production.py`,
`hmm_cv.py`); modifying it risks side effects in active backtests. A wrapper
isolates the regional path.

**Skeleton:**

```python
# scripts/hmm_intl.py
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts')
import hmm_model  # use its fitting helpers; do not rely on its main()

REGION_FEATURES = ['DD_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z']

parser = argparse.ArgumentParser()
parser.add_argument('--region', choices=['UK', 'JP'], required=True)
parser.add_argument('--seeds', type=int, default=200)
parser.add_argument('--smoke', action='store_true', help='5 seeds for wiring check')
args = parser.parse_args()

if args.smoke:
    args.seeds = 5

market = pd.read_parquet(f'data/{args.region.lower()}_market_panel.parquet')
# Drop rows where any feature is NaN (early sample where rolling not yet warm)
market = market.dropna(subset=REGION_FEATURES).reset_index(drop=True)

# Run fitting using helpers from hmm_model — exact API depends on the file;
# see hmm_model.py for the entry points it exposes (likely fit_hmm or similar).
# Output: pi_filter, pi_smooth, pi_next per month + MCMC draws.

# Save outputs
out_pi    = f'data/{args.region.lower()}_panel_with_regimes.parquet'
out_mcmc  = f'data/{args.region.lower()}_mcmc_draws.npz'
# market_with_regimes.to_parquet(out_pi)
# np.savez(out_mcmc, ...)
```

**Validation criterion:** smoke run completes, `pi_filter` is bounded in
[0, 1], and the panic-state probability spikes at known crisis dates (UK:
2008-09, 2008-12, 2020-03; JP: 2001-2002, 2020-03 — but NOT 2008 in Japan
since Japanese banks did not underperform in 2008).

---

### Step 2 — Fit regional HMMs

**Smoke test (~5 seeds, ~1-2 min):**

```bash
python scripts/hmm_intl.py --region UK --smoke
python scripts/hmm_intl.py --region JP --smoke
```

Validate `pi_filter` looks sane (see criterion above) before committing CPU.

**Production fit (~200 seeds, longer):**

```bash
python scripts/hmm_intl.py --region UK --seeds 200
python scripts/hmm_intl.py --region JP --seeds 200
```

**CPU budget:** if backtest is still running on 6 workers, use single-thread
or limit parallel seeds to 2. If backtest is finished, use full
multiprocessing.

**Outputs:**
- `data/uk_panel_with_regimes.parquet`, `data/jp_panel_with_regimes.parquet`
- `data/uk_mcmc_draws.npz`, `data/jp_mcmc_draws.npz`

---

### Step 3 — Adapt `scripts/cross_sectional_model.py` for regional inputs

Same wrapper pattern. The production script reads:
- `data/panel_with_regimes.parquet` (regime labels)
- and (in the panel-builder upstream) `data/panel.parquet` (stock-level data,
  including fundamentals)

Regional version reads:
- `data/{region}_panel_with_regimes.parquet`
- `data/{region}_stock_panel.parquet`
- Drops fundamentals from the feature list. Keeps only `pi_filter` +
  `mom_1..mom_12`.

Use SIC-based decile breakpoints (no NYSE-equivalent for UK/JP); top decile by
month × predicted-return for the long leg, bottom for short. Value-weighted.

**Skeleton:** `scripts/cross_sectional_intl.py` modeled after
`scripts/cross_sectional_model.py`. Read the existing script for entry points;
key functions to reuse: portfolio construction, decile sorts, performance
metrics, factor regressions.

---

### Step 4 — Run cross-sectional model and report

```bash
python scripts/cross_sectional_intl.py --region UK
python scripts/cross_sectional_intl.py --region JP
```

**Outputs:**
- `results/intl_uk_returns.csv`
- `results/intl_jp_returns.csv`
- `results/intl_summary.csv` — Sharpe, mean return, vol, max DD,
  factor-model alphas vs Fama-French 3F (Asness 2013 international FF dataset
  if available; otherwise CAPM via regional market index).

**Reporting target:** report Sharpe and alpha for each region under each
method (Method 0 deterministic formula, Method 1 LR, Method 2 XGB) — same
benchmarks as US production. Compare against:
- Regional market portfolio Sharpe
- Fixed 12-mo momentum Sharpe (no regime conditioning)
- Fixed 1-mo momentum Sharpe

If regional Sharpe materially beats fixed momentum, the regime mechanism
generalizes. If similar, regime conditioning doesn't add international value.

---

### Step 5 — Test A (global financial cycle): apply US `pi_filter` to regional momentum

Stronger scientific claim. Take the existing US-trained `pi_filter` from
`data/panel_with_regimes.parquet` and apply it as the regime input to the
regional cross-sectional model:

```bash
python scripts/cross_sectional_intl.py --region UK --use-us-pi
python scripts/cross_sectional_intl.py --region JP --use-us-pi
```

**Hypothesis:** US `pi_filter` predicts regional momentum crashes via the
global financial cycle (Rey 2015). If yes, write up as the headline finding.
If only the regional `pi_filter` works, that's still a generalization result
just a weaker one.

---

### Step 6 — Update `RESULTS_LOG.md` with the numbers

Append a new section "International validation" with:
- Sharpe table (UK / JP × Methods 0/1/2 × benchmarks)
- Factor alphas (CAPM / FF3 international if available)
- Subperiod stability (does the result hold in 2011-2017 vs 2018-2025?)
- Key BANK_REL crisis-month identifications (validation of regime selection)

DO NOT edit any `.tex` until Gilad reviews the results (memory:
`feedback_thesis_edits_await_data.md`).

---

## Re-running the data ingestion (if needed)

The current panels were built with a minor bug fix applied via `--from-
checkpoint` (mom_k boundary fix). The 0.005-0.04% follow-up `ret` artifact is
the only remaining minor known issue. To eliminate it:

```bash
# Wipe old outputs (NOT the checkpoints — they're already deleted)
rm -f data/uk_stock_panel.parquet data/uk_market_panel.parquet
rm -f data/jp_stock_panel.parquet data/jp_market_panel.parquet

# Re-pull from WRDS — needs ~30 min total + memory budget; system load 7+ from
# the running backtest means this is best done after backtest finishes
python scripts/import_intl_stocks.py --region UK
python scripts/import_intl_stocks.py --region JP
```

The current panels are good enough for a first-pass validation. Re-pull only
if a final paper-grade run is needed.

---

## Session-recovery checklist (if this session disconnects)

A future Claude session should:

1. **Read this file** (`INTL_VALIDATION_PLAN.md`) and `data/INTL_PANEL_README.md`.
2. **Verify data presence:** `ls -la data/uk_stock_panel.parquet data/uk_market_panel.parquet data/jp_stock_panel.parquet data/jp_market_panel.parquet`. If any missing → run Step 0 (re-ingestion).
3. **Verify panel quality:**
   ```python
   import pandas as pd
   for r in ['uk', 'jp']:
       m = pd.read_parquet(f'data/{r}_market_panel.parquet')
       assert {'DD_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z'}.issubset(m.columns)
       assert len(m) >= 430
       s = pd.read_parquet(f'data/{r}_stock_panel.parquet')
       first = s.sort_values(['secid', 'date']).groupby('secid').head(1)
       assert first['mom_1'].isna().all()  # mom_k bug fixed
   ```
4. **Check what's done vs pending:**
   - `ls data/*panel_with_regimes.parquet` → if `uk_panel_with_regimes.parquet` and
     `jp_panel_with_regimes.parquet` exist, Step 2 is done.
   - `ls scripts/hmm_intl.py` → if exists, Step 1 is done.
   - `ls scripts/cross_sectional_intl.py` → if exists, Step 3 partially done.
   - `ls results/intl_*.csv` → if any present, Step 4 is partially done.
5. **Check backtest status:** `ps -p $(pgrep -f expanding_window_backtest)` —
   if alive, use ≤2 cores for HMM/CS work.
6. **Resume from the first incomplete step.** Each step's command is in this
   file.

---

## File map

| Path | Status | Purpose |
|---|---|---|
| `scripts/import_intl_stocks.py` | ✅ written, tested | UK/JP data ingestion from WRDS Compustat Global |
| `data/uk_stock_panel.parquet` | ✅ 514K rows | UK monthly stock panel |
| `data/uk_market_panel.parquet` | ✅ 434 months | UK HMM features |
| `data/jp_stock_panel.parquet` | ✅ 1.42M rows | JP monthly stock panel |
| `data/jp_market_panel.parquet` | ✅ 434 months | JP HMM features |
| `data/INTL_PANEL_README.md` | ✅ written | Schema + caveats |
| `archive/scripts/wrds_diag_*.py` | ✅ archived | WRDS schema diagnostics (one-off) |
| `scripts/hmm_intl.py` | ❌ TODO | Regional HMM wrapper |
| `data/{uk,jp}_panel_with_regimes.parquet` | ❌ TODO | Regional regime probabilities |
| `data/{uk,jp}_mcmc_draws.npz` | ❌ TODO | Regional MCMC draws |
| `scripts/cross_sectional_intl.py` | ❌ TODO | Regional cross-sectional model |
| `results/intl_*.csv` | ❌ TODO | UK/JP returns + summary |

---

## Memory references

- `project_state` — high-level project structure
- `project_cs_definition` — CS feature in US HMM is Moody's BAA-AAA, not HY OAS
  (relevant when explaining why BANK_REL substitutes for CS internationally)
- `project_ls_hmm_features` — production HMM uses DD+CS+DISP+REL_N
- `feedback_thesis_edits_await_data` — don't edit `.tex` until Gilad sees numbers
- `feedback_push_after_change` — push after each change; pull before starting
