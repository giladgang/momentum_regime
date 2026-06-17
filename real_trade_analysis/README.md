# real_trade_analysis

Exploratory analysis (May 2026) comparing the thesis XGB regime-momentum strategy
against the **S&P 500 Momentum index / SPMO ETF**, and characterising its behaviour
on large-cap ("real-tradeable") universes. **Throwaway / experiment-grade** — not part
of the production pipeline, not promoted to `scripts/`, not in `RESULTS_LOG.md`.

All scripts use **production inputs** (200-seed HMM `pi_filter` from
`data/panel_with_regimes.parquet`, production XGB config from `config.py`, production
CRSP). The harness was reconciled bit-for-bit against the production artefact:
momentum features, sample, row order, and XGB are identical; the only difference vs
the logged headline (1.11 → ~1.08) is that the regime panel was regenerated May-9
*after* the May-7 production cross-sectional run (`pi_filter` MCMC noise ~1e-4).

## Layout
- `scripts/` — analysis scripts (run from anywhere; paths self-resolve to project root)
- `results/` — summary + monthly-return CSVs
- `plots/` — figures (PNG)
- `logs/` — run logs
- `data/` — helper inputs (`_spmo_real_monthly.csv` = real SPMO total return via Yahoo;
  `_top500_permnos_2010_2024.csv` = permnos ever in Top-500)

## Headline findings (test 2011-2024 unless noted; long-only/value-weighted; net 10bps)
- **Universe breadth drives the edge.** Full-universe L/S Sharpe ~1.08; restricting the
  *trading* universe to Top-500 (S&P 500 proxy) collapses L/S to 0.28 but long-only holds ~0.86-1.07.
- **Train broad, trade narrow.** The Top-500 long-only weakness was a *training* artifact:
  train on Top-1000-2000 (liquid, no microcaps) and trade the S&P 500 long-only -> Sharpe
  ~1.00 (full) / ~1.22 (2015-2024), matching/beating SPMO without learning on untradeable names.
- **vs SPMO:** SPMO replica = 15.1%/Sharpe 1.00 (2011-2024); real SPMO ETF = 16.9%/1.04
  (2015-2024). XGB long-only (train Top-1000, trade S&P 500) ties SPMO on the full period
  (Sharpe ~1.0, higher raw return at higher vol) and beats it over 2015-2024 (26.4%/1.22).
- **Size profile:** holdings are large-cap *by capital* (value-weighted long leg ~84th
  size pctile); the equal-weighted view (36th pctile) overstated the small-cap tilt.
  Consistent with the thesis's near-zero SMB loading.
- **log_me as a feature hurts slightly** (L/S 1.08 -> 0.95) — leave it out.

## Key scripts
| Script | Purpose |
|---|---|
| `2026-05-30-xgb-russell1000.py` | base module: build_features, apply_size_screen, run_xgb_ls; full vs Top-1000 L/S |
| `2026-05-30-xgb-size-sweep.py` | L/S Sharpe vs universe size (Full…Top-500) |
| `2026-05-31-spmo-replica-vs-xgb.py` | faithful S&P 500 Momentum (SPMO) replica + XGB through same machinery |
| `2026-05-31-xgb-longonly-sp500.py` | XGB long-only (top-decile VW) on S&P 500 |
| `2026-05-31-xgb-longonly-universe-sweep.py` | long-only Sharpe vs universe size, vs SPMO |
| `2026-05-31-train-universe-sweep.py` | **train universe swept, trade fixed Top-500** (the train-broad/trade-narrow result) |
| `2026-05-31-sp500-consolidated.py` | all strategies on one Top-500 universe + size profile |
| `2026-05-31-size-profile-valueweighted.py` | EW vs VW holdings size profile |
| `2026-05-31-xgb-logme-feature-and-size.py` | does adding log_me help? + per-year size |
| `2026-05-31-weighting-mechanism.py` | vw/ew/cap weighting — breadth vs small-name weight |
| `2026-05-31-debug-top500-dip.py` | diagnostic: Top-500 dip = per-universe retraining artifact |
| `2026-05-31-reconcile-production.py` / `-retrain.py` | reconcile harness vs production (read-only) |
| `2026-05-31-final-spmo-vs-xgb1000.py` | clean head-to-head: SPMO (replica+real) vs XGB train-1000/trade-S&P500 |
| `_pull_daily_top500.py` | (staged) WRDS daily pull for true risk-adjusted-momentum sigma — blocked on VPN |

## Caveats
Experiment-grade (±~0.03 Sharpe regime-vintage slack); single test period; Sharpe diffs
often within sampling noise; SPMO replica uses a monthly-sigma proxy (WRDS daily unreachable);
universe is a market-cap proxy, not real S&P 500 membership; L/S has no short-borrow costs;
real-SPMO calm/panic split has an unresolved date-alignment oddity.
