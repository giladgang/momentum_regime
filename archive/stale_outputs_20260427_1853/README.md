# Stale outputs archived 2026-04-27

100 files moved here from `plots/` and `results/` because they are:
- Pre-Shumway dated (before Apr 26 2026 15:00, the post-Shumway pipeline cutoff)
- AND not referenced by any `\input` or `\includegraphics` in `latex/*.tex` or `main.tex`
- AND not mentioned in `TODO.md`
- AND not consumed as input by any active script in `scripts/`

## Why archived

Each file was generated during exploratory research (mostly an Apr 14 plotting
session) and is either superseded by a newer plot or never made it into the
thesis. None of them are cited in published prose.

## Provenance

The audit that produced this list:

```bash
python3 -c "..."  # see scripts/build_metrics.py and scripts/_config_deps.py
```

For the exact selection logic, see the audit block in the conversation log.

## Categories

- ~95 exploratory plots (chart variants, shap variants, score_vs_momentum
  variants, portfolio_chars options, risk_aversion options, etc.)
- 2 historical-OOS summary CSVs at older train-end dates (1990-1999, 1990-2004)
  — superseded by the 30-year expanding-window backtest in `expanding_*.csv`
- 1 shap_portfolio_analysis.csv — superseded by `shap_dependence_*.csv`

## Restoration

If something turns out to be needed:

```bash
mv archive/stale_outputs_20260427_1853/plots/<file> plots/
mv archive/stale_outputs_20260427_1853/results/<file> results/
```
