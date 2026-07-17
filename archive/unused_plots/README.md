# Unused plots

39 figures moved here from the repository root (`unused_plots/`) on 2026-07-17.

## Why archived

These were already segregated as unused at the root level (see
`results/reports/ORGANIZATION_PLAN.md`). They are exploratory figures that were
generated during research but never `\includegraphics`'d from `latex/*.tex` or
`main.tex`, so none of them appear in the thesis.

They are kept rather than deleted, per the `archive/` convention in `CLAUDE.md`
("retired code / outputs move to `archive/` via `git mv`; don't delete"). Full
history is preserved — the move was recorded as a rename, so
`git log --follow archive/unused_plots/<file>` still resolves.

## Relationship to the other archives

- `archive/stale_outputs_20260427_1853/` — 100 plots and results retired in the
  post-Shumway sweep, filtered on the pre-Shumway date cutoff.
- `archive/unused_plots/` — this directory. Never thesis-cited in the first
  place; predates that sweep and sat at the repo root.

## Provenance

Most originate from an Apr 2026 exploratory plotting session. The generating
scripts are in `archive/scripts/` (`zscore_long_*.py`, `cs_plots.py`,
`complementarity_analysis.py`, `plot_optimal_pi.py`, and peers).
