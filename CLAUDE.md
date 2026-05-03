# Momentum Regime Shifts — Claude Guidelines

## What this project is
Master's thesis: HMM regime classifier + cross-sectional L/S momentum model.
Code in `scripts/`, results in `results/`, plots in `plots/`, LaTeX tables in
`tables/`, paper in `main.pdf`.

## Workflow
- Pull before push and before starting work — Gilad edits on Overleaf in parallel.
- Push after every meaningful change.
- Never use em-dashes (—, ---) in `.tex` files.
- Don't edit thesis `.tex` for data-dependent claims until Gilad has reviewed the numbers.
- Never force-push, amend pushed commits, `--no-verify`, or skip hooks.

## Domain conventions
- Strategy is always long/short. Don't caveat sub-test Sharpes as "L/S vs long-only".
- CS = Moody's BAA–AAA spread (US IG). Not HY OAS.
- L/S HMM features: DD + CS + DISP + REL_N (CS replaces VOL).
- Mechanism: model selects by momentum term-structure *shape*, not per-horizon slopes.

## Code layout
- `experiments/` — exploratory / throwaway. Name `YYYY-MM-DD-short-desc.py`. No tests, no reproducibility expectation.
- `scripts/` — production pipeline. Stable names, no date prefix, config-driven paths, has a test.
- `archive/scripts/` — retired code. Move superseded scripts here via `git mv`; don't delete.
- Shared helpers in `scripts/_*.py` (`_canonical_metrics.py`, `_config_deps.py`, `bootstrap_helpers.py`). Import, don't inline.
- Portfolio construction lives in `src/utils.py`, not `scripts/`.
- All paths come from `config.py`. No hardcoded paths.

## Pipeline
- `run_pipeline.py` is the canonical orchestrator. Its `steps = {…}` dict is the DAG; `sorted(steps.keys())` is execution order.
- Step number ranges (preserve these):
  - 1–8 core (HMM → CS → portfolio → main results / robustness / validity / plots)
  - 9–16 stress / mechanism / extensions
  - 17–35 auxiliary thesis-cited generators
  - 80–83 canonical aggregation chain (`build_metrics` → `build_canonical_macros` → `build_thesis_tables` → `verify_thesis_consistency`)
  - 95–102 tests (pytest)
- New generators go in 17–35 (or extend 9–16 if thematic). Pick a number that sorts after inputs and before consumers.
- Never insert in 80–83. The chain consumes everything upstream; if a generator sorts after step 80, `PRODUCTION_METRICS.json` lags by one run (incident 2026-04-28).
- State dependencies in the step's description string ("depends on Step N").
- `_chain_phase2*.sh` is a separate orchestrator for long-running CV/intl ablations, not the canonical pipeline.

## "Thesis-worthy" — the single trigger
A script or result is **thesis-worthy** when Gilad explicitly declares it so ("this goes in the thesis", "promote this", "add this to the log"). Until then it stays exploratory.

Once declared thesis-worthy, ALL of the following must happen, atomically in one commit:

1. **Promote `experiments/` → `scripts/`** via `git mv` (preserve history). Drop date prefix, match peer naming, refactor to use `config.py` paths, strip exploratory cruft.
2. **Add a test** under `tests/` (mirror the closest existing `test_*.py`). Smoke test minimum: runs on a fixture, checks output schema + row count.
3. **Register in `run_pipeline.py`** `steps` dict. Pick a step number per the ranges above. Run-test the pipeline end-to-end with the new step.
4. **Register in `RESULTS_LOG.md`** — numbered section, `table_*` LaTeX label, source CSV path, numbers in a markdown table, one-line interpretation.
5. **Update `build_metrics.py`** if the script produces a number cited in thesis prose, so the metric flows into `PRODUCTION_METRICS.json` → `canonical_macros.tex`. Otherwise prose can't reference it.

If thesis-worthy is unclear, ask before promoting. If something *seems* thesis-worthy but hasn't been declared, surface it for a decision; don't promote silently.

Non-thesis-worthy (smoke, CV-only, exploratory) results do NOT go in `RESULTS_LOG.md` or `run_pipeline.py`.

### Plots and figures
A plot is thesis-worthy when `\includegraphics`'d from `latex/*.tex`. Plus:
- Save both `.png` and `.pdf` to `plots/thesis/`. PDF for text/lines, PNG for dense raster.
- Register in `run_pipeline.py`, step number sorting after data inputs.
- Add entry to `results/reports/PROVENANCE.md` (asset, type, source script, config knob, cost to rerun).
- Most thesis plots come from `scripts/generate_plots.py` (Step 8). New plots sharing that data source belong inside it; new plots with their own data prep get their own step.
- `plots/diagnostic/` and `plots/` (root) are NOT thesis-worthy.

### Why — silent staleness incident, 2026-04-27/28
Two thesis-cited z-score CSVs stayed pre-Shumway because their generators weren't in `run_pipeline.py`. Same week: the canonical chain (steps 21–24) sorted *before* its inputs (25–33), so `PRODUCTION_METRICS.json` lagged by one run; renumbered to 80–83. Both: a thesis number existed but wasn't produced by the canonical run. See bb9545c, 2ba191c.

## Sources of truth
- `RESULTS_LOG.md` — narrative reference; every thesis-worthy *number* has an entry.
- `results/PRODUCTION_METRICS.json` — machine-readable canonical metrics. Built by `build_metrics.py` (step 80) from `tables/` + `results/`. Drives `latex/canonical_macros.tex` (`\newcommand` per metric for thesis prose) and `build_thesis_tables.py` (headline tables).
- `results/reports/PROVENANCE.md` — chapter-by-chapter map of every thesis *asset* (`\input{}` / `\includegraphics{}`) back to its source script + config knob + cost to rerun.
- A thesis-worthy number lives in BOTH `RESULTS_LOG.md` and `PRODUCTION_METRICS.json`. A thesis-worthy table or figure lives in `PROVENANCE.md`. Numbers and assets are tracked separately.
- Superseded `RESULTS_LOG.md` entries get struck through and linked to the replacement — never deleted.

## Results layout
- `results/thesis/` — CSVs backing numeric claims in `main.tex`.
- `results/cv/`, `results/oos_historical/`, `results/reports/` — non-thesis outputs.
- `tables/` — LaTeX-ready tables (built by `build_thesis_tables.py`, step 82).
- `plots/thesis/` — thesis figures. `plots/` — exploratory.
- `artefacts/` — pickles / npz / large regenerable binaries. Gitignored.
- Smoke / pytest outputs match gitignored patterns (`*_smoke_*.csv`, `*_pytest_integration.csv`, `results_smoke/`).

## Key files
- `TODO.md` — active task list. Render as markdown checklist; don't reformat.
- `NEXT_RUN_PLAN.md` — runbook for Shumway + full rerun + CV + leg-betas. Resumable.
- `INTL_VALIDATION_PLAN.md` — international replication plan.
- `architecture_diagram.tex` — visual DAG of the canonical pipeline.

## Git
- Local folder: `momentum_regime_shifts` (descriptive). GitHub remote: `giladgang/momentum_regime`. The old name `thesis_git` is auto-redirected by GitHub forever.

## Reproducibility
- Run scripts in the env defined by `requirements.lock`. Don't `pip install` a new
  package without updating the lock and flagging it.
- Every stochastic run logs its seed in the output filename or a sidecar.
- Never rerun a thesis-published number without the original seed unless the
  rerun *is* the experiment.

## Data integrity
- Features at time *t* use only information available at *t* (or earlier, if
  there's a publication lag).
- Train/test splits are time-ordered, never random.
- If unsure whether a column is point-in-time, ask before assuming.

## Backups
- Before any run that overwrites `results/` or `tables/`, snapshot to
  `baseline_pre_<change>_<YYYYMMDD_HHMMSS>/`.
- Don't delete prior baselines without asking.

## Verification — strict
Nothing is "done" until double-checked AND evidence shown to Gilad.
- Scripts: run end-to-end after any change. Row-diff CSVs when output shouldn't move. Sanity-check Sharpes/alphas/t-stats vs published numbers.
- LaTeX: every numeric claim cites a specific `results/*.csv` or `tables/*.csv` row. Re-read the figure/table; never paraphrase from memory.
- Plots: after generating, describe axes, sample range, anomalies (sign flips, gaps, suspicious smoothness).
- Commits: show diff + verification evidence before claiming done. Re-run checks after any "fix".
- Red flags — stop and recheck: a number moved when it shouldn't have; a failing test now passes without cause; result is suspiciously clean.
- Exempt: pure typo / whitespace / comment edits.

## Long jobs and parallelism
- Long pipeline runs → background. Don't block.
- Independent ablations → dispatch in parallel via subagents.
- Don't poll background jobs; wait for completion notification.
