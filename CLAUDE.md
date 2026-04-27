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

## Key files
- `TODO.md` — active task list. Render as markdown checklist; don't reformat.
- `NEXT_RUN_PLAN.md` — runbook for Shumway + full rerun + CV + leg-betas. Resumable.
- `INTL_VALIDATION_PLAN.md` — international replication plan.

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

## Verification — strict, applies to everything
**Default: nothing is "done" until double-checked AND evidence is shown to Gilad.**

**Scripts:** Run end-to-end after any change. Diff new vs prior `results/*.csv`
row-by-row when output shouldn't have moved. Sanity-check Sharpes/alphas/t-stats
vs published thesis numbers.

**LaTeX:** Every numeric claim cites a specific `results/*.csv` or `tables/*.csv`
row. Every qualitative claim traces to a specific analysis output. Re-read the
figure/table before describing it; never paraphrase from memory.

**Plots:** After generating, describe axes, sample range, and what jumps out.
Flag anomalies (sign flips, gaps, suspicious smoothness).

**Commits:** Show diff AND verification evidence (test output, numeric diffs,
plot inspection) before claiming done. If a "fix" was applied, re-run the check.

**Red flags — stop and recheck:** a number changed when it shouldn't have;
a previously failing test now passes without clear cause; the new result is
suspiciously clean.

**Exemptions (only):** pure typo / whitespace / comment edits with no semantic effect.

## Long jobs and parallelism
- Long pipeline runs → background. Don't block.
- Independent ablations → dispatch in parallel via subagents.
- Don't poll background jobs; wait for completion notification.
