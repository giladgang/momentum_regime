# Redundant results — archived 2026-04-27

These 7 files were moved out of `results/` because nothing in the thesis,
scripts, tables, or tests depends on them. All are gitignored
(see `.gitignore` lines 33-39), so they were never tracked in version
control and could not have backed any thesis number.

## What's here

### Group A: Pytest-integration expanding smoke (3 files)
- `expanding_returns_pytest_integration.csv`
- `expanding_pi_filter_pytest_integration.csv`
- `expanding_summary_pytest_integration.csv`

Written by `tests/test_expanding_window.py` (uses `tag='pytest_integration'`)
as integration-test smoke output. Regenerate with:

    python -m pytest tests/test_expanding_window.py

### Group B: CV smoke serial/parallel A/B variants (4 files)
- `hmm_cv_smoke_serial.csv`
- `hmm_cv_smoke_parallel.csv`
- `xgb_cv_smoke_serial.csv`
- `xgb_cv_smoke_parallel.csv`

Stragglers from a one-off serial-vs-parallel speedup comparison. The live
CV scripts (`scripts/hmm_cv.py`, `scripts/xgb_cv.py`) only write canonical
`hmm_cv_smoke.csv` / `xgb_cv_smoke.csv`. The canonical xgb smoke output
remains in `results/`; the canonical hmm smoke is regenerated on demand
with:

    python scripts/hmm_cv.py --smoke
    python scripts/xgb_cv.py --smoke

## Verification trail (before move)

- `grep -rn` across `scripts/`, `tests/`, `tables/`, `latex/`, `main.tex`,
  `NEXT_RUN_PLAN.md`, `TODO.md`, `CHECKLIST.md`,
  `results/THESIS_EDITS_TODO.md`: zero references to any of the 7 names.
- `.gitignore` lines 33-39 explicitly mark them as "kept locally, not in git".
- Live CV-test script `tests/test_cv_scripts.py` reads only the canonical
  `hmm_cv_smoke.csv` / `xgb_cv_smoke.csv` paths (lines 244-245).
