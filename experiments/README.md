# experiments/

One-off / exploratory scripts live here, not in `scripts/`.

## Convention

Name files `YYYY-MM-DD-short-description.py` (e.g. `2026-04-26-fwdvol-bucket-test.py`).

Anything in this directory is treated as throwaway: no expectation of test coverage, reproducibility, or maintenance. Once an idea proves itself, promote it: rename, refactor, drop the date prefix, and move it under the appropriate `scripts/` subdirectory with tests.

This keeps `scripts/` tight and prevents the recurring "move to `archive/`" cycle.
