#!/bin/bash
# Top-500 robustness rerun: full honest chain (fold grid -> walk -> report)
# in its own results tree. Resume-safe.
# nohup caffeinate -dims bash paper/run_top500.sh >> paper/results_top500.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export PAPER_UNIVERSE_N=500
export PAPER_RESULTS_DIR=paper/results_top500
PY=.venv/bin/python
$PY -m paper.pipeline --stage universe_check
$PY -m paper.pipeline --stage folds --workers 7
$PY -m paper.pipeline --stage walk --workers 7
$PY -m paper.pipeline --stage report --workers 2
echo "=== top500 chain DONE $(date) ==="
