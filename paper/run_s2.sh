#!/bin/bash
# S2 overnight: applied-objective fold grid. Resume-safe relaunch.
# nohup caffeinate -dims bash paper/run_s2.sh >> paper/results/s2_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m paper.pipeline --stage folds --workers 7
