#!/bin/bash
# S3 overnight: yearly walk (50x20 seeds) + report. Resume-safe relaunch.
# nohup caffeinate -dims bash paper/run_s3.sh >> paper/results/s3_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m paper.pipeline --stage walk --workers 7
.venv/bin/python -m paper.pipeline --stage report --workers 2
