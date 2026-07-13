#!/bin/bash
# Live-2026 appendix chain: wait for fold-115 cells, then walk + report.
# nohup caffeinate -dims bash paper/run_live2026.sh >> paper/results/live2026_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== live2026 chain start $(date) ==="
n=0
until $PY -c "
import pandas as pd, sys
n = (pd.read_csv('paper/results/fold_cells.csv')['fold'] == 115).sum()
sys.exit(0 if n == 75 else 1)"; do
  sleep 120
  n=$((n + 1))
  if [ "$n" -ge 60 ]; then
    echo "TIMEOUT (2h) waiting for fold 115 — is S2 running?"
    exit 1
  fi
done
echo "--- fold 115 complete $(date) ---"
$PY -m paper.live2026 --stage walk --workers 7
$PY -m paper.live2026 --stage report
echo "=== live2026 chain DONE $(date) ==="
