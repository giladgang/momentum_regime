#!/usr/bin/env bash
# N2-only restart after the original chain crashed on bash 3.2's lack of
# ${var,,} lowercase expansion. Step J + N1 UK + N1 JP already completed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== Phase 2 N2-only restart: $(date) ===" | tee -a logs/phase2_chain.log

for region in UK JP; do
  region_lc=$(echo "$region" | tr '[:upper:]' '[:lower:]')
  for variant in regional uspi; do
    flag=""
    suffix=""
    if [ "$variant" = "uspi" ]; then
      flag="--use-us-pi"
      suffix="_uspi"
    fi
    echo "[chain] Launching Step N2 ${region} ${variant} ..." \
      | tee -a logs/phase2_chain.log
    python scripts/cross_sectional_intl.py --region "$region" \
      --xgb-seeds 50 ${flag} \
      > "logs/cs_intl_${region_lc}${suffix}_prod.log" 2>&1
    echo "[chain] Step N2 ${region} ${variant} done: $(date)" \
      | tee -a logs/phase2_chain.log
  done
done

echo "=== Phase 2 N2 restart COMPLETE: $(date) ===" | tee -a logs/phase2_chain.log
