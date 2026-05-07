#!/usr/bin/env bash
# Phase 2 autonomous chain. Waits for Step I (xgb_cv), then runs:
#   Step J  (hmm_cv full, --workers 6)
#   Step N1 UK (hmm_intl --region UK --workers 6)
#   Step N1 JP (hmm_intl --region JP --workers 6)
#   Step N2  (4× cross_sectional_intl)
# Each subsequent step's launch is gated on the previous one's success-marker
# file (xgb_cv_winner.json, hmm_cv_winner.json, etc.).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

echo "=== Phase 2 chain started: $(date) ===" | tee -a logs/phase2_chain.log

# ---- Step J: HMM CV (gated on Step I's xgb_cv_winner.json) ------------------
echo "[chain] Waiting for Step I to complete (xgb_cv_winner.json) ..." \
  | tee -a logs/phase2_chain.log
until [ -f results/xgb_cv_winner.json ]; do sleep 60; done
echo "[chain] Step I done: $(date). Launching Step J ..." \
  | tee -a logs/phase2_chain.log
python scripts/hmm_cv.py --workers 6 \
  > logs/hmm_cv_full.log 2>&1
echo "[chain] Step J done: $(date)" | tee -a logs/phase2_chain.log

# ---- Step N1 UK: production HMM ---------------------------------------------
echo "[chain] Launching Step N1 UK ..." | tee -a logs/phase2_chain.log
python scripts/hmm_intl.py --region UK --workers 6 \
  > logs/hmm_intl_uk_prod.log 2>&1
echo "[chain] Step N1 UK done: $(date)" | tee -a logs/phase2_chain.log

# ---- Step N1 JP: production HMM ---------------------------------------------
echo "[chain] Launching Step N1 JP ..." | tee -a logs/phase2_chain.log
python scripts/hmm_intl.py --region JP --workers 6 \
  > logs/hmm_intl_jp_prod.log 2>&1
echo "[chain] Step N1 JP done: $(date)" | tee -a logs/phase2_chain.log

# ---- Step N2: 4 cross-sectional production runs ------------------------------
# Sequential to honor 6-core cap (cross_sectional_intl uses XGB internally).
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

echo "=== Phase 2 chain COMPLETE: $(date) ===" | tee -a logs/phase2_chain.log
