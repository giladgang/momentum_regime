#!/bin/bash
# Refresh all stale plots by running each generator serially.
# These are mostly fast (read cs_artefacts, render matplotlib).
# Total expected wall time: ~10-30 minutes serial.
set -e
cd "$(dirname "$0")/.."

mkdir -p logs/refresh

SCRIPTS=(
    horizon_weight_share
    long_leg_all_horizons
    long_leg_all_horizons_bar
    long_leg_momentum_timeseries
    momentum_share_chart
    portfolio_by_horizon_long_short
    portfolio_characteristics_chart
    portfolio_chars_option_c_v2
    portfolio_chars_options
    portfolio_momentum_profile
    risk_aversion_analysis
    score_vs_momentum
    score_vs_momentum_v2
    selection_rate_by_decile
    selection_rate_long_short
    shap_by_leg_regime
    shap_contribution_by_horizon
    shap_dependence_mom12
    shap_horizon_long_short_final
    shap_horizon_pct_final
    shap_signed_long_short
    zscore_pi_only
)

echo "Refreshing $(echo ${SCRIPTS[@]} | wc -w) plot/analysis scripts serially..."
echo "Started: $(date)"
echo ""

i=0
for s in "${SCRIPTS[@]}"; do
    i=$((i+1))
    echo "[$i/${#SCRIPTS[@]}] scripts/${s}.py"
    if python -u "scripts/${s}.py" > "logs/refresh/${s}.log" 2>&1; then
        echo "  ok"
    else
        echo "  FAILED — see logs/refresh/${s}.log"
    fi
done

echo ""
echo "Finished: $(date)"
echo "All logs in logs/refresh/"
