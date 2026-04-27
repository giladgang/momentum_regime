"""
Canonical post-Shumway expected values for thesis numbers.
Single source of truth shared by test_thesis_consistency.py and
test_pipeline_technical.py so the two suites cannot drift apart
(as they did between the bb9545c rerun and the 1264754 recalibration,
where pipeline_technical kept the old expected values while
thesis_consistency was updated).

Tolerances are tight enough to catch real drift but loose enough to
absorb seed noise within the published 200-HMM x 50-XGB ensemble.
Update this file when a thesis-published number is intentionally
recalibrated.
"""

# M2 production performance (results/thesis/expanding_summary_prod.csv +
# tables/table_performance.tex):
M2_SHARPE_FULL    = 1.11   # ann Sharpe over full 2011-2025 OOS window
M2_ANN_RET_PCT    = 21.7   # ann return %
M2_ANN_VOL_PCT    = 19.5   # ann vol %
M2_MAX_DD_PCT     = -72.2  # signed max drawdown %

# Tolerances
TOL_SHARPE        = 0.02
TOL_ANN_PCT       = 0.1

# Regime-conditional Sharpes (tab:regime_sharpe / canonical macros):
M2_SHARPE_PANIC   = 1.53
M2_SHARPE_CALM    = 0.84
TOL_REGIME_SHARPE = 0.02

# HMM ablation table (table_regime_signal_ablation.tex):
ABLATION_HMM_SHARPE = 1.107  # current measured value; tab:ablation row
TOL_ABLATION        = 0.01

# Subperiod Sharpes (tab:subperiod):
SUBPERIOD_M2 = [0.59, 1.04, 1.70]
TOL_SUBPERIOD = 0.01

# Factor alphas (table_factor_alphas.tex):
FF6_ALPHA_PCT = 24.1
FF6_TSTAT     = 4.81
TOL_ALPHA     = 0.05
TOL_TSTAT     = 0.01

# Depth analysis (results/thesis/depth_results.csv):
DEPTH4_ANN_RET_PCT = 21.7
DEPTH4_SHARPE      = 1.11
TOL_DEPTH          = 0.1
