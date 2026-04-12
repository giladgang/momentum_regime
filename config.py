"""
config.py
=========
Central configuration for the momentum regime shifts pipeline.
Edit parameters here and run run_pipeline.py to regenerate all results.
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  HMM SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Features used as HMM inputs (standardized z-scores)
# Selected via exhaustive search over C(9,4)=126 combinations
HMM_FEATURES = ['DD_z', 'DISP_z', 'REL_N_z', 'CS_z']

# MCMC seeds for Gibbs sampler (pi_filter averaged across these)
HMM_SEEDS = list(range(1, 201))  # 200 seeds for maximum stability

# Gibbs sampler settings
HMM_ITERATIONS = 2000
HMM_BURNIN = 500

# Number of hidden states
K_STATES = 2

# HMM training data start (limited by stock-level data availability)
HMM_START = '1990-01-01'

# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Portfolio construction type: 'long_short' or 'long_only'
PORTFOLIO_TYPE = 'long_short'

# One-way transaction cost (10 bps)
TRADING_FEE = 0.001

# Train/test split date
TRAIN_END = '2011-01-01'

# ═══════════════════════════════════════════════════════════════════════════════
#  XGBOOST SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# XGB seeds for ensemble (predictions averaged across these, then one portfolio built)
XGB_SEEDS = list(range(1, 51))  # 50 seeds to match heavy variance test

N_ESTIMATORS = 500
MAX_DEPTH = 4
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8
COLSAMPLE = 0.8

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Whether to include fundamental features in the cross-sectional model
# Set to False for the D&M comparison (momentum + pi_filter only)
USE_FUNDAMENTALS = True

# Momentum lookback horizons (months)
MOM_LOOKBACKS = list(range(1, 13))

# Fundamental features (used only if USE_FUNDAMENTALS = True)
FUND_FEATURES = [
    'bm',               # Book-to-market
    'roe',              # Return on equity
    'earnings_growth',  # Earnings growth
    'leverage',         # Leverage
    'asset_growth',     # Asset growth
    'gross_profit_a',   # Gross profitability
    'log_me',           # Log market equity
]

# Derived feature lists
MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LOOKBACKS]

if USE_FUNDAMENTALS:
    CS_FEATURES = MOM_FEATURES + ['pi_filter'] + FUND_FEATURES
else:
    CS_FEATURES = MOM_FEATURES + ['pi_filter']

# ═══════════════════════════════════════════════════════════════════════════════
#  ROBUSTNESS CHECK SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Multi-state HMM: number of states to test
K_STATES_ROBUSTNESS = [3, 4, 5]

# Transaction cost sensitivity levels (in bps)
COST_LEVELS_BPS = [0, 5, 10, 20, 30, 50]

# Regime threshold sensitivity
PI_THRESHOLDS = [0.25, 0.50, 0.75]

# Alternative train/test splits
ALT_SPLITS = {
    '1990-2004 / 2005-2025': '2005-01-01',
    '1990-2007 / 2008-2025': '2008-01-01',
    '1990-2010 / 2011-2025 (baseline)': '2011-01-01',
    '1990-2014 / 2015-2025': '2015-01-01',
}

# XGBoost hyperparameter sensitivity configs
XGB_CONFIGS = [
    ('depth=3, lr=0.05, n=500', {'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 500}),
    ('depth=4, lr=0.05, n=500', {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500}),  # baseline
    ('depth=5, lr=0.05, n=500', {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 500}),
    ('depth=6, lr=0.05, n=500', {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500}),
    ('depth=4, lr=0.01, n=500', {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 500}),
    ('depth=4, lr=0.10, n=500', {'max_depth': 4, 'learning_rate': 0.10, 'n_estimators': 500}),
    ('depth=4, lr=0.05, n=200', {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 200}),
    ('depth=4, lr=0.05, n=1000', {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 1000}),
]

# Sub-period definitions
SUB_PERIODS = [
    ('2011-2015', '2011-01-01', '2016-01-01'),
    ('2016-2020', '2016-01-01', '2021-01-01'),
    ('2021-2025', '2021-01-01', '2026-01-01'),
    ('Full', '2011-01-01', '2026-01-01'),
]

# Student-t emission degrees of freedom to test
STUDENT_T_NU = [3, 5, 7, 10, 15, 20, 30, 50, 100]

# Number of placebo shuffles/random runs
N_PLACEBO_RUNS = 5

# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

TABLES_DIR = 'tables'
PLOTS_DIR = '.'  # plots saved to project root (matching current convention)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PATHS
# ═══════════════════════════════════════════════════════════════════════════════

PANEL_PATH = 'data/panel.parquet'
PANEL_WITH_REGIMES_PATH = 'data/panel_with_regimes.parquet'
STOCK_DATA_PATH = 'data/crsp_msf_raw.parquet'
ARTEFACTS_PATH = 'cs_artefacts_data.pkl'
FF_FACTORS_PATH = 'data/ff_factors.parquet'
EXTRA_HMM_FEATURES_PATH = 'data/extra_hmm_features.pkl'
