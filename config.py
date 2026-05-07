"""
config.py
=========
Central configuration for the momentum regime shifts pipeline.
Edit parameters here and run run_pipeline.py to regenerate all results.
"""

import os as _os

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

# ── HMM priors ────────────────────────────────────────────────────────────────
# NIW prior on regime emissions (mu_k, Sigma_k). Hyperparameters:
#   m_0       : prior mean of mu_k (features are z-scored, so 0)
#   kappa_0   : prior strength on the mean (nearly flat at 0.01 pseudo-obs)
#   nu_0_off  : additive offset so nu_0 = D + nu_0_off (default D+2, the
#               minimum integer giving a proper Inverse-Wishart)
# The scale matrix Psi_0 is computed in hmm_model.py as eye(D)*(nu_0-D-1)
# so that E[Sigma_k] = I_D before observing data.
HMM_PRIOR_M0       = 0.0
HMM_PRIOR_KAPPA0   = 0.01
HMM_PRIOR_NU0_OFF  = 2   # nu_0 = D + HMM_PRIOR_NU0_OFF

# Dirichlet prior on each row of the transition matrix. Diagonal is 9,
# off-diagonal is 1, so E[P] has 0.90 persistence in each regime.
HMM_PRIOR_DIRICHLET_ALPHA = [[9.0, 1.0],
                             [1.0, 9.0]]

# Crisis windows used for the production HMM sign-correction step.
# Two-window definition (dot-com + GFC) matches hmm_model.py and
# hmm_feature_selection.py. Expanding-window OOS scripts build their
# own per-training-window masks and intentionally do not use this list.
CRISIS_WINDOWS = [
    ('2000-03-01', '2002-10-01'),
    ('2007-10-01', '2009-06-01'),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Portfolio construction type: 'long_short' or 'long_only'.
# Honour MOMENTUM_PORTFOLIO_TYPE env var so ad-hoc sandbox runs (e.g.,
# long-only sensitivity, portfolio-type ablations) can override the
# production setting without editing this file. Pair with
# MOMENTUM_OUTPUT_ROOT to redirect outputs and avoid clobbering production
# artefacts.
PORTFOLIO_TYPE = _os.environ.get('MOMENTUM_PORTFOLIO_TYPE', 'long_short')
assert PORTFOLIO_TYPE in ('long_short', 'long_only'), (
    f"PORTFOLIO_TYPE must be 'long_short' or 'long_only', got {PORTFOLIO_TYPE!r}"
)

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

# Per-XGB-fit thread cap. Ensemble loops are serial across seeds, so each
# fit gets these threads. Capped at 6 to leave 2 cores free for the OS
# and the user's other work on an 8-core machine.
XGB_N_JOBS = 6

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Whether to include fundamental features in the cross-sectional model
# Set to False for baseline (momentum + pi_filter only)
USE_FUNDAMENTALS = False

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
    'cfo_a',            # Operating cash flow / assets (TTM)
    'fcf_a',            # Free cash flow / assets (TTM, CFO - capex)
    'accruals',         # Sloan accruals: (ibq - oancf) / atq (TTM)
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

# When MOMENTUM_OUTPUT_ROOT is set in the environment (e.g. by run_pipeline.py
# --repro-smoke or by tests/thesis/make_smoke_fixture.py), every output path
# below is silently re-rooted under it. Pure data inputs (CRSP raw, FF factors,
# extra HMM features) are NOT redirected. This lets smoke runs and tests
# isolate from production results without scripts needing to know about the mode.
_OUTPUT_ROOT = _os.environ.get('MOMENTUM_OUTPUT_ROOT')


def _o(path: str) -> str:
    if _OUTPUT_ROOT and not _os.path.isabs(path):
        return _os.path.join(_OUTPUT_ROOT, path)
    return path


TABLES_DIR = _o('tables')
PLOTS_DIR = _o('plots')
RESULTS_DIR = _o('results')
ARTEFACTS_DIR = _o('artefacts')

# Subdirectory layout introduced in the production/cv/diagnostic restructure.
# Production thesis artifacts live in RESULTS_THESIS_DIR / PLOTS_THESIS_DIR.
# CV outputs (not cited in thesis) live in RESULTS_CV_DIR.
# Historical OOS sensitivities live in RESULTS_OOS_DIR.
# Status / drift reports live in RESULTS_REPORTS_DIR.
RESULTS_THESIS_DIR  = _o('results/thesis')
RESULTS_CV_DIR      = _o('results/cv')
RESULTS_OOS_DIR     = _o('results/oos_historical')
RESULTS_REPORTS_DIR = _o('results/reports')
PLOTS_THESIS_DIR    = _o('plots/thesis')
PLOTS_DIAGNOSTIC_DIR = _o('plots/diagnostic')

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PATHS
# ═══════════════════════════════════════════════════════════════════════════════

# Pipeline-built artefacts (honor MOMENTUM_OUTPUT_ROOT so smoke runs build
# fresh copies under the smoke root instead of overwriting production).
ARTEFACTS_PATH = _o('artefacts/cs_artefacts_data.pkl')
XGB_MODEL_PATH = _o('artefacts/cs_artefacts_xgb.pkl')
LR_MODEL_PATH = _o('artefacts/cs_artefacts_lr.pkl')

# Pure data inputs (read-only; never redirected, even in smoke mode).
# panel.parquet is built offline from CRSP+Compustat (not by run_pipeline).
# panel_with_regimes.parquet is technically a Step-1 output, but in smoke
# mode Step 1 is skipped when it already exists (hmm_ready check), so the
# smoke run reuses the production HMM panel — the smoke fixture's purpose
# is to validate cross-sectional + downstream determinism, not HMM.
PANEL_PATH = 'data/panel.parquet'
PANEL_WITH_REGIMES_PATH = 'data/panel_with_regimes.parquet'
STOCK_DATA_PATH = 'data/crsp_msf_raw.parquet'
FF_FACTORS_PATH = 'data/ff_factors.parquet'
EXTRA_HMM_FEATURES_PATH = 'data/extra_hmm_features.pkl'
