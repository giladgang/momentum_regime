"""
test_config.py
==============
Sanity checks for config.py. These tests catch misconfigurations that
would silently corrupt downstream results: date-ordering bugs, duplicated
features, inconsistent seed lists, malformed hyperparameter dictionaries.

Fast and pure — no data loading. Run with:

    pytest tests/test_config.py -v
"""

import os
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import config as cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Date ordering
# ═══════════════════════════════════════════════════════════════════════════════

class TestDates:
    def test_hmm_start_before_train_end(self):
        assert pd.to_datetime(cfg.HMM_START) < pd.to_datetime(cfg.TRAIN_END)

    def test_train_end_is_parseable(self):
        dt = pd.to_datetime(cfg.TRAIN_END)
        assert dt.year >= 1990 and dt.year <= 2030

    def test_alt_splits_all_after_hmm_start(self):
        hmm_start = pd.to_datetime(cfg.HMM_START)
        for label, cut in cfg.ALT_SPLITS.items():
            assert pd.to_datetime(cut) > hmm_start, \
                f"ALT_SPLITS[{label!r}] = {cut} is not after HMM_START"

    def test_alt_splits_keys_and_values_unique(self):
        assert len(cfg.ALT_SPLITS) == len(set(cfg.ALT_SPLITS))
        assert len(set(cfg.ALT_SPLITS.values())) == len(cfg.ALT_SPLITS), \
            "Duplicate cut dates in ALT_SPLITS"

    def test_sub_periods_start_before_end(self):
        for label, start, end in cfg.SUB_PERIODS:
            assert pd.to_datetime(start) < pd.to_datetime(end), \
                f"SUB_PERIODS {label}: start {start} >= end {end}"

    def test_sub_periods_full_spans_others(self):
        full = next((s, e) for label, s, e in cfg.SUB_PERIODS if label == 'Full')
        full_start, full_end = pd.to_datetime(full[0]), pd.to_datetime(full[1])
        for label, s, e in cfg.SUB_PERIODS:
            if label == 'Full':
                continue
            assert pd.to_datetime(s) >= full_start, f"{label} starts before Full"
            assert pd.to_datetime(e) <= full_end, f"{label} ends after Full"


# ═══════════════════════════════════════════════════════════════════════════════
# Feature lists
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatures:
    def test_hmm_features_unique(self):
        assert len(cfg.HMM_FEATURES) == len(set(cfg.HMM_FEATURES))

    def test_fund_features_unique(self):
        assert len(cfg.FUND_FEATURES) == len(set(cfg.FUND_FEATURES))

    def test_mom_features_unique(self):
        assert len(cfg.MOM_FEATURES) == len(set(cfg.MOM_FEATURES))

    def test_cs_features_unique(self):
        assert len(cfg.CS_FEATURES) == len(set(cfg.CS_FEATURES))

    def test_mom_lookbacks_monotonic_and_positive(self):
        assert all(lb > 0 for lb in cfg.MOM_LOOKBACKS)
        assert cfg.MOM_LOOKBACKS == sorted(cfg.MOM_LOOKBACKS)

    def test_mom_features_match_lookbacks(self):
        expected = [f'mom_{lb}' for lb in cfg.MOM_LOOKBACKS]
        assert cfg.MOM_FEATURES == expected

    def test_pi_filter_in_cs_features(self):
        assert 'pi_filter' in cfg.CS_FEATURES, \
            "pi_filter must always be in CS_FEATURES (regime signal)"

    def test_mom_features_all_in_cs_features(self):
        for f in cfg.MOM_FEATURES:
            assert f in cfg.CS_FEATURES, f"MOM feature {f} missing from CS_FEATURES"

    def test_fund_features_respected_by_flag(self):
        if cfg.USE_FUNDAMENTALS:
            for f in cfg.FUND_FEATURES:
                assert f in cfg.CS_FEATURES, \
                    f"USE_FUNDAMENTALS=True but {f} missing from CS_FEATURES"
        else:
            for f in cfg.FUND_FEATURES:
                assert f not in cfg.CS_FEATURES, \
                    f"USE_FUNDAMENTALS=False but {f} is in CS_FEATURES"

    def test_target_column_not_in_features(self):
        # Forward return must never leak in as a feature
        for forbidden in ('ret_fwd', 'ret_next', 'ret', 'ret_adj'):
            assert forbidden not in cfg.CS_FEATURES, \
                f"Forward/contemporaneous return {forbidden!r} leaked into CS_FEATURES"
            assert forbidden not in cfg.HMM_FEATURES, \
                f"Forward/contemporaneous return {forbidden!r} leaked into HMM_FEATURES"


# ═══════════════════════════════════════════════════════════════════════════════
# HMM / MCMC settings
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMSettings:
    def test_k_states_reasonable(self):
        assert 2 <= cfg.K_STATES <= 10

    def test_hmm_seeds_non_empty_and_unique(self):
        assert len(cfg.HMM_SEEDS) > 0
        assert len(cfg.HMM_SEEDS) == len(set(cfg.HMM_SEEDS))

    def test_hmm_burnin_less_than_iterations(self):
        assert cfg.HMM_BURNIN < cfg.HMM_ITERATIONS, \
            "HMM_BURNIN must be strictly less than HMM_ITERATIONS"

    def test_hmm_burnin_non_negative(self):
        assert cfg.HMM_BURNIN >= 0

    def test_k_states_robustness_excludes_baseline(self):
        # K_STATES_ROBUSTNESS should be alternatives to the baseline K_STATES
        assert cfg.K_STATES not in cfg.K_STATES_ROBUSTNESS, \
            "K_STATES_ROBUSTNESS should not include the baseline K_STATES"


# ═══════════════════════════════════════════════════════════════════════════════
# HMM priors (must match methodology.tex §3.1.3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMPriors:
    """Guard the Bayesian priors documented in methodology.tex §3.1.3.

    The thesis specifies m_0 = 0, kappa_0 = 0.01, nu_0 = D+2, and a
    Dirichlet transition prior with diagonal 9 and off-diagonal 1.
    Drifting these silently would invalidate the methodology description
    and potentially the headline numbers, so we pin them here.
    """

    def test_niw_prior_mean_is_zero(self):
        assert cfg.HMM_PRIOR_M0 == 0.0, \
            "HMM_PRIOR_M0 must be 0 (z-scored features: prior mean at origin)"

    def test_niw_prior_kappa_is_nearly_flat(self):
        assert cfg.HMM_PRIOR_KAPPA0 == 0.01, \
            "HMM_PRIOR_KAPPA0 must be 0.01 (nearly flat: data dominates)"

    def test_niw_prior_nu_offset_gives_proper_iw(self):
        # nu_0 = D + nu_0_off. The minimum for a proper IW is D+2, so
        # nu_0_off must be >= 2. The thesis uses exactly D+2.
        assert cfg.HMM_PRIOR_NU0_OFF == 2, \
            ("HMM_PRIOR_NU0_OFF must be 2 so nu_0 = D+2, the minimum "
             "integer giving a proper Inverse-Wishart prior")

    def test_dirichlet_prior_is_2x2(self):
        alpha = cfg.HMM_PRIOR_DIRICHLET_ALPHA
        assert len(alpha) == 2 and all(len(row) == 2 for row in alpha), \
            "HMM_PRIOR_DIRICHLET_ALPHA must be 2x2 (one row per state)"

    def test_dirichlet_prior_all_positive(self):
        for row in cfg.HMM_PRIOR_DIRICHLET_ALPHA:
            for v in row:
                assert v > 0, f"Dirichlet pseudocounts must be > 0, got {v}"

    def test_dirichlet_prior_favors_persistence(self):
        # Diagonal must exceed off-diagonal so the prior mean transition
        # matrix has regime persistence > regime switching.
        alpha = cfg.HMM_PRIOR_DIRICHLET_ALPHA
        assert alpha[0][0] > alpha[0][1], \
            "Calm-row Dirichlet must favor persistence: alpha[0][0] > alpha[0][1]"
        assert alpha[1][1] > alpha[1][0], \
            "Panic-row Dirichlet must favor persistence: alpha[1][1] > alpha[1][0]"

    def test_dirichlet_prior_matches_thesis(self):
        # Methodology.tex equation (dirichlet_prior) sets these exact values.
        assert cfg.HMM_PRIOR_DIRICHLET_ALPHA == [[9.0, 1.0], [1.0, 9.0]], \
            ("HMM_PRIOR_DIRICHLET_ALPHA must be [[9,1],[1,9]] to match "
             "methodology.tex eq:dirichlet_prior (E[persistence] = 0.9)")


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost settings
# ═══════════════════════════════════════════════════════════════════════════════

class TestXGBSettings:
    def test_xgb_seeds_non_empty_and_unique(self):
        assert len(cfg.XGB_SEEDS) > 0
        assert len(cfg.XGB_SEEDS) == len(set(cfg.XGB_SEEDS))

    def test_n_estimators_positive(self):
        assert cfg.N_ESTIMATORS > 0

    def test_max_depth_reasonable(self):
        assert 1 <= cfg.MAX_DEPTH <= 12

    def test_learning_rate_in_unit_interval(self):
        assert 0 < cfg.LEARNING_RATE <= 1.0

    def test_subsample_ratios_in_unit_interval(self):
        assert 0 < cfg.SUBSAMPLE <= 1.0
        assert 0 < cfg.COLSAMPLE <= 1.0

    def test_xgb_configs_valid(self):
        baseline_found = False
        for label, params in cfg.XGB_CONFIGS:
            assert 'max_depth' in params
            assert 'learning_rate' in params
            assert 'n_estimators' in params
            assert params['max_depth'] >= 1
            assert params['learning_rate'] > 0
            assert params['n_estimators'] >= 1
            if (params['max_depth'] == cfg.MAX_DEPTH
                    and params['learning_rate'] == cfg.LEARNING_RATE
                    and params['n_estimators'] == cfg.N_ESTIMATORS):
                baseline_found = True
        assert baseline_found, "XGB_CONFIGS must include the baseline hyperparameter set"


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio settings
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioSettings:
    def test_portfolio_type_valid(self):
        assert cfg.PORTFOLIO_TYPE in ('long_short', 'long_only')

    def test_trading_fee_non_negative(self):
        assert cfg.TRADING_FEE >= 0

    def test_trading_fee_realistic(self):
        # Fee in decimal form; 1% is an extreme upper bound for a realistic config
        assert cfg.TRADING_FEE < 0.01, \
            f"TRADING_FEE = {cfg.TRADING_FEE} looks too high — expected decimal, not bps"

    def test_cost_levels_non_negative_and_sorted(self):
        assert all(c >= 0 for c in cfg.COST_LEVELS_BPS)
        assert cfg.COST_LEVELS_BPS == sorted(cfg.COST_LEVELS_BPS)

    def test_pi_thresholds_in_open_unit_interval(self):
        for t in cfg.PI_THRESHOLDS:
            assert 0 < t < 1, f"pi threshold {t} outside (0,1)"


# ═══════════════════════════════════════════════════════════════════════════════
# Output dirs and data paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaths:
    def test_output_dirs_are_non_empty_strings(self):
        for d in (cfg.TABLES_DIR, cfg.PLOTS_DIR, cfg.RESULTS_DIR, cfg.ARTEFACTS_DIR):
            assert isinstance(d, str) and len(d) > 0

    def test_output_dirs_distinct(self):
        dirs = [cfg.TABLES_DIR, cfg.PLOTS_DIR, cfg.RESULTS_DIR, cfg.ARTEFACTS_DIR]
        assert len(set(dirs)) == len(dirs), "Output dirs must be distinct"

    def test_data_paths_are_relative_strings(self):
        paths = (cfg.PANEL_PATH, cfg.PANEL_WITH_REGIMES_PATH, cfg.STOCK_DATA_PATH,
                 cfg.ARTEFACTS_PATH, cfg.XGB_MODEL_PATH, cfg.LR_MODEL_PATH,
                 cfg.FF_FACTORS_PATH, cfg.EXTRA_HMM_FEATURES_PATH)
        for p in paths:
            assert isinstance(p, str) and len(p) > 0
            assert not os.path.isabs(p), f"{p} should be relative to repo root"

    def test_artefacts_path_under_artefacts_dir(self):
        assert cfg.ARTEFACTS_PATH.startswith(cfg.ARTEFACTS_DIR + '/')
        assert cfg.XGB_MODEL_PATH.startswith(cfg.ARTEFACTS_DIR + '/')
        assert cfg.LR_MODEL_PATH.startswith(cfg.ARTEFACTS_DIR + '/')


# ═══════════════════════════════════════════════════════════════════════════════
# Robustness settings
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustness:
    def test_student_t_nu_all_positive(self):
        assert all(nu > 0 for nu in cfg.STUDENT_T_NU)

    def test_student_t_nu_unique(self):
        assert len(cfg.STUDENT_T_NU) == len(set(cfg.STUDENT_T_NU))

    def test_placebo_runs_positive(self):
        assert cfg.N_PLACEBO_RUNS >= 1
