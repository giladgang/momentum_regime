"""
test_cross_sectional_lookahead.py
=================================
Causality audit for scripts/cross_sectional_model.py.

This is the analogue of tests/test_expanding_window.py's lookahead audit
but for the cross-sectional model. We inspect the source to verify:

  - Training data uses `date < TRAIN_END` (strict `<`, not `<=`)
  - Test data uses `date >= TRAIN_END`
  - Momentum features are built from lagged returns (shift(1)), never
    contemporaneous
  - Target `ret_fwd` is built from shift(-1), never shift(0) or shift(+1)
  - log_me uses shift(1) — no contemporaneous size leak
  - Target is never present in the feature list
  - XGB/LR fit only on train, predict only on test

Fast — pure source inspection plus a symbolic run of the artefacts. Run:

    pytest tests/test_cross_sectional_lookahead.py -v
"""

import os
import re
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

SCRIPT_PATH = os.path.join(REPO, 'scripts', 'cross_sectional_model.py')


@pytest.fixture(scope='module')
def source():
    with open(SCRIPT_PATH) as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════════
# Train / test split boundary
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainTestSplit:
    def test_train_mask_uses_strict_less_than(self, source):
        # A `<=` would leak the cutoff month into training
        assert "train_mask = df['date'] < TRAIN_END" in source, \
            "train_mask must use strict `<` vs TRAIN_END"

    def test_test_mask_uses_ge(self, source):
        assert "test_mask  = df['date'] >= TRAIN_END" in source \
            or "test_mask = df['date'] >= TRAIN_END" in source, \
            "test_mask must use `>= TRAIN_END`"

    def test_no_le_on_train_end(self, source):
        # Any `df['date'] <= TRAIN_END` pattern would be a lookahead bug
        assert not re.search(r"df\['date'\]\s*<=\s*TRAIN_END", source), \
            "Found `df['date'] <= TRAIN_END` — must be strict `<`"

    def test_train_test_are_disjoint_by_construction(self, source):
        # The complementary masks must cover the two branches exactly
        assert "train = df[train_mask]" in source
        assert "test  = df[test_mask]" in source or "test = df[test_mask]" in source


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum / feature lagging
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureLagging:
    def test_momentum_uses_shifted_log_returns(self, source):
        # The script pre-shifts log returns by 1 before rolling -> no lookahead
        assert "_log_ret_s1" in source, \
            "Momentum must be computed from pre-shifted returns"
        assert re.search(r"shift\(1\)", source), \
            "Expected shift(1) for momentum lag"

    def test_log_me_is_lagged(self, source):
        # log_me must be based on me shifted by 1
        assert re.search(r"log_me.*shift\(1\)", source, re.DOTALL), \
            "log_me must be computed from shift(1) of me"

    def test_ret_fwd_is_forward_shift(self, source):
        # Target is ret_adj shifted by -1
        assert re.search(r"ret_fwd.*shift\(-1\)", source, re.DOTALL), \
            "ret_fwd must be computed with shift(-1) of ret_adj"

    def test_no_positive_shift_on_ret_adj_used_for_target(self, source):
        # Would be a catastrophic bug to use shift(+1) as the target
        m = re.search(r"ret_fwd.*=.*shift\(\+?1\)", source)
        assert m is None, \
            "ret_fwd must never be shift(1) — that would be a lagged return, not forward"


# ═══════════════════════════════════════════════════════════════════════════════
# Target vs features isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTargetIsolation:
    def test_ret_fwd_not_in_cs_features(self):
        import config as cfg
        assert 'ret_fwd' not in cfg.CS_FEATURES
        assert 'ret_next' not in cfg.CS_FEATURES

    def test_model_fit_uses_only_X_train(self, source):
        # Any `fit(X_test` or fit on concatenated train+test would be lookahead
        assert not re.search(r"\.fit\(\s*X_test", source), \
            "Model must never be fit on X_test"
        assert not re.search(r"\.fit\(\s*X_full", source), \
            "Model must never be fit on X_full (train + test concatenated)"

    def test_xgb_predictions_only_on_test(self, source):
        # XGB ensemble must predict on X_test only
        assert "xgb_i.predict(X_test)" in source

    def test_lr_predictions_only_on_test(self, source):
        # LR must predict on X_te_s (the test standardized features)
        assert "lr.predict_proba(X_te_s)" in source

    def test_imputer_scaler_fit_on_train_only(self, source):
        # fit_transform on X_train, transform (not fit) on X_test
        assert "imputer.fit_transform(X_train)" in source
        assert "imputer.transform(X_test)" in source
        assert "scaler" in source
        # No `imputer.fit(X_test` or `scaler.fit(X_test`
        assert not re.search(r"imputer\.fit\([^)]*X_test", source)
        assert not re.search(r"scaler\.fit\([^)]*X_test", source)


# ═══════════════════════════════════════════════════════════════════════════════
# pi_filter merge and fill
# ═══════════════════════════════════════════════════════════════════════════════

class TestPiFilterMerge:
    def test_pi_filter_ffill_only(self, source):
        # ffill uses past values only; bfill would be lookahead
        assert "stocks['pi_filter'] = stocks['pi_filter'].ffill()" in source
        assert not re.search(r"pi_filter.*\.bfill\(\)", source), \
            "pi_filter must never be bfill'd — that leaks future regime info"

    def test_pi_filter_merge_on_date_only(self, source):
        # Must merge regime data keyed on date only (no lookahead into ret_next)
        assert "regimes[['date', 'pi_filter']]" in source, \
            "Must merge only date+pi_filter into stocks (not ret_next)"


# ═══════════════════════════════════════════════════════════════════════════════
# Verified against actual artefact
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtefactAgreesWithSource:
    @pytest.fixture(scope='class')
    def artefacts(self):
        path = os.path.join(REPO, 'artefacts', 'cs_artefacts_data.pkl')
        if not os.path.exists(path):
            pytest.skip("artefacts/cs_artefacts_data.pkl not built yet")
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def test_train_dates_strictly_before_train_end(self, artefacts):
        import config as cfg
        train_end = pd.to_datetime(cfg.TRAIN_END)
        assert artefacts['train']['date'].max() < train_end, \
            f"Train data has dates >= TRAIN_END ({train_end}) — lookahead!"

    def test_test_dates_ge_train_end(self, artefacts):
        import config as cfg
        train_end = pd.to_datetime(cfg.TRAIN_END)
        assert artefacts['test']['date'].min() >= train_end, \
            f"Test data has dates before TRAIN_END ({train_end})"

    def test_no_overlap_train_test_dates(self, artefacts):
        train_dates = set(artefacts['train']['date'])
        test_dates = set(artefacts['test']['date'])
        overlap = train_dates & test_dates
        assert len(overlap) == 0, \
            f"Train/test date overlap: {len(overlap)} dates appear in both"

    def test_features_list_matches_config(self, artefacts):
        import config as cfg
        assert artefacts['FEATURES'] == cfg.CS_FEATURES, \
            "Saved FEATURES list diverges from current config.CS_FEATURES"

    def test_ret_fwd_never_in_features(self, artefacts):
        assert 'ret_fwd' not in artefacts['FEATURES']
        assert 'ret_next' not in artefacts['FEATURES']
