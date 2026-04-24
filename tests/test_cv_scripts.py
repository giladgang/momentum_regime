"""
test_cv_scripts.py
==================
Tests for scripts/xgb_cv.py and scripts/hmm_cv.py.

Two categories:

  1. Unit tests for pure helper functions (fold definitions,
     enumerate_combinations, zscore_train, crisis_mask_for_training,
     completed_cells, combo_key). Fast, no compute, no data files.

  2. Post-smoke regression checks on the smoke-run CSVs. These auto-skip
     if the smoke CSVs don't exist, so the file is safe to run anywhere.
     They verify that a smoke run produced a well-formed CSV with finite
     Sharpes, sane month counts, and the expected columns.

Usage
-----
    # Unit tests only (no smoke run needed)
    pytest tests/test_cv_scripts.py::TestXGBHelpers -v
    pytest tests/test_cv_scripts.py::TestHMMHelpers -v

    # After running smoke tests, validate the output CSVs
    python scripts/xgb_cv.py --smoke
    python scripts/hmm_cv.py --smoke
    pytest tests/test_cv_scripts.py -v
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script(name):
    """Load scripts/<name>.py as a module without executing its __main__."""
    path = os.path.join(REPO, 'scripts', name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def xgb_cv():
    return _load_script('xgb_cv')


@pytest.fixture(scope='module')
def hmm_cv():
    return _load_script('hmm_cv')


# ═══════════════════════════════════════════════════════════════════════════════
# XGB CV: pure unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestXGBHelpers:

    def test_fold_count(self, xgb_cv):
        assert len(xgb_cv.FOLDS) == 5, \
            "XGB CV should have 5 expanding-window folds"

    def test_folds_disjoint_train_val(self, xgb_cv):
        """Each fold's validation must start at its train_end (no overlap)."""
        for fold_id, train_end, val_start, val_end in xgb_cv.FOLDS:
            assert train_end == val_start, \
                f"Fold {fold_id}: train_end={train_end}, val_start={val_start} — must be equal"
            assert pd.Timestamp(val_start) < pd.Timestamp(val_end), \
                f"Fold {fold_id}: val_start must be before val_end"

    def test_folds_expanding(self, xgb_cv):
        """Each subsequent fold must have a later train_end than the previous."""
        ends = [pd.Timestamp(tr_end) for _, tr_end, _, _ in xgb_cv.FOLDS]
        assert ends == sorted(ends) and len(set(ends)) == len(ends), \
            "Fold train_ends must be strictly increasing"

    def test_folds_never_touch_test_period(self, xgb_cv):
        """No fold's validation window may reach TRAIN_END (2011-01-01)."""
        from config import TRAIN_END
        train_end_ts = pd.Timestamp(TRAIN_END)
        for fold_id, _, _, val_end in xgb_cv.FOLDS:
            assert pd.Timestamp(val_end) <= train_end_ts, (
                f"Fold {fold_id} val_end={val_end} leaks into test period "
                f"({TRAIN_END} and later)"
            )

    def test_min_val_months_constant(self, xgb_cv):
        assert xgb_cv.MIN_VAL_MONTHS >= 12

    def test_smoke_grid_is_small(self, xgb_cv):
        assert len(xgb_cv.GRID_SMOKE['max_depth']) <= 2
        assert len(xgb_cv.GRID_SMOKE['learning_rate']) == 1
        assert len(xgb_cv.GRID_SMOKE['n_estimators']) == 1

    def test_full_grid_includes_production_config(self, xgb_cv):
        """The baseline (depth=4, lr=0.05, n=500) must be in the grid so
        we can confirm whether CV selects it or something different."""
        from config import MAX_DEPTH, LEARNING_RATE, N_ESTIMATORS
        assert MAX_DEPTH in xgb_cv.GRID_FULL['max_depth']
        assert LEARNING_RATE in xgb_cv.GRID_FULL['learning_rate']
        assert N_ESTIMATORS in xgb_cv.GRID_FULL['n_estimators']

    def test_completed_cells_empty_when_no_file(self, xgb_cv, tmp_path):
        assert xgb_cv.completed_cells(str(tmp_path / 'nonexistent.csv')) == set()

    def test_completed_cells_reads_csv(self, xgb_cv, tmp_path):
        path = tmp_path / 'x.csv'
        df = pd.DataFrame([{
            'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 500,
            'fold': 1, 'n_seeds': 5, 'fee': 0.001,
            'val_sharpe': 0.5, 'n_val_months': 24, 'elapsed_sec': 1.0,
        }])
        df.to_csv(path, index=False)
        done = xgb_cv.completed_cells(str(path))
        assert (4, 0.05, 500, 1) in done
        assert (4, 0.05, 500, 2) not in done

    def test_append_row_creates_then_appends(self, xgb_cv, tmp_path):
        path = str(tmp_path / 'x.csv')
        row1 = [3, 0.05, 200, 1, 5, 0.001, 0.4, 24, 1.2]
        row2 = [4, 0.05, 500, 1, 5, 0.001, 0.5, 24, 1.5]
        xgb_cv.append_row(path, row1)
        xgb_cv.append_row(path, row2)
        df = pd.read_csv(path)
        assert len(df) == 2
        assert list(df.columns) == xgb_cv.COLS

    def test_smoke_output_path_isolated_from_full(self, xgb_cv):
        assert xgb_cv.SMOKE_RESULTS_PATH != xgb_cv.RESULTS_PATH, \
            "Smoke output must not collide with full-run output"


# ═══════════════════════════════════════════════════════════════════════════════
# HMM CV: pure unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMMHelpers:

    def test_candidates_are_nine_features(self, hmm_cv):
        assert len(hmm_cv.CANDIDATES) == 9
        assert 'DD' in hmm_cv.CANDIDATES

    def test_dd_is_required(self, hmm_cv):
        assert hmm_cv.REQUIRED == 'DD'

    def test_enumerate_combinations_count(self, hmm_cv):
        """DD + 0..3 others should give C(8,0)+C(8,1)+C(8,2)+C(8,3) = 93."""
        combos = hmm_cv.enumerate_combinations()
        assert len(combos) == 93, f"Expected 93 DD-inclusive combos, got {len(combos)}"

    def test_every_combination_contains_dd(self, hmm_cv):
        for combo in hmm_cv.enumerate_combinations():
            assert 'DD' in combo, f"DD missing from combination {combo}"

    def test_combinations_are_unique(self, hmm_cv):
        combos = hmm_cv.enumerate_combinations()
        # Canonicalise to frozensets (order-invariant)
        s = set(frozenset(c) for c in combos)
        assert len(s) == len(combos), "Duplicate combinations found"

    def test_combinations_sizes(self, hmm_cv):
        sizes = [len(c) for c in hmm_cv.enumerate_combinations()]
        from collections import Counter
        count = Counter(sizes)
        assert count[1] == 1  # DD alone
        assert count[2] == 8  # DD + 1 of 8
        assert count[3] == 28  # DD + 2 of 8
        assert count[4] == 56  # DD + 3 of 8

    def test_folds_match_xgb_cv(self, hmm_cv, xgb_cv):
        """Both CV scripts must use the same fold definitions so results
        are comparable."""
        assert hmm_cv.FOLDS == xgb_cv.FOLDS

    def test_zscore_train_uses_train_partition_only(self, hmm_cv):
        """zscore_train must standardise using statistics from the
        TRAIN partition only (no leakage)."""
        dates = pd.date_range('1990-01-31', periods=200, freq='ME')
        # Fabricate a feature whose validation-period mean differs hugely
        # from training-period mean. If z-scoring uses train stats only,
        # validation z-scores should be far from 0.
        vals = np.concatenate([np.zeros(150), np.full(50, 100.0)])
        panel = pd.DataFrame({'date': dates, 'X': vals})
        train_end = dates[150]  # validation = last 50 months
        z = hmm_cv.zscore_train(panel, ['X'], train_end)
        # Training portion has constant value 0 -> std=0 -> treated as 1.0.
        # Validation values are 100, far above training mean (0).
        val_z = z.loc[z['date'] >= pd.Timestamp(train_end), 'X_z']
        assert (val_z > 10).all(), (
            "Validation z-scores should be far from 0 when training mean "
            "is 0 and validation values are 100. Got: "
            f"mean={val_z.mean():.1f}, min={val_z.min():.1f}"
        )

    def test_zscore_train_handles_zero_std(self, hmm_cv):
        """Constant training series must not produce inf/NaN."""
        dates = pd.date_range('1990-01-31', periods=100, freq='ME')
        panel = pd.DataFrame({'date': dates, 'X': 5.0})  # constant everywhere
        z = hmm_cv.zscore_train(panel, ['X'], dates[50])
        assert z['X_z'].notna().all(), "z-scores contain NaN"
        assert np.isfinite(z['X_z']).all(), "z-scores contain inf"

    def test_crisis_mask_covers_known_crises(self, hmm_cv):
        """A training window spanning 1990-2010 must pick up multiple crises."""
        dates = pd.date_range('1990-01-31', '2010-12-31', freq='ME').values
        mask = hmm_cv.crisis_mask_for_training(dates)
        assert mask.sum() > 40, (
            f"Crisis mask should have >40 months across 1990-2010 "
            f"(1990 recession + LTCM + dot-com + GFC). Got {mask.sum()}"
        )

    def test_crisis_mask_empty_on_calm_window(self, hmm_cv):
        """A calm window (2012-2018) should have ZERO crisis months."""
        dates = pd.date_range('2012-01-31', '2018-12-31', freq='ME').values
        mask = hmm_cv.crisis_mask_for_training(dates)
        assert mask.sum() == 0, \
            f"Calm window 2012-2018 should have 0 crisis months, got {mask.sum()}"

    def test_combo_key_is_stable(self, hmm_cv):
        k1 = hmm_cv.combo_key(('DD', 'CS'))
        k2 = hmm_cv.combo_key(('DD', 'CS'))
        assert k1 == k2

    def test_smoke_output_path_isolated_from_full(self, hmm_cv):
        assert hmm_cv.SMOKE_RESULTS_PATH != hmm_cv.RESULTS_PATH, \
            "Smoke output must not collide with full-run output"


# ═══════════════════════════════════════════════════════════════════════════════
# Post-smoke regression checks — auto-skip if smoke CSV not present
# ═══════════════════════════════════════════════════════════════════════════════

SMOKE_XGB_PATH = os.path.join(REPO, 'results', 'xgb_cv_smoke.csv')
SMOKE_HMM_PATH = os.path.join(REPO, 'results', 'hmm_cv_smoke.csv')


class TestXGBSmokeOutput:

    @pytest.fixture(scope='class')
    def smoke_df(self):
        if not os.path.exists(SMOKE_XGB_PATH):
            pytest.skip(f"{SMOKE_XGB_PATH} not present; run "
                        "`python scripts/xgb_cv.py --smoke` first")
        return pd.read_csv(SMOKE_XGB_PATH)

    def test_smoke_csv_nonempty(self, smoke_df):
        assert len(smoke_df) > 0

    def test_smoke_csv_has_expected_columns(self, smoke_df, xgb_cv):
        assert list(smoke_df.columns) == xgb_cv.COLS

    def test_smoke_sharpes_are_finite(self, smoke_df):
        assert smoke_df['val_sharpe'].apply(np.isfinite).all()

    def test_smoke_sharpes_in_plausible_range(self, smoke_df):
        # Noisy smoke runs can land anywhere, but not outside [-5, 5]
        # which would indicate a bug (e.g., returns scaled wrong).
        assert smoke_df['val_sharpe'].between(-5, 5).all(), \
            f"Smoke Sharpes outside sanity band: {smoke_df['val_sharpe'].tolist()}"

    def test_smoke_val_months_meet_minimum(self, smoke_df, xgb_cv):
        assert (smoke_df['n_val_months'] >= xgb_cv.MIN_VAL_MONTHS).all()

    def test_smoke_fee_matches_config(self, smoke_df):
        from config import TRADING_FEE
        # All smoke rows should have the same fee (the default)
        assert smoke_df['fee'].nunique() == 1
        assert smoke_df['fee'].iloc[0] == pytest.approx(TRADING_FEE)

    def test_smoke_n_seeds_recorded(self, smoke_df):
        assert (smoke_df['n_seeds'] >= 1).all()


class TestHMMSmokeOutput:

    @pytest.fixture(scope='class')
    def smoke_df(self):
        if not os.path.exists(SMOKE_HMM_PATH):
            pytest.skip(f"{SMOKE_HMM_PATH} not present; run "
                        "`python scripts/hmm_cv.py --smoke` first")
        return pd.read_csv(SMOKE_HMM_PATH)

    def test_smoke_csv_nonempty(self, smoke_df):
        assert len(smoke_df) > 0

    def test_smoke_csv_has_expected_columns(self, smoke_df, hmm_cv):
        assert list(smoke_df.columns) == hmm_cv.COLS

    def test_smoke_sharpes_are_finite(self, smoke_df):
        # NaN is allowed for HMM fit failures; check the non-failing rows.
        good = smoke_df.dropna(subset=['val_sharpe'])
        assert good['val_sharpe'].apply(np.isfinite).all()

    def test_smoke_sharpes_in_plausible_range(self, smoke_df):
        good = smoke_df.dropna(subset=['val_sharpe'])
        assert good['val_sharpe'].between(-5, 5).all()

    def test_smoke_contains_dd_in_every_combo(self, smoke_df):
        assert smoke_df['combo'].str.contains('DD').all()

    def test_smoke_n_features_matches_combo_string(self, smoke_df):
        # n_features should equal the number of + separators + 1
        for combo, n_features in zip(smoke_df['combo'], smoke_df['n_features']):
            assert combo.count('+') + 1 == n_features, \
                f"combo={combo!r} n_features={n_features} inconsistent"

    def test_smoke_val_months_meet_minimum(self, smoke_df, hmm_cv):
        good = smoke_df[smoke_df['n_val_months'] > 0]
        assert (good['n_val_months'] >= hmm_cv.MIN_VAL_MONTHS).all()

    def test_smoke_fee_matches_config(self, smoke_df):
        from config import TRADING_FEE
        assert smoke_df['fee'].nunique() == 1
        assert smoke_df['fee'].iloc[0] == pytest.approx(TRADING_FEE)

    def test_smoke_hmm_elapsed_positive(self, smoke_df):
        """HMM fit should have taken some time (non-zero)."""
        assert (smoke_df['hmm_elapsed_sec'] > 0).all()
