"""
test_reproducibility.py
=======================
Reproducibility / determinism guards.

Verifies that:
  - Pure helpers (metrics, compute_sharpe, long_short_port) are deterministic
    for identical inputs (no hidden RNG).
  - A small XGB ensemble with a fixed seed reproduces identical predictions
    on identical training data.
  - Config-derived feature lists are stable across re-imports.
  - Stored strategies in artefacts/cs_artefacts_data.pkl have Sharpe numbers
    that match those reported in the thesis (regression guard against
    accidental artefact corruption).

Run with:
    pytest tests/test_reproducibility.py -v
"""

import importlib
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from src import utils


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-helper determinism
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelperDeterminism:
    def test_metrics_deterministic(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.005, 0.04, 180))
        assert utils.metrics(r) == utils.metrics(r)

    def test_compute_sharpe_deterministic(self):
        rng = np.random.default_rng(43)
        r = pd.Series(rng.normal(0.005, 0.04, 180))
        assert utils.compute_sharpe(r) == utils.compute_sharpe(r)

    def test_long_short_port_deterministic(self):
        # Same synthetic panel, called twice -> bit-identical results.
        rng = np.random.default_rng(44)
        dates = pd.date_range('2020-01-31', periods=12, freq='ME')
        rows = []
        for d in dates:
            for p in range(40):
                rows.append({
                    'date': d, 'permno': p, 'exchcd': 1,
                    'me': float(rng.uniform(1e8, 1e10)),
                    'ret_fwd': float(rng.normal(0.01, 0.05)),
                    'score': float(rng.normal()),
                })
        df = pd.DataFrame(rows)
        r1 = utils.long_short_port(df, 'score', fee=0.001)
        r2 = utils.long_short_port(df, 'score', fee=0.001)
        pd.testing.assert_series_equal(r1, r2)


# ═══════════════════════════════════════════════════════════════════════════════
# Config stability under reload
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigStability:
    def test_cs_features_stable_across_reimports(self):
        import config as cfg1
        before = list(cfg1.CS_FEATURES)
        importlib.reload(cfg1)
        after = list(cfg1.CS_FEATURES)
        assert before == after

    def test_mom_features_match_lookbacks_after_reload(self):
        import config as cfg
        importlib.reload(cfg)
        assert cfg.MOM_FEATURES == [f'mom_{lb}' for lb in cfg.MOM_LOOKBACKS]


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost ensemble reproducibility (small, fast)
# ═══════════════════════════════════════════════════════════════════════════════

class TestXGBReproducibility:
    def test_fixed_seed_gives_identical_predictions(self):
        """A tiny XGB fit with a fixed seed must produce bit-identical
        predictions across runs on identical data."""
        pytest.importorskip('xgboost')
        from xgboost import XGBRegressor

        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 8))
        y = X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 0.1, 500)

        def fit_predict(seed):
            model = XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                random_state=seed, verbosity=0,
            )
            model.fit(X, y)
            return model.predict(X)

        p1 = fit_predict(seed=7)
        p2 = fit_predict(seed=7)
        np.testing.assert_allclose(p1, p2, rtol=0, atol=0)

    def test_different_seeds_give_different_predictions(self):
        """Sanity check: seeds actually matter. If the two match
        bit-for-bit, our reproducibility test above is meaningless."""
        pytest.importorskip('xgboost')
        from xgboost import XGBRegressor

        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 8))
        y = X[:, 0] + rng.normal(0, 0.1, 500)

        def fit_predict(seed):
            model = XGBRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                random_state=seed, verbosity=0,
            )
            model.fit(X, y)
            return model.predict(X)

        p1 = fit_predict(seed=7)
        p2 = fit_predict(seed=8)
        assert not np.allclose(p1, p2), \
            "Different seeds produced identical predictions — seed isn't flowing through"


# ═══════════════════════════════════════════════════════════════════════════════
# Artefact regression guard: Sharpe of stored strategies is stable
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtefactSharpes:
    @pytest.fixture(scope='class')
    def artefacts(self):
        path = os.path.join(REPO, 'artefacts', 'cs_artefacts_data.pkl')
        if not os.path.exists(path):
            pytest.skip("artefacts/cs_artefacts_data.pkl not built yet")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def test_strategies_lo_has_expected_names(self, artefacts):
        expected = {
            'Market (buy & hold)', 'Fixed 12-mo mom', 'Fixed 1-mo mom',
            'Method 0: Formula', 'Method 1: LR', 'Method 2: XGB',
        }
        assert set(artefacts['strategies_lo'].keys()) == expected

    def test_xgb_sharpe_above_market(self, artefacts):
        """The XGB strategy's Sharpe must exceed the passive market Sharpe.
        If this fails, either the strategy has regressed or the artefacts
        were rebuilt with a broken config."""
        xgb_sharpe = utils.compute_sharpe(artefacts['strategies_lo']['Method 2: XGB'])
        mkt_sharpe = utils.compute_sharpe(artefacts['strategies_lo']['Market (buy & hold)'])
        assert xgb_sharpe > mkt_sharpe, \
            f"XGB Sharpe {xgb_sharpe:.2f} <= market Sharpe {mkt_sharpe:.2f}"

    def test_metrics_on_stored_strategies_stable(self, artefacts):
        """Calling metrics() twice on the same stored series must give
        bit-identical results. This is a no-RNG guarantee check."""
        for name, r in artefacts['strategies_lo'].items():
            m1 = utils.metrics(r)
            m2 = utils.metrics(r)
            assert m1 == m2, f"metrics() not deterministic for {name}"

    def test_all_stored_sharpes_finite(self, artefacts):
        for name, r in artefacts['strategies_lo'].items():
            sr = utils.compute_sharpe(r)
            assert np.isfinite(sr), f"Sharpe for {name} is not finite: {sr}"

    def test_xgb_sharpe_within_known_band(self, artefacts):
        """Regression guard: the XGB test-period Sharpe should sit close to
        the published 1.11. The previous [0.3, 2.5] band tolerated a 70%
        drop in headline performance and so could not catch a real
        regression. Tightened to ±0.10 (well above 50-seed ensemble noise)."""
        from tests import _expected as EXP
        xgb_sharpe = utils.compute_sharpe(artefacts['strategies_lo']['Method 2: XGB'])
        lo, hi = EXP.M2_SHARPE_FULL - 0.10, EXP.M2_SHARPE_FULL + 0.10
        assert lo < xgb_sharpe < hi, \
            f"XGB Sharpe {xgb_sharpe:.3f} outside band [{lo:.2f}, {hi:.2f}] — " \
            f"published value is {EXP.M2_SHARPE_FULL}"
