"""
conftest.py
===========
Shared pytest fixtures for the test suite.

The artefacts pickle (`artefacts/cs_artefacts_data.pkl`) is 943 MB and
takes several seconds to deserialise. Without fixtures, every test class
that needs it reloads from disk, multiplying that cost across the suite.
Session-scoped fixtures load each artefact ONCE per pytest invocation
and hand the same object to every test that requests it.

All data-dependent fixtures auto-skip their consumers when the
underlying files are absent. This lets the same test files run in CI
(no 943 MB pickle) and locally (full suite) without modification.

Usage in a test:

    class TestFoo:
        def test_something(self, artefacts):
            assert 'test' in artefacts
"""

import os
import pickle
import sys

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(scope='session')
def repo_root():
    """Absolute path to the project root."""
    return REPO


@pytest.fixture(scope='session')
def artefacts_path():
    """Absolute path to cs_artefacts_data.pkl, auto-skip if missing."""
    import config as cfg
    path = os.path.join(REPO, cfg.ARTEFACTS_PATH)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present; run scripts/cross_sectional_model.py "
                    "to build artefacts.")
    return path


@pytest.fixture(scope='session')
def artefacts(artefacts_path):
    """The cs_artefacts_data.pkl dict, loaded ONCE per pytest session."""
    with open(artefacts_path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='session')
def panel_with_regimes_path():
    import config as cfg
    path = os.path.join(REPO, cfg.PANEL_WITH_REGIMES_PATH)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present; run scripts/hmm_model.py first.")
    return path


@pytest.fixture(scope='session')
def panel_with_regimes(panel_with_regimes_path):
    """panel_with_regimes.parquet as a DataFrame, loaded ONCE per session."""
    return pd.read_parquet(panel_with_regimes_path)


@pytest.fixture(scope='session')
def ff_factors_path():
    import config as cfg
    path = os.path.join(REPO, cfg.FF_FACTORS_PATH)
    if not os.path.exists(path):
        pytest.skip(f"{path} not present.")
    return path


@pytest.fixture(scope='session')
def ff_factors(ff_factors_path):
    """Fama-French factors DataFrame, loaded ONCE per session."""
    return pd.read_parquet(ff_factors_path)
