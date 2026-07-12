"""Bit-identity gate: paper.src.hmm must reproduce the bit-validated
canonical harness exactly (same inputs, same seed => identical pi)."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

from paper import config as C
from paper.src import hmm as H

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.slow
def test_fit_seed_identical_to_canon():
    spec = importlib.util.spec_from_file_location(
        'canon_gate', os.path.join(
            _ROOT, 'experiments', '2026-07-09-prod-budget-dd-vol-reln.py'))
    canon = importlib.util.module_from_spec(spec)
    sys.modules['canon_gate'] = canon
    spec.loader.exec_module(canon)

    p = pd.read_parquet(C.PANEL_PARQUET)
    p['date'] = pd.to_datetime(p['date'])
    p = p[(p['date'] >= C.PANEL_START) & (p['date'] < '2011-01-01')]
    feats = ['DD_z', 'VOL_z', 'REL_N_z']
    Z = p.dropna(subset=feats)[feats].values.astype(float)
    signs = np.array([1.0, 1.0, 1.0])

    canon._init_worker(Z, Z, signs, 400, 100)
    H._init_worker(Z, Z, signs, 400, 100)
    _, pi_c, panic_c, _ = canon.fit_seed(7)
    _, pi_h, panic_h, _ = H.fit_seed(7)
    assert panic_c == panic_h
    np.testing.assert_array_equal(pi_c, pi_h)   # BIT-identical, not approx
