import numpy as np
import pandas as pd

from paper.src.regime_labels import (CALM, CRASH, RECOVERY,
                                     label_states, label_states_dd)


def _fixture():
    dates = pd.date_range('2020-01-31', periods=6, freq='ME')
    pi = pd.Series([0.1, 0.2, 0.9, 0.8, 0.9, 0.7], index=dates)
    mkt = pd.Series([0.01, 0.01, -0.08, -0.04, 0.06, 0.07], index=dates)
    return dates, pi, mkt


def test_three_state_codes():
    dates, pi, mkt = _fixture()
    s = label_states(dates, pi, mkt)
    assert list(s.values) == [CALM, CALM, CRASH, CRASH, RECOVERY, RECOVERY]
    assert s.dtype == np.int8


def test_dd_definition_matches_sign_definition_on_fixture():
    # dd heals exactly when mkt_ret > 0 while below peak -> identical here
    dates, pi, mkt = _fixture()
    assert (label_states(dates, pi, mkt)
            == label_states_dd(dates, pi, mkt)).all()
