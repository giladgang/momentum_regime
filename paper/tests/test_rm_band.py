import numpy as np
import pandas as pd

from paper import execution as X
from paper.tests.test_hold_losers import _fixture


def test_rm_band_equal_reduces_to_nmv():
    x, _ = _fixture()
    a, _ = X.simulate(x, ('rm_band', (20, 20, 20, 20)), score_col='score')
    b, _ = X.simulate(x, ('nmv_band', (20, 20)), score_col='score')
    assert np.allclose(a['gross'].values, b['gross'].values)
    assert np.allclose(a['turnover'].values, b['turnover'].values)


def test_rm_band_wide_for_panic_losers():
    x, d2 = _fixture()
    # E_panic_low = 100 (hold beaten-down in panic regardless of rank);
    # everything else tight = 10. losers 102,103 (low mom, panic) kept;
    # winners 100,101 (high mom) traded out.
    _, led = X.simulate(x, ('rm_band', (10, 10, 10, 100)), score_col='score')
    exits = set(led[(led['date'] == d2)
                    & (led['cause'] == 'exit_rank')]['permno'])
    assert 102 not in exits and 103 not in exits
    assert 100 in exits and 101 in exits
