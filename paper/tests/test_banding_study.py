import numpy as np
import pandas as pd

from paper import banding_study as B


def _cells_fixture():
    # two strategies, diagonal + one off-diagonal cell each
    return pd.DataFrame({
        'strategy': ['a', 'a', 'a', 'b', 'b', 'b'],
        'family': ['diag', 'diag', 'regime2', 'diag', 'diag', 'regime2'],
        'params': ['(10, 10)', '(20, 20)', '(20, 10)',
                   '(10, 10)', '(20, 20)', '(20, 10)'],
        'net_ir_meas': [0.10, 0.20, 0.35, 0.30, 0.28, 0.29],
        'to_mo': [0.70, 0.50, 0.55, 0.20, 0.15, 0.16],
        'mean_hs_bp': [10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
    })


def test_best_cell_selection():
    best = B.select_best(_cells_fixture())
    a = best[best['strategy'] == 'a'].iloc[0]
    assert a['static_params'] == '(20, 20)' and a['regime_params'] == '(20, 10)'
    assert np.isclose(a['delta_net_ir'], 0.15)


def test_gate2_spearman_direction():
    best = B.select_best(_cells_fixture())
    rho = B.gate2_spearman(best)
    # strategy a: high cost intensity, big delta; b: low, small -> rho = +1
    assert rho > 0
