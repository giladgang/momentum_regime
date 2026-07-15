import numpy as np
import pandas as pd

from paper.crash_recovery import decompose


def _walk_fixture():
    # 6 months: 2 calm, 2 panic-crash (market down), 2 panic-recovery (up)
    dates = pd.date_range('2020-01-31', periods=6, freq='ME')
    return pd.DataFrame({
        'date': dates,
        'strat_ret': [0.01, 0.02, -0.10, -0.05, 0.08, 0.09],
        'bench_ret': [0.01, 0.01, -0.08, -0.04, 0.06, 0.07],
        'pi':        [0.1,  0.2,   0.9,   0.8,  0.9,  0.7],
    })


def test_buckets_and_shares():
    out = decompose(_walk_fixture()).set_index('bucket')
    assert out.loc['calm', 'n_mo'] == 2
    assert out.loc['panic_crash', 'n_mo'] == 2
    assert out.loc['panic_recovery', 'n_mo'] == 2
    # active means, hand-computed
    assert np.isclose(out.loc['calm', 'active_mo'], 0.005)
    assert np.isclose(out.loc['panic_crash', 'active_mo'], -0.015)
    assert np.isclose(out.loc['panic_recovery', 'active_mo'], 0.02)
    # shares sum to 100%
    assert np.isclose(out['share_of_total_active'].sum(), 1.0)
