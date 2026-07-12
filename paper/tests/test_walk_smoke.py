import pandas as pd
import pytest

from paper import config as C


@pytest.mark.slow
def test_walk_2011_smoke():
    from paper import walk_report as W
    te = W._walk_year(2011, 'DD', workers=4,
                      hmm_seeds=C.EVAL_HMM_SEEDS[:3],
                      xgb_seeds=C.EVAL_XGB_SEEDS[:3],
                      n_iter=400, n_burnin=100)
    months = pd.to_datetime(te['date']).dt.to_period('M').nunique()
    assert months == 12
    assert {'score_pi', 'score_nopi', 'pi', 'me', 'ret_fwd'} <= set(te.columns)
    assert te.groupby('date').size().max() <= C.UNIVERSE_N
