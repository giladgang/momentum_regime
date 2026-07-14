"""S6 nmv_band policy: literature banding (NMV 2016 / DNMV 2023) —
enter at the decile, hold until rank falls out of E, count floats."""
import pandas as pd

from paper import execution as X


def _month():
    # 20 names, score descending in permno order -> ranked = [1, 2, ..., 20],
    # rank_pct[p] = p / 20; k = int(20 * 0.10) = 2 -> decile = {1, 2}
    return pd.DataFrame({'permno': range(1, 21),
                         'me': [10.0] * 20,
                         'score': [21.0 - p for p in range(1, 21)],
                         'mom_12': [0.0] * 20})


def test_e10_equals_monthly():
    g = _month()
    prev = {3, 9}                      # held, but outside the top decile
    mem = X._members(g, 'score', ('nmv_band', (10, 10)), prev, 0.2, None)
    mon = X._members(g, 'score', ('monthly', None), prev, 0.2, None)
    assert mem == mon == {1, 2}


def test_floating_count_no_fill():
    g = _month()
    prev = {3, 7, 9, 15}               # rank_pct .15 .35 .45 .75
    mem = X._members(g, 'score', ('nmv_band', (40, 40)), prev, 0.2, None)
    # keep held within E=40% (permnos 3, 7); union decile; NO fill-to-k
    assert mem == {1, 2, 3, 7}


def test_regime_switches_band():
    g = _month()
    prev = {3, 7, 9, 15}
    calm = X._members(g, 'score', ('nmv_band', (40, 10)), prev, 0.2, None)
    panic = X._members(g, 'score', ('nmv_band', (40, 10)), prev, 0.8, None)
    assert calm == {1, 2, 3, 7}        # E_calm=40 retains 3 and 7
    assert panic == {1, 2}             # E_panic=10 -> decile only


def test_held_name_out_of_universe_dropped():
    g = _month()
    prev = {3, 99}                     # 99 delisted from the universe
    mem = X._members(g, 'score', ('nmv_band', (40, 40)), prev, 0.2, None)
    assert mem == {1, 2, 3}
