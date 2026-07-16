import numpy as np
import pandas as pd

from paper import execution as X


def _fixture():
    # 40 stocks, k=4 (decile). Month 1 calm: stocks 100-103 are the top-4 book.
    # Momentum fixed per stock: 100,101 winners (+); 102,103 losers (-).
    # Month 2 panic: 100-103 all rank low; 104-107 become top-4.
    mom = {100: 0.5, 101: 0.4, 102: -0.3, 103: -0.4}
    rows = []
    d1, d2 = pd.Timestamp('2020-01-31'), pd.Timestamp('2020-02-29')
    for i in range(40):
        p = 100 + i
        rows.append({'date': d1, 'permno': p, 'me': 1.0, 'pi': 0.0,
                     'score': 40 - i, 'mom_12': mom.get(p, 0.0),
                     'ret_fwd': 0.0, 'state3': 0})
    m2 = {104: 40, 105: 39, 106: 38, 107: 37,
          100: 4, 101: 3, 102: 2, 103: 1}
    for i in range(40):
        p = 100 + i
        rows.append({'date': d2, 'permno': p, 'me': 1.0, 'pi': 0.9,
                     'score': m2.get(p, 20 - (i * 0.1)),
                     'mom_12': mom.get(p, 0.0), 'ret_fwd': 0.0, 'state3': 1})
    return pd.DataFrame(rows), d2


def test_panic_holds_beaten_down_names_only():
    x, d2 = _fixture()
    _, led = X.simulate(x, ('panic_hold_losers', None), score_col='score')
    exits = set(led[(led['date'] == d2)
                    & (led['cause'] == 'exit_rank')]['permno'])
    # losers 102,103 are HELD through panic (not sold); winners 100,101 traded
    assert 102 not in exits and 103 not in exits
    assert 100 in exits and 101 in exits


def test_calm_month_trades_normally():
    x, _ = _fixture()
    a, _ = X.simulate(x, ('panic_hold_losers', None), score_col='score')
    b, _ = X.simulate(x, ('monthly', None), score_col='score')
    # month 1 is calm -> panic_hold_losers == plain monthly there
    assert np.isclose(a['gross'].iloc[0], b['gross'].iloc[0])
