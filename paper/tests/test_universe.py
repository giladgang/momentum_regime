import pandas as pd

from paper.src import universe as U


def _toy():
    rows = []
    for d in ['2020-01-31', '2020-02-29']:
        for p in range(1, 6):
            rows.append({'date': pd.Timestamp(d), 'permno': p, 'me': float(p)})
    return pd.DataFrame(rows)


def test_top_n_membership():
    s = _toy()
    m = U.top_n(s, 2)
    assert set(m[m['date'] == '2020-01-31']['permno']) == {4, 5}
    assert len(m) == 4


def test_cap_coverage():
    s = _toy()
    m = U.top_n(s, 2)
    cov = U.cap_coverage(s, m)
    assert abs(cov.iloc[0] - 9.0 / 15.0) < 1e-12
