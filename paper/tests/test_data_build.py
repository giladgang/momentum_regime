import os

import pandas as pd
import pytest

from paper import config as C
from paper.src import data_build


@pytest.fixture(scope='module')
def built():
    if not (os.path.exists(C.PANEL_PARQUET) and os.path.exists(C.STOCKS_PARQUET)):
        data_build.build()
    return True


def test_panel_schema_and_range(built):
    p = pd.read_parquet(C.PANEL_PARQUET)
    zs = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z', 'CS_z', 'LVIX_z', 'TERM_z', 'SKEW_z']
    assert all(c in p.columns for c in zs)
    assert str(p['date'].min())[:7] <= '1990-12'
    assert str(p['date'].max())[:7] == '2025-11'
    sub = p[p['date'] >= C.PANEL_START]
    assert sub[zs].notna().all().all()


def test_stocks_schema_and_range(built):
    s = pd.read_parquet(C.STOCKS_PARQUET,
                        columns=['date', 'permno', 'me', 'ret_fwd', 'mom_12',
                                 'exchcd', 'prc'])
    assert s['me'].notna().mean() > 0.95
    assert str(s['date'].max())[:7] == '2025-11'    # last formation month
    m = s[s['date'] == s['date'].max()]
    assert len(m) > 2000                            # full universe pre-filter
    assert m['ret_fwd'].notna().all()               # Dec-2025 return exists
