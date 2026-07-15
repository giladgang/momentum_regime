import numpy as np
import pandas as pd
import pytest

from paper.src.fundamentals import compute_signals


def _funda_fixture():
    # one firm, two fiscal years; hand-checkable numbers
    return pd.DataFrame({
        'permno': [111, 111], 'gvkey': ['001', '001'],
        'datadate': pd.to_datetime(['2014-12-31', '2015-12-31']),
        'avail_date': pd.to_datetime(['2015-06-30', '2016-06-30']),
        'fyear': [2014, 2015],
        'seq': [100.0, 120.0], 'ceq': [95.0, 115.0],
        'txditc': [10.0, np.nan], 'pstkrv': [5.0, np.nan],
        'pstkl': [np.nan, 4.0], 'pstk': [3.0, 3.0],
        'at': [500.0, 550.0], 'at_lag': [np.nan, 500.0], 'lt': [400.0, 430.0],
        'revt': [300.0, 330.0], 'cogs': [200.0, 210.0],
        'xsga': [50.0, np.nan], 'xint': [10.0, 12.0],
        'rect': [40.0, 44.0], 'invt': [30.0, 27.0], 'ap': [20.0, 26.0],
        'xacc': [5.0, 6.0], 'xpp': [2.0, 3.0], 'che': [15.0, 18.0],
    })


def test_book_equity_and_profitability():
    out = compute_signals(_funda_fixture()).set_index('fyear')
    # fy2014: be = 100 + 10 - 5 = 105; op_raw = 300-200-50-10 = 40
    assert np.isclose(out.loc[2014, 'be'], 105.0)
    assert np.isclose(out.loc[2014, 'op'], 40.0 / 105.0)
    # fy2014 has no prior year: deltas -> 0, cop == op
    assert np.isclose(out.loc[2014, 'cop'], 40.0 / 105.0)
    assert np.isnan(out.loc[2014, 'asset_growth'])
    # fy2015: be = 120 + 0 - 4 = 116; op_raw = 330-210-0-12 = 108
    assert np.isclose(out.loc[2015, 'be'], 116.0)
    assert np.isclose(out.loc[2015, 'op'], 108.0 / 116.0)
    # cop_raw = 108 - (44-40) - (27-30) - (3-2) + (26-20) + (6-5) = 113
    assert np.isclose(out.loc[2015, 'cop'], 113.0 / 116.0)
    assert np.isclose(out.loc[2015, 'asset_growth'], 550.0 / 500.0 - 1)


def test_pit_lag_is_six_months():
    out = compute_signals(_funda_fixture())
    lag = ((out['avail_date'].dt.year * 12 + out['avail_date'].dt.month)
           - (out['datadate'].dt.year * 12 + out['datadate'].dt.month))
    assert (lag == 6).all()


def test_negative_be_is_nan():
    f = _funda_fixture()
    f.loc[0, ['seq', 'ceq']] = np.nan
    f.loc[0, 'at'] = 390.0            # at - lt = -10 -> be <= 0 -> NaN
    out = compute_signals(f).set_index('fyear')
    assert np.isnan(out.loc[2014, 'be'])
    assert np.isnan(out.loc[2014, 'op'])
