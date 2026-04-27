"""
test_audit_compustat_delisting.py
=================================
Tests for `scripts/audit_compustat_delisting.py` — the diagnostic that
characterises delisting events in the UK/JP panels and recommends a
strategy for `apply_shumway_intl.py`.

The audit reads the existing parquets and prints diagnostics. Tests
cover the per-region diagnostic on synthetic mini-panels.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script():
    path = os.path.join(REPO, 'scripts', 'audit_compustat_delisting.py')
    spec = importlib.util.spec_from_file_location('audit_compustat_delisting',
                                                    path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def m():
    return _load_script()


def _panel(secid_records, panel_end='2024-12-31'):
    """secid_records: list of (secid, dates list, prc list, ret list)."""
    rows = []
    for secid, dates, prices, rets in secid_records:
        for d, p, r in zip(dates, prices, rets):
            rows.append({'secid': secid, 'date': d,
                         'prc_close': p, 'ret': r})
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Audit mechanics — exercised on tmpdir parquets
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuditRegion:

    def test_returns_none_if_panel_missing(self, m, tmp_path, capsys):
        # No data file -> returns None and prints SKIP
        old_repo = m.REPO
        m.REPO = tmp_path
        try:
            out = m.audit_region('UK')
            assert out is None
            captured = capsys.readouterr()
            assert 'SKIP' in captured.out or 'not present' in captured.out
        finally:
            m.REPO = old_repo

    def test_returns_summary_when_panel_present(self, m, tmp_path):
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'data').mkdir()
        df = _panel([
            ('A', pd.date_range('2020-01-31', '2024-12-31', freq='ME'),
             [100.0] * 60, [0.0] * 60),
            ('B', pd.date_range('2020-01-31', '2020-03-31', freq='ME'),
             [100.0, 50.0, 25.0], [0.0, -0.5, np.nan]),
        ])
        df.to_parquet(tmp_path / 'data' / 'uk_stock_panel.parquet', index=False)
        try:
            out = m.audit_region('UK')
            assert out is not None
            assert out['region'] == 'UK'
            assert out['unique_securities'] == 2
            # B disappeared (last in 2020-03), A is active through 2024-12.
            # Both have last_date < (panel_end - 90d) only for B.
            assert out['n_disappeared'] == 1
            # B's last row has ret=NaN
            assert out['n_last_ret_missing'] == 1
            # B has prev_close (50→25 = drop of 50%) → strong drop signal
            assert out['n_with_price_drop_signal'] == 1
        finally:
            m.REPO = old_repo

    def test_no_disappeared_in_active_only_panel(self, m, tmp_path):
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'data').mkdir()
        df = _panel([
            ('A', pd.date_range('2020-01-31', '2024-12-31', freq='ME'),
             [100.0] * 60, [0.0] * 60),
        ])
        df.to_parquet(tmp_path / 'data' / 'jp_stock_panel.parquet', index=False)
        try:
            out = m.audit_region('JP')
            assert out['n_disappeared'] == 0
            assert out['n_last_ret_missing'] == 0
            assert out['n_with_price_drop_signal'] == 0
        finally:
            m.REPO = old_repo

    def test_disappeared_with_already_negative_ret_counted(self, m, tmp_path):
        """Securities that disappear with a strongly negative final ret
        are counted in `n_already_negative` (no imputation needed)."""
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'data').mkdir()
        df = _panel([
            ('A', pd.date_range('2020-01-31', '2024-12-31', freq='ME'),
             [100.0] * 60, [0.0] * 60),
            ('B', pd.date_range('2020-01-31', '2020-03-31', freq='ME'),
             [100.0, 50.0, 25.0], [0.0, -0.5, -0.5]),
        ])
        df.to_parquet(tmp_path / 'data' / 'uk_stock_panel.parquet', index=False)
        try:
            out = m.audit_region('UK')
            assert out['n_already_negative'] == 1
        finally:
            m.REPO = old_repo


# ═══════════════════════════════════════════════════════════════════════════════
# CSV output
# ═══════════════════════════════════════════════════════════════════════════════

class TestCSVOutput:

    def test_csv_written_when_flag_set(self, m, tmp_path):
        old_repo = m.REPO
        m.REPO = tmp_path
        (tmp_path / 'data').mkdir()
        # Provide both UK and JP panels so the script runs through to CSV
        df_uk = _panel([
            ('A', pd.date_range('2020-01-31', '2024-12-31', freq='ME'),
             [100.0] * 60, [0.0] * 60),
        ])
        df_uk.to_parquet(tmp_path / 'data' / 'uk_stock_panel.parquet',
                          index=False)
        df_jp = _panel([
            ('A', pd.date_range('2020-01-31', '2024-12-31', freq='ME'),
             [200.0] * 60, [0.01] * 60),
        ])
        df_jp.to_parquet(tmp_path / 'data' / 'jp_stock_panel.parquet',
                          index=False)
        out_csv = tmp_path / 'audit.csv'
        old_argv = sys.argv[:]
        sys.argv = ['audit_compustat_delisting.py',
                    '--csv', str(out_csv),
                    '--regions', 'UK', 'JP']
        try:
            m.main()
        finally:
            sys.argv = old_argv
            m.REPO = old_repo
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert set(df['region']) == {'UK', 'JP'}
        assert {'panel_rows', 'unique_securities', 'n_disappeared'} \
            .issubset(df.columns)
