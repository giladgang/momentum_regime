"""
test_apply_shumway_intl.py
==========================
Tests for `scripts/apply_shumway_intl.py`. The Compustat Global delisting
treatment is heuristic (price-drop based, no native dlret/dlstcd fields).
These tests pin the rule logic on synthetic fixtures plus a dry-run
against the real UK panel.
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
    path = os.path.join(REPO, 'scripts', 'apply_shumway_intl.py')
    spec = importlib.util.spec_from_file_location('apply_shumway_intl', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def sh():
    return _load_script()


def _make_security(secid, dates, prices, rets):
    """Build a 1-security DataFrame with the columns the script expects.

    Heuristic-mode panel (no secstat/dldte). Use _make_strict for strict
    mode."""
    return pd.DataFrame({
        'secid': [secid] * len(dates),
        'date': pd.to_datetime(dates),
        'prc_close': prices,
        'ret': rets,
    })


def _make_strict_panel(rows):
    """Build a panel including secstat/dldte. Each row is a dict with
    keys: secid, date, prc_close, ret, secstat, dldte."""
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['dldte'] = pd.to_datetime(df['dldte'], errors='coerce')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Rule-level synthetic tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRuleFirings:

    def test_active_security_passthrough(self, sh):
        """A security observed at panel-end has ret_adj = ret."""
        df = _make_security('A',
                            ['2020-01-31', '2020-02-29', '2020-03-31'],
                            [100, 110, 121],
                            [np.nan, 0.10, 0.10])
        out, c = sh.apply_shumway_intl(df)
        assert c['mode'] == 'heuristic'
        assert c['rule_4_unchanged'] == 3
        assert c['rule_1_compound_ret_with_drop'] == 0
        assert c['rule_2_drop_only'] == 0
        assert (out['ret_adj'].fillna(-99) == out['ret'].fillna(-99)).all()

    def test_disappeared_with_drop_and_observed_ret_compounds(self, sh):
        """Rule 1: ret present + final-month price drop > 10% → compound."""
        # Two securities; A continues, B disappears in early 2020 with a big drop.
        rows = []
        for d in pd.date_range('2020-01-31', '2024-12-31', freq='ME'):
            rows.append({'secid': 'A', 'date': d, 'prc_close': 100.0, 'ret': 0.0})
        # B: ends 2020-02 with a 50% price drop AND observed ret of -0.05
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-01-31'),
                     'prc_close': 100.0, 'ret': 0.01})
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-02-29'),
                     'prc_close': 50.0, 'ret': -0.05})
        df = pd.DataFrame(rows)
        out, c = sh.apply_shumway_intl(df)
        assert c['rule_1_compound_ret_with_drop'] == 1
        # Rule 1: ret_adj = (1 + -0.05) * (1 + (50/100 - 1)) - 1 = 0.95 * 0.5 - 1 = -0.525
        b_last = out[(out['secid'] == 'B') & (out['date'] == '2020-02-29')]
        assert b_last['ret_adj'].iloc[0] == pytest.approx(0.95 * 0.5 - 1.0,
                                                            abs=1e-9)

    def test_disappeared_with_drop_and_missing_ret_uses_drop(self, sh):
        """Rule 2: ret missing + strong price drop → ret_adj = drop."""
        rows = []
        for d in pd.date_range('2020-01-31', '2024-12-31', freq='ME'):
            rows.append({'secid': 'A', 'date': d, 'prc_close': 100.0, 'ret': 0.0})
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-01-31'),
                     'prc_close': 100.0, 'ret': 0.01})
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-02-29'),
                     'prc_close': 50.0, 'ret': np.nan})
        df = pd.DataFrame(rows)
        out, c = sh.apply_shumway_intl(df)
        assert c['rule_2_drop_only'] == 1
        b_last = out[(out['secid'] == 'B') & (out['date'] == '2020-02-29')]
        assert b_last['ret_adj'].iloc[0] == pytest.approx(-0.5, abs=1e-9)

    def test_disappeared_no_signal_passthrough(self, sh):
        """Rule 3: terminal-disappeared row with no usable signal → leave alone.

        Rule 3 counts the FINAL row of disappeared securities that didn't
        match rule 1 or rule 2. Earlier rows of the same security are
        rule 4 (unchanged). So one terminal row → rule_3 = 1."""
        rows = []
        for d in pd.date_range('2020-01-31', '2024-12-31', freq='ME'):
            rows.append({'secid': 'A', 'date': d, 'prc_close': 100.0, 'ret': 0.0})
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-01-31'),
                     'prc_close': 100.0, 'ret': 0.01})
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-02-29'),
                     'prc_close': 99.0, 'ret': np.nan})  # mild dip, ret missing
        df = pd.DataFrame(rows)
        out, c = sh.apply_shumway_intl(df)
        assert c['rule_3_disappeared_no_signal'] == 1
        b_last = out[(out['secid'] == 'B') & (out['date'] == '2020-02-29')]
        assert pd.isna(b_last['ret_adj'].iloc[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Strict-mode rules (panel includes secstat / dldte)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrictMode:
    """Strict mode: panel includes secstat + dldte (+ optionally dlrsn).

    With dlrsn, only performance codes ({02 bankruptcy, 03 liquidation})
    trigger imputation. Without dlrsn, all inactive-in-window rows
    trigger (legacy path)."""

    def test_inactive_performance_no_ret_imputes_minus_thirty(self, sh):
        """Strict-1: secstat='I', dlrsn=02 (bankruptcy), in window, ret missing."""
        df = _make_strict_panel([
            {'secid': 'A', 'date': '2020-01-31', 'prc_close': 100.0,
             'ret': 0.01, 'secstat': 'I', 'dldte': '2020-02-15', 'dlrsn': '02'},
            {'secid': 'A', 'date': '2020-02-29', 'prc_close': 50.0,
             'ret': np.nan, 'secstat': 'I', 'dldte': '2020-02-15', 'dlrsn': '02'},
        ])
        out, c = sh.apply_shumway_intl(df)
        assert c['mode'] == 'strict'
        assert c['strict_1_impute_minus30_no_ret'] == 1
        assert out.iloc[-1]['ret_adj'] == pytest.approx(-0.30, abs=1e-9)

    def test_inactive_performance_with_ret_compounds_minus_thirty(self, sh):
        """Strict-2: secstat='I', dlrsn=03 (liquidation), in window, ret present."""
        df = _make_strict_panel([
            {'secid': 'A', 'date': '2020-01-31', 'prc_close': 100.0,
             'ret': 0.01, 'secstat': 'I', 'dldte': '2020-01-31', 'dlrsn': '03'},
        ])
        out, c = sh.apply_shumway_intl(df)
        assert c['mode'] == 'strict'
        assert c['strict_2_compound_ret_with_minus30'] == 1
        assert out.iloc[0]['ret_adj'] == pytest.approx(1.01 * 0.7 - 1,
                                                          abs=1e-9)

    def test_inactive_nonperformance_passthrough(self, sh):
        """dlrsn=01 (acquisition) is non-performance — must NOT impute."""
        df = _make_strict_panel([
            {'secid': 'A', 'date': '2020-02-29', 'prc_close': 50.0,
             'ret': np.nan, 'secstat': 'I', 'dldte': '2020-02-15', 'dlrsn': '01'},
        ])
        out, c = sh.apply_shumway_intl(df)
        assert c['mode'] == 'strict'
        assert c['strict_1_impute_minus30_no_ret'] == 0
        assert c['strict_2_compound_ret_with_minus30'] == 0
        assert c['strict_3_passthrough_nonperformance'] == 1
        # ret_adj == ret (NaN passthrough)
        assert pd.isna(out.iloc[0]['ret_adj'])

    def test_active_security_in_strict_passthrough(self, sh):
        """Strict-4: active securities are untouched."""
        df = _make_strict_panel([
            {'secid': 'A', 'date': '2020-01-31', 'prc_close': 100.0,
             'ret': 0.05, 'secstat': 'A', 'dldte': pd.NaT, 'dlrsn': pd.NA},
            {'secid': 'A', 'date': '2020-02-29', 'prc_close': 105.0,
             'ret': 0.05, 'secstat': 'A', 'dldte': pd.NaT, 'dlrsn': pd.NA},
        ])
        out, c = sh.apply_shumway_intl(df)
        assert c['mode'] == 'strict'
        assert c['strict_1_impute_minus30_no_ret'] == 0
        assert c['strict_2_compound_ret_with_minus30'] == 0
        assert c['strict_4_passthrough_other'] == 2
        assert (out['ret_adj'].values == out['ret'].values).all()

    def test_inactive_outside_window_passthrough(self, sh):
        """Inactive but date >>90d from dldte must NOT trigger imputation
        — this is the security's pre-distress history, not the delisting
        event itself."""
        df = _make_strict_panel([
            {'secid': 'A', 'date': '2018-01-31', 'prc_close': 100.0,
             'ret': 0.05, 'secstat': 'I', 'dldte': '2020-02-15', 'dlrsn': '02'},
        ])
        out, c = sh.apply_shumway_intl(df)
        assert c['mode'] == 'strict'
        assert c['strict_1_impute_minus30_no_ret'] == 0
        assert c['strict_2_compound_ret_with_minus30'] == 0

    def test_strict_without_dlrsn_legacy_fallback(self, sh):
        """When dlrsn column is absent, fall back to imputing all
        inactive-in-window rows (less precise but documented)."""
        df = _make_strict_panel([
            {'secid': 'A', 'date': '2020-02-29', 'prc_close': 50.0,
             'ret': np.nan, 'secstat': 'I', 'dldte': '2020-02-15'},
        ])
        # Drop dlrsn explicitly (it wasn't added in this fixture)
        assert 'dlrsn' not in df.columns
        out, c = sh.apply_shumway_intl(df)
        assert c['mode'] == 'strict'
        # Without dlrsn, ALL in-window inactives trigger
        assert c['strict_1_impute_minus30_no_ret'] == 1

    def test_strict_idempotent(self, sh):
        """Running strict mode twice on the source must be deterministic."""
        df = _make_strict_panel([
            {'secid': 'A', 'date': '2020-01-31', 'prc_close': 100.0,
             'ret': 0.01, 'secstat': 'I', 'dldte': '2020-02-15', 'dlrsn': '02'},
            {'secid': 'A', 'date': '2020-02-29', 'prc_close': 50.0,
             'ret': np.nan, 'secstat': 'I', 'dldte': '2020-02-15', 'dlrsn': '02'},
        ])
        out1, c1 = sh.apply_shumway_intl(df)
        out2, c2 = sh.apply_shumway_intl(out1.drop(columns=['ret_adj']))
        pd.testing.assert_series_equal(out1['ret_adj'], out2['ret_adj'])
        assert c1 == c2


# ═══════════════════════════════════════════════════════════════════════════════
# Invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvariants:

    def test_row_count_preserved(self, sh):
        df = pd.DataFrame({
            'secid': ['A'] * 5,
            'date': pd.date_range('2020-01-31', periods=5, freq='ME'),
            'prc_close': [100.0] * 5,
            'ret': [0.0] * 5,
        })
        out, _ = sh.apply_shumway_intl(df)
        assert len(out) == 5

    def test_ret_column_preserved(self, sh):
        """The original `ret` column must NOT be overwritten."""
        df = pd.DataFrame({
            'secid': ['A'] * 5,
            'date': pd.date_range('2020-01-31', periods=5, freq='ME'),
            'prc_close': [100.0] * 5,
            'ret': [0.01, 0.02, 0.03, 0.04, 0.05],
        })
        out, _ = sh.apply_shumway_intl(df)
        assert (out['ret'] == df['ret'].values).all()

    def test_ret_adj_column_added(self, sh):
        df = pd.DataFrame({
            'secid': ['A'] * 3,
            'date': pd.date_range('2020-01-31', periods=3, freq='ME'),
            'prc_close': [100.0] * 3,
            'ret': [0.0] * 3,
        })
        out, _ = sh.apply_shumway_intl(df)
        assert 'ret_adj' in out.columns

    def test_idempotent_in_memory(self, sh):
        rows = []
        for d in pd.date_range('2020-01-31', '2024-12-31', freq='ME'):
            rows.append({'secid': 'A', 'date': d, 'prc_close': 100.0, 'ret': 0.0})
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-01-31'),
                     'prc_close': 100.0, 'ret': 0.01})
        rows.append({'secid': 'B', 'date': pd.Timestamp('2020-02-29'),
                     'prc_close': 50.0, 'ret': -0.05})
        df = pd.DataFrame(rows)
        out1, c1 = sh.apply_shumway_intl(df)
        # Pass output back in. Rules use ret/prc_close (not ret_adj), so the
        # second application must produce the same ret_adj.
        df2 = out1.drop(columns=['ret_adj'])
        out2, c2 = sh.apply_shumway_intl(df2)
        pd.testing.assert_series_equal(out1['ret_adj'], out2['ret_adj'])
        assert c1 == c2

    def test_missing_required_columns_raises(self, sh):
        df = pd.DataFrame({'secid': ['A'], 'date': ['2020-01-31']})
        with pytest.raises(ValueError, match='missing required columns'):
            sh.apply_shumway_intl(df)


# ═══════════════════════════════════════════════════════════════════════════════
# Backup + atomic write
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackupAndAtomicWrite:

    def test_backup_created_on_first_run(self, sh, tmp_path):
        src = tmp_path / 'panel.parquet'
        bak = tmp_path / 'panel.bak'
        df = pd.DataFrame({
            'secid': ['A'] * 3,
            'date': pd.date_range('2020-01-31', periods=3, freq='ME'),
            'prc_close': [100.0] * 3,
            'ret': [0.0] * 3,
        })
        df.to_parquet(src, index=False)
        assert not bak.exists()
        assert sh._ensure_backup(str(src), str(bak))
        assert bak.exists()

    def test_backup_preserved_on_second_run(self, sh, tmp_path):
        src = tmp_path / 'panel.parquet'
        bak = tmp_path / 'panel.bak'
        df = pd.DataFrame({
            'secid': ['A'] * 3,
            'date': pd.date_range('2020-01-31', periods=3, freq='ME'),
            'prc_close': [100.0] * 3,
            'ret': [0.0] * 3,
        })
        df.to_parquet(src, index=False)
        sh._ensure_backup(str(src), str(bak))
        first = bak.read_bytes()
        # Corrupt the source and try to re-backup; backup must NOT update
        df2 = pd.DataFrame({
            'secid': ['A'] * 3,
            'date': pd.date_range('2020-01-31', periods=3, freq='ME'),
            'prc_close': [99.0] * 3,
            'ret': [0.0] * 3,
        })
        df2.to_parquet(src, index=False)
        assert not sh._ensure_backup(str(src), str(bak))
        assert bak.read_bytes() == first

    def test_load_source_prefers_backup(self, sh, tmp_path):
        src = tmp_path / 'panel.parquet'
        bak = tmp_path / 'panel.bak'
        orig = pd.DataFrame({
            'secid': ['A'], 'date': pd.to_datetime(['2020-01-31']),
            'prc_close': [100.0], 'ret': [0.05],
        })
        orig.to_parquet(bak, index=False)
        # Different content in source
        new = pd.DataFrame({
            'secid': ['A'], 'date': pd.to_datetime(['2020-01-31']),
            'prc_close': [99.0], 'ret': [0.99],
        })
        new.to_parquet(src, index=False)
        df, used = sh._load_source(str(src), str(bak))
        assert str(used).endswith('.bak')
        assert df['ret'].iloc[0] == pytest.approx(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# Real-panel dry run (skips if not present)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDryRunOnRealPanels:

    @pytest.mark.parametrize('region', ['UK', 'JP'])
    def test_real_panel_apply_is_safe(self, sh, region):
        path = os.path.join(REPO, 'data', f'{region.lower()}_stock_panel.parquet')
        if not os.path.exists(path):
            pytest.skip(f'{path} not present')
        df = pd.read_parquet(path)
        out, counts = sh.apply_shumway_intl(df)
        # Row count preserved
        assert len(out) == len(df)
        # Sum of touched-rule counts + passthrough/unchanged = total rows
        if counts['mode'] == 'strict':
            total = (counts['strict_1_impute_minus30_no_ret']
                     + counts['strict_2_compound_ret_with_minus30']
                     + counts['strict_3_passthrough_nonperformance']
                     + counts['strict_4_passthrough_other'])
            assert total == len(df)
        else:
            total_touched = (counts['rule_1_compound_ret_with_drop']
                             + counts['rule_2_drop_only'])
            assert counts['rule_4_unchanged'] + total_touched == len(df)
        # ret_adj column added
        assert 'ret_adj' in out.columns
        # ret column values unchanged (dtype-tolerant)
        pre_vals = pd.Series(df['ret']).astype(float).values
        post_vals = pd.Series(out['ret']).astype(float).values
        assert np.array_equal(pre_vals, post_vals, equal_nan=True)
