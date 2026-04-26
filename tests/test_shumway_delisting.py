"""
test_shumway_delisting.py
=========================
Tests for scripts/apply_shumway_delisting.py.

Covers each of the five mutually-exclusive rules on synthetic fixtures,
plus invariants (idempotency, row partitioning, no drop of unrelated
columns, column-missing error handling).

Does NOT touch the real ``data/crsp_msf_raw.parquet`` — all fixtures
are fabricated DataFrames.
"""

import importlib.util
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script():
    """Load scripts/apply_shumway_delisting.py as a module."""
    path = os.path.join(REPO, 'scripts', 'apply_shumway_delisting.py')
    spec = importlib.util.spec_from_file_location('apply_shumway_delisting', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def sh():
    return _load_script()


def _base_row(ret=0.05, dlret=np.nan, dlstcd=np.nan, **extras):
    """Synthetic one-row panel with all required columns + filler."""
    row = {
        'permno': 10001, 'date': pd.Timestamp('2000-01-31'),
        'ret': ret, 'ret_adj': ret,   # ret_adj starts equal to ret (incoming state)
        'dlret': dlret, 'dlstcd': dlstcd,
    }
    row.update(extras)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 1: ret + dlret -> compound
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule1CompoundRetDlret:

    def test_compounds_multiplicatively(self, sh):
        df = pd.DataFrame([_base_row(ret=0.10, dlret=-0.20)])
        out, counts = sh.apply_shumway(df)
        expected = (1.10) * (0.80) - 1.0   # = -0.12
        assert out['ret_adj'].iloc[0] == pytest.approx(expected, abs=1e-12)
        assert counts['rule_1_compound_ret_dlret'] == 1
        for k in ('rule_2_dlret_only', 'rule_3_compound_ret_shumway',
                  'rule_4_shumway_only', 'rule_5_unchanged'):
            assert counts[k] == 0

    def test_zero_dlret_preserves_ret(self, sh):
        """dlret = 0 should leave ret_adj = ret (compound with 1)."""
        df = pd.DataFrame([_base_row(ret=0.05, dlret=0.0)])
        out, _ = sh.apply_shumway(df)
        assert out['ret_adj'].iloc[0] == pytest.approx(0.05, abs=1e-12)

    def test_minus_one_dlret_gives_minus_one(self, sh):
        """Total loss delisting return: (1+ret)(1-1)-1 = -1."""
        df = pd.DataFrame([_base_row(ret=0.05, dlret=-1.0)])
        out, _ = sh.apply_shumway(df)
        assert out['ret_adj'].iloc[0] == pytest.approx(-1.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 2: ret NaN + dlret present -> dlret
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule2DlretOnly:

    def test_uses_dlret_when_ret_missing(self, sh):
        df = pd.DataFrame([_base_row(ret=np.nan, dlret=-0.15)])
        out, counts = sh.apply_shumway(df)
        assert out['ret_adj'].iloc[0] == pytest.approx(-0.15, abs=1e-12)
        assert counts['rule_2_dlret_only'] == 1

    def test_rule_2_fires_regardless_of_dlstcd(self, sh):
        """dlstcd being performance or not doesn't matter once dlret exists."""
        # Non-performance dlstcd
        df = pd.DataFrame([_base_row(ret=np.nan, dlret=-0.05, dlstcd=200)])
        out, counts = sh.apply_shumway(df)
        assert out['ret_adj'].iloc[0] == pytest.approx(-0.05, abs=1e-12)
        assert counts['rule_2_dlret_only'] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 3: ret + perf delisting + no dlret -> compound with -30%
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule3CompoundRetShumway:

    def test_performance_delisting_with_ret_but_no_dlret(self, sh):
        df = pd.DataFrame([_base_row(ret=0.10, dlret=np.nan, dlstcd=575)])
        out, counts = sh.apply_shumway(df)
        expected = (1.10) * (0.70) - 1.0   # = -0.23
        assert out['ret_adj'].iloc[0] == pytest.approx(expected, abs=1e-12)
        assert counts['rule_3_compound_ret_shumway'] == 1

    @pytest.mark.parametrize('code', [500, 520, 550, 575, 580, 599])
    def test_all_performance_codes_trigger(self, sh, code):
        df = pd.DataFrame([_base_row(ret=0.0, dlret=np.nan, dlstcd=code)])
        out, counts = sh.apply_shumway(df)
        assert counts['rule_3_compound_ret_shumway'] == 1
        assert out['ret_adj'].iloc[0] == pytest.approx(-0.30, abs=1e-12)

    @pytest.mark.parametrize('code', [100, 200, 300, 400, 499, 600, 700])
    def test_non_performance_codes_do_not_trigger(self, sh, code):
        df = pd.DataFrame([_base_row(ret=0.05, dlret=np.nan, dlstcd=code)])
        out, counts = sh.apply_shumway(df)
        assert counts['rule_3_compound_ret_shumway'] == 0
        assert counts['rule_5_unchanged'] == 1
        # ret_adj unchanged (stays at ret since rule 5 fires)
        assert out['ret_adj'].iloc[0] == pytest.approx(0.05, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 4: both ret and dlret NaN + perf delisting -> -30%
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule4ShumwayOnly:

    def test_both_null_performance_delisting(self, sh):
        df = pd.DataFrame([_base_row(ret=np.nan, dlret=np.nan, dlstcd=550)])
        out, counts = sh.apply_shumway(df)
        assert out['ret_adj'].iloc[0] == pytest.approx(-0.30, abs=1e-12)
        assert counts['rule_4_shumway_only'] == 1

    def test_both_null_non_performance_delisting(self, sh):
        """Non-performance code with both null -> rule 5, ret_adj stays NaN."""
        df = pd.DataFrame([_base_row(ret=np.nan, dlret=np.nan, dlstcd=200)])
        out, counts = sh.apply_shumway(df)
        assert pd.isna(out['ret_adj'].iloc[0])
        assert counts['rule_5_unchanged'] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 5: fallback -> unchanged
# ═══════════════════════════════════════════════════════════════════════════════

class TestRule5Unchanged:

    def test_no_delisting_info_passthrough(self, sh):
        df = pd.DataFrame([_base_row(ret=0.07, dlret=np.nan, dlstcd=np.nan)])
        out, counts = sh.apply_shumway(df)
        assert out['ret_adj'].iloc[0] == pytest.approx(0.07, abs=1e-12)
        assert counts['rule_5_unchanged'] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Invariants across all rules
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvariants:

    def test_rules_partition_rows(self, sh):
        """Every row matches EXACTLY one rule: counts must sum to len(df)."""
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(1000):
            ret = rng.choice([np.nan, rng.normal(0.01, 0.05)])
            dlret = rng.choice([np.nan, rng.normal(-0.1, 0.1)])
            dlstcd = rng.choice([np.nan, 100, 500, 575, 600])
            rows.append(_base_row(ret=ret, dlret=dlret, dlstcd=dlstcd))
        df = pd.DataFrame(rows)
        _, counts = sh.apply_shumway(df)
        total = sum(counts.values())
        assert total == len(df)

    def test_row_count_preserved(self, sh):
        df = pd.DataFrame([_base_row() for _ in range(42)])
        out, _ = sh.apply_shumway(df)
        assert len(out) == 42

    def test_other_columns_preserved(self, sh):
        """Non-touched columns (permno, date, anything else) must not change."""
        df = pd.DataFrame([_base_row(ret=0.10, dlret=-0.05, my_col='hello')])
        out, _ = sh.apply_shumway(df)
        assert out['permno'].iloc[0] == df['permno'].iloc[0]
        assert out['date'].iloc[0] == df['date'].iloc[0]
        assert out['my_col'].iloc[0] == 'hello'

    def test_idempotent_on_in_memory_rerun(self, sh):
        """Applying twice on the same source must give the same output.
        (Mathematically: rules are deterministic and derive from the
        same input columns, not from the current ret_adj.)"""
        df = pd.DataFrame([
            _base_row(ret=0.10, dlret=-0.20),
            _base_row(ret=np.nan, dlret=-0.15),
            _base_row(ret=0.10, dlret=np.nan, dlstcd=575),
            _base_row(ret=np.nan, dlret=np.nan, dlstcd=550),
            _base_row(ret=0.07, dlret=np.nan, dlstcd=np.nan),
        ])
        out1, c1 = sh.apply_shumway(df)
        # Pass the OUTPUT back in: ret_adj is now different, but rules
        # use ret / dlret / dlstcd, which haven't changed. Result must
        # match.
        out2, c2 = sh.apply_shumway(out1)
        pd.testing.assert_series_equal(out1['ret_adj'], out2['ret_adj'])
        assert c1 == c2

    def test_missing_column_raises(self, sh):
        df = pd.DataFrame([{'permno': 1, 'ret': 0.01}])  # missing dlret, dlstcd, ret_adj
        with pytest.raises(ValueError, match='missing required columns'):
            sh.apply_shumway(df)


# ═══════════════════════════════════════════════════════════════════════════════
# I/O: backup + atomic write (uses tmp_path, does not touch real data)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackupAndAtomicWrite:

    def test_backup_created_on_first_run(self, sh, tmp_path):
        panel = tmp_path / 'panel.parquet'
        backup = tmp_path / 'panel.bak'
        pd.DataFrame([_base_row()]).to_parquet(panel, index=False)
        assert not backup.exists()
        created = sh._ensure_backup(str(panel), str(backup))
        assert created
        assert backup.exists()

    def test_backup_preserved_on_second_run(self, sh, tmp_path):
        panel = tmp_path / 'panel.parquet'
        backup = tmp_path / 'panel.bak'
        pd.DataFrame([_base_row()]).to_parquet(panel, index=False)
        sh._ensure_backup(str(panel), str(backup))
        # First-run backup contents.
        first_backup_bytes = backup.read_bytes()
        # Corrupt the main file and try to re-ensure the backup —
        # backup must NOT be updated.
        pd.DataFrame([_base_row(ret=99.0)]).to_parquet(panel, index=False)
        created = sh._ensure_backup(str(panel), str(backup))
        assert not created
        assert backup.read_bytes() == first_backup_bytes, (
            "Backup must only be created once — otherwise idempotency "
            "breaks (second run would back up an already-modified file)."
        )

    def test_load_source_prefers_backup(self, sh, tmp_path):
        """If backup exists, it's the source of truth (supports rerun)."""
        panel = tmp_path / 'panel.parquet'
        backup = tmp_path / 'panel.bak'
        orig = pd.DataFrame([_base_row(ret=0.05)])
        orig.to_parquet(backup, index=False)
        # Main file has a DIFFERENT value (e.g. a prior Shumway run)
        pd.DataFrame([_base_row(ret=0.99)]).to_parquet(panel, index=False)
        df, src = sh._load_source(str(panel), str(backup))
        assert str(src).endswith('panel.bak')
        assert df['ret'].iloc[0] == pytest.approx(0.05)

    def test_atomic_write_overwrites(self, sh, tmp_path):
        panel = tmp_path / 'panel.parquet'
        tmp = tmp_path / 'panel.tmp'
        pd.DataFrame([_base_row(ret=0.01)]).to_parquet(panel, index=False)
        new = pd.DataFrame([_base_row(ret=0.99)])
        sh._atomic_write(new, str(panel), str(tmp))
        reloaded = pd.read_parquet(panel)
        assert reloaded['ret'].iloc[0] == pytest.approx(0.99)
        assert not tmp.exists(), "tmp file should be renamed away, not left behind"


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end sanity on real data (DRY-RUN only — does not modify files)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDryRunOnRealPanel:
    """Smoke test against the real CRSP panel, but NEVER writes."""

    @pytest.fixture(scope='class')
    def real_panel(self):
        path = os.path.join(REPO, 'data', 'crsp_msf_raw.parquet')
        if not os.path.exists(path):
            pytest.skip("real panel not present")
        return pd.read_parquet(path)

    def test_real_panel_apply_is_clean(self, sh, real_panel):
        """Apply rules in memory on real data; rules must partition and
        not crash."""
        out, counts = sh.apply_shumway(real_panel)
        total = sum(counts.values())
        assert total == len(real_panel)
        assert len(out) == len(real_panel)

    def test_real_panel_rule_counts_are_plausible(self, sh, real_panel):
        """Sanity bounds on what we expect to see on the real data."""
        _, counts = sh.apply_shumway(real_panel)
        # We established during the audit that the real panel has:
        #   ~1,294 rows with dlret and ret both present    (rule 1)
        #   ~2 rows with ret NaN and dlret present         (rule 2)
        #   ~11 performance delistings with ret+no dlret   (rule 3)
        #   ~1 performance delisting with both NaN         (rule 4)
        # These bounds are loose enough to absorb small data-pull shifts.
        assert 1000 < counts['rule_1_compound_ret_dlret'] < 2000
        assert 0 <= counts['rule_2_dlret_only'] < 50
        assert 0 < counts['rule_3_compound_ret_shumway'] < 50
        assert 0 <= counts['rule_4_shumway_only'] < 50
        # Everything else is unchanged (vast majority).
        assert counts['rule_5_unchanged'] > 0.99 * len(real_panel)

    def test_real_panel_rule_1_and_5_dominate(self, sh, real_panel):
        """Rule 1 (compound ret+dlret) + Rule 5 (passthrough) must account
        for >99% of all rows. This pins the relative magnitude of
        delisting-related rows so a regression that mis-routes them into
        a different rule (e.g. rule 3 firing on non-performance codes)
        would shift these counts visibly."""
        _, counts = sh.apply_shumway(real_panel)
        rule_1_5 = counts['rule_1_compound_ret_dlret'] + counts['rule_5_unchanged']
        share = rule_1_5 / len(real_panel)
        assert share > 0.999, (
            f"Rule 1 + Rule 5 share is {share:.4%}, expected > 99.9%. "
            f"Counts: {counts}. Suggests rule routing changed."
        )

    def test_real_panel_rule_1_negative_dlret_yields_lower_ret_adj(self, sh, real_panel):
        """Rule 1 fires when both ret and dlret are present and compounds
        them: ret_adj = (1+ret)(1+dlret) - 1. When dlret < 0 (typical for
        delisting), the compounded result must be <= ret. A sign-flip bug
        in the compounding (e.g., (1+ret)(1-dlret)) would silently
        reverse this."""
        out, _ = sh.apply_shumway(real_panel)
        # Find rule-1 rows: both ret and dlret present
        merged = real_panel[['ret', 'dlret']].copy()
        merged['ret_adj_new'] = out['ret_adj'].values
        rule1_mask = merged['ret'].notna() & merged['dlret'].notna()
        rule1 = merged[rule1_mask]
        if len(rule1) == 0:
            pytest.skip('no rule-1 rows in real panel')
        # When dlret <= 0: ret_adj = (1+ret)(1+dlret)-1 <= ret  iff  ret >= -1.
        # Filter to dlret < 0 and ret > -1 for a clean comparison.
        sub = rule1[(rule1['dlret'] < 0) & (rule1['ret'] > -1)]
        if len(sub) == 0:
            pytest.skip('no negative-dlret rows in rule 1')
        violations = sub[sub['ret_adj_new'] > sub['ret'] + 1e-9]
        assert len(violations) == 0, (
            f"{len(violations)} rule-1 rows have ret_adj > ret despite "
            f"dlret < 0. Suggests a sign-flip bug in the compounding."
        )

    def test_real_panel_rule_4_yields_exactly_minus_thirty(self, sh, real_panel):
        """Rule 4 (both ret and dlret missing, dlstcd in performance
        bucket) imputes exactly -0.30. A regression to a wrong constant
        (e.g., -0.10 from an older draft, or +0.30 from a sign flip)
        would fail this. The constant is the central Shumway claim."""
        out, _ = sh.apply_shumway(real_panel)
        rule4_mask = (real_panel['ret'].isna() & real_panel['dlret'].isna()
                      & real_panel['dlstcd'].between(sh.PERF_DLSTCD_MIN,
                                                     sh.PERF_DLSTCD_MAX))
        if rule4_mask.sum() == 0:
            pytest.skip('no rule-4 rows in real panel')
        adj = out.loc[rule4_mask, 'ret_adj']
        # pytest.approx vs a Series doesn't broadcast cleanly, so do it
        # numerically: every value must be within 1e-9 of -0.30.
        assert ((adj - (-0.30)).abs() < 1e-9).all(), (
            f"Rule 4 rows produce ret_adj != -0.30: "
            f"{adj.unique().tolist()}"
        )

    def test_real_panel_rule_3_compounds_with_minus_thirty(self, sh, real_panel):
        """Rule 3 (performance dlst, ret present, dlret missing) compounds
        ret with -30%: ret_adj = (1+ret)(0.7)-1. For any ret < +1/0.3,
        this must be lower than ret. Sign-flip in the constant
        (e.g., +0.30 instead of -0.30) would invert this."""
        out, _ = sh.apply_shumway(real_panel)
        rule3_mask = (real_panel['ret'].notna() & real_panel['dlret'].isna()
                      & real_panel['dlstcd'].between(sh.PERF_DLSTCD_MIN,
                                                     sh.PERF_DLSTCD_MAX))
        if rule3_mask.sum() == 0:
            pytest.skip('no rule-3 rows in real panel')
        ret = real_panel.loc[rule3_mask, 'ret']
        adj = out.loc[rule3_mask, 'ret_adj']
        expected = (1 + ret) * 0.7 - 1
        diff = (adj - expected).abs()
        assert (diff < 1e-12).all(), (
            f"Rule 3 compounding deviates from (1+ret)(0.7)-1; "
            f"max diff={diff.max():.2e}"
        )
