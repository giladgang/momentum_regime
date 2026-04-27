"""
test_intl_scripts.py
====================
Tests for the international (UK + Japan) chain scripts:

  - scripts/cross_sectional_intl.py — regional cross-sectional model
    Specifically the new ret_adj-aware logic (Phase 1b/N2 dependency).
  - scripts/hmm_intl.py — regional HMM wrapper (Step N1)
    Smoke-level: imports cleanly, has correct feature set, panel paths.
  - scripts/import_intl_stocks.py — keep-list / column propagation
    (the WRDS pull is not unit-testable without WRDS access; we test the
    static-config invariants only).

Heavy production scripts are auto-skipped when their data files aren't
present in the working tree.
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


# ═══════════════════════════════════════════════════════════════════════════════
# cross_sectional_intl.py — source-level invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossSectionalIntlSource:
    """Source-level checks that don't require running the script.

    Importing the script triggers data loads + model fits, which we don't
    want in unit tests; instead we read the source and verify key
    invariants are present."""

    @pytest.fixture(scope='class')
    def src(self):
        path = os.path.join(REPO, 'scripts', 'cross_sectional_intl.py')
        with open(path) as f:
            return f.read()

    def test_prefers_ret_adj_when_present(self, src):
        """When ret_adj is in the panel, ret_fwd must be recomputed from it."""
        assert "if 'ret_adj' in stocks.columns" in src
        assert "stocks.groupby('secid')['ret_adj'].shift(-1)" in src

    def test_does_not_overwrite_ret_column(self, src):
        """The contract with apply_shumway_intl is that `ret` is preserved
        for diagnostic comparison. Verify no `stocks['ret'] = ...` write
        outside of the original load."""
        # No assignment overwriting `ret` after the load. Allow only
        # comparisons/reads (`stocks['ret']`).
        # An assignment like `stocks['ret'] = ...` is the regression we want
        # to prevent.
        bad = re.search(r"stocks\['ret'\]\s*=\s*", src)
        assert bad is None, (
            f"cross_sectional_intl.py overwrites stocks['ret'] at "
            f"{bad.start()}; should leave the column untouched."
        )

    def test_uses_ffill_no_bfill_for_pi_filter(self, src):
        """Pi_filter merge must use ffill only (causal); bfill would leak
        future regime information."""
        assert "stocks['pi_filter'].ffill()" in src
        assert ".bfill()" not in src

    def test_uses_unconditional_decile_breakpoints(self, src):
        """No NYSE filter — use percentile of ALL stocks each month."""
        assert "scores.quantile(0.10)" in src
        assert "scores.quantile(0.90)" in src
        # Ensure no `exchcd == 1` filter snuck in
        assert "exchcd" not in src or ("# " in src and "exchcd" not in
            ''.join(line for line in src.split('\n')
                    if not line.strip().startswith('#')))

    def test_no_fundamentals_in_features(self, src):
        """Per INTL_VALIDATION_PLAN.md: international model is momentum-only."""
        assert "MOM_FEATURES = [f'mom_{k}' for k in range(1, 13)]" in src
        assert "FEATURES     = MOM_FEATURES + ['pi_filter']" in src

    def test_uses_train_end_from_config(self, src):
        """Train/test split must come from config.TRAIN_END (not hardcoded)."""
        assert "from config import" in src and "TRAIN_END" in src
        assert "pd.Timestamp(TRAIN_END)" in src

    def test_use_us_pi_flag_supported(self, src):
        """Test A: --use-us-pi swaps regional regime panel for US."""
        assert "'--use-us-pi'" in src
        assert "args.use_us_pi" in src

    def test_market_benchmark_always_regional(self, src):
        """Even when --use-us-pi is set, the market return benchmark must
        be the REGIONAL market — we're testing whether US regimes predict
        REGIONAL momentum, not US-vs-US."""
        # The regional market panel (not US) is loaded for benchmarking
        assert "REGIONAL_REG" in src
        assert "mkt_panel = pd.read_parquet(REGIONAL_REG)" in src


# ═══════════════════════════════════════════════════════════════════════════════
# hmm_intl.py — source-level invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestHmmIntlSource:

    @pytest.fixture(scope='class')
    def src(self):
        path = os.path.join(REPO, 'scripts', 'hmm_intl.py')
        with open(path) as f:
            return f.read()

    def test_uses_four_feature_regional_set(self, src):
        """Regional HMM features: DD, DISP, REL_N, BANK_REL (NOT CS)."""
        assert "['DD_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z']" in src
        # Should NOT use the US CS feature
        # (it's OK for the string 'CS_z' to appear in comments documenting
        # WHY we substitute BANK_REL_z; just not as an active feature)
        active_lines = [l for l in src.split('\n')
                        if 'CS_z' in l and not l.strip().startswith('#')
                        and 'BANK_REL' not in l]
        assert len(active_lines) == 0, \
            f"hmm_intl.py references CS_z in active code: {active_lines}"

    def test_does_not_import_hmm_model(self, src):
        """hmm_intl.py must be self-contained — importing hmm_model.py
        would trigger its US fit at import time, blocking parallel runs."""
        assert "import hmm_model" not in src
        # 'from hmm_model import' would be the other failure mode
        assert "from hmm_model" not in src

    def test_writes_region_specific_outputs(self, src):
        """Outputs go to data/{region}_panel_with_regimes.parquet,
        NOT panel_with_regimes.parquet (which is the US output)."""
        assert "{region}_panel_with_regimes" in src \
            or "_panel_with_regimes.parquet" in src
        # Must NOT overwrite the US panel
        assert "data/panel_with_regimes.parquet'" not in src.replace(
            'US_REG', '<placeholder>')


# ═══════════════════════════════════════════════════════════════════════════════
# import_intl_stocks.py — keep-list / column propagation
# ═══════════════════════════════════════════════════════════════════════════════

class TestImportIntlStocksSource:

    @pytest.fixture(scope='class')
    def src(self):
        path = os.path.join(REPO, 'scripts', 'import_intl_stocks.py')
        with open(path) as f:
            return f.read()

    def test_sql_pulls_delisting_fields(self, src):
        """Phase 1b requires secstat + dldte + dlrsn from comp.g_security."""
        assert "sec.secstat" in src
        assert "sec.dldtei AS dldte" in src
        assert "sec.dlrsni AS dlrsn" in src

    def test_dldte_in_date_cols(self, src):
        """dldte must be parsed as a datetime by db.raw_sql; otherwise it
        comes back as object dtype and apply_shumway_intl's pd.to_datetime
        does double work."""
        assert "'dldte'" in src
        # date_cols list must include 'dldte'
        m = re.search(r"date_cols\s*=\s*\[([^\]]*)\]", src)
        assert m is not None
        assert "'dldte'" in m.group(1)

    def test_keep_list_includes_delisting_fields(self, src):
        """Stock keep-list must propagate secstat/dldte/dlrsn to the
        parquet output."""
        # Find the stock_keep assignment block
        m = re.search(r"stock_keep\s*=\s*\[(.*?)\]", src, re.DOTALL)
        assert m is not None, "Could not locate stock_keep list"
        keep_block = m.group(1)
        for col in ('secstat', 'dldte', 'dlrsn'):
            assert f"'{col}'" in keep_block, \
                f"stock_keep is missing '{col}'"

    def test_documented_in_docstring(self, src):
        """Docstring should reflect the new columns so users discover them."""
        # First docstring of the module
        assert 'secstat' in src.split('\"\"\"', 2)[1]
        assert 'dldte' in src.split('\"\"\"', 2)[1]


# ═══════════════════════════════════════════════════════════════════════════════
# Real-data smoke (auto-skip)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntlPanelsOnDisk:

    @pytest.mark.parametrize('region', ['UK', 'JP'])
    def test_panel_columns_present(self, region):
        path = os.path.join(REPO, 'data',
                             f'{region.lower()}_stock_panel.parquet')
        if not os.path.exists(path):
            pytest.skip(f'{path} not present')
        df = pd.read_parquet(path)
        for col in ('secid', 'date', 'prc_close', 'ret', 'ret_fwd', 'me',
                    'is_bank'):
            assert col in df.columns, f"{path} missing column {col}"

    @pytest.mark.parametrize('region', ['UK', 'JP'])
    def test_market_panel_features_present(self, region):
        path = os.path.join(REPO, 'data',
                             f'{region.lower()}_market_panel.parquet')
        if not os.path.exists(path):
            pytest.skip(f'{path} not present')
        df = pd.read_parquet(path)
        for col in ('date', 'mkt_ret',
                    'DD_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z'):
            assert col in df.columns, f"{path} missing column {col}"
