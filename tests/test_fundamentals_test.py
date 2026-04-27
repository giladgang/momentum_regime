"""
test_fundamentals_test.py
=========================
Tests for `scripts/fundamentals_test.py`.

The script is heavyweight (50-seed XGB ensembles across feature variants,
~15 min runtime) so we don't exercise main() in unit tests. We pin:

  - the NW t-stat methodology fix (uses raw mean t, NOT mean − market t)
    — this aligns with the rest of the codebase post commit 3e27a80
  - the LaTeX fragment emit picks the correct column from the cached CSV
  - core invariants (uses ret_adj, has cached CSV with both t-stat
    columns, etc.)
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


@pytest.fixture(scope='module')
def src():
    path = os.path.join(REPO, 'scripts', 'fundamentals_test.py')
    with open(path) as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════════
# NW t-stat methodology
# ═══════════════════════════════════════════════════════════════════════════════

class TestNWTStatMethodology:
    """The Apr 4 commit 3e27a80 changed `newey_west_t` everywhere from
    'mean excess return vs benchmark' to 'mean return = 0'. The
    fundamentals script must follow suit in its LaTeX emit (the CSV may
    keep both for diagnostic comparison)."""

    def test_latex_fragment_uses_nw_t_not_excess(self, src):
        """The published LaTeX row must use raw-mean nw_t, NOT
        nw_t_excess (which would be inconsistent with table_performance
        produced by main_results_analysis.py)."""
        # Find the section that writes table_performance_fund_row.tex
        m = re.search(
            r"fund_row\s*=\s*next.*?with open\('tables/table_performance_fund_row\.tex'",
            src, re.DOTALL,
        )
        assert m is not None, "Could not locate fund_row LaTeX emit block"
        block = m.group(0)
        # Block must reference fund_row['nw_t'] (raw mean) NOT
        # fund_row['nw_t_excess']
        assert "fund_row['nw_t']" in block, (
            "LaTeX emit must use nw_t (raw mean t-stat)"
        )
        # nw_t_excess may appear in the CSV/diagnostic block but not in
        # the LaTeX nw_str line specifically
        nw_str_line = next(
            line for line in block.split('\n')
            if 'nw_str = f"' in line
        )
        assert 'nw_t_excess' not in nw_str_line, (
            "LaTeX nw_str must not pull from nw_t_excess column"
        )

    def test_csv_keeps_both_for_diagnostic(self, src):
        """The CSV stores both `nw_t` and `nw_t_excess` so diagnostic
        comparison stays available."""
        assert "nw_t=nw_t" in src
        assert "nw_t_excess=nw_t_excess" in src

    def test_doc_comment_explains_market_control(self, src):
        """The methodology choice should be documented inline."""
        # The fix note at the LaTeX-emit site should mention market exposure
        # being controlled in the factor-alpha table, not subtracted.
        assert "raw mean t-stat" in src or "mean = 0" in src
        # Should reference the factor-alpha table convention
        assert "factor" in src.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# CSV contract — outputs the columns downstream consumers expect
# ═══════════════════════════════════════════════════════════════════════════════

class TestCSVContract:

    def test_csv_path_unchanged(self, src):
        assert "results/fundamentals_test_results.csv" in src

    def test_emits_required_columns(self, src):
        """Downstream consumers (Step F report composer; thesis tables)
        rely on these columns being in the CSV."""
        for col in ('name', 'ann_ret', 'sharpe', 'mdd', 'beta',
                     'nw_t', 'nw_t_excess'):
            assert col in src, f"CSV column {col!r} not emitted by script"


# ═══════════════════════════════════════════════════════════════════════════════
# Returns-source contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestReturnsSource:

    def test_uses_ret_fwd_post_shumway(self, src):
        """Script must read ret_fwd (which is derived from ret_adj
        post-Shumway by the upstream cross_sectional_model)."""
        assert "ret_fwd" in src
        # The portfolio construction must use ret_fwd, not ret directly
        assert "(longs['ret_fwd']" in src or "longs['ret_fwd']" in src

    def test_loads_artefacts_from_step_d(self, src):
        """Must consume the Step D artefacts pickle, NOT recompute the panel."""
        # Either via config.ARTEFACTS_PATH or a literal path
        assert ("ARTEFACTS_PATH" in src
                or "cs_artefacts_data.pkl" in src), \
            "fundamentals_test.py must reference the Step D artefacts pickle"
        assert "pickle.load" in src


# ═══════════════════════════════════════════════════════════════════════════════
# Cached CSV state (auto-skip if absent)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCachedCSV:

    def test_cached_csv_has_expected_columns(self):
        """If the CSV exists from a prior run, it must contain both nw_t
        and nw_t_excess so we can regenerate the LaTeX without rerunning."""
        path = os.path.join(REPO, 'results', 'fundamentals_test_results.csv')
        if not os.path.exists(path):
            pytest.skip('cached CSV not present')
        df = pd.read_csv(path)
        for col in ('name', 'sharpe', 'nw_t', 'nw_p',
                    'nw_t_excess', 'nw_p_excess', 'beta'):
            assert col in df.columns, f'cached CSV missing {col}'
        # Full+pi+fund variant must be in the rows
        assert any(df['name'].astype(str).str.contains(
            'full_mom_pi_fund', regex=False))

    def test_latex_fragment_value_matches_csv_nw_t(self):
        """If both the CSV and the regenerated LaTeX fragment exist,
        verify the LaTeX shows the nw_t (not nw_t_excess) value."""
        csv_path = os.path.join(REPO, 'results',
                                  'fundamentals_test_results.csv')
        tex_path = os.path.join(REPO, 'tables',
                                  'table_performance_fund_row.tex')
        if not (os.path.exists(csv_path) and os.path.exists(tex_path)):
            pytest.skip('cached CSV or LaTeX fragment not present')
        df = pd.read_csv(csv_path)
        fund = df[df['name'] == 'full_mom_pi_fund']
        if fund.empty:
            pytest.skip('full_mom_pi_fund row not in CSV')
        nw_t = fund.iloc[0]['nw_t']
        nw_t_excess = fund.iloc[0]['nw_t_excess']
        with open(tex_path) as f:
            tex = f.read()
        # The LaTeX should contain nw_t to 2dp (and NOT nw_t_excess to 2dp,
        # unless they happen to round to the same value)
        nw_t_str = f"{nw_t:.2f}"
        assert nw_t_str in tex, (
            f"LaTeX fragment must show nw_t={nw_t_str}; current content:\n"
            f"{tex.strip()}"
        )
