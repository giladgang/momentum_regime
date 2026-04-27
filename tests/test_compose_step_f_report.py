"""
test_compose_step_f_report.py
=============================
Tests for `scripts/compose_step_f_report.py` — the composer that builds
`results/STEP_F_REPORT.md` from pre-Shumway baseline tables, post-Shumway
live tables, and the Step E leg-betas CSV.

The script is a "stitcher" — it doesn't compute numbers, it just reads
existing tables and emits a markdown report. So the tests focus on:
  - cell parsing correctness on synthetic LaTeX tables
  - LaTeX-to-markdown sanitisation
  - delta computation
  - end-to-end smoke that produces a well-formed report
"""

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script():
    path = os.path.join(REPO, 'scripts', 'compose_step_f_report.py')
    spec = importlib.util.spec_from_file_location('compose_step_f_report', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def m():
    return _load_script()


# ═══════════════════════════════════════════════════════════════════════════════
# parse_table_row
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseTableRow:

    def test_finds_label_returns_correct_column(self, m, tmp_path):
        tex = tmp_path / 't.tex'
        tex.write_text(
            r'''\begin{tabular}{l r r r}
\toprule
Strategy & Ann.Ret & Sharpe & MDD \\
\midrule
M2: XGB & 21.7\% & 1.11 & -22.8\% \\
\bottomrule
\end{tabular}
''')
        assert m.parse_table_row(str(tex), 'M2: XGB', 1) == r'21.7\%'
        assert m.parse_table_row(str(tex), 'M2: XGB', 2) == '1.11'
        assert m.parse_table_row(str(tex), 'M2: XGB', 3) == r'-22.8\%'

    def test_returns_none_when_label_missing(self, m, tmp_path):
        tex = tmp_path / 't.tex'
        tex.write_text(
            r'''\begin{tabular}{l r}
M2: XGB & 1.11 \\
\end{tabular}
''')
        assert m.parse_table_row(str(tex), 'Method 99', 1) is None

    def test_returns_none_when_path_missing(self, m, tmp_path):
        assert m.parse_table_row(str(tmp_path / 'nonexistent.tex'), 'X', 1) is None

    def test_handles_multiple_rows_picks_first_match(self, m, tmp_path):
        tex = tmp_path / 't.tex'
        tex.write_text(
            r'''\begin{tabular}{l r}
M2: XGB & 1.10 \\
M2: XGB & 1.11 \\
\end{tabular}
''')
        # First match wins (current contract)
        assert m.parse_table_row(str(tex), 'M2: XGB', 1) == '1.10'

    def test_skips_comment_lines(self, m, tmp_path):
        tex = tmp_path / 't.tex'
        tex.write_text(
            '% This is a comment about M2: XGB & 99.99 \\\\' + '\n'
            r'M2: XGB & 1.11 \\' + '\n'
        )
        assert m.parse_table_row(str(tex), 'M2: XGB', 1) == '1.11'


# ═══════════════════════════════════════════════════════════════════════════════
# _md_clean
# ═══════════════════════════════════════════════════════════════════════════════

class TestMdClean:

    def test_strips_escaped_percent(self, m):
        assert m._md_clean(r'21.7\%') == '21.7%'

    def test_strips_dollar_minus_wrapper(self, m):
        assert m._md_clean(r'$-$24.8\%') == '-24.8%'

    def test_strips_trailing_double_backslash(self, m):
        assert m._md_clean(r'1.11 \\') == '1.11'

    def test_none_passthrough(self, m):
        assert m._md_clean(None) is None

    def test_already_clean_text(self, m):
        assert m._md_clean('1.11') == '1.11'

    def test_strips_combined_decoration(self, m):
        assert m._md_clean(r'$-$24.8\% \\') == '-24.8%'


# ═══════════════════════════════════════════════════════════════════════════════
# maybe_float
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaybeFloat:

    def test_parses_plain_number(self, m):
        assert m.maybe_float('1.11') == pytest.approx(1.11)

    def test_parses_percent(self, m):
        assert m.maybe_float(r'21.9\%') == pytest.approx(21.9)

    def test_parses_dollar_minus_negative(self, m):
        assert m.maybe_float(r'$-$24.8\%') == pytest.approx(-24.8)

    def test_strips_significance_stars(self, m):
        assert m.maybe_float('4.37***') == pytest.approx(4.37)

    def test_returns_none_for_non_numeric(self, m):
        assert m.maybe_float('CAPM') is None

    def test_returns_none_for_none(self, m):
        assert m.maybe_float(None) is None


# ═══════════════════════════════════════════════════════════════════════════════
# fmt_delta
# ═══════════════════════════════════════════════════════════════════════════════

class TestFmtDelta:

    def test_both_present_no_marker(self, m):
        a, b, marker = m.fmt_delta('1.10', '1.11')
        assert a == '1.10' and b == '1.11' and marker == ''

    def test_pre_missing_marker_new(self, m):
        a, b, marker = m.fmt_delta(None, '1.11')
        assert a == '—' and marker == 'NEW'

    def test_post_missing_marker_dropped(self, m):
        a, b, marker = m.fmt_delta('1.10', None)
        assert b == '—' and marker == 'DROPPED'

    def test_both_missing(self, m):
        assert m.fmt_delta(None, None) == ('—', '—', '—')

    def test_cleans_latex_decoration(self, m):
        a, b, _ = m.fmt_delta(r'$-$24.8\%', r'$-$22.8\%')
        assert a == '-24.8%' and b == '-22.8%'


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end smoke
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def _make_minimal_tables(self, base_dir, sharpe_pre, sharpe_post):
        """Drop just enough tables for the composer to run end-to-end."""
        for d in (base_dir / 'baseline' / 'tables',
                  base_dir / 'live' / 'tables'):
            d.mkdir(parents=True, exist_ok=True)
        # table_performance — only the M2 row matters
        for tex_dir, sharpe in [
            (base_dir / 'baseline' / 'tables', sharpe_pre),
            (base_dir / 'live' / 'tables', sharpe_post),
        ]:
            (tex_dir / 'table_performance.tex').write_text(
                r'''\begin{tabular}{l r r r r r r r}
M2: XGB & 21.7\% & 19.5\% & ''' + f'{sharpe}'
                r''' & -22.8\% & 0.45 & 4.37*** & 15.3 \\
\end{tabular}
''')
            (tex_dir / 'table_regime_sharpe.tex').write_text(
                r'''\begin{tabular}{l r r r}
M2: XGB & 1.11 & 0.84 & 1.53 \\
\end{tabular}
''')
            (tex_dir / 'table_subperiod.tex').write_text(
                r'''\begin{tabular}{l r r r r}
M2: XGB & 0.59 & 1.04 & 1.70 & 1.11 \\
\end{tabular}
''')
            (tex_dir / 'table_factor_alphas.tex').write_text(
                r'''\begin{tabular}{l r r}
CAPM & 23.4 & 4.08 \\
FF6 & 24.1 & 4.81 \\
\end{tabular}
''')
            (tex_dir / 'table_shap.tex').write_text(
                r'''\begin{tabular}{l r r r}
Momentum & 0.0207 & 0.0170 & 0.0273 \\
pi_filter & 0.0173 & 0.0165 & 0.0187 \\
\end{tabular}
''')

    def test_smoke_run_produces_well_formed_report(self, m, tmp_path):
        self._make_minimal_tables(tmp_path, sharpe_pre='1.11', sharpe_post='1.11')
        out = tmp_path / 'REPORT.md'
        old_argv = sys.argv[:]
        old_cwd = os.getcwd()
        # The composer hardcodes `tables/` and `results/` to REPO. We bypass
        # those by chdir'ing into a workspace that mimics that layout.
        ws = tmp_path / 'live'
        os.chdir(ws)
        # Patch the REPO/tables resolution via monkeypatch on the module-
        # level constant
        old_repo = m.REPO
        m.REPO = ws
        sys.argv = ['compose_step_f_report.py',
                    '--baseline-dir', str(tmp_path / 'baseline'),
                    '--output', str(out)]
        try:
            m.main()
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
            m.REPO = old_repo

        text = out.read_text()
        # Header
        assert 'Step F — First Checkpoint Report' in text
        # Sections present
        for section in ['Headline (M2: XGB',
                        'Regime-conditional Sharpe',
                        'Sub-period Sharpe',
                        'Factor alphas',
                        'SHAP feature importance',
                        'Leg-betas finding (Step E)',
                        'Sanity gates',
                        'Gates respected']:
            assert section in text, f'missing section: {section}'
        # LaTeX decoration sanitised — should not see escaped percent
        assert r'\%' not in text
        # Sharpe shift sanity gate
        assert 'Sharpe shift' in text

    def test_anomalous_sharpe_shift_flagged(self, m, tmp_path):
        """When Sharpe shifts >0.15 the report must say ANOMALY."""
        self._make_minimal_tables(tmp_path, sharpe_pre='1.11', sharpe_post='1.50')
        out = tmp_path / 'REPORT.md'
        old_argv = sys.argv[:]
        old_cwd = os.getcwd()
        old_repo = m.REPO
        ws = tmp_path / 'live'
        os.chdir(ws)
        m.REPO = ws
        sys.argv = ['compose_step_f_report.py',
                    '--baseline-dir', str(tmp_path / 'baseline'),
                    '--output', str(out)]
        try:
            m.main()
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
            m.REPO = old_repo

        text = out.read_text()
        assert 'ANOMALY' in text or 'flag' in text

    def test_missing_baseline_dir_exits(self, m, tmp_path):
        old_argv = sys.argv[:]
        sys.argv = ['compose_step_f_report.py',
                    '--baseline-dir', str(tmp_path / 'does_not_exist'),
                    '--output', str(tmp_path / 'r.md')]
        try:
            with pytest.raises(SystemExit):
                m.main()
        finally:
            sys.argv = old_argv
