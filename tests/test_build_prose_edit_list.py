"""
test_build_prose_edit_list.py
==============================
Tests for `scripts/build_prose_edit_list.py` (Step L). Pin the cell
parser, the diff logic, and the prose-grep behaviour.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)


def _load_script():
    path = os.path.join(REPO, 'scripts', 'build_prose_edit_list.py')
    spec = importlib.util.spec_from_file_location('build_prose_edit_list', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def m():
    return _load_script()


# ═══════════════════════════════════════════════════════════════════════════════
# Cell parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseCells:

    def test_extracts_simple_table(self, m, tmp_path):
        tex = tmp_path / 't.tex'
        tex.write_text(
            r'''\begin{table}
\begin{tabular}{l r r}
\toprule
Foo & 1.10 & 2.20 \\
Bar & 3.30 & 4.40 \\
\bottomrule
\end{tabular}
\end{table}
''')
        cells = m.parse_tex_table_cells(str(tex))
        # Cells: 1.10, 2.20, 3.30, 4.40 (label "Foo"/"Bar" doesn't match NUM_RE)
        vals = sorted(c[3] for c in cells)
        assert vals == [1.10, 2.20, 3.30, 4.40]

    def test_handles_percent_and_negative(self, m, tmp_path):
        tex = tmp_path / 't.tex'
        tex.write_text(
            r'''\begin{tabular}{l r}
M2 & 21.9\% \\
M1 & $-$24.8\% \\
\end{tabular}
''')
        cells = m.parse_tex_table_cells(str(tex))
        vals = sorted(c[3] for c in cells)
        # 21.9 and -24.8
        assert vals == [-24.8, 21.9]

    def test_skips_environment_lines(self, m, tmp_path):
        tex = tmp_path / 't.tex'
        tex.write_text(
            r'''\begin{tabular}{l r}
\toprule
\midrule
Foo & 1.0 \\
\bottomrule
\end{tabular}
''')
        cells = m.parse_tex_table_cells(str(tex))
        assert len(cells) == 1
        assert cells[0][3] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Cell diff
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiffCells:

    def test_no_diff_when_identical(self, m):
        pre = [(1, 0, '1.10', 1.10), (2, 0, '2.20', 2.20)]
        post = list(pre)
        diffs, structural = m.diff_table_cells(pre, post)
        assert diffs == []
        assert structural == []

    def test_diff_detects_value_change(self, m):
        pre = [(1, 0, '1.10', 1.10), (2, 0, '2.20', 2.20)]
        post = [(1, 0, '1.11', 1.11), (2, 0, '2.20', 2.20)]
        diffs, _ = m.diff_table_cells(pre, post)
        assert len(diffs) == 1
        assert diffs[0][2:6] == ('1.10', '1.11', 1.10, 1.11)

    def test_structural_diff_when_lengths_differ(self, m):
        pre = [(1, 0, '1.10', 1.10), (2, 0, '2.20', 2.20)]
        post = [(1, 0, '1.10', 1.10)]
        _, structural = m.diff_table_cells(pre, post)
        assert any('structural' in s for s in structural)

    def test_tolerance_ignores_floating_noise(self, m):
        pre = [(1, 0, '1.10', 1.10)]
        post = [(1, 0, '1.10', 1.10 + 1e-12)]
        diffs, _ = m.diff_table_cells(pre, post, tol=1e-9)
        assert diffs == []


# ═══════════════════════════════════════════════════════════════════════════════
# Prose grep
# ═══════════════════════════════════════════════════════════════════════════════

class TestGrepLatex:

    def test_finds_token_in_prose(self, m, tmp_path):
        latex = tmp_path / 'main.tex'
        latex.write_text(
            'Some prose says the Sharpe is 1.11 and the alpha is 23.8\\%.\n'
            'A second line with no number.\n'
        )
        matches = m.grep_latex_for_token('1.11', [latex])
        assert len(matches) == 1
        assert matches[0][1] == 1  # line 1

    def test_skips_caption_and_input_lines(self, m, tmp_path):
        latex = tmp_path / 'main.tex'
        latex.write_text(
            r'\caption{This caption mentions 1.11 explicitly.}' + '\n'
            r'A real prose line says 1.11 again.' + '\n'
        )
        matches = m.grep_latex_for_token('1.11', [latex])
        assert len(matches) == 1
        assert matches[0][1] == 2  # the prose line, not the caption

    def test_handles_dollar_minus_wrapper(self, m, tmp_path):
        latex = tmp_path / 'main.tex'
        latex.write_text(
            'The MDD was -24.8%.\n'
            'Another sentence with $-$24.8\\% syntax.\n'
        )
        # Token from a table is $-$24.8\% — should match both the table-style
        # latex and a plain-text negative if both forms appear
        matches = m.grep_latex_for_token(r'$-$24.8\%', [latex])
        assert len(matches) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end smoke
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_smoke_run_produces_output(self, m, tmp_path):
        # Set up baseline + live + latex dirs
        baseline = tmp_path / 'baseline_pre_shumway_test'
        baseline_tables = baseline / 'tables'
        baseline_tables.mkdir(parents=True)
        live_tables = tmp_path / 'tables_live'
        live_tables.mkdir()
        latex_dir = tmp_path / 'latex'
        latex_dir.mkdir()

        # Pre table has 1.10 in cell, post has 1.11
        (baseline_tables / 'foo.tex').write_text(
            r'''\begin{tabular}{l r}
M2 & 1.10 \\
\end{tabular}
''')
        (live_tables / 'foo.tex').write_text(
            r'''\begin{tabular}{l r}
M2 & 1.11 \\
\end{tabular}
''')
        (latex_dir / 'main.tex').write_text(
            'The Sharpe is 1.10 in the literature.\n'
        )

        # Save and restore sys.argv
        out = tmp_path / 'PROSE_EDITS.md'
        old_argv = sys.argv[:]
        sys.argv = ['build_prose_edit_list.py',
                    '--baseline-dir', str(baseline),
                    '--output', str(out),
                    '--latex-dir', str(latex_dir),
                    '--tables-dir', str(live_tables)]
        try:
            m.main()
        finally:
            sys.argv = old_argv

        text = out.read_text()
        assert 'foo.tex' in text
        assert '1.10' in text
        assert '1.11' in text
        # Should reference the latex file path
        assert 'main.tex' in text
