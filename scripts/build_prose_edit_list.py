"""
build_prose_edit_list.py
========================
Step L of the post-Shumway chain: build a structured checklist of every
prose number in the thesis that needs updating because the underlying
table cell changed between the pre-Shumway baseline and the post-Shumway
rerun.

Pipeline
--------
1. For each `tables/*.tex` file, parse the cells in the baseline snapshot
   and in the live table. Diff cell values (numeric only, ignoring
   formatting noise like trailing zeros).
2. For each changed cell, grep `latex/*.tex` for the pre-Shumway numeric
   string. Emit every match with file:line:context.
3. Write `results/PROSE_EDITS.md` as a markdown checklist that the user
   reviews and greenlights line-by-line before any `.tex` edit.

Caveats
-------
- Numeric strings like "0.84" or "1.11" appear in many unrelated places
  in prose. Each row is reviewed by a human; this script surfaces
  candidates, not commands.
- Skips environment markers (`\\begin{...}`, `\\end{...}`, `\\caption{...}`)
  on the assumption their numbers are macro arguments, not data.
- Compares numeric tokens not whole table content, so layout differences
  (column reordering, dropped rows) are reported separately.

Usage
-----
    python scripts/build_prose_edit_list.py \\
        --baseline-dir baseline_pre_shumway_20260426_150517 \\
        --output results/PROSE_EDITS.md

Run AFTER Step D + Step E complete (so post-Shumway tables are written)
and BEFORE any thesis edit (gate G3).
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LATEX_DIR = REPO / 'latex'
TABLES_DIR = REPO / 'tables'
RESULTS_DIR = REPO / 'results'

# Numeric token patterns in tables (handle %, $-$, $ wrappers, scientific)
# Examples: "21.9\%", "$-$24.8\%", "1.11", "-0.66", "$-$0.30", "4.37***"
NUM_RE = re.compile(
    r'(?<![\w.])'                       # word boundary on the left
    r'(\$-\$|-)?'                       # optional unary minus (latex or plain)
    r'(\d+(?:,\d{3})*(?:\.\d+)?)'       # core number with optional thousands sep
    r'(\\?%)?'                          # optional percent (escaped or not)
    r'(?![\w.])'                        # word boundary on the right
)

# Patterns to skip when emitting prose matches (these are macro args, not text)
SKIP_LATEX_LINE = re.compile(
    r'^\s*\\(begin|end|caption|label|input|cite|ref|usepackage|documentclass|'
    r'newcommand|title|author|date|section|subsection|subsubsection)\b'
)


def parse_tex_table_cells(path):
    """Extract numeric cells from a single .tex table file.

    Returns list of (line_no, cell_index, raw_token, numeric_value).
    Handles `&`-separated cells inside tabular environments. Ignores
    macro/environment lines (begin/end/caption etc.)."""
    cells = []
    with open(path) as f:
        lines = f.readlines()
    in_tabular = False
    for line_no, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(r'\begin{tabular}'):
            in_tabular = True
            continue
        if stripped.startswith(r'\end{tabular}'):
            in_tabular = False
            continue
        if not in_tabular:
            continue
        if stripped.startswith('%'):
            continue
        # Remove trailing \\
        body = re.sub(r'\\\\\s*$', '', stripped)
        # Skip pure rule rows
        if body.replace(r'\toprule', '').replace(r'\midrule', '').replace(
                r'\bottomrule', '').strip() == '':
            continue
        for cell_idx, cell in enumerate(body.split('&')):
            for m in NUM_RE.finditer(cell):
                sign, mantissa, pct = m.group(1), m.group(2), m.group(3)
                try:
                    val = float(mantissa.replace(',', ''))
                except ValueError:
                    continue
                if sign in ('-', r'$-$'):
                    val = -val
                token = m.group(0)
                cells.append((line_no, cell_idx, token, val))
    return cells


def diff_table_cells(pre_cells, post_cells, tol=1e-9):
    """Diff cells aligned by (line_no, cell_index, token-position).

    Returns list of (line_no, cell_idx, pre_token, post_token, pre_val, post_val)
    for cells where the numeric value changed beyond `tol`.

    If row counts differ (e.g., a table dropped a row), reports the missing
    rows as a structural diff."""
    structural = []
    if len(pre_cells) != len(post_cells):
        structural.append(
            f'    structural: pre had {len(pre_cells)} numeric tokens, '
            f'post has {len(post_cells)}'
        )
    diffs = []
    # Align by index. Imperfect when rows are added/removed, but caller
    # is expected to inspect structural diffs separately.
    for pc, qc in zip(pre_cells, post_cells):
        pre_line, pre_idx, pre_tok, pre_val = pc
        post_line, post_idx, post_tok, post_val = qc
        if abs(pre_val - post_val) > tol:
            diffs.append((pre_line, pre_idx, pre_tok, post_tok,
                          pre_val, post_val))
    return diffs, structural


def grep_latex_for_token(token, latex_files):
    """Search every latex file for the literal token (or close variants).

    Returns list of (file, line_no, line_text). Skips lines that look like
    macro/environment headers."""
    matches = []
    # Build a tolerant search pattern: handle spacing inside $ wrappers
    candidates = {token}
    # Strip $-$ wrapper for plain-text alternative
    if token.startswith('$-$'):
        candidates.add('-' + token[3:])
    # Also try without escaped percent
    if r'\%' in token:
        candidates.add(token.replace(r'\%', '%'))
    for path in latex_files:
        try:
            try:
                shown = str(path.relative_to(REPO))
            except ValueError:
                shown = str(path)
            with open(path) as f:
                for line_no, raw in enumerate(f, start=1):
                    if SKIP_LATEX_LINE.match(raw):
                        continue
                    if any(c in raw for c in candidates):
                        matches.append((shown, line_no, raw.rstrip('\n')))
        except OSError:
            continue
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', required=True,
                        help='Path to baseline_pre_shumway_<stamp>/ snapshot')
    parser.add_argument('--output',
                        default=str(RESULTS_DIR / 'PROSE_EDITS.md'),
                        help='Where to write the checklist (default: '
                             'results/PROSE_EDITS.md)')
    parser.add_argument('--latex-dir', default=str(LATEX_DIR),
                        help='Thesis latex directory (default: latex/)')
    parser.add_argument('--tables-dir', default=str(TABLES_DIR),
                        help='Live tables directory (default: tables/)')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Numeric tolerance for considering a cell '
                             'unchanged (default 1e-6)')
    args = parser.parse_args()

    baseline_tables_dir = Path(args.baseline_dir) / 'tables'
    if not baseline_tables_dir.is_dir():
        sys.exit(f'baseline tables dir not found: {baseline_tables_dir}')

    live_tables_dir = Path(args.tables_dir)
    latex_dir = Path(args.latex_dir)
    if not latex_dir.is_dir():
        sys.exit(f'latex dir not found: {latex_dir}')

    latex_files = sorted(latex_dir.rglob('*.tex'))
    if not latex_files:
        sys.exit(f'no .tex files under {latex_dir}')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pre_files = {p.name: p for p in baseline_tables_dir.glob('*.tex')}
    post_files = {p.name: p for p in live_tables_dir.glob('*.tex')}

    common = sorted(set(pre_files) & set(post_files))
    only_pre = sorted(set(pre_files) - set(post_files))
    only_post = sorted(set(post_files) - set(pre_files))

    lines = [
        '# Prose-edit checklist (Step L)',
        '',
        f'Baseline: `{args.baseline_dir}`',
        f'Live:     `{args.tables_dir}`',
        f'Latex:    `{args.latex_dir}`',
        '',
        '**Each row below is a CANDIDATE.** Numeric strings can match '
        'unrelated places. Check the line context and tick the box if the '
        'edit is correct; skip if false-positive.',
        '',
        '---',
        '',
    ]

    if only_pre:
        lines.append('## Tables present in baseline only (DROPPED)')
        for n in only_pre:
            lines.append(f'- `{n}` — review whether prose still references it')
        lines.append('')
    if only_post:
        lines.append('## Tables added (NEW)')
        for n in only_post:
            lines.append(f'- `{n}`')
        lines.append('')

    total_diffs = 0
    total_matches = 0
    for tex in common:
        pre_path = pre_files[tex]
        post_path = post_files[tex]
        pre_cells = parse_tex_table_cells(pre_path)
        post_cells = parse_tex_table_cells(post_path)
        diffs, structural = diff_table_cells(pre_cells, post_cells, tol=args.tol)
        if not diffs and not structural:
            continue
        lines.append(f'## `{tex}`')
        if structural:
            for s in structural:
                lines.append(s)
        for (line_no, cell_idx, pre_tok, post_tok,
             pre_val, post_val) in diffs:
            total_diffs += 1
            lines.append(
                f'### Cell change at line {line_no} '
                f'(token `{pre_tok}` → `{post_tok}`)'
            )
            matches = grep_latex_for_token(pre_tok, latex_files)
            if not matches:
                lines.append('  No prose matches found (table-only change).')
                continue
            for path, mline, ctx in matches:
                total_matches += 1
                short_ctx = ctx.strip()[:140]
                lines.append(
                    f'- [ ] `{path}:{mline}` — `{pre_tok}` → `{post_tok}`'
                )
                lines.append(f'    context: {short_ctx}')
        lines.append('')

    lines.append('---')
    lines.append('')
    lines.append(
        f'**Summary:** {total_diffs} cell-level changes across '
        f'{len(common)} tables → {total_matches} candidate prose edits.'
    )

    output_path.write_text('\n'.join(lines) + '\n')
    print(f'Wrote: {output_path}')
    print(f'Tables changed: {sum(1 for _ in common)}, '
          f'cells: {total_diffs}, prose candidates: {total_matches}')


if __name__ == '__main__':
    main()
