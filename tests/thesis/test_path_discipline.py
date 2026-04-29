"""
test_path_discipline.py
=======================
Layer 5b — script-level path-discipline check.

Many scripts in ``scripts/`` historically hardcode output paths as
string literals (e.g. ``'artefacts/cs_artefacts_data.pkl'``,
``'tables/foo.tex'``) instead of reading them from ``config.py``.
When the pipeline is invoked with ``MOMENTUM_OUTPUT_ROOT`` set
(``--repro-smoke``, smoke-fixture generation, isolated CI runs),
hardcoded paths silently bypass the redirection and write to
**production** directories — clobbering live artefacts. This was
witnessed on 2026-04-29 when the smoke fixture overwrote production
``cs_artefacts_data.pkl`` and the canonical tex tables.

This test AST-scans every ``scripts/*.py`` for short string literals
that begin with a production output prefix and reports the file:line
of every offender. Whitelist legitimate or temporarily-acceptable
cases via ``tests/thesis/path_discipline_whitelist.txt``.

Run with:

    pytest tests/thesis/test_path_discipline.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCRIPTS = REPO / "scripts"
WHITELIST = HERE / "path_discipline_whitelist.txt"

# Production output prefixes that scripts must not hardcode.
HARDCODED_PREFIXES = (
    "artefacts/",
    "tables/",
    "results/",
    "plots/",
)
# Inputs that are technically built by the pipeline but treated as inputs
# downstream (panel files). Hardcoding these also breaks isolation.
HARDCODED_FILE_LITERALS = (
    "data/panel.parquet",
    "data/panel_with_regimes.parquet",
)


def _whitelist() -> set[str]:
    """Whitelist file format:  <file:line>  # rationale
    Match is by `<file:line>` substring against the offender string.
    """
    if not WHITELIST.exists():
        return set()
    out: set[str] = set()
    for line in WHITELIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split("#", 1)[0].strip()
        if token:
            out.add(token)
    return out


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Collect ids of Constant nodes that are module/class/function docstrings."""
    out: set[int] = set()
    body_holders = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if isinstance(node, body_holders) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                out.add(id(first.value))
    return out


def _hardcoded_path_offenders(rel_path: str) -> list[tuple[str, int, str]]:
    full = REPO / rel_path
    if not full.exists():
        return []
    try:
        tree = ast.parse(full.read_text(encoding="utf-8"), filename=rel_path)
    except SyntaxError:
        return []
    docstring_ids = _docstring_node_ids(tree)
    offenders: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstring_ids:
            continue
        s = node.value
        # Skip multi-line strings (almost always comments/docs/log messages).
        if "\n" in s or len(s) > 200:
            continue
        matched = False
        for prefix in HARDCODED_PREFIXES:
            if s.startswith(prefix):
                matched = True
                break
        if not matched:
            for lit in HARDCODED_FILE_LITERALS:
                if s == lit:
                    matched = True
                    break
        if matched:
            offenders.append((rel_path, node.lineno, s))
    return offenders


def test_no_hardcoded_output_paths():
    """Every script must read output paths from cfg.* (RESULTS_DIR,
    TABLES_DIR, ARTEFACTS_PATH, ARTEFACTS_DIR, PLOTS_DIR, PANEL_PATH,
    PANEL_WITH_REGIMES_PATH, etc.) so MOMENTUM_OUTPUT_ROOT redirection
    works. Hardcoded literals like ``'artefacts/cs_artefacts_data.pkl'``
    bypass the redirection and clobber production state under
    ``--repro-smoke`` / fixture generation / isolated CI runs.
    """
    if not SCRIPTS.exists():
        pytest.skip("scripts/ missing")
    wl = _whitelist()
    bad: dict[str, list[str]] = {}
    for path in sorted(SCRIPTS.glob("*.py")):
        rel = str(path.relative_to(REPO))
        offenders = []
        for f, line, s in _hardcoded_path_offenders(rel):
            tag = f"{f}:{line}"
            if any(w == tag or w == f"{f}:{line} '{s}'" for w in wl):
                continue
            offenders.append(f"{tag}  '{s}'")
        if offenders:
            bad[rel] = offenders
    if not bad:
        return
    total = sum(len(v) for v in bad.values())
    msg_lines = [
        f"{total} hardcoded output paths in {len(bad)} scripts.",
        "These bypass config.py path redirection and break "
        "MOMENTUM_OUTPUT_ROOT-based isolation (--repro-smoke, "
        "make_smoke_fixture, CI sandboxes).",
        "",
        "Offenders (first 5 per file):",
    ]
    for rel, offenders in sorted(bad.items()):
        msg_lines.append(f"  {rel}:")
        for o in offenders[:5]:
            msg_lines.append(f"    {o}")
        if len(offenders) > 5:
            msg_lines.append(f"    ...and {len(offenders) - 5} more")
    msg_lines.append("")
    msg_lines.append(
        "Fix by importing the appropriate cfg.* attribute "
        "(RESULTS_DIR, TABLES_DIR, ARTEFACTS_PATH, PLOTS_DIR, etc.) "
        "and using f-strings against it. Or whitelist with rationale "
        "in tests/thesis/path_discipline_whitelist.txt."
    )
    pytest.fail("\n".join(msg_lines))
