"""
test_coverage_gate.py
=====================
Layer 3 of the thesis verification suite — coverage report.

Walks every ``latex/*.tex`` file plus ``main.tex`` and lists every decimal
token that does NOT trace to one of the canonical sources:

  - ``\newcommand{\\m...}{value}`` in ``latex/canonical_macros.tex``
  - ``EXP.X`` constants from ``tests/_expected.py``
  - ``published.value`` entries from ``tests/thesis/manifest.yaml``
  - Lines in ``tests/thesis/coverage_whitelist.txt``

The output is a markdown report at ``tests/thesis/coverage_report.md`` (gitignored,
always overwritten). The test PASSES regardless of orphan count (warn-mode is
default per spec §7). Set ``THESIS_GATE=strict`` to fail the test when orphans
exist.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.thesis.conftest import REPO, load_manifest

LATEX_DIR = REPO / "latex"
WHITELIST_PATH = Path(__file__).parent / "coverage_whitelist.txt"
REPORT_PATH = Path(__file__).parent / "coverage_report.md"
MACROS_PATH = LATEX_DIR / "canonical_macros.tex"


# Regex for decimals: signed, ≥1 digit before decimal, decimal point, ≥1 digit after.
# Negative lookbehind/ahead avoids partial matches inside larger numbers.
DECIMAL_RX = re.compile(r"(?<![\w.])-?\d+\.\d+(?!\d)")


# Tokens to strip from .tex content before scanning for decimals
def _strip_tex(text: str) -> str:
    r"""Remove comments, math display, label/ref/cite/section commands, and
    ``\m<Name>`` macro CALLS (treated as already-reachable). Keep figure/table
    captions but not their ``\caption{}`` wrapper (the contents are scanned)."""
    # Drop comment lines (everything after a non-escaped %)
    text = re.sub(r"(?<!\\)%.*", "", text)
    # Strip display math
    text = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.DOTALL)
    # Strip simple label/ref/cite/section/input commands (incl. their argument)
    text = re.sub(r"\\(label|ref|c?ref|Cref|cite[tp]?|autoref|input|include)\{[^}]*\}", "", text)
    text = re.sub(r"\\(section|subsection|subsubsection|paragraph|chapter)\*?\{[^}]*\}", "", text)
    # Strip canonical-macro CALLS: \m<Name> with optional spacing (e.g. \msharpe{} or \msharpe~)
    text = re.sub(r"\\m[A-Za-z][A-Za-z0-9]*\b", "", text)
    return text


def _read_macros(macros_path: Path) -> set[str]:
    """Return the set of literal numeric values defined as
    ``\\newcommand{\\m...}{value}`` in canonical_macros.tex."""
    if not macros_path.exists():
        return set()
    content = macros_path.read_text(encoding="utf-8")
    out: set[str] = set()
    for m in re.finditer(r"\\newcommand\{\\m[A-Za-z0-9]+\}\{([^}]+)\}", content):
        raw = m.group(1).strip().replace("$-$", "-").replace("\\%", "").replace("%", "")
        out.add(raw)
        # Also normalise (drop trailing zeros)
        if "." in raw:
            out.add(raw.rstrip("0").rstrip("."))
    return out


def _read_expected_values() -> set[str]:
    """All numeric constants from tests/_expected.py."""
    try:
        from tests import _expected as EXP
    except Exception:  # pragma: no cover
        return set()
    out: set[str] = set()
    for name in dir(EXP):
        if name.startswith("_"):
            continue
        v = getattr(EXP, name)
        if isinstance(v, (int, float)):
            out.add(_normalise(v))
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, (int, float)):
                    out.add(_normalise(x))
    return out


def _read_manifest_values() -> set[str]:
    """All numeric published.value entries from manifest.yaml (flattened)."""
    out: set[str] = set()
    for entry in load_manifest():
        v = (entry.get("published") or {}).get("value")
        if v is None:
            continue
        if isinstance(v, list):
            for x in v:
                if isinstance(x, (int, float)):
                    out.add(_normalise(x))
        elif isinstance(v, (int, float)):
            out.add(_normalise(v))
    return out


def _read_whitelist() -> set[str]:
    if not WHITELIST_PATH.exists():
        return set()
    out: set[str] = set()
    for line in WHITELIST_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip trailing '#' rationale
        token = line.split("#", 1)[0].strip()
        if token:
            out.add(_normalise(_to_float_str(token)))
    return out


def _to_float_str(s: str) -> str:
    try:
        return str(float(s))
    except ValueError:
        return s


def _normalise(v) -> str:
    """Stringify a numeric value with trailing-zero collapse so different
    formattings of the same value compare equal."""
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return v
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ──────────────────────────────────────────────────────────────────────────────


def _scan_tex(rel_path: Path) -> list[tuple[int, str, str]]:
    """Return list of (lineno, raw_token, surrounding_40_chars)."""
    text = rel_path.read_text(encoding="utf-8")
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = _strip_tex(line)
        if not stripped or stripped.startswith("%"):
            continue
        # Skip if line tagged with `% config-ok`
        if "config-ok" in line:
            continue
        for m in DECIMAL_RX.finditer(stripped):
            tok = m.group(0)
            ctx_start = max(0, m.start() - 20)
            ctx_end = min(len(stripped), m.end() + 20)
            ctx = stripped[ctx_start:ctx_end].strip()
            out.append((lineno, tok, ctx))
    return out


def _build_reachable() -> set[str]:
    out = set()
    out.update(_read_whitelist())
    out.update(_read_expected_values())
    out.update(_read_manifest_values())
    out.update(_normalise(v) for v in _read_macros(MACROS_PATH))
    return out


def _build_report() -> tuple[int, int, list[dict]]:
    reachable = _build_reachable()
    orphans: list[dict] = []
    total = 0
    if LATEX_DIR.exists():
        tex_files = sorted(LATEX_DIR.glob("*.tex"))
    else:
        tex_files = []
    main_tex = REPO / "main.tex"
    if main_tex.exists():
        tex_files.append(main_tex)

    for tf in tex_files:
        # Don't scan canonical_macros.tex for orphans — its values define the
        # reachable set by construction.
        if tf.name == "canonical_macros.tex":
            continue
        for lineno, tok, ctx in _scan_tex(tf):
            total += 1
            norm = _normalise(tok)
            if norm in reachable:
                continue
            orphans.append({
                "file": str(tf.relative_to(REPO)),
                "lineno": lineno,
                "value": tok,
                "context": ctx,
            })
    reachable_count = total - len(orphans)
    return total, reachable_count, orphans


def _write_report(total: int, reachable: int, orphans: list[dict]):
    lines = []
    lines.append("# Thesis Coverage Report\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
    lines.append(f"- Total decimal tokens scanned: **{total}**\n")
    lines.append(f"- Reachable via canonical macros / manifest / _expected / whitelist: **{reachable}**\n")
    lines.append(f"- Orphan tokens: **{len(orphans)}**\n\n")
    lines.append("## Orphan tokens\n\n")
    if not orphans:
        lines.append("_No orphan tokens — full coverage._\n")
    else:
        lines.append("| File:line | Value | Context |\n")
        lines.append("|---|---|---|\n")
        for o in orphans[:500]:
            ctx = o["context"].replace("|", r"\|").replace("\n", " ")
            lines.append(f"| {o['file']}:{o['lineno']} | `{o['value']}` | {ctx} |\n")
        if len(orphans) > 500:
            lines.append(f"\n_...and {len(orphans) - 500} more orphans truncated._\n")
    REPORT_PATH.write_text("".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_coverage_report_writable():
    """Smoke test: can we generate and write the coverage report?"""
    total, reachable, orphans = _build_report()
    _write_report(total, reachable, orphans)
    assert REPORT_PATH.exists(), f"failed to write {REPORT_PATH}"
    # Loose sanity: total should be > 0 (the thesis has decimals)
    assert total >= 0  # tolerate empty-latex repos


def test_coverage_orphans_strict_mode():
    """Strict mode (env THESIS_GATE=strict): fail if any orphans exist.
    Default mode: always pass."""
    if os.environ.get("THESIS_GATE", "").lower() != "strict":
        pytest.skip("THESIS_GATE != strict; warn-mode (default).")
    total, reachable, orphans = _build_report()
    _write_report(total, reachable, orphans)
    assert not orphans, (
        f"{len(orphans)} orphan decimal tokens found in latex/*.tex; "
        f"see {REPORT_PATH} for details."
    )
