"""
tests/thesis/conftest.py
========================
Session fixtures for the thesis verification suite.

Inherits from the parent ``tests/conftest.py`` (artefacts, panel_with_regimes,
ff_factors fixtures) automatically because pytest walks up the directory tree
collecting conftest.py files. We add manifest loading and small helpers that
multiple tests in this folder share.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

try:
    import yaml  # PyYAML
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "tests/thesis/ requires PyYAML; install with `pip install pyyaml`"
    ) from e


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


MANIFEST_PATH = HERE / "manifest.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Manifest loading
# ──────────────────────────────────────────────────────────────────────────────


def load_manifest() -> list[dict]:
    """Read manifest.yaml. Returns [] if missing (so test files import cleanly)."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH) as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("entries", []) if isinstance(data, dict) else (data or [])
    if not isinstance(entries, list):
        raise ValueError(f"manifest.yaml: expected list under 'entries', got {type(entries)}")
    return entries


@pytest.fixture(scope="session")
def manifest() -> list[dict]:
    return load_manifest()


@pytest.fixture(scope="session")
def repo_root_path() -> Path:
    return REPO


# ──────────────────────────────────────────────────────────────────────────────
# CSV cache (session-scoped, keyed by absolute path)
# ──────────────────────────────────────────────────────────────────────────────

_CSV_CACHE: dict[str, pd.DataFrame] = {}


@pytest.fixture(scope="session")
def csv_loader():
    """Returns a function that loads a CSV (relative or absolute path) and caches it."""

    def _load(rel_or_abs: str) -> pd.DataFrame:
        path = rel_or_abs
        if not os.path.isabs(path):
            path = str(REPO / rel_or_abs)
        if path not in _CSV_CACHE:
            _CSV_CACHE[path] = pd.read_csv(path)
        return _CSV_CACHE[path]

    return _load


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table parsing helper
# ──────────────────────────────────────────────────────────────────────────────


def find_in_table(table_text: str, row_substr: str, col_index: int) -> str:
    """
    Mirror of tests/test_thesis_consistency.py::_find_in_table. Locate the
    first row in a tabular environment containing ``row_substr`` and return
    the value at 0-based ``col_index``. Strips LaTeX formatting and handles
    the ``$-$`` minus-sign quirk.
    """
    for line in table_text.splitlines():
        if row_substr in line:
            cols = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            if col_index < len(cols):
                raw = cols[col_index]
                raw = raw.replace("$-$", "-").replace(r"\%", "").replace("%", "")
                raw = raw.replace("$", "").strip()
                # Strip trailing significance stars (*, **, ***) common in t-stat columns.
                raw = raw.rstrip("*").strip()
                return raw
    return ""


@pytest.fixture(scope="session")
def tex_table_finder():
    return find_in_table


# ──────────────────────────────────────────────────────────────────────────────
# Compare-with-tolerance helper
# ──────────────────────────────────────────────────────────────────────────────


def within_tolerance(observed: float, expected: float, tolerance: dict) -> tuple[bool, float, str]:
    """
    Returns (ok, abs_delta, mode_used). ``tolerance`` may have one of:
      - ``abs``: absolute delta
      - ``rel``: relative delta
      - ``stars``: categorical (handled separately by callers)
    """
    delta = abs(float(observed) - float(expected))
    if "abs" in tolerance:
        tol = float(tolerance["abs"])
        return delta <= tol, delta, f"abs<={tol}"
    if "rel" in tolerance:
        tol = float(tolerance["rel"])
        scale = max(abs(float(expected)), 1e-9)
        return delta / scale <= tol, delta, f"rel<={tol}"
    if "stars" in tolerance:
        # Stars handled outside this helper; treat as exact match here.
        return delta < 1e-9, delta, "stars(exact)"
    # Default fallback: 1% relative or 0.01 absolute, whichever larger
    fallback = max(0.01, 0.01 * abs(float(expected)))
    return delta <= fallback, delta, f"default(abs<={fallback:.4g})"


@pytest.fixture(scope="session")
def tolerance_checker():
    return within_tolerance


# ──────────────────────────────────────────────────────────────────────────────
# Selector application
# ──────────────────────────────────────────────────────────────────────────────


def apply_selector(df: pd.DataFrame, selector: dict | None, selector_expr: str | None) -> pd.DataFrame:
    """
    Apply a manifest entry's selector (dict of column=value) or selector_expr
    (string evaluated as a pandas boolean mask in the df namespace) to a df.
    Returns the filtered df.
    """
    if selector_expr:
        # Evaluate against df. Only df is exposed.
        mask = eval(selector_expr, {"__builtins__": {}}, {"df": df, "pd": pd})  # noqa: S307
        return df[mask]
    if not selector:
        return df
    out = df
    for col, val in selector.items():
        if col not in out.columns:
            return out.iloc[0:0]  # empty
        out = out[out[col] == val]
    return out


@pytest.fixture(scope="session")
def selector_applier():
    return apply_selector


# ──────────────────────────────────────────────────────────────────────────────
# Tex content cache
# ──────────────────────────────────────────────────────────────────────────────

_TEX_CACHE: dict[str, str] = {}


def read_tex(rel_path: str) -> str:
    abs_path = str(REPO / rel_path) if not os.path.isabs(rel_path) else rel_path
    if abs_path not in _TEX_CACHE:
        with open(abs_path, encoding="utf-8") as f:
            _TEX_CACHE[abs_path] = f.read()
    return _TEX_CACHE[abs_path]


@pytest.fixture(scope="session")
def tex_reader():
    return read_tex


# ──────────────────────────────────────────────────────────────────────────────
# _expected.py canonical accessor
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def expected_module():
    """The shared canonical-constants module used by the legacy suite."""
    from tests import _expected as EXP
    return EXP
