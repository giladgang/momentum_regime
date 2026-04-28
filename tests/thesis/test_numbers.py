"""
test_numbers.py
===============
Layer 1 of the thesis verification suite.

For each entry in ``manifest.yaml``:

  1. (CSV / TEX row check) If ``source`` is non-null, load the CSV (or LaTeX
     table), apply the selector, read ``column``, compare to
     ``published.value`` within ``tolerance``.
  2. (Citation check) For each item in ``cites[]``, verify the cite locator
     matches in the cited file.
  3. (Canonical-consistency check) If ``canonical`` is set (e.g.
     ``EXP.M2_SHARPE_FULL``), assert published.value matches the constant
     in ``tests/_expected.py`` exactly.

A failure on any sub-check fails the parametrized case for that entry.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import pytest

from tests.thesis.conftest import (
    REPO,
    apply_selector,
    find_in_table,
    load_manifest,
    read_tex,
    within_tolerance,
)


MANIFEST = load_manifest()
MANIFEST_IDS = [e["id"] for e in MANIFEST]


# Selection helper for parametrize
def _params() -> list:
    if not MANIFEST:
        return [pytest.param({"id": "manifest_missing"}, id="manifest_missing", marks=pytest.mark.skip(reason="manifest.yaml empty or missing"))]
    return [pytest.param(e, id=e["id"]) for e in MANIFEST]


# ──────────────────────────────────────────────────────────────────────────────
# Source-value extraction
# ──────────────────────────────────────────────────────────────────────────────


def _read_source_value(entry: dict, csv_loader_fn) -> tuple[float | list[float] | None, str]:
    """
    Return (observed_value, debug_str). observed_value is None if the source
    spec is unusable. ``debug_str`` describes what was attempted (for failure
    messages).
    """
    src = entry.get("source")
    if not src:
        return None, "source=null"

    csv_path = src.get("csv")
    if not csv_path:
        return None, "source.csv missing"

    column = src.get("column")
    selector = src.get("selector")
    selector_expr = src.get("selector_expr")

    abs_path = csv_path if os.path.isabs(csv_path) else str(REPO / csv_path)
    if not os.path.exists(abs_path):
        return None, f"file missing: {csv_path}"

    # Tex tables vs CSVs: detect by extension
    if csv_path.endswith(".tex"):
        text = read_tex(csv_path)
        if not selector or "row_substr" not in selector:
            return None, f"tex source needs selector.row_substr: {csv_path}"
        col_idx = column if isinstance(column, int) else None
        if col_idx is None:
            return None, f"tex source needs integer column index: {csv_path}"
        raw = find_in_table(text, selector["row_substr"], col_idx)
        if raw == "":
            return None, f"row '{selector['row_substr']}' not found in {csv_path}"
        try:
            return float(raw), f"tex {csv_path} row='{selector['row_substr']}' col={col_idx}"
        except ValueError:
            return None, f"tex value not numeric: '{raw}' from {csv_path}"

    # CSV path
    df = csv_loader_fn(csv_path)
    filtered = apply_selector(df, selector, selector_expr)
    if column is None:
        return None, f"csv source needs column: {csv_path}"
    if column not in filtered.columns:
        return None, f"column '{column}' not in {csv_path} ({list(filtered.columns)})"
    series = filtered[column]
    if len(series) == 0:
        return None, f"selector matched 0 rows in {csv_path}"
    if len(series) == 1:
        try:
            return float(series.iloc[0]), f"csv {csv_path} 1 row matched"
        except (TypeError, ValueError):
            return None, f"non-numeric value '{series.iloc[0]}' in {csv_path}"
    # Multiple rows (grouped claim): return list
    try:
        return [float(x) for x in series.tolist()], f"csv {csv_path} {len(series)} rows matched"
    except (TypeError, ValueError):
        return None, f"non-numeric values in {csv_path}"


# ──────────────────────────────────────────────────────────────────────────────
# Cite verification
# ──────────────────────────────────────────────────────────────────────────────


def _verify_cite(cite: dict) -> tuple[bool, str]:
    file_path = cite.get("file")
    locator = cite.get("locator", "")
    if not file_path or not locator:
        return False, f"malformed cite: {cite}"
    abs_path = file_path if os.path.isabs(file_path) else str(REPO / file_path)
    if not os.path.exists(abs_path):
        return False, f"cite file missing: {file_path}"
    content = read_tex(file_path)
    if locator.startswith("regex="):
        pattern = locator[len("regex="):]
        try:
            ok = re.search(pattern, content) is not None
        except re.error as e:
            return False, f"bad regex {pattern!r}: {e}"
        return ok, f"regex {pattern!r} -> {'hit' if ok else 'miss'} in {file_path}"
    if locator.startswith("row="):
        # locator format: row=<row_substr>; col=<int>
        m = re.match(r"row=([^;]+);\s*col=(\d+)", locator)
        if not m:
            return False, f"malformed row locator: {locator}"
        row_substr = m.group(1).strip()
        col_idx = int(m.group(2))
        raw = find_in_table(content, row_substr, col_idx)
        return raw != "", f"row='{row_substr}' col={col_idx} -> {'hit' if raw else 'miss'} in {file_path}"
    return False, f"unknown locator scheme: {locator}"


# ──────────────────────────────────────────────────────────────────────────────
# Canonical consistency
# ──────────────────────────────────────────────────────────────────────────────


def _verify_canonical(entry: dict) -> tuple[bool, str]:
    canonical = entry.get("canonical")
    if not canonical:
        return True, "no canonical ref"
    # Expect form "EXP.SOMETHING"
    if not canonical.startswith("EXP."):
        return False, f"unsupported canonical form: {canonical}"
    attr = canonical[len("EXP."):]
    from tests import _expected as EXP
    if not hasattr(EXP, attr):
        return False, f"_expected.py has no attribute {attr}"
    expected_val = getattr(EXP, attr)
    published = entry["published"]["value"]
    if isinstance(expected_val, (list, tuple)):
        if not isinstance(published, list):
            return False, f"manifest value is scalar but EXP.{attr} is list"
        if len(expected_val) != len(published):
            return False, f"length mismatch list canonical: manifest={len(published)} EXP={len(expected_val)}"
        for i, (a, b) in enumerate(zip(published, expected_val)):
            if abs(a - b) > 1e-9:
                return False, f"EXP.{attr}[{i}] = {b}, manifest = {a}"
        return True, "list canonical match"
    try:
        ok = abs(float(published) - float(expected_val)) < 1e-9
    except (TypeError, ValueError):
        return False, f"canonical compare failed: published={published} EXP={expected_val}"
    if not ok:
        return False, f"manifest published.value={published} != EXP.{attr}={expected_val}"
    return True, f"matches EXP.{attr}={expected_val}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("entry", _params())
def test_manifest_entry(entry: dict, csv_loader, tolerance_checker):
    """Run all sub-checks for one manifest entry. Reports the first failure."""
    failures: list[str] = []

    published = entry.get("published", {})
    pub_val = published.get("value")
    tol = published.get("tolerance", {})

    # 1. Source-row check
    src = entry.get("source")
    if src is not None and pub_val is not None:
        observed, debug = _read_source_value(entry, csv_loader)
        if observed is None:
            failures.append(f"[source] could not read: {debug}")
        elif "stars" in tol:
            # Stars are categorical; numeric value is the underlying t-stat.
            # Skip numeric source check; nothing to do here.
            pass
        elif isinstance(pub_val, list):
            if not isinstance(observed, list):
                failures.append(
                    f"[source] manifest value is list but source returned scalar ({observed}); {debug}"
                )
            elif len(observed) != len(pub_val):
                failures.append(
                    f"[source] length mismatch: published={len(pub_val)} observed={len(observed)}; {debug}"
                )
            else:
                for i, (po, pe) in enumerate(zip(observed, pub_val)):
                    ok, delta, mode = tolerance_checker(po, pe, tol)
                    if not ok:
                        failures.append(
                            f"[source][{i}] published={pe} csv={po} |Δ|={delta:.4g} mode={mode}; {debug}"
                        )
        else:
            obs_scalar = observed if not isinstance(observed, list) else (observed[0] if observed else None)
            if obs_scalar is None:
                failures.append(f"[source] no observed value; {debug}")
            else:
                ok, delta, mode = tolerance_checker(obs_scalar, pub_val, tol)
                if not ok:
                    failures.append(
                        f"[source] published={pub_val} csv={obs_scalar} |Δ|={delta:.4g} mode={mode}; {debug}"
                    )

    # 2. Citation checks
    cites = entry.get("cites") or []
    for i, cite in enumerate(cites):
        ok, dbg = _verify_cite(cite)
        if not ok:
            failures.append(f"[cite#{i}] {dbg}")

    # 3. Canonical consistency
    ok, dbg = _verify_canonical(entry)
    if not ok:
        failures.append(f"[canonical] {dbg}")

    if failures:
        msg = f"\nManifest entry {entry['id']} failed {len(failures)} check(s):\n"
        msg += "\n".join(f"  - {f}" for f in failures)
        pytest.fail(msg)


# Sanity — collection itself succeeds even when manifest is empty.
def test_manifest_loadable():
    entries = load_manifest()
    assert isinstance(entries, list), "manifest entries must be a list"
    if entries:
        ids = [e.get("id") for e in entries]
        assert all(isinstance(i, str) and i for i in ids), "every entry needs a string id"
        assert len(set(ids)) == len(ids), f"duplicate entry ids in manifest: {len(ids) - len(set(ids))} dupes"
