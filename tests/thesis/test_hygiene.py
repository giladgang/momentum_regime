"""
test_hygiene.py
===============
Layer 5 — plumbing-hygiene checks.

Four independent checks, all deterministic:

  1. **Pickle staleness** — every cached pickle in ``artefacts/`` must
     be at least as fresh as its declared parquet inputs.
  2. **Plot-vs-CSV parity** — for a small set of thesis figures,
     verify the plot-generator script reads the CSV that the manifest
     points at (consistency, not value comparison).
  3. **RNG hygiene** — no module-level ``np.random.seed`` /
     ``random.seed`` / ``torch.manual_seed`` in any ``scripts/*.py``,
     except those whitelisted in ``rng_hygiene_whitelist.txt``.
  4. **dtype invariants** — extends ``test_critical_column_dtypes`` to
     cover ``artefacts['train']`` and ``panel_with_regimes`` parquet.

Skipped tests show up explicitly when the underlying file is absent.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SCRIPTS = REPO / "scripts"
WHITELIST = HERE / "rng_hygiene_whitelist.txt"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ──────────────────────────────────────────────────────────────────────────────
# 9.1 Pickle staleness
# ──────────────────────────────────────────────────────────────────────────────


PICKLE_DEPS: dict[str, list[str]] = {
    "artefacts/cs_artefacts_data.pkl": [
        "data/panel.parquet",
        "data/panel_with_regimes.parquet",
        "data/ff_factors.parquet",
    ],
}


@pytest.mark.parametrize("pkl_rel", list(PICKLE_DEPS.keys()))
def test_pickle_not_stale(pkl_rel):
    pkl = REPO / pkl_rel
    if not pkl.exists():
        pytest.skip(f"{pkl_rel} not present")
    pkl_mtime = pkl.stat().st_mtime
    older = []
    for dep_rel in PICKLE_DEPS[pkl_rel]:
        dep = REPO / dep_rel
        if not dep.exists():
            continue
        if dep.stat().st_mtime > pkl_mtime:
            older.append(f"{dep_rel} mtime={dep.stat().st_mtime} > pkl mtime={pkl_mtime}")
    assert not older, (
        f"{pkl_rel} is older than its declared inputs: {older}. "
        f"Re-run the upstream script to refresh the pickle."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9.2 Plot vs CSV parity
# ──────────────────────────────────────────────────────────────────────────────


PLOT_CSV_PAIRS: list[dict] = [
    {
        "plot_script": "scripts/zscore_subregime_analysis.py",
        "expected_csv": "results/thesis/zscore_subregime_summary.csv",
    },
    {
        "plot_script": "scripts/zscore_subregime_xgb_seed_robustness.py",
        "expected_csv": "results/thesis/zscore_subregime_xgb_seed_robustness.csv",
    },
    {
        "plot_script": "scripts/seed_convergence.py",
        "expected_csv": "results/thesis/seed_convergence.csv",
    },
]


@pytest.mark.parametrize("pair", PLOT_CSV_PAIRS, ids=[p["plot_script"] for p in PLOT_CSV_PAIRS])
def test_plot_script_writes_expected_csv(pair):
    """The script must write to (or read) the canonical CSV path."""
    script = REPO / pair["plot_script"]
    if not script.exists():
        pytest.skip(f"{pair['plot_script']} missing")
    text = script.read_text(encoding="utf-8")
    expected = pair["expected_csv"]
    base_name = os.path.basename(expected)  # e.g. "seed_convergence.csv"
    assert base_name in text, (
        f"{pair['plot_script']} does not reference the expected CSV "
        f"'{base_name}' — possible plot/CSV mismatch."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9.3 RNG hygiene — no module-level seeding in any scripts/*.py
# ──────────────────────────────────────────────────────────────────────────────


def _whitelist() -> set[str]:
    if not WHITELIST.exists():
        return set()
    out: set[str] = set()
    for line in WHITELIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = line.split("#", 1)[0].strip()
        if path:
            out.add(path)
    return out


def _module_level_seed_offenders(rel_path: str) -> list[str]:
    full = REPO / rel_path
    if not full.exists():
        return []
    try:
        tree = ast.parse(full.read_text(encoding="utf-8"), filename=rel_path)
    except SyntaxError:
        return []
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            name = _dotted(call.func)
            if name in ("np.random.seed", "numpy.random.seed", "random.seed", "torch.manual_seed"):
                offenders.append(f"{name} @ line {node.lineno}")
        if isinstance(node, (ast.If, ast.With, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    name = _dotted(sub.func)
                    if name in ("np.random.seed", "numpy.random.seed", "random.seed", "torch.manual_seed"):
                        offenders.append(f"{name} @ line {sub.lineno}")
    return offenders


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def test_no_module_level_rng_seed_anywhere():
    """No script under scripts/ may call ``np.random.seed`` etc. at module
    level. Whitelist via ``tests/thesis/rng_hygiene_whitelist.txt`` if a
    specific file is genuinely safe."""
    if not SCRIPTS.exists():
        pytest.skip("scripts/ missing")
    wl = _whitelist()
    bad: dict[str, list[str]] = {}
    for path in sorted(SCRIPTS.glob("*.py")):
        rel = str(path.relative_to(REPO))
        if rel in wl:
            continue
        offenders = _module_level_seed_offenders(rel)
        if offenders:
            bad[rel] = offenders
    assert not bad, (
        "Module-level RNG seeding pollutes import-time state. "
        f"Offenders ({len(bad)}):\n"
        + "\n".join(f"  {k}: {v}" for k, v in bad.items())
        + "\n\nWhitelist via tests/thesis/rng_hygiene_whitelist.txt if intentional."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9.4 dtype invariants
# ──────────────────────────────────────────────────────────────────────────────


def test_artefacts_train_dtypes(artefacts):
    """``art['train']`` should mirror the dtype invariants applied to
    ``art['test']`` in the legacy suite."""
    if "train" not in artefacts:
        pytest.skip("'train' missing from artefacts")
    train = artefacts["train"]
    if "date" in train.columns:
        import pandas as pd
        assert pd.api.types.is_datetime64_any_dtype(train["date"]) or pd.api.types.is_datetime64_dtype(train["date"]), (
            f"train['date'] dtype = {train['date'].dtype}, expected datetime64"
        )
    if "permno" in train.columns:
        import pandas as pd
        assert pd.api.types.is_integer_dtype(train["permno"]) or pd.api.types.is_numeric_dtype(train["permno"]), (
            f"train['permno'] dtype = {train['permno'].dtype}, expected integer-ish"
        )


def test_panel_with_regimes_dtypes(panel_with_regimes):
    """panel_with_regimes parquet: date is datetime, pi_filter (when present) is float."""
    import pandas as pd
    df = panel_with_regimes
    if "date" in df.columns:
        assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
            f"panel_with_regimes['date'] dtype = {df['date'].dtype}, expected datetime"
        )
    if "pi_filter" in df.columns:
        assert pd.api.types.is_float_dtype(df["pi_filter"]), (
            f"panel_with_regimes['pi_filter'] dtype = {df['pi_filter'].dtype}, expected float"
        )


def test_top_results_csv_dtypes():
    """Spot-check a few results/thesis/*.csv for sane dtypes."""
    import pandas as pd
    csvs = [
        "results/thesis/expanding_summary_prod.csv",
        "results/thesis/depth_results.csv",
        "results/thesis/seed_convergence.csv",
    ]
    for rel in csvs:
        p = REPO / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        # Any float column must be float (no string-typed numerics)
        for col in df.columns:
            if df[col].dtype == object:
                # Empty / mixed columns are tolerated; flag only if 100% numeric strings
                sample = df[col].dropna().astype(str).head(5).tolist()
                # If every sample parses as float, dtype should be float — flag
                try:
                    [float(s) for s in sample]
                    pytest.fail(
                        f"{rel}:{col} is dtype object but every value parses as float; "
                        f"sample: {sample}"
                    )
                except ValueError:
                    pass  # genuinely non-numeric; OK
