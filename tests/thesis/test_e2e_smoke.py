"""
test_e2e_smoke.py
=================
Layer 4 — end-to-end smoke test.

Runs ``python run_pipeline.py --repro-smoke`` in a temp output root and
compares the resulting ``PRODUCTION_METRICS.json`` and
``cs_artefacts_data.pkl`` Sharpe to a committed fixture.

**Skipped** unless both fixtures exist:
  - ``tests/thesis/fixtures/smoke_baseline.json``
  - ``tests/thesis/fixtures/smoke_artefacts.pkl``

Generate them once, locally, with:

    python tests/thesis/make_smoke_fixture.py

The cloud agent that authored this suite **deliberately does NOT run**
``make_smoke_fixture.py`` — fixture provenance must be a clean local
run pinned to the documented requirements.lock environment.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURES = HERE / "fixtures"
BASELINE = FIXTURES / "smoke_baseline.json"
ARTEFACTS = FIXTURES / "smoke_artefacts.pkl"

SHARPE_TOL = 1e-6


def _fixtures_present() -> bool:
    return BASELINE.exists() and ARTEFACTS.exists()


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Invoke the pipeline in --repro-smoke mode under a tempdir output root."""
    if not _fixtures_present():
        pytest.skip(
            "Smoke fixtures missing. Run `python tests/thesis/make_smoke_fixture.py` "
            "locally to generate them."
        )
    out_root = tmp_path_factory.mktemp("smoke_pipeline")
    env = os.environ.copy()
    env["MOMENTUM_OUTPUT_ROOT"] = str(out_root)
    env["PYTHONHASHSEED"] = "0"
    cmd = [sys.executable, "-u", str(REPO / "run_pipeline.py"), "--repro-smoke"]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
        timeout=60 * 60,  # 1 hr cap
    )
    if proc.returncode != 0:
        pytest.fail(
            "run_pipeline.py --repro-smoke failed:\n"
            f"stdout (tail):\n{proc.stdout[-2000:]}\n"
            f"stderr (tail):\n{proc.stderr[-2000:]}"
        )
    return out_root


# ──────────────────────────────────────────────────────────────────────────────


def test_fixtures_present_or_skip():
    if not _fixtures_present():
        pytest.skip(
            "Smoke fixtures missing. Run `python tests/thesis/make_smoke_fixture.py` "
            "locally first; the agent does not generate them."
        )
    assert BASELINE.is_file()
    assert ARTEFACTS.is_file()


def test_pipeline_emits_production_metrics(smoke_run):
    metrics_path = smoke_run / "results" / "PRODUCTION_METRICS.json"
    assert metrics_path.exists(), (
        f"Pipeline did not emit {metrics_path}. The --repro-smoke flag may not "
        f"be wiring MOMENTUM_OUTPUT_ROOT into config.py."
    )


def test_metrics_byte_identical_to_baseline(smoke_run):
    baseline = BASELINE.read_bytes()
    actual = (smoke_run / "results" / "PRODUCTION_METRICS.json").read_bytes()
    assert baseline == actual, (
        "PRODUCTION_METRICS.json drifted from the smoke baseline. Either the "
        "fixture is stale (regenerate it) or the pipeline emitted different "
        "numbers — investigate before refreshing."
    )


def test_artefacts_sharpe_within_tolerance(smoke_run):
    """Compare Sharpe of ``strategies_lo['M2 (XGB)']`` (or first strategy) to
    fixture Sharpe within 1e-6."""
    actual_path = smoke_run / "artefacts" / "cs_artefacts_data.pkl"
    if not actual_path.exists():
        pytest.skip(f"{actual_path} not produced by smoke run")
    with open(actual_path, "rb") as f:
        actual = pickle.load(f)
    with open(ARTEFACTS, "rb") as f:
        baseline = pickle.load(f)

    def _sharpe(art):
        strategies = art.get("strategies_lo") or {}
        if not strategies:
            return None
        # Pick the first strategy whose name contains 'M2' or 'XGB'; else first.
        target = None
        for name, ret in strategies.items():
            if "M2" in name or "XGB" in name.lower():
                target = ret
                break
        if target is None:
            target = next(iter(strategies.values()))
        import numpy as np  # noqa: WPS433
        r = np.asarray(target)
        s = float(r.std())
        return float(r.mean() / s * np.sqrt(12)) if s > 0 else 0.0

    a = _sharpe(actual)
    b = _sharpe(baseline)
    assert a is not None and b is not None, "missing strategies_lo in pickle(s)"
    assert abs(a - b) < SHARPE_TOL, (
        f"Smoke artefact Sharpe drifted: actual={a!r} baseline={b!r}, "
        f"|Δ|={abs(a-b):.3e} > {SHARPE_TOL:.0e}"
    )
