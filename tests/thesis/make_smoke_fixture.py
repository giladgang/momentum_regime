"""
make_smoke_fixture.py
=====================
Run once locally (NOT by the cloud agent) to populate the smoke fixtures
consumed by ``tests/thesis/test_e2e_smoke.py``.

Workflow:

  1. Sets ``MOMENTUM_OUTPUT_ROOT`` to a temp directory.
  2. Invokes ``python run_pipeline.py --repro-smoke``.
  3. Copies the resulting ``PRODUCTION_METRICS.json`` and
     ``cs_artefacts_data.pkl`` to ``tests/thesis/fixtures/``.
  4. Writes ``tests/thesis/fixtures/PROVENANCE.md`` with timestamp,
     git SHA, env hash, and runtime.

Target wall time: < 30 minutes on a 6-core box.

The fixtures are gitignored by default — commit them only after
verifying the run is clean and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURES = HERE / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO),
            text=True, capture_output=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _env_hash() -> str:
    """Hash of pip freeze output as a coarse env identifier."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            text=True, capture_output=True, check=True,
        )
        return hashlib.sha256(out.stdout.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep the smoke output dir for inspection (default: delete after copy)",
    )
    args = ap.parse_args()

    out_root = Path(tempfile.mkdtemp(prefix="momentum_smoke_"))
    env = os.environ.copy()
    env["MOMENTUM_OUTPUT_ROOT"] = str(out_root)
    env["PYTHONHASHSEED"] = "0"

    sha = _git_sha()
    print(f"[make_smoke_fixture] git SHA: {sha}")
    print(f"[make_smoke_fixture] output root: {out_root}")
    print(f"[make_smoke_fixture] running run_pipeline.py --repro-smoke ...")

    t0 = time.time()
    cmd = [sys.executable, "-u", str(REPO / "run_pipeline.py"), "--repro-smoke"]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[make_smoke_fixture] FAILED (exit {proc.returncode}); "
              f"output preserved at {out_root}", file=sys.stderr)
        return proc.returncode

    metrics_src = out_root / "results" / "PRODUCTION_METRICS.json"
    art_src = out_root / "artefacts" / "cs_artefacts_data.pkl"
    if not metrics_src.exists() or not art_src.exists():
        print(f"[make_smoke_fixture] expected outputs not found:\n"
              f"  {metrics_src} (exists={metrics_src.exists()})\n"
              f"  {art_src} (exists={art_src.exists()})\n"
              f"Has --repro-smoke been wired into run_pipeline.py?",
              file=sys.stderr)
        return 1

    shutil.copy2(metrics_src, FIXTURES / "smoke_baseline.json")
    shutil.copy2(art_src, FIXTURES / "smoke_artefacts.pkl")

    provenance = (
        f"# Smoke Fixture Provenance\n\n"
        f"- Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"- git SHA: `{sha}`\n"
        f"- env hash (pip freeze sha256[0:12]): `{_env_hash()}`\n"
        f"- Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n"
        f"- Output root (deleted): {out_root}\n"
        f"- Files copied:\n"
        f"  - `tests/thesis/fixtures/smoke_baseline.json`\n"
        f"  - `tests/thesis/fixtures/smoke_artefacts.pkl`\n"
    )
    (FIXTURES / "PROVENANCE.md").write_text(provenance)
    print(f"[make_smoke_fixture] wrote fixtures/ in {elapsed:.1f}s")

    if not args.keep_output:
        shutil.rmtree(out_root, ignore_errors=True)
        print(f"[make_smoke_fixture] cleaned up {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
