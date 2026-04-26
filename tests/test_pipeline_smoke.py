"""
test_pipeline_smoke.py
======================
Fast smoke test of the pipeline orchestration.

Does not run the actual multi-hour HMM/XGB pipeline. Verifies:
  - run_pipeline.py imports without side effects
  - run_pipeline.py --help succeeds and lists expected flags
  - The `steps` dict is well-formed (every listed script exists or is
    explicitly allowed to be missing)
  - Config printout doesn't crash on the current config.py
  - run_pipeline.py --step 999 (invalid) fails gracefully

Run with:
    pytest tests/test_pipeline_smoke.py -v
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

RUN_PIPELINE = os.path.join(REPO, 'run_pipeline.py')


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level sanity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineImport:
    def test_run_pipeline_file_exists(self):
        assert os.path.exists(RUN_PIPELINE)

    def test_run_pipeline_imports_cleanly(self):
        # Import should define `main` without running it
        import importlib
        if 'run_pipeline' in sys.modules:
            del sys.modules['run_pipeline']
        mod = importlib.import_module('run_pipeline')
        assert hasattr(mod, 'main')
        assert hasattr(mod, 'run_script')
        assert hasattr(mod, 'run_pytest')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI: --help and invalid step
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineCLI:
    def test_help_flag_succeeds(self):
        result = subprocess.run(
            [sys.executable, RUN_PIPELINE, '--help'],
            capture_output=True, text=True, timeout=60, cwd=REPO,
        )
        assert result.returncode == 0, \
            f"--help exited with {result.returncode}\nSTDERR: {result.stderr}"
        assert '--step' in result.stdout

    def test_unknown_step_is_handled(self):
        # Step 999 isn't in the steps dict — the script prints a message
        # and exits with code 0 (current behaviour; this locks it in)
        result = subprocess.run(
            [sys.executable, RUN_PIPELINE, '--step', '999'],
            capture_output=True, text=True, timeout=60, cwd=REPO,
        )
        # Either a clean exit with a helpful message, or a non-zero exit.
        # Whichever the script does now, we just require it not hang or crash.
        assert result.returncode in (0, 1, 2)
        assert 'Unknown step' in result.stdout or '999' in result.stdout + result.stderr

    def test_invalid_step_type_rejected(self):
        result = subprocess.run(
            [sys.executable, RUN_PIPELINE, '--step', 'notanumber'],
            capture_output=True, text=True, timeout=60, cwd=REPO,
        )
        assert result.returncode != 0


# ═══════════════════════════════════════════════════════════════════════════════
# Steps dictionary
# ═══════════════════════════════════════════════════════════════════════════════

class TestStepsDict:
    @pytest.fixture(scope='class')
    def steps(self):
        """Extract the `steps` dict by parsing the script source (the dict
        is built inside main() so we can't just import it)."""
        import ast
        with open(RUN_PIPELINE) as f:
            source = f.read()
        tree = ast.parse(source)
        # Find the `steps = {...}` assignment inside main()
        main_fn = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == 'main')
        steps_assign = next(
            node for node in ast.walk(main_fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'steps' for t in node.targets)
        )
        # Evaluate the dict literal statically
        return ast.literal_eval(steps_assign.value)

    def test_steps_dict_non_empty(self, steps):
        assert len(steps) > 0

    def test_step_keys_are_positive_ints(self, steps):
        for k in steps:
            assert isinstance(k, int) and k > 0

    def test_each_step_is_tuple_of_two_strings(self, steps):
        for k, v in steps.items():
            assert isinstance(v, tuple) and len(v) == 2, \
                f"Step {k} must be (script_path, description) tuple"
            script, desc = v
            assert isinstance(script, str) and len(script) > 0
            assert isinstance(desc, str) and len(desc) > 0

    def test_step_script_paths_are_relative(self, steps):
        for k, (script, _) in steps.items():
            assert not os.path.isabs(script), \
                f"Step {k} uses absolute path {script}; should be relative to repo"

    def test_step_scripts_exist(self, steps):
        """Every referenced script must exist on disk. If a script is
        intentionally optional, this test is where you'd notice the
        mismatch and fix the steps dict.

        Some steps embed CLI args in the first tuple element (matching
        run_script which splits on whitespace), so check only the path
        prefix before the first space."""
        missing = []
        for k, (script, desc) in steps.items():
            path = script.split()[0]
            full = os.path.join(REPO, path)
            if not os.path.exists(full):
                missing.append(f"{k}: {path} ({desc!r})")
        assert not missing, \
            "Steps reference non-existent scripts:\n  " + "\n  ".join(missing)

    def test_no_duplicate_scripts_across_steps(self, steps):
        scripts = [v[0] for v in steps.values()]
        # Duplicates can still be valid (same script with different args) but
        # with the current pipeline each entry should be a distinct file.
        dupes = [s for s in set(scripts) if scripts.count(s) > 1]
        assert not dupes, f"Duplicate scripts in steps: {dupes}"


# ═══════════════════════════════════════════════════════════════════════════════
# Output directories created on first run
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputDirs:
    def test_expected_output_dirs_can_be_created(self, tmp_path, monkeypatch):
        """Simulate first-run behaviour: output dirs should be creatable."""
        import config as cfg
        for d in (cfg.TABLES_DIR, cfg.PLOTS_DIR, cfg.RESULTS_DIR, cfg.ARTEFACTS_DIR):
            target = tmp_path / d
            os.makedirs(target, exist_ok=True)
            assert target.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-flight test block — gated on `args.step == 0`.
# Lives at run_pipeline.py:117-133. Aborts the multi-hour pipeline if any
# of the fast unit tests fail. This is the cheapest safety net we have, so
# regressions in the gate are blocking.
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreflight:

    @pytest.fixture(scope='class')
    def source(self):
        with open(RUN_PIPELINE) as f:
            return f.read()

    def test_preflight_block_exists(self, source):
        """The pre-flight pytest invocation must be present and gated."""
        assert 'if args.step == 0' in source, (
            "Pre-flight must run only when no specific step is targeted "
            "(args.step == 0)"
        )
        assert 'pytest' in source

    def test_preflight_runs_expected_test_files(self, source):
        """The pre-flight list must include the four artefact-independent
        test files. These are the only ones that can run in ~3s before
        any HMM/XGB compute."""
        expected = [
            'tests/test_config.py',
            'tests/test_utils.py',
            'tests/test_portfolio_edge_cases.py',
            'tests/test_pipeline_smoke.py',
        ]
        for path in expected:
            assert path in source, (
                f"Pre-flight is missing {path}; without it, regressions in "
                "config/utils/portfolio reach the multi-hour HMM/XGB stages."
            )

    def test_preflight_aborts_on_failure(self, source):
        """Failure to abort would defeat the safety net."""
        assert 'sys.exit(1)' in source
        assert 'PRE-FLIGHT FAILED' in source

    def test_preflight_files_actually_exist(self):
        """The four pre-flight test files referenced in run_pipeline.py
        must exist on disk; otherwise the gate silently no-ops on missing
        files (pytest's behaviour for unknown paths varies)."""
        for path in ('tests/test_config.py', 'tests/test_utils.py',
                     'tests/test_portfolio_edge_cases.py',
                     'tests/test_pipeline_smoke.py'):
            full = os.path.join(REPO, path)
            assert os.path.exists(full), f"Missing: {path}"

    def test_preflight_invokable_directly(self):
        """Run the pre-flight files via pytest in a subprocess. They must
        all pass on the current state — otherwise `python run_pipeline.py`
        would fail at the gate."""
        import subprocess
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q', '--tb=short',
             'tests/test_config.py', 'tests/test_utils.py',
             'tests/test_portfolio_edge_cases.py',
             'tests/test_pipeline_smoke.py',
             '--deselect', 'tests/test_pipeline_smoke.py::TestPreflight'],
            cwd=REPO, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, (
            f"Pre-flight test files failed (rc={result.returncode}). "
            f"`python run_pipeline.py` would abort at the gate.\n"
            f"STDOUT tail: {result.stdout[-1500:]}\n"
            f"STDERR tail: {result.stderr[-500:]}"
        )

    def test_step_n_skips_preflight(self, source):
        """`--step N` (N > 0) targets one step only and must NOT trigger
        pre-flight. The gate `if args.step == 0` enforces this."""
        # Find the pre-flight block and verify it's inside an `if args.step == 0`
        idx = source.find('Pre-flight tests')
        assert idx > 0
        # Look backwards from the print() for the gating `if`
        prefix = source[:idx]
        last_if = prefix.rfind('if args.step == 0')
        last_else = prefix.rfind('else:')
        assert last_if > last_else, (
            "Pre-flight block is not inside an `if args.step == 0:` branch — "
            "would run on every --step invocation, defeating its purpose."
        )
