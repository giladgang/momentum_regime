"""
test_methodology.py
===================
Layer 2 of the thesis verification suite.

Static-source audits, mirroring the pattern of
``tests/test_cross_sectional_lookahead.py``. We inspect the source of
six previously-unaudited thesis-cited scripts to catch:

  - i.i.d. bootstrap on time-series without justification
  - per-date z-scoring leaking across dates
  - hard-coded ``random_state=None`` or unseeded ``np.random.seed()``
  - ``<=`` instead of ``<`` on train cutoffs
  - module-level RNG seeding (pollutes import-time state)

The six audited scripts (and ONLY these — see §12 of the design spec):

  1. scripts/bootstrap_analysis.py
  2. scripts/zscore_subregime_analysis.py
  3. scripts/zscore_subregime_xgb_seed_robustness.py
  4. scripts/historical_oos_posthoc.py
  5. scripts/seed_convergence.py
  6. scripts/run_rolling_experiment.py

Scripts EXPLICITLY NOT audited here (already covered elsewhere or out
of scope per spec §2 / §12):

  - scripts/cross_sectional_model.py
  - scripts/expanding_window_backtest.py
  - scripts/expanding_window_backtest_parallel.py
  - scripts/historical_oos_production.py
  - scripts/xgb_cv.py
  - scripts/hmm_cv.py
  - scripts/leg_betas_by_regime.py
  - scripts/apply_shumway_delisting.py
  - scripts/apply_shumway_intl.py
  - scripts/audit_compustat_delisting.py
  - scripts/cross_sectional_intl.py
  - scripts/hmm_intl.py
  - scripts/import_intl_stocks.py
  - scripts/fundamentals_test.py
  - scripts/random_forest_test.py

Run:

    pytest tests/thesis/test_methodology.py -v
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────


def _read(rel_path: str) -> str:
    p = REPO / rel_path
    if not p.exists():
        pytest.skip(f"{rel_path} missing")
    return p.read_text(encoding="utf-8")


def _parse(rel_path: str) -> ast.Module:
    return ast.parse(_read(rel_path), filename=rel_path)


def _module_level_seed_calls(tree: ast.Module) -> list[str]:
    """
    Find ``np.random.seed(...)``, ``random.seed(...)``, ``torch.manual_seed(...)``
    invoked at module level (outside function/class bodies). Returns a list of
    short string descriptors (e.g. ``"np.random.seed @ line 23"``).
    """
    offenders: list[str] = []
    for node in tree.body:
        # Top-level Expr with Call
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            name = _call_dotted_name(call.func)
            if name in ("np.random.seed", "numpy.random.seed", "random.seed", "torch.manual_seed"):
                offenders.append(f"{name} @ line {node.lineno}")
        # Top-level If/With blocks could also contain seed calls — recurse
        # only into top-level if-blocks (not function defs / class defs).
        if isinstance(node, (ast.If, ast.With, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    name = _call_dotted_name(sub.func)
                    if name in ("np.random.seed", "numpy.random.seed", "random.seed", "torch.manual_seed"):
                        offenders.append(f"{name} @ line {sub.lineno}")
    return offenders


def _call_dotted_name(node: ast.AST) -> str:
    """Reconstruct a dotted attribute chain into a string. Returns '' if not a Name/Attribute chain."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _all_calls(tree: ast.Module):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            yield n


# ──────────────────────────────────────────────────────────────────────────────
# Cross-cutting: no module-level seeding in any audited script
# ──────────────────────────────────────────────────────────────────────────────


AUDITED_SCRIPTS = [
    "scripts/bootstrap_analysis.py",
    "scripts/zscore_subregime_analysis.py",
    "scripts/zscore_subregime_xgb_seed_robustness.py",
    "scripts/historical_oos_posthoc.py",
    "scripts/seed_convergence.py",
    "scripts/run_rolling_experiment.py",
]


@pytest.mark.parametrize("rel_path", AUDITED_SCRIPTS)
def test_no_module_level_rng_seed(rel_path):
    """No ``np.random.seed(...)`` / ``random.seed(...)`` at module load time.
    These pollute import-time state and silently affect any other script
    that imports the offender. RNG calls must live inside functions or
    ``if __name__ == '__main__':`` blocks."""
    tree = _parse(rel_path)
    offenders = _module_level_seed_calls(tree)
    assert not offenders, (
        f"{rel_path} has module-level RNG seeding (pollutes import time): {offenders}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.1 bootstrap_analysis.py
# ──────────────────────────────────────────────────────────────────────────────


class TestBootstrapAnalysis:
    PATH = "scripts/bootstrap_analysis.py"

    def test_seed_is_explicit(self):
        src = _read(self.PATH)
        # Reject `np.random.seed()` (no arg) and `np.random.seed(None)`
        bad = re.search(r"np\.random\.seed\s*\(\s*(None\s*)?\)", src)
        assert bad is None, (
            f"{self.PATH} calls np.random.seed() with no/None arg — non-deterministic."
        )
        # At least one of: --seed arg, SEED constant, np.random.seed(<int>),
        # np.random.RandomState(<int>), np.random.default_rng(<int>).
        has_seed = (
            re.search(r"--seed", src)
            or re.search(r"\bSEED[\w]*\s*=\s*\d+", src)
            or re.search(r"np\.random\.seed\(\s*\d+\s*\)", src)
            or re.search(r"np\.random\.RandomState\(\s*\d+\s*\)", src)
            or re.search(r"np\.random\.default_rng\(\s*\d+\s*\)", src)
            or re.search(r"\brng\s*=\s*np\.random\.default_rng\(\s*\d+\s*\)", src)
        )
        assert has_seed, (
            f"{self.PATH} has no explicit integer seed source (--seed flag, SEED const, "
            f"or np.random.RandomState(<int>) / default_rng(<int>))."
        )

    def test_block_bootstrap_or_justification(self):
        src = _read(self.PATH)
        block_tokens = ("block_size", "block_bootstrap", "circular_block",
                        "stationary_bootstrap", "block_len", "block_bootstrap_indices",
                        "BLOCK_LEN")
        has_block = any(tok in src for tok in block_tokens)
        has_justification = re.search(r"#\s*i\.i\.d\..*acceptable", src) is not None
        assert has_block or has_justification, (
            f"{self.PATH} uses i.i.d. resampling on time-series data without an "
            f"explicit block-bootstrap mechanism or justifying comment."
        )

    def test_no_date_index_shuffle(self):
        src = _read(self.PATH)
        # df.sample(frac=1) on a date-indexed frame would shuffle dates.
        bad1 = re.search(r"\.sample\(\s*frac\s*=\s*1", src)
        bad2 = re.search(r"np\.random\.shuffle\([^)]*\.index", src)
        assert bad1 is None, f"{self.PATH} uses .sample(frac=1) — likely shuffles dates"
        assert bad2 is None, f"{self.PATH} uses np.random.shuffle on a frame index"


# ──────────────────────────────────────────────────────────────────────────────
# 6.2 zscore_subregime_analysis.py & zscore_subregime_xgb_seed_robustness.py
# ──────────────────────────────────────────────────────────────────────────────


class _ZScoreCommonMixin:
    PATH: str  # set by subclass

    def test_per_date_zscore_or_no_panel_zscore(self):
        """If the script computes a z-score, it must be inside groupby('date')
        (per-date cross-sectional) or use train-only stats. Reject any
        ``(x - x.mean()) / x.std()`` pattern that operates on the full panel
        without a date groupby in surrounding context."""
        src = _read(self.PATH)
        # Heuristic: count z-score idioms and groupby('date') idioms
        # If z-score idiom present, require a groupby('date') somewhere in the file
        zscore_pattern = re.search(r"\([^()\n]+\.mean\(\)\)\s*/\s*[^()\n]+\.std\(\)", src)
        groupby_date = "groupby('date')" in src or 'groupby("date")' in src
        if zscore_pattern is not None:
            assert groupby_date, (
                f"{self.PATH} computes a (x - x.mean()) / x.std() z-score but never "
                f"groups by date — this would pool across dates and leak."
            )

    def test_xgb_random_state_explicit(self):
        """Every XGB instantiation must pass random_state=<not None>."""
        tree = _parse(self.PATH)
        offenders: list[str] = []
        for call in _all_calls(tree):
            name = _call_dotted_name(call.func)
            if name not in ("XGBRegressor", "XGBClassifier", "RandomForestRegressor"):
                # Also accept fully qualified xgboost.XGBRegressor etc.
                if not name.endswith(".XGBRegressor") and not name.endswith(".XGBClassifier") and not name.endswith(".RandomForestRegressor"):
                    continue
            kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            if "random_state" not in kwargs:
                offenders.append(f"{name} @ line {call.lineno} missing random_state=")
                continue
            v = kwargs["random_state"]
            if isinstance(v, ast.Constant) and v.value is None:
                offenders.append(f"{name} @ line {call.lineno} has random_state=None")
        assert not offenders, f"{self.PATH}: {offenders}"


class TestZscoreSubregimeAnalysis(_ZScoreCommonMixin):
    PATH = "scripts/zscore_subregime_analysis.py"

    def test_kmeans_random_state_set(self):
        """KMeans calls must use a non-None random_state."""
        tree = _parse(self.PATH)
        offenders: list[str] = []
        for call in _all_calls(tree):
            name = _call_dotted_name(call.func)
            if name != "KMeans" and not name.endswith(".KMeans"):
                continue
            kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            v = kwargs.get("random_state")
            if v is None or (isinstance(v, ast.Constant) and v.value is None):
                offenders.append(f"KMeans @ line {call.lineno} has no random_state / None")
        assert not offenders, f"{self.PATH}: {offenders}"

    def test_seed_in_filename_or_sidecar(self):
        """Stochastic outputs should log the seed in the filename or via a
        sidecar — reproducibility convention from CLAUDE.md."""
        src = _read(self.PATH)
        # Either filename embeds seed OR a sidecar CSV mentions 'seed'
        has_seed_const = re.search(r"\bRNG_SEED\s*=\s*\d+", src) is not None
        writes_seed_sidecar = re.search(r"to_csv\([^)]*robustness", src) is not None or "seed" in src.lower()
        assert has_seed_const, f"{self.PATH} does not pin a stochastic seed (RNG_SEED const)."
        assert writes_seed_sidecar, f"{self.PATH} does not write any seed-tagged output."


class TestZscoreSubregimeXgbSeedRobustness(_ZScoreCommonMixin):
    PATH = "scripts/zscore_subregime_xgb_seed_robustness.py"

    def test_robust_seeds_disjoint_from_production(self):
        """The robustness ensemble must use seeds disjoint from the production
        (1..50) set, otherwise it isn't actually testing seed sensitivity."""
        src = _read(self.PATH)
        # Look for ROBUST_SEEDS = list(range(101, 151)) or similar
        m = re.search(r"ROBUST_SEEDS\s*=\s*list\(range\(\s*(\d+)\s*,\s*(\d+)\s*\)\)", src)
        assert m, f"{self.PATH} must define ROBUST_SEEDS = list(range(...))"
        start = int(m.group(1))
        end = int(m.group(2))
        # Production seeds are 1..50, so robust must start >= 51
        assert start >= 51, (
            f"{self.PATH} ROBUST_SEEDS overlaps production (production=1..50, "
            f"robust={start}..{end-1})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6.3 historical_oos_posthoc.py
# ──────────────────────────────────────────────────────────────────────────────


class TestHistoricalOOSPosthoc:
    PATH = "scripts/historical_oos_posthoc.py"

    def test_no_loose_le_on_dates(self):
        """No ``date <= cutoff`` patterns on training masks."""
        src = _read(self.PATH)
        # Allow `<= end` for closed sub-period slicing, but flag any train_mask /
        # train= mention with <= against a year/cutoff variable.
        bad = re.search(r"train[^=\n]*<=\s*\b(\d{4}-\d{2}-\d{2}|TRAIN_END)", src)
        assert bad is None, (
            f"{self.PATH} uses `train ... <= cutoff` (potential lookahead). "
            f"Use strict `<` instead."
        )

    def test_feature_lag_shift1(self):
        """Momentum / log-return features must be built from .shift(1) of the
        per-permno series, never on contemporaneous values."""
        src = _read(self.PATH)
        assert "shift(1)" in src, (
            f"{self.PATH} should use shift(1) for momentum features"
        )

    def test_target_uses_shift_minus_1(self):
        """Forward target must be shift(-1) of returns."""
        src = _read(self.PATH)
        assert "shift(-1)" in src, (
            f"{self.PATH} should use shift(-1) for forward target"
        )

    def test_subperiod_slicing_uses_inclusive_window(self):
        """Sub-period decomposition uses (idx >= start) & (idx <= end). This is
        an inclusive window for sub-period reporting (NOT a train cutoff), so
        the `<=` here is correct. Sanity-check the construction is what we
        expect, not a bug."""
        src = _read(self.PATH)
        assert re.search(r"\(idx\s*>=\s*start\)\s*&\s*\(idx\s*<=\s*end\)", src), (
            f"{self.PATH} subperiod_slice should use closed [start, end] window"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6.4 seed_convergence.py
# ──────────────────────────────────────────────────────────────────────────────


class TestSeedConvergence:
    PATH = "scripts/seed_convergence.py"

    def test_seeds_deterministic_enumeration(self):
        """SEED_OFFSET/N_MAX or seeds = list(range(...)) — never sampled."""
        src = _read(self.PATH)
        has_deterministic = (
            re.search(r"\bSEED_OFFSET\s*=\s*\d+", src)
            or re.search(r"\bseeds\s*=\s*list\(range\(", src)
            or re.search(r"\bSEEDS\s*=\s*list\(range\(", src)
        )
        assert has_deterministic, (
            f"{self.PATH} must enumerate seeds deterministically (range/literal), "
            f"not stochastically."
        )

        bad = re.search(r"seeds\s*=\s*np\.random\.choice", src)
        assert bad is None, (
            f"{self.PATH} samples seeds stochastically — non-reproducible."
        )

    def test_subset_rng_is_seeded(self):
        """The subset-of-seeds RNG must be seeded (default_rng(<int>) or
        RandomState(<int>))."""
        src = _read(self.PATH)
        assert (
            re.search(r"np\.random\.default_rng\(\s*\d+\s*\)", src)
            or re.search(r"np\.random\.RandomState\(\s*\d+\s*\)", src)
        ), f"{self.PATH} subset RNG is not explicitly seeded."

    def test_seed_logged_in_output(self):
        """Output CSV / pickle filename references seed enumeration so the
        run is auditable."""
        src = _read(self.PATH)
        # Either to_csv mentions a seed-bearing filename or k-column is logged
        has_k_column = "'k'" in src or '"k"' in src
        assert has_k_column, (
            f"{self.PATH} should log subset size 'k' (seed enumeration unit) "
            f"in its CSV output."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6.5 run_rolling_experiment.py
# ──────────────────────────────────────────────────────────────────────────────


class TestRunRollingExperiment:
    PATH = "scripts/run_rolling_experiment.py"

    def test_no_random_train_test_split(self):
        """Time-series train/test splits must be by date, not random."""
        src = _read(self.PATH)
        bad = re.search(r"train_test_split\([^)]*shuffle\s*=\s*True", src)
        assert bad is None, (
            f"{self.PATH} uses train_test_split(shuffle=True) — random split on "
            f"time-series."
        )
        # Generic train_test_split is also suspect on time-series; flag if present
        generic = re.search(r"\btrain_test_split\(", src)
        assert generic is None, (
            f"{self.PATH} imports train_test_split — time-ordered splits should "
            f"be done by date, not by sklearn's random splitter."
        )

    def test_restores_production_data(self):
        """The experiment must restore production data after running so the
        next pipeline rerun isn't poisoned."""
        src = _read(self.PATH)
        assert "Restoring production data" in src or "PRODUCTION_BACKUPS" in src, (
            f"{self.PATH} must restore production data after the experiment."
        )

    def test_validates_after_restore(self):
        """The experiment must call a validation step after restoring."""
        src = _read(self.PATH)
        assert "validate_production" in src or "Validating production" in src, (
            f"{self.PATH} must run validate_production.py after restoring."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Sanity: every audited script exists
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rel_path", AUDITED_SCRIPTS)
def test_audited_script_exists(rel_path):
    assert (REPO / rel_path).exists(), f"{rel_path} not found"
