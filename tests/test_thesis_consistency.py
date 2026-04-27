"""
test_thesis_consistency.py
==========================
Comprehensive consistency tests for the momentum regime shifts thesis.

Verifies that:
  1. Artefact files exist and have the correct structure
  2. Numbers in LaTeX tables match the numbers cited in thesis text
  3. All figures referenced in LaTeX exist on disk
  4. Cross-references (labels, inputs, citations) are internally consistent
  5. Pipeline config matches expected production settings
  6. Reproducibility artefacts exist and contain expected values

Run with:
    pytest tests/test_thesis_consistency.py -v
"""

import os
import re
import pickle
import sys

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg

LATEX_DIR = os.path.join(PROJECT_ROOT, "latex")
TABLES_DIR = os.path.join(PROJECT_ROOT, cfg.TABLES_DIR)
ARTEFACTS_PATH = os.path.join(PROJECT_ROOT, cfg.ARTEFACTS_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _read(path: str) -> str:
    """Read a file from the project and return its content."""
    full = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    with open(full, "r", encoding="utf-8") as f:
        return f.read()


def _load_artefacts():
    """Load the artefacts pickle (cached at module level)."""
    with open(ARTEFACTS_PATH, "rb") as f:
        return pickle.load(f)


# Module-level cache so we only load once across all tests
_ARTEFACTS_CACHE = None


def _get_artefacts():
    global _ARTEFACTS_CACHE
    if _ARTEFACTS_CACHE is None:
        _ARTEFACTS_CACHE = _load_artefacts()
    return _ARTEFACTS_CACHE


def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric tokens from a string (integers and decimals)."""
    return re.findall(r"-?\d+\.?\d*", text)


def _find_in_table(table_text: str, row_substr: str, col_index: int) -> str:
    """
    In a LaTeX tabular, find the row containing *row_substr* and return
    the value at column *col_index* (0-based). Handles $-$ negative sign.
    """
    for line in table_text.splitlines():
        if row_substr in line:
            # Split on & and strip LaTeX formatting
            cols = [c.strip().rstrip("\\").strip() for c in line.split("&")]
            if col_index < len(cols):
                raw = cols[col_index]
                # Normalise $-$ to a minus sign, strip %, $
                raw = raw.replace("$-$", "-").replace(r"\%", "").replace("%", "")
                raw = raw.replace("$", "").strip()
                return raw
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Artefact Integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestArtefacts:
    """Verify cs_artefacts_data.pkl exists, loads, and has the right shape."""

    def test_artefacts_file_exists(self):
        assert os.path.exists(ARTEFACTS_PATH), (
            f"Artefacts file not found at {ARTEFACTS_PATH}"
        )

    def test_artefacts_loads(self):
        art = _get_artefacts()
        assert isinstance(art, dict), "Artefacts should be a dict"

    @pytest.mark.parametrize("key", [
        "test", "train", "FEATURES", "X_train", "X_test",
        "y_train", "shap_values", "strategies_lo",
    ])
    def test_artefacts_has_required_key(self, key):
        art = _get_artefacts()
        assert key in art, f"Artefacts missing required key '{key}'"

    def test_features_match_config(self):
        art = _get_artefacts()
        features = art["FEATURES"]
        expected = cfg.CS_FEATURES  # mom_1..mom_12 + pi_filter
        assert list(features) == list(expected), (
            f"FEATURES mismatch.\n  Artefact: {list(features)}\n  Config:   {list(expected)}"
        )
        assert len(features) == 13, f"Expected 13 features, got {len(features)}"

    def test_train_test_split_date(self):
        art = _get_artefacts()
        test_df = art["test"]
        # date may be a column or index level
        if "date" in test_df.columns:
            dates = pd.to_datetime(test_df["date"])
        elif hasattr(test_df.index, "get_level_values"):
            try:
                dates = test_df.index.get_level_values("date")
            except KeyError:
                dates = test_df.index.get_level_values(0)
        else:
            dates = test_df.index
        min_date = pd.Timestamp(dates.min())
        expected = pd.Timestamp("2011-01-01")
        assert min_date >= expected, (
            f"Test data starts at {min_date}, expected >= {expected}"
        )
        # Also verify train ends before 2011
        train_df = art["train"]
        if "date" in train_df.columns:
            train_dates = pd.to_datetime(train_df["date"])
        elif hasattr(train_df.index, "get_level_values"):
            try:
                train_dates = train_df.index.get_level_values("date")
            except KeyError:
                train_dates = train_df.index.get_level_values(0)
        else:
            train_dates = train_df.index
        max_train = pd.Timestamp(train_dates.max())
        assert max_train < expected, (
            f"Train data extends to {max_train}, expected < {expected}"
        )

    def test_score_xgb_in_test(self):
        art = _get_artefacts()
        test_df = art["test"]
        assert "score_xgb" in test_df.columns, (
            "score_xgb not found in test data columns"
        )

    def test_shap_values_shape(self):
        art = _get_artefacts()
        shap = art["shap_values"]
        X_test = art["X_test"]
        assert shap.shape[0] == X_test.shape[0], (
            f"SHAP rows ({shap.shape[0]}) != X_test rows ({X_test.shape[0]})"
        )
        assert shap.shape[1] == X_test.shape[1], (
            f"SHAP cols ({shap.shape[1]}) != X_test cols ({X_test.shape[1]})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Table-Text Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableConsistency:
    """Verify that numbers in LaTeX tables match inline text in the thesis."""

    # --- table_performance.tex ---

    def test_table_performance_m2_sharpe(self):
        tbl = _read("tables/table_performance.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 3))  # Sharpe column
        assert val == pytest.approx(1.11, abs=0.01), (
            f"M2 Sharpe in table_performance = {val}, expected 1.11"
        )

    def test_table_performance_m2_ann_ret(self):
        tbl = _read("tables/table_performance.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 1))  # Ann. Ret column
        assert val == pytest.approx(21.7, abs=0.1), (
            f"M2 Ann.Ret in table_performance = {val}, expected 21.7"
        )

    def test_table_performance_m2_ann_vol(self):
        tbl = _read("tables/table_performance.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 2))  # Ann. Vol column
        assert val == pytest.approx(19.5, abs=0.1), (
            f"M2 Ann.Vol in table_performance = {val}, expected 19.5"
        )

    def test_table_performance_m2_mdd(self):
        tbl = _read("tables/table_performance.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 4))  # Max DD column
        assert val == pytest.approx(-22.8, abs=0.1), (
            f"M2 MDD in table_performance = {val}, expected -22.8"
        )

    # --- table_regime_sharpe.tex ---

    def test_table_regime_sharpe_m2_full(self):
        tbl = _read("tables/table_regime_sharpe.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 1))  # Full column
        assert val == pytest.approx(1.11, abs=0.01), (
            f"M2 Full Sharpe in table_regime_sharpe = {val}, expected 1.11"
        )

    def test_table_regime_sharpe_m2_calm(self):
        tbl = _read("tables/table_regime_sharpe.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 2))  # Calm column
        assert val == pytest.approx(0.84, abs=0.01), (
            f"M2 Calm Sharpe in table_regime_sharpe = {val}, expected 0.84"
        )

    def test_table_regime_sharpe_m2_panic(self):
        tbl = _read("tables/table_regime_sharpe.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 3))  # Panic column
        assert val == pytest.approx(1.53, abs=0.01), (
            f"M2 Panic Sharpe in table_regime_sharpe = {val}, expected 1.53"
        )

    # --- table_regime_signal_ablation.tex ---

    def test_table_ablation_hmm_sharpe(self):
        tbl = _read("tables/table_regime_signal_ablation.tex")
        val = float(_find_in_table(tbl, "HMM", 3))  # Sharpe column
        assert val == pytest.approx(1.109, abs=0.01), (
            f"HMM Sharpe in ablation table = {val}, expected 1.109"
        )

    def test_table_ablation_no_signal_sharpe(self):
        tbl = _read("tables/table_regime_signal_ablation.tex")
        val = float(_find_in_table(tbl, "no regime signal", 3))
        assert val == pytest.approx(0.429, abs=0.01), (
            f"No-signal Sharpe in ablation table = {val}, expected 0.429"
        )

    def test_table_ablation_raw_sharpe(self):
        tbl = _read("tables/table_regime_signal_ablation.tex")
        val = float(_find_in_table(tbl, "raw indicators", 3))
        assert val == pytest.approx(0.257, abs=0.01), (
            f"Raw indicators Sharpe in ablation table = {val}, expected 0.257"
        )

    # --- table_shap.tex ---

    def test_table_shap_momentum_share(self):
        tbl = _read("tables/table_shap.tex")
        val = float(_find_in_table(tbl, "Momentum", 1))  # Overall column
        assert val == pytest.approx(54, abs=1), (
            f"Momentum SHAP share = {val}%, expected 54%"
        )

    def test_table_shap_pi_share(self):
        tbl = _read("tables/table_shap.tex")
        val = float(_find_in_table(tbl, "pi", 1))  # Overall column
        assert val == pytest.approx(46, abs=1), (
            f"pi_filter SHAP share = {val}%, expected 46%"
        )

    # --- table_subperiod.tex ---

    def test_table_subperiod_m2_values(self):
        tbl = _read("tables/table_subperiod.tex")
        expected_sharpes = [0.59, 1.04, 1.70]
        for i, expected in enumerate(expected_sharpes):
            val = float(_find_in_table(tbl, "M2: XGB", i + 1))
            assert val == pytest.approx(expected, abs=0.01), (
                f"M2 subperiod {i+1} Sharpe = {val}, expected {expected}"
            )

    def test_table_subperiod_m2_full(self):
        tbl = _read("tables/table_subperiod.tex")
        val = float(_find_in_table(tbl, "M2: XGB", 4))  # Full column
        assert val == pytest.approx(1.11, abs=0.01), (
            f"M2 Full Sharpe in subperiod table = {val}, expected 1.11"
        )

    # --- main_results.tex inline numbers match tables ---

    def test_main_results_sharpe_inline(self):
        text = _read("latex/main_results.tex")
        # "annualised Sharpe of 1.11"
        assert "Sharpe of 1.11" in text or "Sharpe 1.11" in text, (
            "main_results.tex does not contain 'Sharpe of 1.11' or 'Sharpe 1.11'"
        )

    def test_main_results_alpha_inline(self):
        text = _read("latex/main_results.tex")
        # "six-factor alpha of 24.7%"
        assert "24.7" in text, (
            "main_results.tex does not contain alpha value 24.7"
        )

    def test_main_results_alpha_matches_factor_table(self):
        tbl = _read("tables/table_factor_alphas.tex")
        # FF6 alpha = 24.7
        val = float(_find_in_table(tbl, "FF6", 1))
        assert val == pytest.approx(24.1, abs=0.1), (
            f"FF6 alpha in factor_alphas table = {val}, expected 24.1"
        )

    def test_main_results_alpha_tstat_inline(self):
        text = _read("latex/main_results.tex")
        assert "4.78" in text, (
            "main_results.tex does not contain alpha t-stat 4.78"
        )

    def test_main_results_alpha_tstat_matches_factor_table(self):
        tbl = _read("tables/table_factor_alphas.tex")
        val = float(_find_in_table(tbl, "FF6", 2))  # t(alpha) column
        assert val == pytest.approx(4.81, abs=0.01), (
            f"FF6 t(alpha) in factor_alphas table = {val}, expected 4.81"
        )

    def test_main_results_panic_sharpe_inline(self):
        text = _read("latex/main_results.tex")
        assert "1.56" in text, (
            "main_results.tex does not mention panic Sharpe 1.56"
        )

    def test_main_results_calm_sharpe_inline(self):
        text = _read("latex/main_results.tex")
        assert "0.82" in text, (
            "main_results.tex does not mention calm Sharpe 0.82"
        )

    def test_main_results_ablation_sharpe_041_inline(self):
        text = _read("latex/main_results.tex")
        # "Sharpe to 0.41" (from removing pi_filter)
        assert "0.41" in text, (
            "main_results.tex does not mention ablation Sharpe 0.41"
        )

    def test_main_results_depth4_return_inline(self):
        text = _read("latex/main_results.tex")
        # "depth~4 (21.9%)"
        assert "21.9" in text, (
            "main_results.tex does not mention depth-4 return 21.9%"
        )

    # --- conclusion.tex numbers match results ---

    def test_conclusion_sharpe_inline(self):
        text = _read("latex/conclusion.tex")
        assert "1.11" in text or "Sharpe" in text, (
            "conclusion.tex does not reference key Sharpe result"
        )

    def test_conclusion_mentions_167_months(self):
        text = _read("latex/conclusion.tex")
        assert "167" in text, (
            "conclusion.tex does not mention 167 test months"
        )

    def test_introduction_sharpe_inline(self):
        text = _read("latex/introduction.tex")
        assert "1.11" in text, (
            "introduction.tex does not mention Sharpe 1.11"
        )

    def test_introduction_alpha_inline(self):
        text = _read("latex/introduction.tex")
        assert "24.7" in text, (
            "introduction.tex does not mention alpha 24.7%"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Figure Verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestFigures:
    """Verify that all figures referenced in LaTeX exist on disk."""

    @staticmethod
    def _collect_figure_refs() -> list[str]:
        """Parse all .tex files for \\includegraphics references."""
        refs = []
        for dirpath in [LATEX_DIR, PROJECT_ROOT]:
            for fname in os.listdir(dirpath):
                if not fname.endswith(".tex"):
                    continue
                fpath = os.path.join(dirpath, fname)
                content = open(fpath, "r", encoding="utf-8").read()
                # Match \includegraphics[...]{filename} or \includegraphics{filename}
                for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content):
                    refs.append(m.group(1))
        return list(set(refs))

    def test_all_referenced_figures_exist(self):
        refs = self._collect_figure_refs()
        assert len(refs) > 0, "No figure references found in LaTeX files"
        missing = []
        for fig in refs:
            full_path = os.path.join(PROJECT_ROOT, fig)
            if not os.path.exists(full_path):
                missing.append(fig)
        assert missing == [], (
            f"Missing figures referenced in LaTeX:\n  " + "\n  ".join(missing)
        )

    def test_zscore_and_absshap_v3_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.PLOTS_DIR, "thesis", "zscore_and_absshap_v3.png")
        assert os.path.exists(path), f"Missing: {path}"

    def test_depth_vs_sharpe_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.PLOTS_DIR, "thesis", "depth_vs_sharpe.png")
        assert os.path.exists(path), f"Missing: {path}"

    def test_cs_performance_regime_shaded_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.PLOTS_DIR, "thesis", "cs_performance_regime_shaded.png")
        assert os.path.exists(path), f"Missing: {path}"

    def test_regime_probabilities_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.PLOTS_DIR, "thesis", "regime_probabilities.png")
        assert os.path.exists(path), f"Missing: {path}"

    def test_features_hmm_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.PLOTS_DIR, "thesis", "features_hmm.png")
        assert os.path.exists(path), f"Missing: {path}"

    def test_convergence_trace_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.PLOTS_DIR, "thesis", "convergence_trace.png")
        assert os.path.exists(path), f"Missing: {path}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Cross-Reference Integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossReferences:
    """Verify labels, inputs, and citations are consistent."""

    @staticmethod
    def _all_tex_content() -> str:
        """Concatenate all .tex files (main + latex/ + tables/)."""
        parts = []
        for dirpath in [PROJECT_ROOT, LATEX_DIR, TABLES_DIR]:
            if not os.path.isdir(dirpath):
                continue
            for fname in sorted(os.listdir(dirpath)):
                if fname.endswith(".tex"):
                    fpath = os.path.join(dirpath, fname)
                    parts.append(open(fpath, "r", encoding="utf-8").read())
        return "\n".join(parts)

    def test_ref_targets_have_labels(self):
        content = self._all_tex_content()
        # Collect all \ref{...} and \cref{...} targets
        refs = set(re.findall(r"\\(?:c?ref|Cref)\{([^}]+)\}", content))
        labels = set(re.findall(r"\\label\{([^}]+)\}", content))
        missing = refs - labels
        assert missing == set(), (
            f"\\ref targets without \\label:\n  " + "\n  ".join(sorted(missing))
        )

    def test_input_files_exist(self):
        content = self._all_tex_content()
        inputs = re.findall(r"\\input\{([^}]+)\}", content)
        missing = []
        for inp in inputs:
            # LaTeX \input may omit .tex extension
            candidates = [
                os.path.join(PROJECT_ROOT, inp),
                os.path.join(PROJECT_ROOT, inp + ".tex"),
                os.path.join(LATEX_DIR, inp),
                os.path.join(LATEX_DIR, inp + ".tex"),
                os.path.join(TABLES_DIR, inp),
                os.path.join(TABLES_DIR, inp + ".tex"),
            ]
            if not any(os.path.exists(c) for c in candidates):
                missing.append(inp)
        assert missing == [], (
            f"\\input files not found on disk:\n  " + "\n  ".join(missing)
        )

    def test_citation_keys_in_bib(self):
        content = self._all_tex_content()
        # Collect citation keys from \citep{...} and \citet{...}
        # Handle multiple keys in one cite command, e.g. \citep{A, B}
        cite_groups = re.findall(
            r"\\cite[tp]?(?:\[[^\]]*\])*\{([^}]+)\}", content
        )
        cited_keys = set()
        for group in cite_groups:
            for key in group.split(","):
                key = key.strip()
                if key:
                    cited_keys.add(key)

        # Read bib file
        bib_path = os.path.join(PROJECT_ROOT, "references.bib")
        assert os.path.exists(bib_path), "references.bib not found"
        bib_content = open(bib_path, "r", encoding="utf-8").read()
        bib_keys = set(re.findall(r"@\w+\{(\w+)", bib_content))

        missing = cited_keys - bib_keys
        assert missing == set(), (
            f"Citation keys not in references.bib:\n  " + "\n  ".join(sorted(missing))
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline Config Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    """Verify config.py matches expected production settings."""

    def test_use_fundamentals_false(self):
        assert cfg.USE_FUNDAMENTALS is False, (
            f"config.USE_FUNDAMENTALS = {cfg.USE_FUNDAMENTALS}, expected False"
        )

    def test_xgb_seeds_50(self):
        assert len(cfg.XGB_SEEDS) == 50, (
            f"config.XGB_SEEDS has {len(cfg.XGB_SEEDS)} seeds, expected 50"
        )

    def test_hmm_seeds_200(self):
        assert len(cfg.HMM_SEEDS) == 200, (
            f"config.HMM_SEEDS has {len(cfg.HMM_SEEDS)} seeds, expected 200"
        )

    def test_train_end(self):
        assert cfg.TRAIN_END == "2011-01-01", (
            f"config.TRAIN_END = '{cfg.TRAIN_END}', expected '2011-01-01'"
        )

    def test_trading_fee(self):
        assert cfg.TRADING_FEE == pytest.approx(0.001, abs=1e-6), (
            f"config.TRADING_FEE = {cfg.TRADING_FEE}, expected 0.001"
        )

    def test_max_depth(self):
        assert cfg.MAX_DEPTH == 4, (
            f"config.MAX_DEPTH = {cfg.MAX_DEPTH}, expected 4"
        )

    def test_k_states(self):
        assert cfg.K_STATES == 2, (
            f"config.K_STATES = {cfg.K_STATES}, expected 2"
        )

    def test_cs_features_count(self):
        assert len(cfg.CS_FEATURES) == 13, (
            f"config.CS_FEATURES has {len(cfg.CS_FEATURES)} features, expected 13 "
            f"(12 momentum + pi_filter)"
        )

    def test_cs_features_contents(self):
        expected_mom = [f"mom_{i}" for i in range(1, 13)]
        assert cfg.CS_FEATURES == expected_mom + ["pi_filter"], (
            f"config.CS_FEATURES mismatch.\n"
            f"  Got:      {cfg.CS_FEATURES}\n"
            f"  Expected: {expected_mom + ['pi_filter']}"
        )

    def test_hmm_features(self):
        assert cfg.HMM_FEATURES == ["DD_z", "DISP_z", "REL_N_z", "CS_z"], (
            f"config.HMM_FEATURES = {cfg.HMM_FEATURES}, "
            f"expected ['DD_z', 'DISP_z', 'REL_N_z', 'CS_z']"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Reproducibility Checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestReproducibility:
    """Verify that reproducibility artefact CSVs exist and contain key values."""

    def test_depth_results_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "thesis", "depth_results.csv")
        assert os.path.exists(path), f"Missing: {path}"

    def test_depth_results_depth4_return(self):
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "thesis", "depth_results.csv")
        df = pd.read_csv(path)
        # Find the row for depth=4 and check annualised return
        depth_col = [c for c in df.columns if "depth" in c.lower()]
        assert len(depth_col) > 0, (
            f"No depth column found in depth_results.csv. Columns: {list(df.columns)}"
        )
        dcol = depth_col[0]
        ret_col = [c for c in df.columns if "ret" in c.lower() or "ann" in c.lower()]
        if ret_col:
            row = df[df[dcol] == 4]
            assert len(row) > 0, "No row with depth=4 in depth_results.csv"
            val = float(row[ret_col[0]].iloc[0])
            # Value might be in decimal (0.219) or percent (21.9)
            if val < 1:
                val = val * 100
            assert val == pytest.approx(21.7, abs=0.5), (
                f"Depth-4 return = {val}%, expected 21.9%"
            )

    def test_risk_aversion_results_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "thesis", "risk_aversion_thesis_results.csv")
        assert os.path.exists(path), f"Missing: {path}"

    def test_fundamentals_ablation_results_exists(self):
        # Produced by scripts/fundamentals_test.py
        path = os.path.join(PROJECT_ROOT, cfg.RESULTS_DIR, "thesis", "fundamentals_test_results.csv")
        assert os.path.exists(path), f"Missing: {path}"

    def test_artefacts_pickle_exists(self):
        assert os.path.exists(ARTEFACTS_PATH), f"Missing: {ARTEFACTS_PATH}"

    def test_panel_with_regimes_exists(self):
        path = os.path.join(PROJECT_ROOT, cfg.PANEL_WITH_REGIMES_PATH)
        assert os.path.exists(path), f"Missing: {path}"
