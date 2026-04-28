"""
make_manifest.py
================
Auto-generate ``tests/thesis/manifest.yaml`` from the three canonical
sources documented in the design spec (§10):

  1. ``tests/_expected.py``                   -- ~10 hand-pinned constants
  2. ``results/PRODUCTION_METRICS.json``     -- ~200 records, may have
                                                  ``value``/``t``/``ci_lo``/
                                                  ``ci_hi``/``p`` per record
  3. ``latex/canonical_macros.tex``          -- ~140 ``\\newcommand``s

Heuristic source-mapping per section + best-effort cite-backfill via
regex search across ``latex/*.tex``. Entries we can't resolve get
``source: null`` and an empty ``cites: []`` and are flagged in the
generation report. Manual followup is expected.

Run from the repo root:

    python tests/thesis/make_manifest.py

Outputs:
  - tests/thesis/manifest.yaml
  - tests/thesis/manifest_generation_report.md
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))


# Make yaml.safe_dump handle OrderedDict like dict (preserves insertion order).
def _odict_representer(dumper, data):
    return dumper.represent_dict(data.items())


yaml.SafeDumper.add_representer(OrderedDict, _odict_representer)


# ─── Source mapping per PRODUCTION_METRICS section ──────────────────────────
SECTION_SOURCE_HINT: dict[str, dict] = {
    "bootstrap": {
        "csv": "results/thesis/bootstrap_sharpe_cis.csv",
        "category": "bootstrap",
    },
    "factor_alphas": {
        "csv": "tables/table_factor_alphas.tex",
        "category": "alpha",
    },
    "fund_alphas": {
        "csv": "tables/table_fund_alphas.tex",
        "category": "alpha",
    },
    "hmm_separation": {
        "csv": "results/thesis/hmm_separation.csv",
        "category": "hmm",
    },
    "ic": {
        "csv": "tables/table_ic.tex",
        "category": "ic",
    },
    "international": {
        "csv": None,  # multi-CSV per region; left null
        "category": "intl",
    },
    "january": {
        "csv": "tables/table_january.tex",
        "category": "other",
    },
    "m2_perf": {
        "csv": "tables/table_performance.tex",
        "category": "performance",
    },
    "panic_subtypes": {
        "csv": "tables/table_panic_subtypes.tex",
        "category": "panic_subtype",
    },
    "regime_sharpe": {
        "csv": "tables/table_regime_sharpe.tex",
        "category": "regime",
    },
    "seed_convergence": {
        "csv": "results/thesis/seed_convergence.csv",
        "category": "seed",
    },
    "subperiod": {
        "csv": "tables/table_subperiod.tex",
        "category": "subperiod",
    },
}


# ─── Hand-pinned canonical constants (mirrors tests/_expected.py) ───────────
EXPECTED_PINS = [
    # (manifest_id, attr_name, value, unit, tolerance_dict, description, source_hint)
    (
        "exp_m2_sharpe_full",
        "M2_SHARPE_FULL",
        1.11,
        "ratio",
        {"abs": 0.02},
        "M2 (XGB) full-sample annualised Sharpe, 2011-2025 OOS",
        {"csv": "tables/table_performance.tex", "selector": {"row_substr": "M2: XGB"}, "column": 3, "category": "performance"},
    ),
    (
        "exp_m2_ann_ret_pct",
        "M2_ANN_RET_PCT",
        21.7,
        "percent",
        {"abs": 0.1},
        "M2 (XGB) annualised return %, full-sample",
        {"csv": "tables/table_performance.tex", "selector": {"row_substr": "M2: XGB"}, "column": 1, "category": "performance"},
    ),
    (
        "exp_m2_ann_vol_pct",
        "M2_ANN_VOL_PCT",
        19.5,
        "percent",
        {"abs": 0.1},
        "M2 (XGB) annualised volatility %, full-sample",
        {"csv": "tables/table_performance.tex", "selector": {"row_substr": "M2: XGB"}, "column": 2, "category": "performance"},
    ),
    (
        "exp_m2_max_dd_pct",
        "M2_MAX_DD_PCT",
        -72.2,
        "percent",
        {"abs": 0.1},
        "M2 (XGB) max drawdown % (signed), full-sample",
        {"csv": None, "category": "performance"},
    ),
    (
        "exp_m2_sharpe_panic",
        "M2_SHARPE_PANIC",
        1.53,
        "ratio",
        {"abs": 0.02},
        "M2 (XGB) regime-conditional Sharpe in panic months",
        {"csv": "tables/table_regime_sharpe.tex", "selector": {"row_substr": "M2: XGB"}, "column": 3, "category": "regime"},
    ),
    (
        "exp_m2_sharpe_calm",
        "M2_SHARPE_CALM",
        0.84,
        "ratio",
        {"abs": 0.02},
        "M2 (XGB) regime-conditional Sharpe in calm months",
        {"csv": "tables/table_regime_sharpe.tex", "selector": {"row_substr": "M2: XGB"}, "column": 2, "category": "regime"},
    ),
    (
        "exp_ablation_hmm_sharpe",
        "ABLATION_HMM_SHARPE",
        1.107,
        "ratio",
        {"abs": 0.01},
        "HMM ablation Sharpe (table_regime_signal_ablation row=HMM)",
        {"csv": "tables/table_regime_signal_ablation.tex", "selector": {"row_substr": "HMM"}, "column": 3, "category": "regime"},
    ),
    (
        "exp_ff6_alpha_pct",
        "FF6_ALPHA_PCT",
        24.1,
        "percent",
        {"abs": 0.05},
        "FF6 monthly alpha % (annualised) for M2",
        {"csv": "tables/table_factor_alphas.tex", "selector": {"row_substr": "FF6"}, "column": 1, "category": "alpha"},
    ),
    (
        "exp_ff6_tstat",
        "FF6_TSTAT",
        4.81,
        "tstat",
        {"abs": 0.01},
        "FF6 alpha t-statistic for M2",
        {"csv": "tables/table_factor_alphas.tex", "selector": {"row_substr": "FF6"}, "column": 2, "category": "alpha"},
    ),
    (
        "exp_depth4_ann_ret_pct",
        "DEPTH4_ANN_RET_PCT",
        21.7,
        "percent",
        {"abs": 0.1},
        "Depth-4 annualised return % from depth_results.csv",
        {"csv": "results/thesis/depth_results.csv", "selector_expr": "df['depth' if 'depth' in df.columns else df.columns[0]] == 4", "column": "ann_ret", "category": "performance"},
    ),
    (
        "exp_depth4_sharpe",
        "DEPTH4_SHARPE",
        1.11,
        "ratio",
        {"abs": 0.1},
        "Depth-4 Sharpe ratio from depth_results.csv",
        {"csv": "results/thesis/depth_results.csv", "selector_expr": "df['depth' if 'depth' in df.columns else df.columns[0]] == 4", "column": "sharpe", "category": "performance"},
    ),
]


# ─── Tolerance defaults by leaf-field type ──────────────────────────────────
def default_tolerance(field: str, value, section: str) -> dict:
    """Pick a sensible tolerance based on field name / section / magnitude."""
    if field == "t":
        return {"abs": 0.05}
    if field == "p":
        return {"abs": 0.005}
    if field in ("ci_lo", "ci_hi"):
        return {"abs": 0.05}
    # value field — depends on section / unit hint
    if section in ("factor_alphas", "fund_alphas"):
        return {"abs": 0.1}  # alphas in percent
    if section == "ic":
        return {"abs": 0.005}
    if section == "international":
        # fractional ratios/returns — 1pp absolute tolerance
        return {"abs": 0.01}
    if section == "hmm_separation":
        return {"abs": 0.05}
    if section == "seed_convergence":
        return {"abs": 0.02}
    if section == "panic_subtypes":
        # mixed: counts (months) need wider, sharpes/returns tighter
        return {"abs": 0.5}
    if section == "subperiod":
        return {"abs": 0.05}
    if section == "m2_perf":
        return {"abs": 0.5}
    if section == "regime_sharpe":
        return {"abs": 0.02}
    if section == "bootstrap":
        return {"abs": 0.05}
    if section == "january":
        return {"abs": 0.5}
    return {"abs": max(0.01, abs(float(value)) * 0.01)}


# ─── Cite backfill: regex-search latex/*.tex for the literal value ─────────
def _format_value_for_search(v: float) -> list[str]:
    """Return regex-safe string forms of a numeric value worth searching for."""
    out = []
    if abs(v) < 1.0 and v != 0:
        # Likely a fractional ratio; search 2 and 3 dp
        for prec in (2, 3, 4):
            s = f"{v:.{prec}f}"
            if s.endswith("0") and prec > 2:
                continue
            out.append(s)
    else:
        # Search a few precision rounds
        out.append(f"{v:.2f}")
        out.append(f"{v:.1f}")
        out.append(f"{v:.3f}".rstrip("0").rstrip("."))
        # also raw int
        if abs(v - round(v)) < 1e-9:
            out.append(str(int(round(v))))
    # Dedupe but preserve order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen and s.strip():
            seen.add(s)
            uniq.append(s)
    return uniq


def find_cites(value: float, tex_files: dict[str, str], max_cites: int = 5) -> list[dict]:
    """For each .tex file, find regex matches with non-digit lookahead. Returns up to max_cites cites."""
    cites: list[dict] = []
    for form in _format_value_for_search(value):
        # Anchor on a non-digit / boundary; lookbehind to avoid partial digit matches
        # Escape dot manually since we want literal '.'
        escaped = re.escape(form)
        # Add (?<!\d) before and (?!\d) after; allow optional '$-$' minus or '-' just before
        pat = rf"(?<![\d.]){escaped}(?!\d)"
        for path, content in tex_files.items():
            for m in re.finditer(pat, content):
                cites.append({
                    "file": path,
                    "locator": f"regex={pat}",
                })
                if len(cites) >= max_cites:
                    return cites
    return cites


def categorise(entry_id: str, default: str) -> str:
    s = entry_id.lower()
    if "sharpe" in s and "subperiod" in s:
        return "subperiod"
    if "subperiod" in s or "_2011_" in s or "_2016_" in s or "_2021_" in s:
        return "subperiod"
    if "alpha" in s or "_t" in s.split("_")[-1:][0:1]:
        # "alpha" wins
        if "alpha" in s:
            return "alpha"
    if "sharpe" in s and "regime" in s:
        return "regime"
    if "panic" in s or "calm" in s:
        return "regime"
    if "bootstrap" in s or "ci_lo" in s or "ci_hi" in s:
        return "bootstrap"
    if "intl" in s or "international" in s or s.startswith(("jp_", "uk_")):
        return "intl"
    if "seed" in s:
        return "seed"
    if "ic" in s.split("_")[:1]:
        return "ic"
    if "hmm" in s:
        return "hmm"
    if "alpha" in s:
        return "alpha"
    if "sharpe" in s:
        return "performance"
    return default or "other"


# ─── Generators ─────────────────────────────────────────────────────────────


def gen_from_expected(tex_files: dict[str, str]) -> list[dict]:
    entries = []
    for entry_id, attr, value, unit, tol, desc, source_hint in EXPECTED_PINS:
        cites = find_cites(value, tex_files)
        e: OrderedDict = OrderedDict()
        e["id"] = entry_id
        e["category"] = source_hint.get("category", "other")
        e["description"] = desc
        e["published"] = OrderedDict([
            ("value", value),
            ("unit", unit),
            ("tolerance", tol),
        ])
        e["source"] = build_source(source_hint)
        e["cites"] = cites
        e["canonical"] = f"EXP.{attr}"
        e["origin"] = "_expected.py"
        entries.append(e)
    return entries


def build_source(hint: dict | None) -> dict | None:
    if not hint or hint.get("csv") is None:
        return None
    out: OrderedDict = OrderedDict()
    out["csv"] = hint["csv"]
    sel = hint.get("selector")
    if sel and "row_substr" in sel:
        # Tex-table style selector
        out["selector"] = OrderedDict([("row_substr", sel["row_substr"])])
    elif sel:
        out["selector"] = OrderedDict(sel)
    if hint.get("selector_expr"):
        out["selector_expr"] = hint["selector_expr"]
    if "column" in hint:
        out["column"] = hint["column"]
    return out


# ─── Per-section selector/column resolvers ──────────────────────────────────
# Each resolver returns (selector_dict_or_None, column_or_None) for a given
# (key, leaf) pair. Returning (None, None) means the entry stays source: null.

# Strategy → CSV "strategy" column (bootstrap)
_BOOTSTRAP_STRAT = {
    "fixed_12": "Fixed 12-mo mom",
    "fixed_1": "Fixed 1-mo mom",
    "m0": "M0 (Formula)",
    "m1": "M1 (LR)",
    "m2": "M2 (XGB)",
}
_BOOTSTRAP_LEAF_COL = {"value": "point_sharpe", "ci_lo": "ci_low", "ci_hi": "ci_high"}

# Bootstrap paired-tests (m2_vs_<benchmark>) live in a different CSV
_BOOTSTRAP_PAIRED_BENCH = {
    "m2_vs_m1": "M1 (LR)",
    "m2_vs_m0": "M0 (Formula)",
    "m2_vs_fixed_12": "Fixed 12-mo mom",
    "m2_vs_fixed_1": "Fixed 1-mo mom",
}
_BOOTSTRAP_PAIRED_LEAF_COL = {
    "value": "point_diff",
    "ci_lo": "diff_ci_low",
    "ci_hi": "diff_ci_high",
    "p": "p_value",
}

# Strategy → tex row_substr (m2_perf, regime_sharpe, subperiod)
_TEX_STRAT = {
    "fixed_12": "Fixed 12-mo",
    "fixed_1": "Fixed 1-mo",
    "m0": "M0: Formula",
    "m1": "M1: LR",
    "m2": "M2: XGB",
    "market": "Market",
}
# m2_perf metric → 0-based column index (after row label)
_M2_PERF_METRIC_COL = {
    "ann_ret": 1, "ann_vol": 2, "sharpe": 3, "max_dd": 4,
    "beta": 5, "nw_t": 6, "final_dollar": 7,
}
# regime_sharpe regime → column
_REGIME_COL = {"full": 1, "calm": 2, "panic": 3}
# subperiod period → column
_SUBPERIOD_COL = {"2011_2015": 1, "2016_2020": 2, "2021_2025": 3, "full": 4}

# panic_subtypes
_PANIC_SUBTYPE_ROW = {
    "calm": "Calm",
    "panic_crash": "Panic: Crash",
    "panic_recovery": "Panic: Recovery",
}
_PANIC_METRIC_COL = {"ann_ret": 1, "ann_vol": 2, "sharpe": 3, "months": 4}

# factor_alphas / fund_alphas
_FACTOR_MODEL_ROW = {
    "capm_alpha": "CAPM",
    "ff3_alpha": "FF3",
    "carhart_alpha": "Carhart",
    "ff5_alpha": "FF5",
    "ff6_alpha": "FF6",
}
_FUND_MODEL_ROW = {
    "fund_capm_alpha": "CAPM",
    "fund_ff3_alpha": "FF3",
    "fund_carhart_alpha": "Carhart",
    "fund_ff5_alpha": "FF5",
    "fund_ff6_alpha": "FF6",
}
_ALPHA_LEAF_COL = {"value": 1, "t": 2}

# hmm_separation: feature row + metric column
_HMM_FEATURE_ROW = {"cs": "CS_z", "dd": "DD_z", "disp": "DISP_z", "rel_n": "REL_N_z"}
_HMM_METRIC_COL = {"calm": "calm_mean", "panic": "panic_mean", "delta": "delta"}
_HMM_LEAF_COL = {"value": None, "ci_lo": "ci_lo", "ci_hi": "ci_hi"}  # value uses metric col

# ic: only m2_ic
_IC_ROW = {"m2_ic": "M2: XGB"}
# IC: only `value` is unambiguous. PRODUCTION_METRICS.json labels the Std
# column (col 2) as `t`, but the actual NW t-stat is col 3. Until that JSON
# label bug is resolved, we don't auto-generate the t entry — the test would
# either pass against the wrong column or fail. See IMPLEMENTATION_NOTES.
_IC_LEAF_COL = {"value": 1}

# seed_convergence: row by k, column by leaf
_SEED_LEAF_COL = {"value": "mean", "p25": "p25", "p75": "p75",
                  "std": "std", "min": "min", "max": "max"}


def resolve_source(section: str, key: str, leaf: str) -> tuple[dict | None, object]:
    """Return (selector_dict, column) for a (section, key, leaf) tuple.
    Returns (None, None) when no mapping is known."""
    if section == "bootstrap":
        # m2_vs_<benchmark>: paired comparison in a different CSV
        if key in _BOOTSTRAP_PAIRED_BENCH:
            col = _BOOTSTRAP_PAIRED_LEAF_COL.get(leaf)
            if col is None:
                return None, None
            return {"_csv_override": "results/thesis/bootstrap_paired_tests.csv",
                    "benchmark": _BOOTSTRAP_PAIRED_BENCH[key]}, col
        # key like "m2_sharpe", "fixed_12_sharpe"
        for prefix, row_val in _BOOTSTRAP_STRAT.items():
            if key.startswith(prefix + "_"):
                col = _BOOTSTRAP_LEAF_COL.get(leaf)
                if col is None:
                    return None, None
                return {"strategy": row_val}, col
        return None, None

    if section == "m2_perf":
        for prefix, row_substr in _TEX_STRAT.items():
            if key.startswith(prefix + "_"):
                metric = key[len(prefix) + 1:]
                col = _M2_PERF_METRIC_COL.get(metric)
                if col is None:
                    return None, None
                if leaf == "value":
                    return {"row_substr": row_substr}, col
                # No t/p/ci on m2_perf table directly
                return None, None
        return None, None

    if section == "regime_sharpe":
        for prefix, row_substr in _TEX_STRAT.items():
            if key.startswith(prefix + "_"):
                regime = key[len(prefix) + 1:]
                col = _REGIME_COL.get(regime)
                if col is None or leaf != "value":
                    return None, None
                return {"row_substr": row_substr}, col
        return None, None

    if section == "subperiod":
        for prefix, row_substr in _TEX_STRAT.items():
            if key.startswith(prefix + "_"):
                period = key[len(prefix) + 1:]
                col = _SUBPERIOD_COL.get(period)
                if col is None or leaf != "value":
                    return None, None
                return {"row_substr": row_substr}, col
        return None, None

    if section == "panic_subtypes":
        for prefix, row_substr in _PANIC_SUBTYPE_ROW.items():
            if key.startswith(prefix + "_"):
                metric = key[len(prefix) + 1:]
                col = _PANIC_METRIC_COL.get(metric)
                if col is None or leaf != "value":
                    return None, None
                return {"row_substr": row_substr}, col
        return None, None

    if section == "factor_alphas":
        row = _FACTOR_MODEL_ROW.get(key)
        if row is None:
            return None, None
        col = _ALPHA_LEAF_COL.get(leaf)
        if col is None:
            return None, None
        return {"row_substr": row}, col

    if section == "fund_alphas":
        row = _FUND_MODEL_ROW.get(key)
        if row is None:
            return None, None
        col = _ALPHA_LEAF_COL.get(leaf)
        if col is None:
            return None, None
        return {"row_substr": row}, col

    if section == "hmm_separation":
        # key like "cs_calm", "dd_delta"
        for prefix, feat in _HMM_FEATURE_ROW.items():
            if key.startswith(prefix + "_"):
                metric = key[len(prefix) + 1:]
                if leaf == "value":
                    col = _HMM_METRIC_COL.get(metric)
                    if col is None:
                        return None, None
                    return {"feature": feat}, col
                if leaf in ("ci_lo", "ci_hi") and metric == "delta":
                    return {"feature": feat}, leaf
                return None, None
        return None, None

    if section == "ic":
        row = _IC_ROW.get(key)
        if row is None:
            return None, None
        col = _IC_LEAF_COL.get(leaf)
        if col is None:
            return None, None
        return {"row_substr": row}, col

    if section == "seed_convergence":
        # key like "k_1", "k_10"
        if key.startswith("k_"):
            try:
                k_val = int(key[2:])
            except ValueError:
                return None, None
            col = _SEED_LEAF_COL.get(leaf)
            if col is None:
                return None, None
            return {"k": k_val}, col
        return None, None

    # january: skipped — has duplicate row labels across "Full" / "Excluding January"
    # sections that need a custom selector. Tracked in IMPLEMENTATION_NOTES.md.
    # international: 72 entries across regions, multi-CSV — left as null per spec.
    return None, None


def gen_from_metrics(metrics: dict, tex_files: dict[str, str], existing_ids: set[str]) -> list[dict]:
    entries = []
    for section, sec_v in metrics.items():
        if section.startswith("_"):
            continue
        hint = SECTION_SOURCE_HINT.get(section, {})
        for key, rec in sec_v.items():
            base_id = f"{section}_{key}".lower()
            if not isinstance(rec, dict):
                # Scalar leaf
                rec = {"value": rec}
            # Identify leaf fields present
            leaves = [f for f in ("value", "t", "ci_lo", "ci_hi", "p") if f in rec]
            for leaf in leaves:
                v = rec[leaf]
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                eid = base_id if leaf == "value" else f"{base_id}_{leaf}"
                if eid in existing_ids:
                    continue
                existing_ids.add(eid)
                desc = f"PRODUCTION_METRICS.{section}.{key}.{leaf}"
                e: OrderedDict = OrderedDict()
                e["id"] = eid
                e["category"] = categorise(eid, hint.get("category", "other"))
                e["description"] = desc
                e["published"] = OrderedDict([
                    ("value", float(v)),
                    ("unit", _infer_unit(rec, leaf)),
                    ("tolerance", default_tolerance(leaf, v, section)),
                ])
                # Per-section selector/column resolver (gives null when unknown).
                sel, col = resolve_source(section, key, leaf)
                if sel is not None and col is not None:
                    # Selector may carry a `_csv_override` to point at a different CSV.
                    csv_path = sel.pop("_csv_override", None) or hint.get("csv")
                    if csv_path:
                        src_hint = {
                            "csv": csv_path,
                            "selector": sel,
                            "column": col,
                        }
                        e["source"] = build_source(src_hint)
                    else:
                        e["source"] = None
                else:
                    e["source"] = None
                e["cites"] = find_cites(float(v), tex_files)
                if leaf != "value":
                    e["pair"] = base_id
                e["origin"] = "PRODUCTION_METRICS.json"
                entries.append(e)
    return entries


def _infer_unit(rec: dict, leaf: str) -> str:
    if leaf == "t":
        return "tstat"
    if leaf == "p":
        return "ratio"  # p-values are ratios
    if leaf in ("ci_lo", "ci_hi"):
        # Inherit from main value if known
        if rec.get("unit") == "percent":
            return "percent"
        return "ratio"
    return rec.get("unit", "ratio")


def gen_from_macros(macros_path: Path, existing_ids: set[str], tex_files: dict[str, str]) -> list[dict]:
    if not macros_path.exists():
        return []
    content = macros_path.read_text()
    pat = re.compile(r"\\newcommand\{\\m([A-Za-z0-9]+)\}\{([^}]+)\}")
    entries = []
    for m in pat.finditer(content):
        name = m.group(1).lower()
        raw = m.group(2).strip()
        # Extract leading numeric (allow leading $-$, %, etc.)
        cleaned = raw.replace("$-$", "-").replace("\\%", "").replace("%", "").strip()
        try:
            val = float(cleaned)
        except ValueError:
            continue
        eid = f"macro_{name}"
        if eid in existing_ids:
            continue
        existing_ids.add(eid)
        e: OrderedDict = OrderedDict()
        e["id"] = eid
        e["category"] = categorise(name, "other")
        e["description"] = f"canonical_macros.tex \\m{name} = {raw}"
        e["published"] = OrderedDict([
            ("value", val),
            ("unit", "ratio"),
            ("tolerance", {"abs": max(0.01, abs(val) * 0.01)}),
        ])
        e["source"] = None  # macro is itself a derived value
        e["cites"] = find_cites(val, tex_files)
        e["origin"] = "canonical_macros.tex"
        entries.append(e)
    return entries


# ─── Main ───────────────────────────────────────────────────────────────────


def collect_tex_files() -> dict[str, str]:
    out = {}
    latex_dir = REPO / "latex"
    for p in sorted(latex_dir.glob("*.tex")):
        rel = str(p.relative_to(REPO))
        out[rel] = p.read_text(encoding="utf-8")
    main = REPO / "main.tex"
    if main.exists():
        out["main.tex"] = main.read_text(encoding="utf-8")
    return out


def main() -> int:
    metrics_path = REPO / "results" / "PRODUCTION_METRICS.json"
    macros_path = REPO / "latex" / "canonical_macros.tex"
    out_yaml = HERE / "manifest.yaml"
    out_report = HERE / "manifest_generation_report.md"

    metrics_present = metrics_path.exists()
    metrics: dict = {}
    if metrics_present:
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        print(f"WARN: {metrics_path} missing — falling back to _expected.py + canonical_macros.tex only.")

    tex_files = collect_tex_files()

    existing_ids: set[str] = set()
    entries: list[dict] = []
    expected_entries = gen_from_expected(tex_files)
    for e in expected_entries:
        existing_ids.add(e["id"])
    entries.extend(expected_entries)

    metrics_entries = gen_from_metrics(metrics, tex_files, existing_ids) if metrics_present else []
    entries.extend(metrics_entries)

    macros_entries = gen_from_macros(macros_path, existing_ids, tex_files)
    entries.extend(macros_entries)

    # Sort by category, then id
    entries.sort(key=lambda e: (e.get("category", "zzz"), e["id"]))

    # Strip the helper "origin" field from the YAML (keep elsewhere for report)
    origins = {e["id"]: e.pop("origin") for e in entries}

    # Write YAML
    header = (
        f"# Auto-generated by tests/thesis/make_manifest.py\n"
        f"# Generated: {datetime.now(timezone.utc).isoformat()}\n"
        f"# Sources: tests/_expected.py, results/PRODUCTION_METRICS.json"
        f"{'' if metrics_present else ' (MISSING)'}, latex/canonical_macros.tex\n"
        f"# Total entries: {len(entries)}\n"
        f"#\n"
        f"# Edit by hand only as a last resort. Re-run make_manifest.py to refresh.\n"
        f"# Entries with `source: null` need manual followup (no CSV/TEX selector).\n"
        f"# Entries with empty `cites: []` need manual followup (value not found in latex/*.tex).\n"
    )
    body = yaml.safe_dump(
        {"entries": [dict(e) for e in entries]},
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    out_yaml.write_text(header + "\n" + body)

    # Stats for report
    by_origin = {"_expected.py": 0, "PRODUCTION_METRICS.json": 0, "canonical_macros.tex": 0}
    by_category: dict[str, int] = {}
    null_source = []
    empty_cites = []
    selector_expr_used = []
    for e in entries:
        by_origin[origins[e["id"]]] = by_origin.get(origins[e["id"]], 0) + 1
        by_category[e.get("category", "other")] = by_category.get(e.get("category", "other"), 0) + 1
        if e.get("source") is None:
            null_source.append(e["id"])
        if not e.get("cites"):
            empty_cites.append(e["id"])
        src = e.get("source") or {}
        if src.get("selector_expr"):
            selector_expr_used.append(e["id"])

    rep = []
    rep.append("# Manifest Generation Report\n")
    rep.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    rep.append(f"Total entries: {len(entries)}\n")
    rep.append("\n## By origin\n")
    for k, v in by_origin.items():
        rep.append(f"- {k}: {v}\n")
    rep.append("\n## By category\n")
    for k in sorted(by_category):
        rep.append(f"- {k}: {by_category[k]}\n")
    rep.append(f"\n## Source: null ({len(null_source)} entries — manual followup needed)\n")
    for x in null_source[:200]:
        rep.append(f"- `{x}`\n")
    if len(null_source) > 200:
        rep.append(f"- ...and {len(null_source) - 200} more\n")
    rep.append(f"\n## Empty cites ({len(empty_cites)} entries — manual followup needed)\n")
    for x in empty_cites[:200]:
        rep.append(f"- `{x}`\n")
    if len(empty_cites) > 200:
        rep.append(f"- ...and {len(empty_cites) - 200} more\n")
    rep.append(f"\n## selector_expr used ({len(selector_expr_used)} entries — escape hatch)\n")
    for x in selector_expr_used:
        rep.append(f"- `{x}`\n")
    rep.append("\n## Notes on de-duplication\n")
    rep.append("- `_expected.py` constants take priority and have `canonical: EXP.X`.\n")
    rep.append("- PRODUCTION_METRICS entries that share an id with an _expected entry are skipped.\n")
    rep.append("- canonical_macros.tex entries that share an id are skipped.\n")
    if not metrics_present:
        rep.append("\n## WARNING\n")
        rep.append("- `results/PRODUCTION_METRICS.json` was missing at generation time.\n")
        rep.append("- Manifest was built from `_expected.py` + `canonical_macros.tex` only.\n")

    out_report.write_text("".join(rep))

    print(f"Wrote {out_yaml} ({len(entries)} entries).")
    print(f"Wrote {out_report}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
