# Thesis Verification Test Suite — Design Spec

**Date:** 2026-04-28
**Author:** brainstormed with Claude (opus 4.7)
**Status:** Draft, awaiting user approval before scheduling implementation

## 1. Goal

A reproducible, single-command test suite that verifies the thesis is internally and methodologically sound:

- **Numeric correctness:** every published number in the thesis traces to a `results/thesis/*.csv` row within tolerance.
- **Methodological soundness:** scripts that produce thesis numbers respect point-in-time data, time-ordered splits, seed logging, and no leakage.
- **Coverage visibility:** un-pinned numbers in the thesis prose are surfaced as a markdown report.
- **Wiring health:** the production pipeline can be re-run end-to-end (at reduced size) and produce byte-identical fixtures.
- **Plumbing hygiene:** stale pickles, plot/table disagreements, module-level RNG pollution are caught.

Single entry point: `pytest tests/thesis/`. Determinism is achieved by reading committed CSVs and committed fixtures; no stochastic re-derivation is required to pass the suite.

## 2. Non-goals

- Re-deriving thesis numbers from scratch (the 30–50 hr full pipeline rerun stays manual).
- Modifying `latex/*.tex` or `main.tex` content (Gilad edits prose on Overleaf).
- Touching any existing test file in `tests/` (this design adds new files only).
- Replacing or refactoring `tests/_expected.py` (the new manifest references it; cleanup is a future PR).
- Adding new methodology audits for already-audited scripts (`cross_sectional_model.py`, `expanding_window_backtest.py`, `historical_oos_production.py`, `xgb_cv.py`, `hmm_cv.py`, `leg_betas_by_regime.py`, `apply_shumway_*.py`, `cross_sectional_intl.py`, `hmm_intl.py`, `import_intl_stocks.py`).

## 3. Architecture

```
tests/thesis/
  manifest.yaml               # ~150 entries; single source of truth for thesis claims
  conftest.py                 # session fixtures: load manifest; reuses parent conftest
  test_numbers.py             # parametrized: each manifest entry → CSV row + tex cite check
  test_methodology.py         # static-source audits of 6 currently-unaudited scripts
  test_coverage_gate.py       # warn-mode: writes coverage_report.md; does NOT fail
  test_e2e_smoke.py           # runs run_pipeline.py --repro-smoke, compares fixtures
  test_hygiene.py             # pickle staleness, plot-vs-CSV, RNG hygiene, dtype invariants
  fixtures/
    .gitkeep
    smoke_baseline.json       # NOT committed by 9pm agent; Gilad runs make_smoke_fixture.py
    smoke_artefacts.pkl       # NOT committed by 9pm agent; Gilad runs make_smoke_fixture.py
  make_smoke_fixture.py       # helper: Gilad runs once locally to populate fixtures/
  README.md                   # short, explains how to run, how to update manifest
```

`tests/thesis/coverage_report.md` is gitignored (build artefact).

## 4. Manifest schema

YAML, stored at `tests/thesis/manifest.yaml`. Each entry:

```yaml
- id: m2_sharpe_full                # stable snake_case key
  category: performance              # performance|alpha|regime|subperiod|ic|hmm|intl|seed|bootstrap|crra|stress|leg_beta|panic_subtype|other
  description: "M2 (XGB) full-sample annualised Sharpe, 2011-2025 OOS"
  published:
    value: 1.11
    unit: ratio                     # ratio|percent|tstat|count|months|usd|beta
    tolerance:
      abs: 0.02                     # absolute tolerance, OR
      # rel: 0.01                   # relative, OR
      # stars: '***'                # categorical significance tier
  source:
    csv: results/thesis/expanding_summary_prod.csv
    selector: { strategy: "M2 (XGB)", subperiod: "Full" }   # dict of column=value filters
    column: sharpe
    # OR escape hatch for derived rows:
    # selector_expr: "df.depth == 4"
  cites:
    - { file: tables/table_performance.tex, locator: "row=M2: XGB; col=Sharpe" }
    - { file: latex/main_results.tex, locator: "regex=Sharpe(?:\\s+of)?\\s+1\\.11(?!\\d)" }
  producer:
    script: scripts/run_full_pipeline.py
    seed_set: HMM200xXGB50          # symbolic; recorded for provenance only
  # Optional cross-references:
  canonical: EXP.M2_SHARPE_FULL    # if pinned in tests/_expected.py; consistency-checked
  pair: ff6_alpha_tstat            # for paired claims (alpha/t-stat)
  group: m2_subperiod_sweep        # for grouped claims (e.g. 3 subperiod rows)
  notes: "Recalibrated 2026-04-27 post-Shumway."
```

### Field semantics

- `id` (required, unique). snake_case. Used in test parametrize IDs.
- `category` (required, enum).
- `description` (required, human-readable, ≤120 chars).
- `published.value` (required). May be int, float, or list (for grouped claims like subperiods).
- `published.unit` (required, enum). For consistency in display only.
- `published.tolerance` (required). Exactly one of: `abs`, `rel`, `stars`. For grouped values, applies element-wise.
- `source.csv` (required). Path relative to repo root. May point at `results/thesis/*.csv` or `tables/*.tex` (for tex-table-derived numbers without a separate CSV; uses the existing `_find_in_table` parser pattern).
- `source.selector` (required for CSV; optional for tex). Dict of `column=value` filters that must yield exactly one row. Use `selector_expr` as escape hatch (string evaluated against the loaded DataFrame).
- `source.column` (required). Column name to read; for tex tables, integer 0-based column index.
- `cites` (optional, list). Each cite is a `{file, locator}` pair. `locator` either `"row=X; col=Y"` (for tex tables) or `"regex=PATTERN"` (for prose). `\d` matches must use lookahead/lookbehind to avoid partial matches (e.g. `1\\.11(?!\\d)`).
- `producer` (optional, but populate where known). Documentation only.
- `canonical` (optional). Python attr path into `tests._expected` (e.g. `EXP.M2_SHARPE_FULL`). When present, `test_numbers.py` adds a sub-assertion `abs(published.value - getattr(EXP, attr)) < 1e-9`.
- `pair`, `group` (optional). Free-text refs, used for grouped reporting.
- `notes` (optional). Free-text.

### Edge-case handling

- **Paired claims** (FF6 alpha + t-stat): two entries with `pair:` cross-reference; tested independently.
- **Grouped claims** (3 subperiod Sharpes): one entry with `published.value: [0.59, 1.04, 1.70]`, `source.selector_expr: "df.subperiod.isin(['2011-2015','2016-2020','2021-2025'])"`, `source.column: sharpe`. The test asserts elementwise within tolerance.
- **CI bounds** (bootstrap_sharpe_cis.csv): split into `_ci_low` and `_ci_high` entries with shared `group:` for cross-row consistency.
- **Categorical (stars)**: `tolerance: { stars: '***' }`. The test reads the column and asserts the count of `*` matches the expected tier. Numeric value in `published.value` is the underlying t-stat or coefficient.
- **No-CSV claims** (in-prose only, e.g. `91 panic months in training`): `source: null`. Test only checks `cites[]`.

## 5. Layer 1 — `test_numbers.py`

Parametrized over the manifest. For each entry:

1. **CSV-row check** (skip if `source` is null):
   - Load `source.csv` (cached via fixture).
   - Apply `selector` (or `selector_expr`); assert exactly one row matches.
   - Read `column`. Compare to `published.value` per `tolerance`.
   - On failure: report `<id>: published <V> vs CSV row <V'>; |Δ|=<X>; csv=<path>; selector=<dict>`.
2. **Citation check** (one sub-assertion per cite):
   - Read the cited file.
   - For `locator="regex=..."`: `assert re.search(pattern, content)`.
   - For `locator="row=...; col=..."`: use the existing `_find_in_table` helper from `test_thesis_consistency.py` (import or duplicate in conftest).
   - On failure: report `<id> cite <i>: <pattern> not found in <file>`.
3. **Canonical-consistency check** (if `canonical:` present):
   - `assert abs(entry.published.value - getattr(EXP, attr)) < 1e-9`.
   - On failure: `<id> drift between manifest and _expected.py: <V_manifest> vs <V_expected>`.

### Implementation notes

- Each entry parametrizes 1 + len(cites) + (1 if canonical else 0) subtests.
- Use `pytest.param(..., id=entry['id'])` so failure IDs are readable.
- Cache loaded CSVs via session-scoped fixture keyed by path.
- Tolerances are conservative; default to `abs: max(0.01, 0.01 * |value|)` only if entry has neither `abs` nor `rel`.

## 6. Layer 2 — `test_methodology.py`

Static-source audits in the spirit of `tests/test_cross_sectional_lookahead.py`. Six target scripts (and ONLY these — explicitly do not re-audit cross_sectional_model, expanding_window_backtest, historical_oos_production, xgb_cv, hmm_cv, leg_betas_by_regime, apply_shumway_*, cross_sectional_intl, hmm_intl, import_intl_stocks):

### 6.1 `scripts/bootstrap_analysis.py`

Risks: i.i.d. vs block bootstrap; seed pollution; date-shuffling.

Checks:
- **Seed source**: assert any of (a) script accepts a `--seed` argparse flag, (b) reads a `BOOTSTRAP_SEED`/`SEED` constant from `config.py`, OR (c) hard-codes `np.random.seed(<int literal>)` at top of the bootstrap function (NOT module level — see cross-cutting rule below). Reject `np.random.seed()` with no argument or `np.random.seed(None)`.
- **Block bootstrap**: assert source contains either a block token (`block_size`, `block_bootstrap`, `circular_block`, `stationary_bootstrap`) OR an explicit justifying comment matching regex `# i\.i\.d\. bootstrap.*acceptable` near the resampling call site. If neither, fail with: "bootstrap_analysis.py uses i.i.d. resampling on time-series data without an explicit justification comment".
- **Date integrity**: assert source does NOT contain `df.sample(frac=1)` or `np.random.shuffle(df.index)` directly on a date-indexed frame.

### 6.2 `scripts/zscore_subregime_analysis.py` and `scripts/zscore_subregime_xgb_seed_robustness.py`

Risks: per-date cross-sectional z-scoring (must groupby date, not pool); train-only normalisation; seed sidecar.

Checks:
- **Per-date z-score**: assert source contains `groupby('date')` or equivalent before calling any `.zscore` / `(x - x.mean()) / x.std()` pattern. Reject any pattern that calls `.mean()/.std()` on the full panel and uses the result for z-scoring.
- **Train-only stats**: if the script computes z-score parameters (mean/std), assert they're computed on a `train_mask` slice, not the full panel. Use AST to find `.mean()` / `.std()` call sites and verify operand has been masked.
- **Seed sidecar**: assert script writes a sidecar JSON or filename suffix containing the seed (consistent with `tests/_expected.py` conventions; pattern: `*_seed{N}.csv` or sidecar with `'seed': N` key).
- **No `random_state=None`**: AST-grep for `XGBRegressor(`, `XGBClassifier(`, `RandomForestRegressor(`. Each instantiation must pass `random_state=` with a non-None argument.

### 6.3 `scripts/historical_oos_posthoc.py`

Risks: reuses production training cutoff; refit cadence may peek; feature lag.

Checks:
- **Train cutoff**: assert source uses `< TRAIN_END` (strict), never `<=`. Mirror the `test_cross_sectional_lookahead.py` pattern.
- **Refit cadence**: assert any rolling-fit loop iterates over `t in test_dates` with `train_data = df[df.date < t]` (strict `<`). Reject `<=`.
- **Feature lag**: assert momentum/return features use `shift(1)` (or are read from a column built with shift(1)); assert target uses `shift(-1)`.
- **No reuse of test predictions in training**: AST-grep that `train` does not import any column built from `test_predictions` or `test.<col>`.

### 6.4 `scripts/seed_convergence.py`

Risks: seeds enumerated must be deterministic; no peek across seeds.

Checks:
- **Seed enumeration logged**: filename or output CSV column contains seed values (assert grep for `seed` in `to_csv` calls).
- **Seeds are deterministic**: source contains `seeds = list(range(...))` or `SEEDS = (1, 2, ..., N)` (literal); reject `seeds = np.random.choice(...)` or any stochastic seed-generation pattern.
- **No cross-seed leakage**: assert the script does not use predictions from seed `i` to inform seed `j` (AST-grep for `predictions[i]` referenced inside the seed-`j` loop iteration). Best-effort; if pattern is hard to detect, assert with regex on common shapes.

### 6.5 `scripts/run_rolling_experiment.py`

Risks: rolling window splits, refit cadence.

Checks:
- **Time-ordered splits**: source uses `< t` in train mask, never `train_test_split(shuffle=True)` (AST-grep).
- **Refit cadence**: rolling fit loop iterates `t` in chronological order; `t` enters train only after it enters test (assert via AST that `train_mask = df.date < t` precedes `test_mask = df.date >= t` at minimum).
- **Output sidecar**: results CSV has a date column or window-id column for traceability (grep `to_csv` calls for `index=` and column names).

### Cross-cutting

For every audited script, ALSO assert:
- `from numpy.random import seed` or module-level `np.random.seed(<literal>)` is NOT called at import time (only inside `if __name__ == '__main__'` or function bodies). Use AST.

## 7. Layer 3 — `test_coverage_gate.py`

Walks `latex/*.tex` and lists every numeric token not reachable through the canonical-macros pipeline. **Warn-mode by default; does not fail.**

### Detection logic

For each `latex/*.tex` file:

1. Strip out:
   - Comment lines (`% ...`)
   - Math display blocks (`\begin{equation}...\end{equation}`, `\[...\]`, `$$...$$`)
   - Inline math (`$...$`) — controversial; default to *include* for now (most claims live in inline math)
   - `\input`, `\ref`, `\label`, `\cite`, `\autoref`, `\section`, `\subsection`, `\caption{...}` (caption keeps its content but argument is rescanned with same rules)
   - Body of any `\m<Name>` macro CALL (e.g. `\msharpe` in `main_results.tex` is treated as already-reachable; the literal value is whatever `canonical_macros.tex` defines and counts as covered)
   - Numbers on lines tagged `% config-ok` (whitelist marker for config values)
2. From what remains, extract decimal tokens via regex: `(?<![\w.])-?\d+\.\d+(?!\d)` (decimals only; integers are usually counts and noisy).
3. For each token, compare against the **reachable set**, defined as the union of:
   - All numeric literals on the right-hand side of `\newcommand{\m...}{value}` parsed from `latex/canonical_macros.tex`
   - All `EXP.X` constants from `tests/_expected.py` (loaded via `import _expected`)
   - All `published.value` entries from `manifest.yaml` (including grouped/list values flattened)
   - All values in `coverage_whitelist.txt`
   Match by string equality after normalizing trailing zeros and `$-$` → `-`. Tolerance for "approximately equal" decimals is intentionally NONE — orphans must match exactly, otherwise the report is noisy.

### Output

Write `tests/thesis/coverage_report.md`:

```markdown
# Thesis Coverage Report
Generated: <timestamp>
Total decimal tokens scanned: <N>
Reachable via canonical macros / manifest: <M>
Orphan tokens: <K>

## Orphan tokens
| File:line | Value | Surrounding text (40 chars) |
|---|---|---|
| latex/main_results.tex:42 | 1.27 | "...long-leg sub-regime β=1.27..." |
...
```

### Strict mode

Env var `THESIS_GATE=strict` → test fails if `K > 0`. Default mode → test always passes; report is the artefact.

### Whitelist file

`tests/thesis/coverage_whitelist.txt`: line-based file of values the user has explicitly classified as "config, not result". E.g.:
```
# format: <value> # rationale
0.025  # significance threshold
0.05   # XGB learning rate
2000   # MCMC iterations
500    # burn-in
```

Whitelist values are excluded from the orphan list.

## 8. Layer 4 — `test_e2e_smoke.py`

Runs the full pipeline at reduced scale and compares to a committed fixture. Skips if fixture is missing.

### Prerequisites

- Add `--repro-smoke` flag to `run_pipeline.py` (see §10).
- Fixture files at `tests/thesis/fixtures/smoke_baseline.json` and `smoke_artefacts.pkl` exist (generated by `make_smoke_fixture.py`, run locally by Gilad once).

### Test flow

1. **Skip if no fixture**: `pytest.skip("Run tests/thesis/make_smoke_fixture.py to generate baseline.")` if either fixture missing.
2. **Tempdir invocation**: invoke `python run_pipeline.py --repro-smoke` from a tempdir (use `tmp_path` pytest fixture). Pipeline output redirection: the agent must add CLI plumbing in `run_pipeline.py` so `--repro-smoke` either (a) accepts `--output-root <path>` directing all `results/`, `tables/`, `plots/`, `artefacts/` under that root, OR (b) sets a `THESIS_OUTPUT_ROOT` env var read by config.py. Pick whichever is least invasive after reading `config.py` and the existing scripts. Document the choice in `run_pipeline.py --help`.
3. **Assertions**:
   - Expected output files exist under the tempdir output root (a hard-coded list of `results/thesis/*.csv` paths derived from the fixture's manifest).
   - `<output_root>/results/PRODUCTION_METRICS.json` is byte-identical to `fixtures/smoke_baseline.json`.
   - `<output_root>/artefacts/cs_artefacts_data.pkl` Sharpe (via `metrics()` from `scripts/utils.py`) is within 1e-6 of fixture pickle Sharpe.
   - No file paths in `<output_root>/` outside the fixture's expected set (catches accidental new-file emission).

### `--repro-smoke` config

When flag is set, `run_pipeline.py` overrides config:
- `HMM_SEEDS = list(range(1, 6))` (5 seeds)
- `HMM_ITERATIONS = 200` (was 2000)
- `XGB_SEEDS = list(range(1, 6))` (5 seeds)
- `FIRST_RETRAIN_YEAR = 2020`, `LAST_RETRAIN_YEAR = 2022`
- Step 19 (12-18 hr expanding window) is **skipped** (or replaced with a 2-year sub-run)
- `_chain_phase2.sh` chain is **skipped**
- All output goes to `MOMENTUM_OUTPUT_DIR` if set, else `results/`

Target wall time: **< 30 min on Gilad's 6-core box.**

### Fixture provenance

`make_smoke_fixture.py`:
1. Sets `--repro-smoke` mode.
2. Runs the pipeline.
3. Copies `results/PRODUCTION_METRICS.json` → `tests/thesis/fixtures/smoke_baseline.json`.
4. Copies `artefacts/cs_artefacts_data.pkl` → `tests/thesis/fixtures/smoke_artefacts.pkl`.
5. Writes a `tests/thesis/fixtures/PROVENANCE.md` with timestamp, git SHA, env hash, runtime.

The 9pm agent does NOT run this script (cloud env mismatch risk). The fixture is generated by Gilad locally after the agent commits.

## 9. Layer 5 — `test_hygiene.py`

Four cross-cutting checks. All deterministic.

### 9.1 Pickle staleness

For each cached pickle in `artefacts/`:
- Get its mtime.
- Get the mtime of every `data/*.parquet` file it depends on.
- Assert `pickle.mtime >= max(input.mtimes)`.
- If not, fail with: "`<pickle>` is older than `<input>` — re-run upstream script."

Dependency map is hardcoded for the known pickles:
- `artefacts/cs_artefacts_data.pkl` ← `data/panel.parquet`, `data/panel_with_regimes.parquet`, `data/ff_factors.parquet`
- (extend per project structure)

### 9.2 Plot vs CSV parity

Spot-check 3 thesis-figure CSVs against the corresponding plot data:
- `plots/thesis/zscore_subregimes.pdf` ← `results/thesis/zscore_subregime_summary.csv`
- `plots/thesis/leg_betas.pdf` ← `results/thesis/leg_betas_by_regime.csv`
- (one more)

For each: parse the corresponding plot-generation script (`scripts/zscore_*.py`, `scripts/leg_betas_by_regime.py`), extract the literal `.csv` path it reads, assert the CSV path matches the manifest's source for the related claim. This is a *consistency check* (plot reads the same CSV as the test), not a value comparison (which would require re-rendering).

Stretch (skip if too complex): for the 3 figures, also assert the latest mtime of the PDF is ≥ the CSV mtime.

### 9.3 RNG hygiene

AST-grep over `scripts/*.py`:
- For each script, parse the AST.
- Find module-level (not function-scoped) calls to `np.random.seed(...)`, `random.seed(...)`, `torch.manual_seed(...)`.
- Fail with the list of offenders.

Whitelist: `tests/thesis/rng_hygiene_whitelist.txt` for known-safe legacy scripts.

### 9.4 dtype invariants

Extend `test_critical_column_dtypes` (currently in `test_pipeline_technical.py:321`) to also check:
- `artefacts['train']` (currently only `art['test']` is checked)
- `panel_with_regimes` parquet
- Top 5 most-cited `results/thesis/*.csv` files (return columns are float64; date columns are datetime64[ns]; permno is int64)

This duplicates one assertion's *spirit* (not its exact values) but generalizes scope. Acceptable.

## 10. Manifest auto-generation

The 9pm agent must produce `manifest.yaml` programmatically, not by hand. Strategy:

**Prerequisite**: if `results/PRODUCTION_METRICS.json` does not exist or has not been refreshed against current `tables/*.tex`, the agent MUST NOT regenerate it (that requires running scripts the agent shouldn't run). Instead, work with whatever is committed. If the file is missing, generate the manifest from `_expected.py` + `latex/canonical_macros.tex` parsing only, and document in `manifest_generation_report.md` that PRODUCTION_METRICS.json was unavailable.

1. **Seed entries from `tests/_expected.py`**: each `EXP.X` constant becomes a manifest entry with `canonical: EXP.X`. ~10 entries.
2. **Seed entries from `results/PRODUCTION_METRICS.json`** (if present): parse the JSON, generate one entry per leaf. ~140 entries. Use the JSON key as `id` (snake_case it if needed). Fill `published.value` from the JSON value. Tolerance defaults: `abs: 0.02` for ratios, `abs: 0.1` for percentages, `abs: 0.05` for tstats.
3. **Seed entries from `latex/canonical_macros.tex`** (always): parse `\newcommand{\m...}{value}` lines. Each becomes an entry with `id` = macro name lowercased without the `\m` prefix. If duplicates with step 2, prefer step 2's metadata.
4. **Backfill `source` for each entry**: heuristic — search `scripts/*.py` for the entry's `id` (and JSON key, if applicable) as a string literal. The script containing it is the candidate producer. Look at `to_csv` calls in that script to find the output CSV. This is best-effort; entries where no match is found get `source: null` and are flagged in the generation report.
5. **Backfill `cites`**: for each `published.value`, scan `latex/*.tex` for the literal value (with surrounding-digit lookahead to avoid partial matches like `1.11` matching inside `1.115`). Each match becomes a cite with `locator="regex=<value-with-lookahead>"`. Limit: max 5 cites per entry. If the value also appears in a `tables/*.tex` row, add a `row=...; col=...` cite by parsing the table.
6. **Categorize**: simple keyword rules on the `id` (e.g. id contains `sharpe` → `category: performance`; contains `alpha` → `category: alpha`; contains `subperiod` → `category: subperiod`). Default to `other` if no match.
7. **Write the YAML**: ordered by category then id. Include header comment with generation timestamp, source files, and the rules used.

The agent should commit `manifest.yaml` AND `manifest_generation_report.md` documenting:
- Total entries by source (`_expected.py` / PRODUCTION_METRICS / canonical_macros)
- Entries with `source: null` (manual followup needed)
- Entries with empty `cites: []` (manual followup needed)
- Any `selector_expr:` used (escape hatches)
- Duplicates resolved across sources

## 11. Determinism & reproducibility

- All test inputs are committed: `manifest.yaml`, fixtures, whitelists.
- All test outputs (`coverage_report.md`) are deterministic given inputs.
- `pytest tests/thesis/` runs in < 5 minutes on Gilad's machine (excluding `test_e2e_smoke` which is < 30 min).
- `test_e2e_smoke.py` sets `PYTHONHASHSEED=0`, all RNG seeds pinned to manifest values.
- Fixtures are < 50 MB total (precedent: `data/` is 762 MB committed).
- No tests depend on system time, network, or unpinned dependencies.

## 12. Hard rules for the 9pm agent (anti-overlap)

The agent **must**:
- Only create files under `tests/thesis/` and one CLI flag in `run_pipeline.py`.
- Read existing tests for pattern reference but **never modify** them.
- Read existing scripts for AST audit but **never modify** them.
- **Never** touch `latex/*.tex`, `main.tex`, `tables/*.tex`, `results/`, `plots/`, `data/`, `artefacts/`.
- Skip Layer 2 audits for: `cross_sectional_model`, `expanding_window_backtest`, `expanding_window_backtest_parallel`, `historical_oos_production`, `xgb_cv`, `hmm_cv`, `leg_betas_by_regime`, `apply_shumway_delisting`, `apply_shumway_intl`, `audit_compustat_delisting`, `cross_sectional_intl`, `hmm_intl`, `import_intl_stocks`, `fundamentals_test`, `random_forest_test`.
- For Layer 1, when manifest entry has `canonical:`, **do not re-pin** the value in a separate hardcoded constant; the manifest is the new source.

The agent **must not**:
- Run the pipeline (cloud env mismatch). The smoke fixture is generated by Gilad later.
- Push to main.
- Open a PR (just push the branch; Gilad opens the PR after review).
- Modify `tests/_expected.py` (referenced via `canonical:`, never replaced).
- Add em-dashes to any `.tex` file (per CLAUDE.md). N/A since no `.tex` edits.
- Skip any hooks or use `--no-verify`.

## 13. Acceptance criteria

After the 9pm run, on branch `tests/thesis-verification`:

1. `git status` clean (all changes committed).
2. `pytest tests/thesis/ --collect-only` lists every test file's parametrized cases without errors.
3. `pytest tests/thesis/ -v --ignore=tests/thesis/test_e2e_smoke.py` runs to completion. Some tests may FAIL — this is expected and acceptable. A failing `test_numbers.py` case means the manifest correctly identified real drift between thesis and CSV (good signal, not agent failure). The agent's job is to wire up the framework, not to verify every number is currently correct. The agent must include the failure summary in its final commit message so Gilad knows what drift was surfaced on first run.
4. `tests/thesis/manifest.yaml` has ≥ 100 entries.
5. `tests/thesis/manifest_generation_report.md` documents coverage and gaps.
6. `tests/thesis/coverage_report.md` is generated by running `pytest tests/thesis/test_coverage_gate.py`.
7. `tests/thesis/README.md` explains how to run, how to add a manifest entry, how to populate fixtures.
8. `run_pipeline.py` has a `--repro-smoke` flag visible in `--help`.
9. Branch `tests/thesis-verification` is pushed to origin (fresh branch; no main conflict).

## 14. Branch + commit strategy

The 9pm agent:
1. Pulls latest main.
2. Creates branch `tests/thesis-verification` from main.
3. Commits work in logical chunks (one commit per layer + one for the manifest + one for run_pipeline.py + one for README/whitelists).
4. Commit messages follow project style (verb-led, lowercase, references issue/section if applicable).
5. Pushes branch to origin.
6. Does **not** open a PR or merge. Posts a final summary message describing what was built.

## 15. Out of scope (future PRs by Gilad)

- Refactoring `test_thesis_consistency.py::TestTableConsistency` and `test_pipeline_technical.py::TestThesisTextMatchesTables` to consume the manifest (collapse duplications).
- Migrating `latex/main_results.tex` from hardcoded decimals to `\m...` macros (enables `THESIS_GATE=strict`).
- Wiring `pytest tests/thesis/` into CI.
- Generating the smoke fixture via `make_smoke_fixture.py`.
- Expanding plot-vs-CSV parity to all thesis figures (Layer 5.2 starts with 3).
- Adding methodology audits for the next tier of unaudited scripts (post-hoc analyses, post-hoc plots).
