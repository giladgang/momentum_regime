# Post-Shumway Finalization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the post-Shumway transition: finalize all data outputs, execute the production/cv/diagnostic directory restructure, and refresh the canonical-store framework so every thesis number traces cleanly to a single source of truth (`results/PRODUCTION_METRICS.json`).

**Architecture:** A pipeline of mechanical steps gated on the last in-flight script (`hmm_model.py`) finishing. Sequence: (1) verify final in-flight outputs land cleanly, (2) Tier B revive `features_hmm.png` from archive, (3) execute the pre-written migration script (`scripts/migrate_to_organized.py`) to move ~101 files into `results/thesis/`, `results/cv/`, `results/oos_historical/`, `results/reports/`, `plots/thesis/`, `plots/diagnostic/`, (4) refresh the canonical chain (build_metrics → build_canonical_macros → build_thesis_tables → verify_thesis_consistency), (5) regenerate `PROSE_DRIFT_REPORT.md`, (6) commit + push.

**Tech Stack:** Python 3.12 (pandas, statsmodels, matplotlib, xgboost), LaTeX (latex/*.tex + main.tex), git, GitHub remote `giladgang/momentum_regime`. CLAUDE.md project conventions apply (no em-dashes in .tex; pull before push; never force-push or skip hooks).

---

## File Structure

This plan modifies / creates:

- **Read-only verification:** `logs/hmm_model.log`, `plots/convergence_trace.png`, `plots/regime_probabilities.png`
- **Tier B revive:** invoke `archive/scripts/regenerate_thesis_plots.py` to (re)generate `plots/features_hmm.png`
- **Migration execution:** `scripts/migrate_to_organized.py` (already committed, executes the restructure)
- **After migration, the canonical chain reads/writes:**
  - `results/PRODUCTION_METRICS.json` (root, path-stable)
  - `results/reports/METRICS_DIFF.md`, `results/reports/PROSE_DRIFT_REPORT.md` (in new locations)
  - `latex/canonical_macros.tex`
  - `tables/*.canonical.tex`
- **Verification:** `pdflatex main.tex` (existing thesis compile), `pytest tests/test_pipeline_smoke.py` (one fast smoke test)

The migration script automatically updates ~26 Python scripts under `scripts/` and ~5 latex files for the new paths, so no per-script edits in this plan.

---

## Task 1: Confirm `hmm_model.py` finished cleanly

The 5 other in-flight scripts already finished. Only `hmm_model` remains (was at iteration 1500/2000 last check). Once it completes, three latex-included plots refresh: `convergence_trace.png`, `regime_probabilities.png`, and `features_hmm.png` (the last via Tier B in Task 2).

**Files:**
- Read: `logs/hmm_model.log`
- Verify: `plots/convergence_trace.png`, `plots/regime_probabilities.png`

- [ ] **Step 1.1: Wait for the hmm_model background task to emit a completion notification**

The agent harness emits a `<task-notification>` when the script's PID exits. Block on that notification; do not poll with `sleep`.

- [ ] **Step 1.2: Read the tail of the log to confirm clean exit**

Run: `tail -30 logs/hmm_model.log`
Expected output ends with something like:
```
HMM diagnostics complete
Saved: artefacts/hmm_panel.parquet
Saved: plots/convergence_trace.png
Saved: plots/regime_probabilities.png
```

If the tail shows a Python traceback instead, **STOP** and inspect. Do not proceed to migration with a half-baked HMM run.

- [ ] **Step 1.3: Verify the plot files have a fresh mtime**

Run: `stat -f "%Sm  %N" plots/convergence_trace.png plots/regime_probabilities.png`
Expected: both files dated **today** (Apr 27 2026), not Apr 18 (the pre-rerun date).

If still pre-rerun: the script ran but didn't write the expected output. Inspect `logs/hmm_model.log` for `Saved:` lines.

---

## Task 2: Revive `features_hmm.png` from archive

`scripts/hmm_model.py` doesn't write `features_hmm.png` — that figure was originally produced by `archive/scripts/regenerate_thesis_plots.py`. It's referenced via `\includegraphics{plots/features_hmm.png}` in `latex/data_section.tex:45`, so it must exist.

**Files:**
- Run: `archive/scripts/regenerate_thesis_plots.py`
- Output: `plots/features_hmm.png`

- [ ] **Step 2.1: Run the archive script with the project root as cwd so its hardcoded relative paths resolve**

Run: `cd /Users/giladgang/momentum_regime && python archive/scripts/regenerate_thesis_plots.py 2>&1 | tail -5`

Expected output ends with:
```
Saved features_hmm.png (correct 4 features)
```

- [ ] **Step 2.2: The archive script writes `features_hmm.png` into cwd, not `plots/`. Move it.**

Run: `[ -f features_hmm.png ] && mv features_hmm.png plots/features_hmm.png`
Then verify: `stat -f "%Sm  %N" plots/features_hmm.png` → expect today's date.

- [ ] **Step 2.3: Sanity-check the figure exists and is non-trivial**

Run: `ls -la plots/features_hmm.png` → expect file size > 50 KB (a real matplotlib PNG, not a 1-byte stub).

---

## Task 3: Snapshot pre-migration state

Before the migration moves 101 files, take an explicit snapshot per CLAUDE.md backup convention. The session's earlier `baseline_pre_full_rerun_20260427_150701/` may not include the latest in-flight outputs.

**Files:**
- Create: `baseline_pre_restructure_<timestamp>/` (full copy of `tables/`, `results/`, `plots/`)

- [ ] **Step 3.1: Snapshot tables/, results/, plots/ to a timestamped directory**

Run:
```bash
TS=$(date +%Y%m%d_%H%M%S)
SNAP="baseline_pre_restructure_${TS}"
mkdir -p "$SNAP"
cp -R tables "$SNAP/tables"
cp -R results "$SNAP/results"
cp -R plots "$SNAP/plots"
du -sh "$SNAP"
echo "Snapshot saved to $SNAP"
```

Expected output: `~ 50 MB    baseline_pre_restructure_20260427_HHMMSS` (size approximate).

- [ ] **Step 3.2: Verify snapshot is readable**

Run: `ls "$SNAP"/tables | head -3 && ls "$SNAP"/plots | head -3 && ls "$SNAP"/results | head -3`
Expected: snapshot directories contain the expected files (table_*.tex, *.png, *.csv).

---

## Task 4: Dry-run the migration to preview the move

`scripts/migrate_to_organized.py` is already committed (commit `7bb6b99`). The dry run reports exactly what it would change without modifying anything.

**Files:**
- Read-only: `scripts/migrate_to_organized.py`

- [ ] **Step 4.1: Run the migration with `--dry-run` and capture the summary**

Run: `python scripts/migrate_to_organized.py --dry-run 2>&1 | tail -20`

Expected output (last lines):
```
[2/4] Moving files ...
  → 101 files moved
[3/4] Updating script output paths (scripts/*.py) ...
  → 26 script(s) updated
[4/4] Updating latex \includegraphics paths ...
  → 5 latex file(s) updated
Dry run complete. Re-run without --dry-run to execute.
```

If counts have changed (e.g., 99 files instead of 101), inspect the log to see which files are missing — likely an in-flight script wrote to a path I didn't list. Update `scripts/migrate_to_organized.py`'s categorization sets if needed before executing.

---

## Task 5: Execute the migration

**Files:**
- Move: ~101 files in `results/` and `plots/`
- Modify: ~26 scripts in `scripts/` (path strings)
- Modify: 5 latex files (`main.tex`, `latex/main_results.tex`, `latex/data_section.tex`, `latex/methodology.tex`, `latex/appendix.tex`)
- Create directories: `results/thesis/`, `results/cv/`, `results/oos_historical/`, `results/reports/`, `plots/thesis/`, `plots/diagnostic/`

- [ ] **Step 5.1: Execute the migration**

Run: `python scripts/migrate_to_organized.py 2>&1 | tail -10`

Expected output (last lines):
```
  → 101 files moved
  → 26 script(s) updated
  → 5 latex file(s) updated
Migration complete. Verify with:
  python scripts/build_metrics.py
  pdflatex main.tex  # check thesis still compiles
```

- [ ] **Step 5.2: Verify directory structure was created and populated**

Run:
```bash
for d in results/thesis results/cv results/oos_historical results/reports plots/thesis plots/diagnostic; do
  printf "%-30s " "$d/:"; ls "$d" 2>/dev/null | wc -l
done
```

Expected:
```
results/thesis/:                32+
results/cv/:                     8
results/oos_historical/:         8
results/reports/:                7+
plots/thesis/:                  11
plots/diagnostic/:              30+
```

- [ ] **Step 5.3: Confirm `PRODUCTION_METRICS.json` did NOT move (it should stay at results/ root for path stability)**

Run: `[ -f results/PRODUCTION_METRICS.json ] && echo "OK" || echo "ERROR — moved"`
Expected: `OK`

If `ERROR — moved`: restore from snapshot per Task 9 rollback.

---

## Task 6: Rebuild the canonical chain post-migration

The framework scripts have been updated by the migration to read/write the new paths. Verify the chain still works end-to-end.

**Files:**
- Run: `scripts/build_metrics.py` (writes `results/PRODUCTION_METRICS.json` and `results/reports/METRICS_DIFF.md`)
- Run: `scripts/build_canonical_macros.py` (writes `latex/canonical_macros.tex`)
- Run: `scripts/build_thesis_tables.py` (writes `tables/table_*.canonical.tex`)
- Run: `scripts/verify_thesis_consistency.py` (writes nothing; prints drift)

- [ ] **Step 6.1: Build canonical metrics from the now-organized files**

Run: `python scripts/build_metrics.py 2>&1 | tail -8`

Expected output (last lines):
```
  international           72 metrics
Total: 220+ metrics in PRODUCTION_METRICS.json
Diff vs last run: <N> metrics changed, 0 config knobs changed
Report: results/reports/METRICS_DIFF.md
```

The metric count should be ≥ 200 (the framework was at 211 metrics before the rerun completed; with fresh post-Shumway data from the in-flight scripts, it will land somewhere ≥ 200).

If the count is **0**: the parser couldn't find tables. Inspect parser paths in `scripts/build_metrics.py` (the migration should have updated them).

- [ ] **Step 6.2: Generate latex macros from the canonical store**

Run: `python scripts/build_canonical_macros.py 2>&1`

Expected output:
```
Saved: latex/canonical_macros.tex (240+ macros)
```

- [ ] **Step 6.3: Render thesis tables from canonical**

Run: `python scripts/build_thesis_tables.py 2>&1 | tail -10`

Expected output (last lines):
```
  ✓ performance       →  tables/table_performance.canonical.tex
  ✓ factor_alphas     →  tables/table_factor_alphas.canonical.tex
  ✓ regime_sharpe     →  tables/table_regime_sharpe.canonical.tex
  ✓ panic_subtypes    →  tables/table_panic_subtypes.canonical.tex
  ✓ hmm_separation    →  tables/table_hmm_separation.canonical.tex
Rendered 5 table(s) from canonical metrics.
```

- [ ] **Step 6.4: Run drift verifier and capture flag count**

Run: `python scripts/verify_thesis_consistency.py 2>&1 | tail -5`

Expected output ends with: `Thesis-consistency check: <N> matches, <M> flagged`

A small `M` (single digits) is fine — those are the prose-edit candidates already documented in `results/reports/PROSE_DRIFT_REPORT.md`. A surprisingly large `M` (> 50) means the verifier's regex over-matched after the migration; investigate before continuing.

---

## Task 7: Verify the thesis still compiles

If the migration broke any `\input` or `\includegraphics` path, `pdflatex` will fail.

**Files:**
- Read: `main.tex` (entrypoint)
- Output: `main.pdf`

- [ ] **Step 7.1: Compile the thesis**

Run: `latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex 2>&1 | tail -20`

Expected output ends with `Output written on main.pdf (NN pages, M bytes).`

If it fails with `! LaTeX Error: File 'plots/foo.png' not found` or `! LaTeX Error: File 'tables/foo.tex' not found`:
1. Run `grep -nE 'plots/foo|tables/foo' latex/*.tex main.tex` to find the broken reference
2. Either move the file to where the latex expects it, OR update the latex path
3. Re-run `latexmk`

- [ ] **Step 7.2: Verify the resulting PDF is non-trivial**

Run: `ls -la main.pdf | awk '{print $5}'`
Expected: > 1,000,000 bytes (a real ~70-page thesis PDF, not a tiny error stub).

If PDF is < 100KB: latexmk likely produced an error PDF. Inspect `main.log`.

---

## Task 8: Run the canonical-store regression smoke test

A minimal sanity check: every metric in `PRODUCTION_METRICS.json` reads cleanly with no parse errors.

**Files:**
- Test: ad-hoc python invocation (no new file)

- [ ] **Step 8.1: Verify the JSON is well-formed and has the expected metadata**

Run:
```bash
python3 -c "
import json
with open('results/PRODUCTION_METRICS.json') as f:
    d = json.load(f)
md = d.get('_metadata', {})
sections = sorted(k for k in d if not k.startswith('_'))
print(f'Sections: {len(sections)}')
print(f'Total metrics: {sum(len(d[k]) for k in sections)}')
print(f'Last updated: {md.get(\"last_updated\")}')
print(f'Config snapshot: {len(md.get(\"config_snapshot\", {}))} knobs')
"
```

Expected output (numbers may vary):
```
Sections: 12
Total metrics: 211+
Last updated: 2026-04-27T...
Config snapshot: 12 knobs
```

If the JSON is malformed or empty: `build_metrics.py` step failed silently. Re-run Step 6.1.

- [ ] **Step 8.2: Spot-check one specific metric known to be present**

Run:
```bash
python3 -c "
import json
d = json.load(open('results/PRODUCTION_METRICS.json'))
m2 = d['m2_perf']['m2_sharpe']
print(f'm2_sharpe = {m2[\"value\"]}')
nwt = d['m2_perf']['m2_nw_t']
print(f'm2_nw_t = {nwt[\"value\"]} (stars: {nwt.get(\"stars\", \"\")})')
"
```

Expected: `m2_sharpe = 1.11` and `m2_nw_t = 4.37 (stars: ***)`.

If `m2_sharpe` is 0.0 or missing: parser broke after the migration. Inspect `scripts/build_metrics.py`'s `parse_table_performance` for path/regex issues.

---

## Task 9: Commit the migration + framework refresh as one atomic commit

A single commit makes rollback clean (revert one commit, all changes reverse).

**Files:**
- Commit all the moved files (git tracks moves as renames automatically)
- Commit modified scripts and latex files
- Commit refreshed canonical artifacts (`PRODUCTION_METRICS.json`, `METRICS_DIFF.md`, `canonical_macros.tex`, `*.canonical.tex`)

- [ ] **Step 9.1: Inspect what's about to be committed**

Run: `git status --short | head -50`
Expected: many `R` (rename) lines for moved files, several `M` lines for modified scripts/latex, possibly `M` for `PRODUCTION_METRICS.json`.

- [ ] **Step 9.2: Stage everything in the relevant directories**

Run:
```bash
git add -A results plots tables scripts latex main.tex
git status --short | head -10
```

Should show all staged. If `main.pdf` shows as modified, leave it OUT of this commit (it's a build artifact, separate concern).

- [ ] **Step 9.3: Commit with a structured message**

Run:
```bash
git commit -m "$(cat <<'EOF'
restructure: production / cv / diagnostic directory split

results/thesis/, results/cv/, results/oos_historical/, results/reports/,
plots/thesis/, plots/diagnostic/ — clean separation of canonical thesis
data from CV/exploratory/diagnostic outputs.

  - 101 files moved into the new layout
  - 26 scripts auto-updated for new output paths via
    scripts/migrate_to_organized.py
  - 5 latex \includegraphics paths updated (main.tex +
    main_results, data_section, methodology, appendix)
  - results/PRODUCTION_METRICS.json stays at results/ root (path-stable
    for tools that import the canonical store)
  - tables/ stays flat (every .tex there is thesis-grade \input target)

After-migration verification:
  ✓ python scripts/build_metrics.py — re-extracts canonical from new paths
  ✓ python scripts/build_canonical_macros.py — emits latex macros
  ✓ python scripts/build_thesis_tables.py — renders 5 tables from canonical
  ✓ python scripts/verify_thesis_consistency.py — drift check
  ✓ latexmk -pdf -halt-on-error main.tex — thesis still compiles

Pre-restructure snapshot:
  baseline_pre_restructure_<timestamp>/ (rollback insurance)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)" 2>&1 | tail -3
```

Expected output: `[main <hash>] restructure: production / cv / diagnostic directory split`.

- [ ] **Step 9.4: Push to GitHub**

Run: `git push origin main 2>&1 | tail -3`
Expected: `<hash>..<new-hash>  main -> main`.

---

## Task 10: Update PROSE_DRIFT_REPORT.md (final pass)

Earlier `PROSE_DRIFT_REPORT.md` was generated against pre-rerun canonical values for some metrics (e.g., panic-subtype Sharpes were wrong because the parser had a bug). After the post-Shumway rerun + migration, regenerate it once more so it reflects the truly final state.

**Files:**
- Modify: `results/reports/PROSE_DRIFT_REPORT.md`

- [ ] **Step 10.1: Regenerate the drift report**

Run: `python scripts/verify_thesis_consistency.py > results/reports/PROSE_DRIFT_REPORT.md 2>&1`

This overwrites the file with the latest verifier output. Open and inspect:
`cat results/reports/PROSE_DRIFT_REPORT.md | head -30`

Expected: a list of `⚠ <file>:<line>` flags, each pointing to a hand-typed prose number that doesn't match the canonical store. These are the Bucket-1 prose edits to make tomorrow.

- [ ] **Step 10.2: Stage and commit only this file**

Run:
```bash
git add results/reports/PROSE_DRIFT_REPORT.md
git commit -m "docs: refresh PROSE_DRIFT_REPORT after migration

Regenerated from the post-migration canonical store. This is the final
walking checklist for tomorrow's Bucket-1 latex prose edits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin main 2>&1 | tail -3
```

Expected: `<hash>..<new-hash>  main -> main`.

---

## Rollback (if anything breaks)

If a post-migration step fails AND the cause is unclear:

```bash
# 1. Revert the migration commit
git log --oneline -5  # find the restructure commit hash
git revert <hash>     # creates a new commit reversing the move

# 2. OR restore from snapshot if git revert is messy
rm -rf results plots
mv "$SNAP/results" results
mv "$SNAP/plots"   plots
# scripts/ and latex/ paths still point to new layout — revert their commit too
git checkout HEAD~1 -- scripts/ latex/ main.tex
```

The pre-restructure snapshot at `baseline_pre_restructure_<timestamp>/` is the source of truth. The restructure was atomic (single commit), so reverting is one command.

---

## Self-Review

**Spec coverage:**
- ✅ Wait for in-flight script (Task 1) → covers `hmm_model.py`
- ✅ Tier B revive (Task 2) → covers `features_hmm.png`
- ✅ Snapshot (Task 3) → covers backup-before-destructive-move per CLAUDE.md
- ✅ Migration dry-run + execute (Tasks 4-5) → covers the 101-file move + 26-script + 5-latex path updates
- ✅ Canonical chain refresh (Task 6) → covers build_metrics, build_canonical_macros, build_thesis_tables, verify_thesis_consistency
- ✅ Latex compile verify (Task 7) → catches any broken `\input` or `\includegraphics`
- ✅ Smoke test (Task 8) → catches malformed JSON or missing critical metrics
- ✅ Atomic commit (Task 9) → clean rollback path
- ✅ Drift report final pass (Task 10) → produces tomorrow's edit checklist

**Placeholder scan:**
- No "TBD" / "fill in" / "implement later"
- No "add error handling" — concrete error paths included (Task 7 fallback, Task 9 inspection step)
- All commands are exact and runnable
- All expected outputs are concrete (specific file dates, specific metric values, specific count ranges)

**Type/path consistency:**
- `scripts/migrate_to_organized.py` is referenced consistently
- `results/PRODUCTION_METRICS.json` stays at `results/` root in every task that mentions it
- Reports paths consistently use `results/reports/` after Task 5 onwards
- Snapshot variable `$SNAP` is set in Task 3 Step 3.1, used in rollback

**One subtle issue caught + fixed:** Task 10's command originally would have written to the OLD `results/PROSE_DRIFT_REPORT.md` path. After migration that file is at `results/reports/PROSE_DRIFT_REPORT.md`. The plan now writes to the new location.
