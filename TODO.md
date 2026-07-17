# TODO

The thesis is complete — see [main.pdf](main.pdf). What follows is the live backlog.

The full thesis-edit record (560 lines: ~207 triaged edits, per-table verification status, citation audit, and the Phase 0-3 findings) is preserved in [`archive/THESIS_TODO_ARCHIVE.md`](archive/THESIS_TODO_ARCHIVE.md). It is a historical record, not an active list.

## Known limitations

Stated here because they are the first things a careful reader should ask about. All are acknowledged in the thesis itself.

- **Feature selection was tuned on the test period.** The 4-pass HMM feature selection ran against the evaluation window. The thesis carries an explicit caveat and flags CV-based selection as future work; the CV artefacts exist (`results/cv/`) but are deliberately not cited.
- **The panic edge is marginal.** The panic-minus-calm Sharpe difference is significant only at p = 0.074, and the panic Sharpe comes entirely from recovery months (+64.4% annualised, n=41), not crash months (-7.2%, n=18). See `RESULTS_LOG.md` §1 and §2.
- **The test window contains no structurally novel regime.** 2011-2024 has no dot-com analogue. The 30-year backtest shows a -64.8% drawdown over 32 months in 2000-2002.
- **Hyperparameters are fixed, not tuned.** The thesis uses a fixed production specification; sensitivity is reported but not optimised over.

## Research

- [ ] Spanning regressions (M2 ~ M1 and M1 ~ M2) — no dedicated CSV; needs a rerun of `scripts/new_ls_analyses.py` to re-confirm post-Shumway
- [ ] Student-t HMM (`tables/table_student_t_hmm.tex`) and Gelman-Rubin (`tables/table_gelman_rubin.tex`) — not separately verified
- [ ] Full cell-by-cell verification of appendix tables against source CSVs (high-risk cells were spot-checked; ~40 tables not exhaustively audited)

## Engineering

- [ ] Hook `verify_thesis_consistency.py` into CI to fail PRs that introduce prose-vs-canonical drift
- [ ] Add a `pytest.skip` guard to `tests/test_utils.py::TestDataLoaders` so the class can run in [.github/workflows/tests.yml](.github/workflows/tests.yml)
- [ ] Tighten the regex patterns in `scripts/verify_thesis_consistency.py` — it occasionally matches a nearby unrelated number
- [ ] Adopt `\mmperfm2sharpe`-style macros for headline numbers in `latex/main_results.tex`, so a numeric update becomes a one-line rerun

## Architecture (deferred, post-thesis)

Ordered by leverage. None are blocking; the pipeline works as-is.

- [ ] **Subdivide `scripts/`** into `hmm/`, `cs/`, `backtest/`, `analysis/`, `plots/`, `intl/`. Pure file moves plus subprocess path updates in `run_pipeline.py`.
- [ ] **Move HMM/CS/portfolio logic into `src/`** as importable modules, reducing `scripts/*.py` to thin CLIs. Biggest single win for testability.
- [ ] **Replace subprocess orchestration with Snakemake.** Eliminates the silent-staleness class of bug (see the 2026-04-27 incident in `CLAUDE.md`); gives an auto-generated DAG and rule-level parallelism.
- [ ] **Split the 943 MB `cs_artefacts_data.pkl`** into parquets (`scores_{train,test}.parquet`, `shap_values.parquet`) plus XGBoost native `.json`. Would also let CI run the data-dependent tests.
- [ ] **Split `scripts/main_results_analysis.py`** (1092 lines) into `performance.py`, `factor_alphas.py`, `ic.py`, `granger.py`, `shap.py`.

## Companion documents

- [`RESULTS_LOG.md`](RESULTS_LOG.md) — table-by-table provenance; the canonical "where does this number come from?"
- [`NEXT_RUN_PLAN.md`](NEXT_RUN_PLAN.md) — resumable execution runbook
- [`INTL_VALIDATION_PLAN.md`](INTL_VALIDATION_PLAN.md) — UK + Japan replication plan
- [`archive/THESIS_TODO_ARCHIVE.md`](archive/THESIS_TODO_ARCHIVE.md) — the full historical thesis-edit record
