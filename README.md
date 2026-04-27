# Momentum and Regime Shifts

Empirical-finance thesis project. A 2-state Bayesian Hidden Markov Model identifies "calm" and "panic" market regimes from a panel of monthly US equity-market features; a regime-conditioned cross-sectional model then scores stocks for a long–short momentum portfolio.

## What this repo contains

The full reproducible pipeline behind the thesis: data loading, HMM regime classifier, cross-sectional momentum model (logistic regression + XGBoost ensemble), backtested long–short portfolio, robustness checks, factor regressions, plots, and the LaTeX manuscript.

The headline performance numbers, factor alphas, sub-period stability, and ablation results all live in [RESULTS_LOG.md](RESULTS_LOG.md), which traces every published table and figure back to the script that produces it.

## Repository structure

```
momentum_regime/
├── run_pipeline.py        # master orchestrator (~102 ordered steps)
├── config.py              # central parameter store (seeds, features, dates, fees)
├── main.tex               # thesis entry point
├── requirements.txt       # human-curated direct dependencies
├── requirements.lock      # full transitive snapshot of the .venv used to produce results
│
├── src/                   # shared library code
│   └── utils.py           # portfolio construction, metrics, data loaders
│
├── scripts/               # analysis & pipeline stages (~85 files)
│   ├── hmm_model.py                       # Bayesian Gibbs-sampler HMM
│   ├── cross_sectional_model.py           # logistic + XGBoost ensemble
│   ├── main_results_analysis.py           # generates LaTeX tables
│   ├── generate_plots.py                  # generates figures
│   ├── expanding_window_backtest_parallel.py  # 30-year OOS backtest
│   ├── hmm_intl.py / cross_sectional_intl.py  # international (UK/JP) variants
│   └── ...
│
├── tests/                 # pytest suite (smoke, lookahead, reproducibility, parallel determinism)
├── experiments/           # dated, throwaway one-off scripts (YYYY-MM-DD-*.py)
├── archive/               # frozen older analyses kept for reproducibility
│
├── data/                  # raw + processed parquets (gitignored)
├── artefacts/             # serialised model outputs (gitignored)
├── results/               # pipeline output CSVs (gitignored)
├── plots/                 # rendered figures
├── tables/                # LaTeX table fragments consumed by the thesis
├── latex/                 # thesis source files (sections + bibliography)
└── .github/workflows/     # CI: pytest on every push, LaTeX compile on every push
```

## Installation

Tested on Python 3.12 (macOS, Ubuntu 22.04 CI).

```bash
git clone https://github.com/giladgang/momentum_regime.git
cd momentum_regime

python3.12 -m venv .venv
source .venv/bin/activate

# Strict reproduction of the env used to generate thesis numbers:
pip install -r requirements.lock

# Or, looser direct-dependency install (versions may drift):
pip install -r requirements.txt
```

Raw CRSP and Fama–French data are not in the repo (licensing); see [data/INTL_PANEL_README.md](data/INTL_PANEL_README.md) for the WRDS pulls used for the international panels and the equivalent path for the US panel.

## Reproducing the results

Run the full pipeline:

```bash
python run_pipeline.py
```

Estimated runtime ~3–5 hours on a modern laptop (the 30-year expanding-window backtest is parallelised across CPU cores; it dominates wall time).

Run a single stage:

```bash
python run_pipeline.py --step 1     # HMM fit
python run_pipeline.py --step 2     # cross-sectional model
python run_pipeline.py --step 3     # main-results tables
python run_pipeline.py --step 8     # plots
python run_pipeline.py --step 19    # 30-year expanding-window backtest
```

Pre-flight smoke tests run automatically before the heavy compute starts (when no `--step` is specified) — they catch config/import errors in seconds.

The full step list is documented inline in [run_pipeline.py](run_pipeline.py) and the resumable execution queue lives in [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md).

## Provenance / map

Every published table and figure is traceable to its source script + output CSV:

- [RESULTS_LOG.md](RESULTS_LOG.md) — table-by-table provenance log; the canonical reference for "where does this number come from?"
- [TODO.md](TODO.md) — current backlog (research, engineering, thesis edits) and granular thesis-edit detail
- [NEXT_RUN_PLAN.md](NEXT_RUN_PLAN.md) — active execution runbook (resumable across sessions)
- [INTL_VALIDATION_PLAN.md](INTL_VALIDATION_PLAN.md) — UK + Japan replication plan

## Tests

```bash
pytest tests/ -v
```

Coverage by stage:
- **HMM:** parallel-determinism, regime-signal ablation, expanding-window correctness
- **Cross-sectional:** lookahead audit, CV-helper unit tests, reproducibility regression
- **Portfolio:** edge cases (NaN handling, zero market cap, regime transitions), metric formulas
- **Pipeline:** orchestration smoke (imports, step dict, CLI), artefact existence
- **Outputs:** thesis consistency (table values match published manuscript), external validity

The fast/artefact-independent subset runs on every push via [.github/workflows/tests.yml](.github/workflows/tests.yml). Data-dependent tests skip cleanly via `conftest.py` fixtures when the 943 MB artefacts pickle isn't present (e.g. on CI).

The LaTeX build also runs on every push via [.github/workflows/latex.yml](.github/workflows/latex.yml); the compiled PDF is saved as a workflow artifact.

## Conventions

- **`scripts/` vs `experiments/`** — `scripts/` is reviewed, tested, and referenced from `run_pipeline.py`. `experiments/` is throwaway code (dated `YYYY-MM-DD-*.py`); promotion to `scripts/` is a deliberate step.
- **No emojis or em-dashes in `latex/*.tex`** — thesis convention.
- **Random seeds are explicit** — `HMM_SEEDS = 200`, `XGB_SEEDS = 50` in [config.py](config.py). No global `np.random.seed()` calls.

## Citation

```bibtex
@mastersthesis{gilad_thesis,
  author  = {Gilad Gang},
  title   = {Momentum and Regime Shifts},
  school  = {Tilburg University},
  year    = {2026},
  note    = {See https://github.com/giladgang/momentum_regime for the reproducible pipeline.}
}
```

## License

Currently unlicensed — all rights reserved. If you would like to reuse any of the code or methods, please get in touch.
