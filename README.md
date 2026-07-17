# Momentum and Regime Shifts

Empirical-finance thesis project. A 2-state Bayesian Hidden Markov Model identifies "calm" and "panic" market regimes from a panel of monthly US equity-market features; a regime-conditioned cross-sectional model then scores stocks for a long–short momentum portfolio.

**The thesis itself is in this repo: [main.pdf](main.pdf).** Combining an HMM regime signal with an XGBoost cross-sectional model is, to the author's knowledge, novel, and extracts a stronger momentum signal than conventional regime indicators such as market drawdown.

## Headline result

Models are estimated on 1990–2010 US equities and evaluated out-of-sample on the 167-month 2011–2024 window. Long–short, NYSE decile breakpoints, value-weighted, 10 bps one-way costs.

| Strategy | Ann. return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| US market | 11.9% | 0.84 | -24.7% |
| Fixed 12-month momentum | -1.9% | 0.06 | -72.3% |
| **HMM regime + XGBoost (M2)** | **21.9%** | **1.11** | **-24.8%** |

M2 earns a **24.7% annual Fama–French 6-factor alpha (t = 4.78)** — the return is not explained by market, size, value, momentum, profitability, or investment exposure. It is Sharpe-positive in all three sub-periods (0.62 / 1.03 / 1.69), its bootstrap 95% CI on Sharpe is [0.66, 1.54] (12-month block, 10,000 resamples), and it beats every benchmark on a paired test at p < 0.05. It also sidesteps the classic momentum crash: -24.8% max drawdown against -72.3% for fixed 12-month momentum.

**The caveat, stated up front.** The edge concentrates in panic regimes (Sharpe 1.56, n=59) over calm ones (0.82, n=108) — but that panic Sharpe comes *entirely from recovery months*. The 41 panic months where the market rose returned +64.4% annualised; the 18 genuine crash months returned -7.2%. The panic-minus-calm Sharpe difference is only marginally significant (p = 0.074). The model is better read as a rebound-timer than a crash-hedge.

Every number above traces to a script and a CSV in [RESULTS_LOG.md](RESULTS_LOG.md).

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
- [TODO.md](TODO.md) — known limitations, plus the live research/engineering backlog. The full historical thesis-edit record lives in [archive/THESIS_TODO_ARCHIVE.md](archive/THESIS_TODO_ARCHIVE.md)
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

Code is released under the MIT License — see [LICENSE](LICENSE). You are free to use, modify, and redistribute it, with attribution and without warranty.

This covers the code only. The thesis manuscript ([main.pdf](main.pdf), `latex/`, `main.tex`) remains © 2026 Gilad Gang, all rights reserved; please cite it rather than reproduce it. The underlying CRSP, Compustat, and Fama–French data are licensed from their respective providers and are not redistributed here.
