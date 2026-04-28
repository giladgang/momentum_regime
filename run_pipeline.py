"""
run_pipeline.py
===============
Master script that runs the full momentum regime shifts pipeline.
Edit config.py to change parameters, then run this file.

Usage:
    python run_pipeline.py              # run everything
    python run_pipeline.py --step 1     # run only step 1 (HMM)
    python run_pipeline.py --step 2     # run only step 2 (cross-sectional model)
    python run_pipeline.py --step 3     # run only step 3 (main results tables)
    python run_pipeline.py --step 4     # run only step 4 (robustness checks)
    python run_pipeline.py --step 5     # run only step 5 (external validity)
    python run_pipeline.py --step 6     # run only step 6 (HMM diagnostics)
    python run_pipeline.py --step 7     # run only step 7 (new L/S analyses)
    python run_pipeline.py --step 8     # run only step 8 (plots)
"""

import subprocess
import sys
import time
import os

def run_script(script_path, description):
    """Run a Python script and report status. `script_path` may be a bare
    path or a path followed by space-separated CLI arguments (e.g.
    'scripts/foo.py --tag prod --seeds 200')."""
    parts = script_path.split()
    script = parts[0]
    script_args = parts[1:]

    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  Script: {script_path}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, '-u', script, *script_args],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  DONE: {description} ({elapsed:.0f}s)")
    else:
        print(f"\n  FAILED: {description} (exit code {result.returncode})")
        return False
    return True


def run_pytest(test_path, description):
    """Run a pytest test file and report status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  Test:   {test_path}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_path, '-v'],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  DONE: {description} ({elapsed:.0f}s)")
    else:
        print(f"\n  FAILED: {description} (exit code {result.returncode})")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run momentum regime shifts pipeline')
    parser.add_argument('--step', type=int, default=0,
                        help='Run only this step (0 = all)')
    args = parser.parse_args()

    # Import config to print current settings
    import config as cfg

    print("="*70)
    print("  MOMENTUM REGIME SHIFTS PIPELINE")
    print("="*70)
    print(f"\n  Configuration:")
    print(f"    HMM features:     {cfg.HMM_FEATURES}")
    print(f"    Portfolio type:   {cfg.PORTFOLIO_TYPE}")
    print(f"    Use fundamentals: {cfg.USE_FUNDAMENTALS}")
    print(f"    HMM seeds:        {cfg.HMM_SEEDS}")
    print(f"    XGB seeds:        {len(cfg.XGB_SEEDS)} seeds")
    print(f"    Trading fee:      {cfg.TRADING_FEE}")
    print(f"    Train end:        {cfg.TRAIN_END}")
    print(f"    K states:         {cfg.K_STATES}")
    print()

    os.makedirs(cfg.TABLES_DIR, exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.ARTEFACTS_DIR, exist_ok=True)

    # Pre-flight: verify Shumway delisting treatment has been applied IF the
    # CRSP panel exists. We only fail-fast when the file is present but
    # missing the `ret_adj` column — that's the silent-inflation scenario
    # the guard is designed to catch. If the file is absent entirely
    # (fresh checkout, CI without data, etc.) we let the individual step
    # scripts emit their own "data not found" errors so users on a clean
    # repo aren't blocked by this guard. Idempotent: re-runs of
    # apply_shumway_delisting load from the .bak_before_shumway backup.
    import pandas as pd
    if os.path.exists('data/crsp_msf_raw.parquet'):
        _crsp_head = pd.read_parquet('data/crsp_msf_raw.parquet').head(1)
        if 'ret_adj' not in _crsp_head.columns:
            print('  [ERROR] data/crsp_msf_raw.parquet is missing the `ret_adj` '
                  'column.\n'
                  '          The pipeline reads ret_adj (Shumway-corrected '
                  'returns); without it,\n'
                  '          downstream artefacts would silently inflate by '
                  'ignoring delisting losses.\n'
                  '          Run:\n'
                  '              python scripts/apply_shumway_delisting.py --force\n'
                  '          before re-running the pipeline. (Idempotent: re-runs '
                  'load from\n'
                  '          .bak_before_shumway, no double-application.)')
            sys.exit(1)
        del _crsp_head

    # Check if HMM results already exist
    try:
        _panel = pd.read_parquet(cfg.PANEL_WITH_REGIMES_PATH)
        hmm_ready = 'pi_filter' in _panel.columns and _panel['pi_filter'].notna().sum() > 100
        del _panel
    except:
        hmm_ready = False

    if hmm_ready:
        print("  HMM results found in panel_with_regimes.parquet -- skipping Step 1")
        print("  (Delete panel_with_regimes.parquet to force re-estimation)")

    # Check if 30-year expanding-window backtest results already exist
    # (canonical path is RESULTS_THESIS_DIR; legacy path probed for
    # back-compat with older runs).
    _exp_path_new = os.path.join(cfg.RESULTS_THESIS_DIR, 'expanding_returns_prod.csv')
    _exp_path_old = os.path.join(cfg.RESULTS_DIR, 'expanding_returns_prod.csv')
    _exp_path = _exp_path_new if os.path.exists(_exp_path_new) else _exp_path_old
    try:
        _exp = pd.read_csv(_exp_path)
        expanding_ready = len(_exp) >= 350  # ~30 years of monthly data
        del _exp
    except Exception:
        expanding_ready = False
    if expanding_ready:
        print("  Expanding-window backtest results found -- skipping Step 19")
        print(f"  (Delete {_exp_path} to force re-run)")
    print()

    # ── Pre-flight: fast artefact-independent tests ──────────────────────────
    # Runs in ~3s. Catches config typos, feature-list bugs, and portfolio
    # edge cases BEFORE the multi-hour HMM/XGB compute.
    # Skipped when --step is specified (user is targeting a single step).
    if args.step == 0:
        print("  Pre-flight tests (config, utils, portfolio, pipeline smoke) ...")
        preflight = [
            'tests/test_config.py',
            'tests/test_utils.py',
            'tests/test_portfolio_edge_cases.py',
            'tests/test_pipeline_smoke.py',
        ]
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', *preflight, '-q', '--tb=short'],
            text=True,
        )
        if result.returncode != 0:
            print("\n  PRE-FLIGHT FAILED. Fix the failing tests before running "
                  "the full pipeline.")
            sys.exit(1)
        print("  Pre-flight OK.\n")

    t_total = time.time()

    steps = {
        1: ('scripts/hmm_model.py',
            'STEP 1: Fit HMM and generate pi_filter'),

        2: ('scripts/cross_sectional_model.py',
            'STEP 2: Train cross-sectional models (LR + XGB) and build portfolios'),

        3: ('scripts/main_results_analysis.py',
            'STEP 3: Generate main results tables'),

        4: ('scripts/robustness_checks.py',
            'STEP 4: Run robustness checks'),

        5: ('scripts/external_validity.py',
            'STEP 5: Run external validity tests'),

        6: ('scripts/hmm_diagnostics.py',
            'STEP 6: Run HMM diagnostics (Student-t, Gelman-Rubin, separation)'),

        7: ('scripts/new_ls_analyses.py',
            'STEP 7: Run new L/S analyses (D&M, factor alphas, IC rotation, spanning)'),

        8: ('scripts/generate_plots.py',
            'STEP 8: Generate all plots'),

        9: ('scripts/stress_test.py',
            'STEP 9: Run stress test (vulnerability, VaR/CVaR, prolonged bear simulation)'),

        10: ('scripts/economic_mechanism.py',
             'STEP 10: Analyze economic mechanism (portfolio characteristics, forward returns, sector rotation)'),

        11: ('scripts/selection_rank_analysis.py',
             'STEP 11: Selection rank analysis (percentile rank of selected stocks by horizon and regime)'),

        12: ('scripts/risk_aversion_thesis_table.py',
             'STEP 12: Risk aversion analysis (all target families, 50 seeds)'),

        13: ('scripts/ridge_baseline_test.py',
             'STEP 13: Ridge regression baseline (linearity vs target confound)'),

        14: ('scripts/depth_vs_sharpe.py',
             'STEP 14: Tree depth analysis (interaction order requirements)'),

        15: ('scripts/tree_path_analysis.py',
             'STEP 15: Tree path analysis (momentum decision paths by regime and leg)'),

        16: ('scripts/tree_combo_grouped.py',
             'STEP 16: Grouped combo analysis (S/M/I/L horizon interactions by regime)'),

        # ── Auxiliary thesis-cited generators (added 2026-04-27 after a code
        # review found these scripts produce numbers cited in the thesis but
        # weren't in the pipeline, leading to silent staleness on rerun). ──

        17: ('scripts/bootstrap_analysis.py',
             'STEP 17: Block-bootstrap Sharpe CIs + paired tests + regime-split '
             '(table_bootstrap.tex + 3 bootstrap_*.csv in §5.1, App.~bootstrap)'),

        18: ('scripts/zscore_time_by_horizon.py',
             'STEP 18: Monthly z-score by horizon — long, short, longshort '
             '(3 zscore_*_by_month.csv in results/thesis/, fed into §5.2 figures)'),

        19: ('scripts/expanding_window_backtest_parallel.py '
             '--first-retrain-year 1995 --last-retrain-year 2024 '
             '--hmm-seeds 200 --xgb-seeds 50 --workers 6 --tag prod',
             'STEP 19: 30-year expanding-window OOS backtest (Sec 5.4.3, ~12-18 hr)'),

        20: ('scripts/build_table_expanding_subperiods.py',
             'STEP 20: Build expanding-window sub-period table (depends on Step 19)'),

        # Auxiliary continued — thesis-cited but separate from main_results_analysis
        # ─ FUNDAMENTALS ABLATION
        25: ('scripts/fundamentals_test.py',
             'STEP 25: Fundamentals ablation (5 variants × full metrics) — '
             'table_fund_alphas.tex, table_fundamentals_ablation.tex, '
             'table_performance_fund_row.tex, fundamentals_test_results.csv'),

        # ─ TREE / PATH / COMBO ANALYSES
        26: ('scripts/tree_combo_analysis.py',
             'STEP 26: Per-stock tree-path combo analysis '
             '(tree_combo_results.csv — §E19 data backing)'),

        27: ('scripts/verify_pi_dominance_stats.py',
             'STEP 27: π_filter tree-dominance statistics '
             '(pi_dominance_stats.csv — §E31 panic-share claims)'),

        # ─ ROBUSTNESS / BENCHMARKS
        28: ('scripts/random_forest_test.py',
             'STEP 28: Random-forest baseline (RF underperforms XGB; '
             'random_forest_results.csv — §E1 ablation backing)'),

        29: ('scripts/january_exclusion.py',
             'STEP 29: January-exclusion robustness '
             '(table_january.tex — App.~january)'),

        30: ('scripts/seed_convergence.py',
             'STEP 30: XGB ensemble seed-convergence sweep '
             '(seed_convergence.csv + plot + table_seed_convergence.tex)'),

        # ─ RISK AVERSION (CRRA + dual utility)
        31: ('scripts/risk_aversion_crra.py',
             'STEP 31: CRRA risk-aversion sweep (risk_aversion_crra_results.csv)'),

        32: ('scripts/two_model_dual_util.py',
             'STEP 32: Two-model dual-utility comparison '
             '(two_model_dual_util.csv — feeds §E32 + plot below)'),

        33: ('scripts/plot_crra_dual_util.py',
             'STEP 33: Render risk-aversion dual-utility plot '
             '(plots/thesis/risk_aversion_dual_util.pdf — depends on Step 32)'),

        # ─ Z-SCORE SUB-REGIME DECOMPOSITION (App. zscore_subregime)
        34: ('scripts/zscore_subregime_analysis.py',
             'STEP 34: K-means sub-regime decomposition of panic months '
             '(table_subregime_shap.tex + zscore_subregimes.{png,pdf} + '
             'zscore_subregime_summary.csv + robustness sidecar). '
             'Depends on Step 18 (zscore_*_by_month) + Step 25 (fundamentals_returns)'),

        35: ('scripts/zscore_subregime_xgb_seed_robustness.py',
             'STEP 35: Sub-regime clustering robustness to XGB seed choice. '
             'Trains a disjoint 50-seed XGB ensemble and re-clusters; reports ARI '
             'vs production clustering (zscore_subregime_xgb_seed_robustness.csv). '
             'Heavy step (~5 min, 50 XGB fits); supports the §5.2.3 robustness claim'),

        # ─ Canonical chain (must run AFTER all generators above).
        # Renumbered 21-24 -> 80-83 so sorted(steps.keys()) places them
        # strictly after the auxiliary generators (steps 25-33). Earlier
        # numbering ran them in slots 21-24 which sorted BEFORE 25-33,
        # so PRODUCTION_METRICS.json lagged by one full pipeline run.
        80: ('scripts/build_metrics.py',
             'STEP 80: Extract canonical metrics from tables/+results/ → results/PRODUCTION_METRICS.json '
             '(also writes results/METRICS_DIFF.md showing what changed since last run)'),

        81: ('scripts/build_canonical_macros.py',
             'STEP 81: Emit latex/canonical_macros.tex from PRODUCTION_METRICS.json '
             '(\\newcommand per metric for use in thesis prose)'),

        82: ('scripts/build_thesis_tables.py',
             'STEP 82: Render headline thesis tables (table_performance, table_factor_alphas, '
             'table_regime_sharpe, table_panic_subtypes, table_hmm_separation) directly from '
             'PRODUCTION_METRICS.json — single source of truth. Writes .canonical.tex siblings '
             'until rename'),

        83: ('scripts/verify_thesis_consistency.py',
             'STEP 83: Flag drift between latex prose and PRODUCTION_METRICS.json '
             '(soft check; non-blocking — surfaces hand-typed numbers that need updating)'),

        95: ('tests/test_config.py',
             'STEP 95: Config sanity checks (dates, features, hyperparameters)'),

        96: ('tests/test_utils.py',
             'STEP 96: Unit tests for src/utils.py (metrics, portfolio, loaders)'),

        97: ('tests/test_portfolio_edge_cases.py',
             'STEP 97: Portfolio construction edge cases (NaN, zero ME, regime transitions)'),

        98: ('tests/test_pipeline_smoke.py',
             'STEP 98: Pipeline orchestration smoke test (imports, steps dict, CLI)'),

        99: ('tests/test_thesis_consistency.py',
             'STEP 99: Run thesis consistency tests (tables, figures, cross-refs)'),

        100: ('tests/test_pipeline_technical.py',
              'STEP 100: Run technical pipeline tests (data integrity, numbers, connections)'),

        101: ('tests/test_cross_sectional_lookahead.py',
              'STEP 101: Cross-sectional model lookahead/causality audit'),

        102: ('tests/test_reproducibility.py',
              'STEP 102: Reproducibility and determinism guards (artefact regression)'),
    }

    # Steps that should be run via pytest instead of plain python
    pytest_steps = {95, 96, 97, 98, 99, 100, 101, 102}

    if args.step > 0:
        if args.step in steps:
            script, desc = steps[args.step]
            if args.step in pytest_steps:
                run_pytest(script, desc)
            else:
                run_script(script, desc)
        else:
            print(f"  Unknown step {args.step}. Valid steps: {sorted(steps.keys())}")
    else:
        # Run all steps in order
        for step_num in sorted(steps.keys()):
            # Skip HMM if results already exist
            if step_num == 1 and hmm_ready:
                print(f"\n  SKIPPING Step 1 (HMM already estimated)")
                continue
            # Skip 30-year expanding-window backtest if results exist
            if step_num == 19 and expanding_ready:
                print(f"\n  SKIPPING Step 19 (expanding-window backtest results exist)")
                continue
            script, desc = steps[step_num]
            # Path may be 'scripts/foo.py' or 'scripts/foo.py --arg val'; check
            # only the script-file portion exists.
            script_file = script.split()[0]
            if not os.path.exists(script_file):
                print(f"\n  SKIPPING: {script_file} (file not found)")
                continue
            if step_num in pytest_steps:
                success = run_pytest(script, desc)
            else:
                success = run_script(script, desc)
            if not success:
                print(f"\n  Pipeline stopped at step {step_num}.")
                print(f"  Fix the error and rerun with: python run_pipeline.py --step {step_num}")
                sys.exit(1)

    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE ({elapsed_total/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
