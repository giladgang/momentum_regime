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
    """Run a Python script and report status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  Script: {script_path}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, '-u', script_path],
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

    # Check if HMM results already exist
    import pandas as pd
    try:
        _panel = pd.read_parquet(cfg.PANEL_WITH_REGIMES_PATH)
        hmm_ready = 'pi_filter' in _panel.columns and _panel['pi_filter'].notna().sum() > 100
        del _panel
    except:
        hmm_ready = False

    if hmm_ready:
        print("  HMM results found in panel_with_regimes.parquet -- skipping Step 1")
        print("  (Delete panel_with_regimes.parquet to force re-estimation)")
    print()

    t_total = time.time()

    steps = {
        1: ('scripts/hmm_model.py',
            'STEP 1: Fit HMM and generate pi_filter'),

        2: ('scripts/cross_sectional_model.py',
            'STEP 2: Train cross-sectional models (LR + XGB) and build portfolios'),

        3: ('scripts/main_results_analysis.py',
            'STEP 3: Generate main results tables'),

        4: ('tests/robustness_checks.py',
            'STEP 4: Run robustness checks'),

        5: ('tests/test_external_validity.py',
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
    }

    if args.step > 0:
        if args.step in steps:
            script, desc = steps[args.step]
            run_script(script, desc)
        else:
            print(f"  Unknown step {args.step}. Valid steps: 1-8")
    else:
        # Run all steps in order
        for step_num in sorted(steps.keys()):
            # Skip HMM if results already exist
            if step_num == 1 and hmm_ready:
                print(f"\n  SKIPPING Step 1 (HMM already estimated)")
                continue
            script, desc = steps[step_num]
            if not os.path.exists(script):
                print(f"\n  SKIPPING: {script} (file not found)")
                continue
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
