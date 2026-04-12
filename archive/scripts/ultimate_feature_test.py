"""
ultimate_feature_test.py
========================
Ultimate validation for the selected HMM feature combination.
Orchestrates existing scripts -- does not re-implement their logic.

Steps:
  0. HMM Quality Gates    → runs hmm_model.py (separation, crisis alignment, ESS)
  1. Heavy Ensemble        → 20 HMM seeds + 50 XGB seeds averaged
  2. Sub-period Consistency → Sharpe for 2011-15, 2016-20, 2021-25
  3. Stability Test        → 20 independent experiments (random 3 HMM + 1 XGB)
  4. Leave-one-year-out    → drop each year, check Sharpe stays positive
  5. Factor Model Alphas   → CAPM through FF5+Mom

Usage:
    python scripts/ultimate_feature_test.py              # run all steps
    python scripts/ultimate_feature_test.py --skip-hmm   # skip step 0 (if already validated)

Reads HMM_FEATURES from config.py.
"""

import subprocess
import sys
import os
import time
import argparse

def run_step(description, script_path=None, command=None):
    """Run a step and report status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")

    t0 = time.time()
    if script_path:
        result = subprocess.run(
            [sys.executable, '-u', script_path],
            capture_output=False, text=True
        )
    elif command:
        result = subprocess.run(
            command, shell=True, capture_output=False, text=True
        )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  DONE ({elapsed:.0f}s)")
        return True
    else:
        print(f"\n  FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description='Ultimate Feature Validation')
    parser.add_argument('--skip-hmm', action='store_true',
                        help='Skip step 0 (HMM quality gates)')
    args = parser.parse_args()

    # Import config to show what we're testing
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import HMM_FEATURES

    print("=" * 70)
    print(f"  ULTIMATE FEATURE VALIDATION")
    print(f"  Testing: {HMM_FEATURES}")
    print("=" * 70)

    t_total = time.time()

    # Step 0: HMM Quality Gates
    if not args.skip_hmm:
        print("\n" + "=" * 70)
        print("  STEP 0: HMM Quality Gates (hmm_model.py)")
        print("  Checks: regime separation, crisis alignment, MCMC convergence")
        print("=" * 70)

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, '-u', 'scripts/hmm_model.py'],
            capture_output=True, text=True
        )
        print(result.stdout)
        elapsed = time.time() - t0

        # Check if all 3 validation checks passed
        if '3/3 passed' in result.stdout:
            print(f"\n  ALL QUALITY GATES PASSED ({elapsed:.0f}s)")
        else:
            # Count how many passed
            pass_count = result.stdout.count('PASS')
            fail_count = result.stdout.count('FAIL')
            print(f"\n  WARNING: {pass_count} passed, {fail_count} failed ({elapsed:.0f}s)")
            print("  Review output above. Continuing with remaining tests ...")
            print("  (Quality gate failures should be addressed before using in thesis)")

    # Step 1-5: Performance and robustness tests
    # These are in a single inline script to avoid file management issues
    print("\n" + "=" * 70)
    print("  STEPS 1-5: Performance & Robustness")
    print("=" * 70)

    run_step(
        "Running performance and robustness tests ...",
        command=f'{sys.executable} -u scripts/run_ultimate_tests.py'
    )

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  ULTIMATE VALIDATION COMPLETE ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
