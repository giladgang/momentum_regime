"""
test_parallel_hmm_determinism.py
================================
Verifies that the parallel HMM seed loop in expanding_window_backtest_parallel.py
produces identical (or numerically equivalent within tolerance) output to the
serial seed loop in expanding_window_backtest.py.

If this test passes, it is safe to resume a serial production run using the
parallel script: the resulting expanding_returns_<tag>.csv will be coherent
across the serial-to-parallel switch.

If it fails, the parallel script must not be mixed into the same output file
as serial results (would have to re-run from scratch).

Run as:
    python -m pytest tests/test_parallel_hmm_determinism.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
os.chdir(REPO)

from config import HMM_FEATURES  # noqa: E402
from historical_oos_production import fit_hmm  # noqa: E402
from expanding_window_backtest_parallel import fit_hmm_parallel  # noqa: E402
from expanding_window_backtest import crisis_mask_for_training  # noqa: E402


def _fit_one_short(seed, Z_train, Z_full, crisis_mask):
    """Module-level worker for the short-iter parallel test (picklable)."""
    return fit_hmm(Z_train, Z_full, seed=seed, crisis_mask=crisis_mask,
                   n_iter=200, n_burnin=50)


@pytest.fixture(scope='module')
def training_setup():
    """Common training data for the determinism test. Uses a recent training
    window with both dot-com and GFC in the crisis mask, so we exercise the
    data-driven sign-correction path (the most numerically delicate)."""
    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    sub = (panel[['date'] + HMM_FEATURES]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date')
           .reset_index(drop=True))

    train_end = pd.Timestamp('2010-01-01')
    train_mask = sub['date'] < train_end
    Z_train = sub.loc[train_mask, HMM_FEATURES].values.astype(float)
    Z_full = sub[HMM_FEATURES].values.astype(float)

    train_dates = sub.loc[train_mask, 'date'].values
    crisis_mask = np.zeros(len(train_dates), dtype=bool)
    for s, e in [('1990-07-01', '1991-03-31'),
                 ('1998-07-01', '1998-10-31'),
                 ('2000-03-01', '2002-10-31'),
                 ('2007-10-01', '2009-06-30')]:
        crisis_mask |= ((train_dates >= np.datetime64(s))
                        & (train_dates <= np.datetime64(e)))

    return Z_train, Z_full, crisis_mask


def test_serial_fit_hmm_is_deterministic(training_setup):
    """Calling fit_hmm twice with the same seed must return identical arrays.
    If this fails, deterministic is broken at the source — nothing else makes
    sense."""
    Z_train, Z_full, crisis_mask = training_setup
    pi_a = fit_hmm(Z_train, Z_full, seed=1, crisis_mask=crisis_mask,
                   n_iter=200, n_burnin=50)
    pi_b = fit_hmm(Z_train, Z_full, seed=1, crisis_mask=crisis_mask,
                   n_iter=200, n_burnin=50)
    np.testing.assert_array_equal(pi_a, pi_b)


def test_parallel_matches_serial_single_seed(training_setup):
    """A pool with one seed averaged should give the same result as a serial
    fit_hmm call."""
    Z_train, Z_full, crisis_mask = training_setup
    pi_serial = fit_hmm(Z_train, Z_full, seed=42, crisis_mask=crisis_mask,
                        n_iter=200, n_burnin=50)

    from multiprocessing import Pool
    from functools import partial

    f = partial(_fit_one_short, Z_train=Z_train, Z_full=Z_full,
                crisis_mask=crisis_mask)
    with Pool(processes=2) as pool:
        results = pool.map(f, [42])
    pi_parallel = np.mean(results, axis=0)

    diff = np.abs(pi_serial - pi_parallel).max()
    print(f"\n  Max abs diff (single seed serial vs parallel): {diff:.2e}")
    assert diff < 1e-10, \
        f"Parallel single-seed diverged from serial by {diff:.2e} > 1e-10"


def test_parallel_matches_serial_multi_seed(training_setup):
    """Average over multiple seeds: serial and parallel should give identical
    arrays (because each seed is deterministic and averaging is associative)."""
    Z_train, Z_full, crisis_mask = training_setup
    seeds = [1, 7, 13, 31, 57]

    pi_serial = np.zeros(len(Z_full))
    for s in seeds:
        pi_serial += fit_hmm(Z_train, Z_full, seed=s, crisis_mask=crisis_mask,
                             n_iter=200, n_burnin=50)
    pi_serial /= len(seeds)

    from multiprocessing import Pool
    from functools import partial

    f = partial(_fit_one_short, Z_train=Z_train, Z_full=Z_full,
                crisis_mask=crisis_mask)
    with Pool(processes=4) as pool:
        results = pool.map(f, seeds)
    pi_parallel = np.mean(results, axis=0)

    diff = np.abs(pi_serial - pi_parallel).max()
    print(f"\n  Max abs diff (5-seed avg serial vs parallel): {diff:.2e}")
    assert diff < 1e-8, \
        f"Parallel 5-seed avg diverged from serial by {diff:.2e} > 1e-8"


def test_parallel_matches_serial_production_iterations(training_setup):
    """Same as above but at production iteration count (2000/500). This is the
    test that matters for whether the parallel script can substitute for the
    serial one in the live production output. Slow — uses 5 seeds at full
    iterations."""
    Z_train, Z_full, crisis_mask = training_setup
    seeds = [1, 2, 3, 4, 5]

    pi_serial = np.zeros(len(Z_full))
    for s in seeds:
        pi_serial += fit_hmm(Z_train, Z_full, seed=s, crisis_mask=crisis_mask)
    pi_serial /= len(seeds)

    pi_parallel = fit_hmm_parallel(Z_train, Z_full, seeds, crisis_mask,
                                    n_workers=4)

    diff = np.abs(pi_serial - pi_parallel).max()
    print(f"\n  Max abs diff (production iterations, 5 seeds): {diff:.2e}")
    assert diff < 1e-8, \
        f"Parallel production-iter diverged from serial by {diff:.2e} > 1e-8"


def test_parallel_matches_existing_production_year():
    """Gold-standard test: pick a year that the live production run has already
    computed and saved, re-run that year's HMM with the parallel script at
    full production iterations and 200 seeds, and compare to the saved
    pi_filter for the predicted year. If they match within tolerance, the
    parallel script is a drop-in replacement for the serial one and can resume
    the production output without breaking continuity.

    We pick the most recent fully-completed predict-year so we exercise the
    largest training window (most numerically delicate). The check excludes
    the currently-running year (which won't be in the saved output yet)."""
    # Probe both new (post-restructure) and legacy paths so the test follows
    # whichever location the live production run wrote to.
    pi_path_new = os.path.join(REPO, 'results', 'thesis', 'expanding_pi_filter_prod.csv')
    pi_path_old = os.path.join(REPO, 'results', 'expanding_pi_filter_prod.csv')
    if os.path.exists(pi_path_new):
        pi_path = pi_path_new
    elif os.path.exists(pi_path_old):
        pi_path = pi_path_old
    else:
        pytest.skip("No expanding_pi_filter_prod.csv yet; production must "
                    "have completed at least one year.")

    saved = pd.read_csv(pi_path, parse_dates=['date']).set_index('date')
    if len(saved) == 0:
        pytest.skip("expanding_pi_filter_prod.csv is empty.")

    # Pick the most recent year fully present in saved output. We'll take the
    # max year and check it has 12 months (a complete predict-year). If not,
    # back off to the previous year.
    saved['year'] = saved.index.year
    completed = saved.groupby('year').size()
    full_years = completed[completed == 12].index.tolist()
    if not full_years:
        pytest.skip("No fully-saved predict-years in production output yet.")
    target_year = max(full_years)

    print(f"\n  Validating against production year {target_year} ...")

    # Reproduce the training setup the production run used for that year
    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    sub = (panel[['date'] + HMM_FEATURES]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date')
           .reset_index(drop=True))
    sub = sub[sub['date'] >= '1990-01-01'].reset_index(drop=True)

    train_end = pd.Timestamp(f'{target_year}-01-01')
    test_end = pd.Timestamp(f'{target_year + 1}-01-01')
    train_mask_dates = sub['date'] < train_end
    Z_train = sub.loc[train_mask_dates, HMM_FEATURES].values.astype(float)
    Z_full = sub[HMM_FEATURES].values.astype(float)
    crisis_mask = crisis_mask_for_training(sub.loc[train_mask_dates, 'date'].values)

    print(f"    Training window: {len(Z_train)} months, "
          f"crisis mask = {crisis_mask.sum()} months")

    # Run parallel HMM with the SAME 200-seed set the production run used
    seeds = list(range(1, 201))
    print(f"    Running parallel fit_hmm with {len(seeds)} seeds, 8 workers...")
    pi_parallel = fit_hmm_parallel(Z_train, Z_full, seeds, crisis_mask,
                                   n_workers=8)

    # Slice to predicted-year months and compare to saved
    pred_mask = (sub['date'] >= train_end) & (sub['date'] < test_end)
    pi_parallel_pred = pi_parallel[pred_mask.values]
    pred_dates = sub.loc[pred_mask, 'date'].values

    # Match saved values by date
    saved_year = saved[saved['year'] == target_year].drop(columns='year')
    saved_year_aligned = saved_year.reindex(pd.to_datetime(pred_dates),
                                            method='nearest')
    pi_saved_pred = saved_year_aligned['pi_filter'].values

    diff = np.abs(pi_parallel_pred - pi_saved_pred).max()
    mean_diff = np.abs(pi_parallel_pred - pi_saved_pred).mean()
    print(f"    Saved pi range:    [{pi_saved_pred.min():.4f}, {pi_saved_pred.max():.4f}]")
    print(f"    Parallel pi range: [{pi_parallel_pred.min():.4f}, {pi_parallel_pred.max():.4f}]")
    print(f"    Max abs diff:      {diff:.2e}")
    print(f"    Mean abs diff:     {mean_diff:.2e}")

    # Tolerance: 1e-6 is strict enough to catch any substantive difference
    # while allowing minor BLAS-thread numerical drift between the original
    # serial run and the parallel reproduction.
    assert diff < 1e-6, (
        f"Parallel reproduction of production year {target_year} diverged by "
        f"{diff:.2e} (> 1e-6). The parallel script cannot safely resume the "
        f"existing production output."
    )
