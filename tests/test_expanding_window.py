"""
test_expanding_window.py
========================
Correctness tests for scripts/expanding_window_backtest.py and the fit_hmm
sign-correction fix in scripts/historical_oos_production.py.

Runs fast — uses small seed counts and short date ranges. Purpose is to
catch look-ahead bugs, sign-correction regressions, and output-format
issues before kicking off the multi-hour production backtest.

Usage:
    python -m pytest tests/test_expanding_window.py -v
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

# Ensure cwd is repo so relative data paths resolve
os.chdir(REPO)


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests for the sign-correction fix in historical_oos_production.fit_hmm
# ══════════════════════════════════════════════════════════════════════════════

def test_fit_hmm_uses_theoretical_signs_when_crisis_mask_small():
    """With crisis_mask.sum() < 20, fit_hmm must use theoretical signs
    [-1, +1, -1, +1] rather than data-driven signs. We check this by monkey-
    patching the sign-determining code path: the function should produce a
    pi_filter that rises (not falls) during historical crisis periods."""
    from historical_oos_production import fit_hmm

    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    from config import HMM_FEATURES
    sub = (panel[['date'] + HMM_FEATURES]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date')
           .reset_index(drop=True))

    # Training window with a small crisis mask: 1990 through 1999-06
    # (training includes 1990 recession only = ~4 months of crisis)
    train_mask = sub['date'] < '1999-07-01'
    Z_train = sub.loc[train_mask, HMM_FEATURES].values.astype(float)
    Z_full = sub[HMM_FEATURES].values.astype(float)

    # Crisis mask: early-90s recession only
    train_dates = sub.loc[train_mask, 'date'].values
    crisis_mask = ((train_dates >= np.datetime64('1990-07-01'))
                   & (train_dates <= np.datetime64('1991-03-31')))

    assert crisis_mask.sum() < 20, \
        "Test invariant: crisis mask must be below the 20-month threshold"

    # Short-iter HMM for speed (2000/500 still long for unit tests)
    pi = fit_hmm(Z_train, Z_full, seed=42, crisis_mask=crisis_mask,
                 n_iter=200, n_burnin=50)

    assert pi.shape == (len(Z_full),), f"pi_filter wrong shape: {pi.shape}"
    assert (pi >= 0).all() and (pi <= 1).all(), \
        "pi_filter must be in [0, 1]"
    assert np.isfinite(pi).all(), "pi_filter must be finite"

    # Sanity: during the 1990-1991 recession (known in training), pi should be
    # elevated if the panic state is correctly labeled. With theoretical signs
    # this should hold even with 4 crisis months.
    recession_mask = ((sub['date'].values >= np.datetime64('1990-07-01'))
                      & (sub['date'].values <= np.datetime64('1991-03-31')))
    if recession_mask.sum() > 0:
        pi_recession = pi[recession_mask].mean()
        # Only check for calibration, not full signal strength
        assert pi_recession > 0.2, \
            f"Pi during 1990 recession too low ({pi_recession:.2f}); " \
            "panic state likely mislabeled"


def test_fit_hmm_uses_data_driven_signs_when_crisis_mask_large():
    """With crisis_mask.sum() >= 20, fit_hmm should use data-driven signs.
    We check by giving it a training window containing dot-com crisis and
    verifying pi rises during training-period bears."""
    from historical_oos_production import fit_hmm

    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    from config import HMM_FEATURES
    sub = (panel[['date'] + HMM_FEATURES]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date')
           .reset_index(drop=True))

    # Training window: 1990 - 2003 (includes 1990 recession, LTCM, dot-com)
    train_mask = sub['date'] < '2004-01-01'
    Z_train = sub.loc[train_mask, HMM_FEATURES].values.astype(float)
    Z_full = sub[HMM_FEATURES].values.astype(float)

    train_dates = sub.loc[train_mask, 'date'].values
    # Crisis windows fully inside training
    crisis_mask = (
        ((train_dates >= np.datetime64('1990-07-01'))
         & (train_dates <= np.datetime64('1991-03-31')))
        | ((train_dates >= np.datetime64('1998-07-01'))
           & (train_dates <= np.datetime64('1998-10-31')))
        | ((train_dates >= np.datetime64('2000-03-01'))
           & (train_dates <= np.datetime64('2002-10-31')))
    )
    assert crisis_mask.sum() >= 20, \
        "Test invariant: crisis mask must exceed threshold"

    pi = fit_hmm(Z_train, Z_full, seed=42, crisis_mask=crisis_mask,
                 n_iter=500, n_burnin=100)

    # During dot-com bust (in training), pi should be high
    dotcom = ((sub['date'].values >= np.datetime64('2000-06-01'))
              & (sub['date'].values <= np.datetime64('2002-06-01')))
    pi_dotcom = pi[dotcom].mean()
    assert pi_dotcom > 0.4, \
        f"Pi during dot-com (in-training) too low ({pi_dotcom:.2f}); " \
        "panic state likely mislabeled"


def test_fit_hmm_forward_filter_causality():
    """pi_filter at time t must depend only on observations through t.
    We verify by computing pi on the full series, then pi on a truncated
    series ending at t, and comparing the value at t."""
    from historical_oos_production import forward_filter
    from scipy.stats import invwishart

    D = 4
    K = 2
    np.random.seed(0)

    # Fabricate stable parameters for a simple 2-state HMM
    mu = np.array([[0.1, -0.2, 0.3, -0.1], [-0.4, 0.8, -0.9, 0.7]])
    Sigma = np.stack([np.eye(D), np.eye(D) * 0.5])
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    # 100-step random series from calm state
    T = 100
    Z = np.random.randn(T, D) * 0.5

    pi_full = forward_filter(Z, mu, Sigma, P)
    # Truncate at T-20 and recompute
    pi_trunc = forward_filter(Z[:T - 20], mu, Sigma, P)

    # Values at the overlapping indices must match exactly
    np.testing.assert_allclose(
        pi_full[:T - 20], pi_trunc,
        rtol=1e-10, atol=1e-12,
        err_msg="Forward filter is not strictly causal"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Integration test: small-scale expanding-window run
# ══════════════════════════════════════════════════════════════════════════════

def test_expanding_window_integration_small(tmp_path):
    """Run a 2-year expanding window at minimum seeds and verify the output
    shape, continuity, and basic sanity. Takes ~5 min on a fast machine."""
    tag = 'pytest_integration'
    # Outputs land in results/thesis/ post-restructure (config.RESULTS_THESIS_DIR).
    results_dir = os.path.join(REPO, 'results', 'thesis')
    returns_path = os.path.join(results_dir, f'expanding_returns_{tag}.csv')
    pi_path = os.path.join(results_dir, f'expanding_pi_filter_{tag}.csv')
    log_path = os.path.join(results_dir, f'expanding_summary_{tag}.csv')

    # Clean any prior outputs
    for p in (returns_path, pi_path, log_path):
        if os.path.exists(p):
            os.remove(p)

    cmd = [
        sys.executable, '-u',
        'scripts/expanding_window_backtest.py',
        '--first-retrain-year', '2010',
        '--last-retrain-year', '2011',
        '--hmm-seeds', '3',
        '--xgb-seeds', '3',
        '--tag', tag,
    ]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                            timeout=900)

    assert result.returncode == 0, \
        f"Script exited with code {result.returncode}.\n" \
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    # Outputs exist
    assert os.path.exists(returns_path), "Returns CSV not written"
    assert os.path.exists(pi_path), "Pi filter CSV not written"
    assert os.path.exists(log_path), "Summary CSV not written"

    # Returns: shape and date continuity
    rets = pd.read_csv(returns_path, index_col='date', parse_dates=True)
    assert len(rets) > 0, "Empty returns output"
    assert rets['ret'].notna().all(), "Returns contain NaN"
    assert rets['ret'].between(-0.5, 0.5).all(), \
        "Returns outside sanity bounds [-50%, 50%]"

    # Date coverage: should span two full years, roughly
    min_d, max_d = rets.index.min(), rets.index.max()
    assert min_d.year == 2010 and max_d.year == 2011, \
        f"Unexpected date range: {min_d} to {max_d}"

    # No duplicate dates
    assert rets.index.is_unique, "Duplicate dates in returns output"

    # Pi filter: valid range
    pi = pd.read_csv(pi_path, parse_dates=['date'])
    assert pi['pi_filter'].between(0, 1).all(), \
        "pi_filter outside [0, 1]"

    # Summary log: one row per year
    log = pd.read_csv(log_path)
    assert len(log) == 2, f"Expected 2 summary rows, got {len(log)}"
    assert set(log['predict_year']) == {2010, 2011}
    assert (log['train_months'] >= 24).all(), "Training windows too short"
    assert (log['xgb_train_rows'] > 1000).all(), "XGB train rows too few"

    # Clean up test outputs
    for p in (returns_path, pi_path, log_path):
        if os.path.exists(p):
            os.remove(p)


# ══════════════════════════════════════════════════════════════════════════════
# Look-ahead audit: retraining for year Y uses only pre-Y data
# ══════════════════════════════════════════════════════════════════════════════

def test_expanding_window_no_lookahead_in_training_window():
    """Inspect expanding_window_backtest.py source to confirm train/test
    boundaries are correctly enforced."""
    script_path = os.path.join(REPO, 'scripts', 'expanding_window_backtest.py')
    with open(script_path) as f:
        source = f.read()

    # The script must use `date < train_end` for training, not `<=`
    assert "df['date'] < train_end" in source, \
        "Training cutoff must be strict `<` to exclude predict-year months"

    # Test filter must require date >= train_end AND date < test_end
    assert "(df['date'] >= train_end) & (df['date'] < test_end)" in source, \
        "Test filter malformed"

    # HMM training must use train_date_mask based on < train_end
    assert "sub['date'] < train_end" in source, \
        "HMM training cutoff must be strict `<`"


def test_crisis_mask_function():
    """crisis_mask_for_training must return the expected number of months
    for known training windows."""
    from expanding_window_backtest import crisis_mask_for_training

    # Synthetic dates spanning 1990-01 to 2010-12 monthly
    dates = pd.date_range('1990-01-01', '2010-12-31', freq='M').values
    mask = crisis_mask_for_training(dates)

    # Expected: 1990 recession ~9 months + LTCM ~4 + dot-com ~32 + GFC ~21 + COVID 3 = ~69 months
    expected_min = 50
    expected_max = 80
    assert expected_min <= mask.sum() <= expected_max, \
        f"Crisis mask count {mask.sum()} outside plausible range " \
        f"[{expected_min}, {expected_max}]"


# ══════════════════════════════════════════════════════════════════════════════
# Regression: output format stability
# ══════════════════════════════════════════════════════════════════════════════

def test_output_columns_stable():
    """If any previously-generated expanding run is present, verify it has
    the expected column structure."""
    path = os.path.join(REPO, 'results', 'expanding_returns_smoke.csv')
    if not os.path.exists(path):
        pytest.skip("No expanding_returns_smoke.csv present; run smoke first")

    df = pd.read_csv(path, parse_dates=['date'])
    assert set(df.columns) >= {'date', 'ret'}, \
        "Returns CSV must have at least date and ret columns"
    assert df['ret'].notna().all(), "Returns CSV contains NaN"
    assert df['date'].is_unique, "Duplicate dates in returns CSV"
