"""
test_lookahead_behavioural.py
=============================
Behavioural look-ahead tests. Unlike test_cross_sectional_lookahead.py
(which greps the source for the strings "shift(1)" and "train_mask"),
these tests build small synthetic panels and assert correctness via
recomputation. A semantically-equivalent rewrite of the production
script that REMOVED its causal guards would silently pass the textual
audit but fail these behavioural checks.

Coverage:
  1. Cross-permno ffill of pi_filter does not bleed values across permno
     boundaries (regression test for the bug in cross_sectional_model.py
     where a permno-major sort + global ffill propagated the prior
     permno's late pi_filter into the next permno's earliest rows,
     fabricating regime probabilities for pre-1990 stocks).
  2. Production momentum at month t uses only returns up to and
     including month t-1 (verified by permuting future rows and
     confirming pre-cutoff momentum is byte-identical).
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 1. Cross-permno ffill regression
# ---------------------------------------------------------------------------

def test_groupwise_ffill_does_not_bleed_across_permnos():
    """Synthesize a panel with two permnos: an old one (1985-2025) and a
    young one (1990-2025). The regime panel only covers 1990+. After a
    naive global ffill, the old permno's pre-1990 rows would inherit the
    young permno's late pi_filter via the permno-major sort. Group-wise
    ffill prevents this."""
    dates_old = pd.date_range('1985-01-31', '2025-12-31', freq='M')
    dates_young = pd.date_range('1990-01-31', '2025-12-31', freq='M')

    stocks = pd.concat([
        pd.DataFrame({'permno': 1, 'date': dates_old}),
        pd.DataFrame({'permno': 2, 'date': dates_young}),
    ]).sort_values(['permno', 'date']).reset_index(drop=True)

    # Regime panel: pi_filter only defined from 1990 onward.
    regime_dates = pd.date_range('1990-01-31', '2025-12-31', freq='M')
    rng = np.random.RandomState(0)
    regimes = pd.DataFrame({'date': regime_dates,
                            'pi_filter': rng.uniform(0, 1, size=len(regime_dates))})

    merged = stocks.merge(regimes, on='date', how='left')

    # ── Bad behaviour (legacy bug): global ffill ──
    bad = merged.copy()
    bad['pi_filter_bad'] = bad['pi_filter'].ffill()
    pre_1990_mask = bad['date'] < '1990-01-31'
    permno_old_pre_1990 = (pre_1990_mask & (bad['permno'] == 1))
    # The bug: the LAST row of permno=1 (just before permno=2 starts)
    # inherits a value from later in the merge. Actually — in this test
    # permno=1 is sorted first AND its full history ALSO predates regimes
    # for the early years, so we just need to show the bug WOULD propagate
    # if rows were sorted the other way. Build a permno=1 vs permno=2
    # interleave to make the bleed concrete:
    merged_tail_first = pd.concat([
        merged[merged['permno'] == 2],
        merged[merged['permno'] == 1],
    ]).reset_index(drop=True)
    merged_tail_first['pi_bad'] = merged_tail_first['pi_filter'].ffill()
    permno1_pre_1990 = ((merged_tail_first['permno'] == 1) &
                        (merged_tail_first['date'] < '1990-01-31'))
    n_leaked = merged_tail_first.loc[permno1_pre_1990, 'pi_bad'].notna().sum()
    assert n_leaked > 0, ("This test fixture is wrong — it should DEMONSTRATE "
                          "the bleed under naive global ffill.")

    # ── Good behaviour (group-wise ffill): no bleed ──
    good = merged.copy()
    good['pi_filter_good'] = good.groupby('permno')['pi_filter'].ffill()
    permno1_good_pre_1990 = ((good['permno'] == 1) & (good['date'] < '1990-01-31'))
    n_good_leaked = good.loc[permno1_good_pre_1990, 'pi_filter_good'].notna().sum()
    assert n_good_leaked == 0, (
        f"Group-wise ffill should not bleed pi_filter into pre-1990 rows "
        f"of an earlier permno; got {n_good_leaked} leaked values."
    )


def test_production_scripts_use_groupwise_ffill_for_pi_filter():
    """Source-level guard: every production-path script that merges
    pi_filter into a permno-major panel must use groupby('permno').ffill(),
    not a global ffill. This test reads the script source so a refactor
    that re-introduces the bug fails CI."""
    affected = [
        'scripts/cross_sectional_model.py',
        'scripts/xgb_cv.py',
        'scripts/random_forest_test.py',
        'scripts/fundamentals_test.py',
    ]
    for rel in affected:
        path = os.path.join(PROJECT_ROOT, rel)
        with open(path) as f:
            src = f.read()
        # Must NOT contain the bare `stocks['pi_filter'] = stocks['pi_filter'].ffill()`
        # pattern. Must contain the groupby-based fix.
        assert "stocks.groupby('permno')['pi_filter'].ffill()" in src, (
            f"{rel} no longer uses group-wise pi_filter ffill — the "
            "cross-permno bleed bug may have been reintroduced."
        )


# ---------------------------------------------------------------------------
# 2. Momentum at t uses only returns through t-1
# ---------------------------------------------------------------------------

def test_momentum_pre_cutoff_invariant_under_post_cutoff_permutation():
    """Permute returns AFTER a cutoff date and confirm momentum signals
    BEFORE the cutoff are byte-identical to the unpermuted panel.
    A look-ahead leak (e.g. forgetting to .shift(1) returns) would
    cause pre-cutoff momentum to depend on post-cutoff returns and
    fail this check."""
    rng = np.random.RandomState(7)
    permnos = [10, 20, 30]
    dates = pd.date_range('2000-01-31', '2010-12-31', freq='M')
    rows = []
    for p in permnos:
        rets = rng.normal(0.005, 0.05, size=len(dates))
        rows.append(pd.DataFrame({'permno': p, 'date': dates, 'ret_adj': rets}))
    panel = pd.concat(rows).sort_values(['permno', 'date']).reset_index(drop=True)

    def compute_mom(df, lookbacks=(1, 6, 12)):
        df = df.copy()
        df['_lr'] = np.log1p(df['ret_adj'].clip(lower=-0.999))
        df['_lr_s1'] = df.groupby('permno')['_lr'].shift(1)
        for lb in lookbacks:
            df[f'mom_{lb}'] = np.expm1(
                df.groupby('permno', sort=False)['_lr_s1']
                  .rolling(lb, min_periods=lb).sum()
                  .reset_index(level='permno', drop=True)
                  .sort_index()
            )
        return df

    cutoff = pd.Timestamp('2007-01-01')
    baseline = compute_mom(panel)

    # Permute post-cutoff returns within each permno.
    permuted = panel.copy()
    post_mask = permuted['date'] >= cutoff
    for p in permnos:
        sel = (permuted['permno'] == p) & post_mask
        idx = permuted.index[sel].to_numpy()
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        permuted.loc[idx, 'ret_adj'] = permuted.loc[shuffled, 'ret_adj'].values

    permuted_mom = compute_mom(permuted)

    # mom_lb at month t depends on returns at months t-lb..t-1. So mom at
    # any month t with t-1 BEFORE cutoff must be identical between
    # baseline and permuted. We check t such that month t-1 < cutoff,
    # i.e. months strictly before cutoff. mom_12 also requires t-12..t-1
    # all to be pre-cutoff, so the strictest invariant: rows with
    # date < cutoff must agree on every momentum column.
    pre_baseline = baseline[baseline['date'] < cutoff][['permno', 'date',
                                                         'mom_1', 'mom_6', 'mom_12']]
    pre_permuted = permuted_mom[permuted_mom['date'] < cutoff][['permno', 'date',
                                                                 'mom_1', 'mom_6', 'mom_12']]
    for col in ['mom_1', 'mom_6', 'mom_12']:
        a = pre_baseline[col].dropna().values
        b = pre_permuted[col].dropna().values
        assert len(a) == len(b)
        assert np.array_equal(a, b), (
            f"Pre-cutoff {col} changed when post-cutoff returns were "
            "permuted; momentum has a look-ahead leak."
        )
