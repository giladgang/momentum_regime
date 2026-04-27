"""
historical_oos_production.py
============================
Single-config out-of-sample test (one training window, one test window) with
production-level seed ensembles. Built originally to compare retrained
configurations (1990-1999, 1990-2004, etc.) against the main production model.

Now serves primarily as a SHARED MODULE: scripts/expanding_window_backtest.py
and scripts/expanding_window_backtest_parallel.py both import fit_hmm and
build_ls from this file. Standalone use is still supported via CLI args
(useful for sensitivity checks at single training cutoffs).

The headline 30-year backtest in Section 5.4.3 uses
expanding_window_backtest_parallel.py (which imports from here), not this
script directly.

Sign correction in fit_hmm uses theoretical fixed signs for [DD, DISP, REL_N, CS]
(-1, +1, -1, +1) when the training crisis mask has fewer than 20 months
of crisis data; otherwise uses data-driven 95th-percentile comparison.

Extends test_2008_oos.py with:
  - CLI-configurable seed counts (for smoke tests vs production runs)
  - CLI-configurable training window (one config per run)
  - Per-month L/S return saving for downstream bootstrap (Option B)
  - Sub-period decomposition (dot-com bust, 2003-2007 bull, GFC crash,
    GFC rebound, post-GFC calm) with Sharpe, cumulative return, and MDD

Usage:
    # Smoke test (1-3 min compute)
    python scripts/historical_oos_production.py --train-end 2000-01-01 \
        --test-end 2011-01-01 --hmm-seeds 5 --xgb-seeds 5 --tag smoke

    # Production primary run (~2.5-3 hours)
    python scripts/historical_oos_production.py --train-end 2000-01-01 \
        --test-end 2011-01-01 --hmm-seeds 200 --xgb-seeds 50 --tag prod_1990_1999

    # Production robustness (1990-2004 train)
    python scripts/historical_oos_production.py --train-end 2005-01-01 \
        --test-end 2011-01-01 --hmm-seeds 200 --xgb-seeds 50 --tag prod_1990_2004

Outputs (per tag):
    results/oos_returns_<tag>.csv         -- per-month L/S returns
    results/oos_subperiods_<tag>.csv      -- sub-period decomposition
    results/oos_summary_<tag>.csv         -- one-line overall summary
"""

import argparse
import numpy as np
import pandas as pd
import warnings
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from xgboost import XGBRegressor

from config import (HMM_FEATURES, HMM_ITERATIONS, HMM_BURNIN, N_ESTIMATORS,
                    MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE, MOM_FEATURES,
                    RESULTS_DIR, HMM_PRIOR_M0, HMM_PRIOR_KAPPA0,
                    HMM_PRIOR_NU0_OFF, HMM_PRIOR_DIRICHLET_ALPHA)

warnings.filterwarnings('ignore')

K = 2
FEE = 0.001

# Sub-period windows for decomposition (apply within whichever test range
# is actually populated; windows outside the test range are dropped).
SUBPERIODS = [
    ('Dot-com bust',   '2000-03-01', '2002-10-31'),
    ('2003-2007 bull', '2002-11-01', '2007-09-30'),
    ('GFC crash',      '2007-10-01', '2009-02-28'),
    ('GFC rebound',    '2009-03-01', '2009-12-31'),
    ('Post-GFC calm',  '2010-01-01', '2010-12-31'),
]


# ── HMM helpers (matches production hmm_model.py logic) ───────────────────────

def log_emission(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])


def forward_filter(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K))
    la[0] = np.log(0.5) + le[0]
    la[0] -= logsumexp(la[0])
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t-1] + np.log(P[:, k]))
        la[t] -= logsumexp(la[t])
    return np.exp(la[:, 1])


def ffbs(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K))
    la[0] = np.log(0.5) + le[0]
    la[0] -= logsumexp(la[0])
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t-1] + np.log(P[:, k]))
        la[t] -= logsumexp(la[t])
    states = np.zeros(n, dtype=int)
    states[-1] = np.random.choice(K, p=np.exp(la[-1]))
    for t in range(n-2, -1, -1):
        lp = la[t] + np.log(P[:, states[t+1]])
        lp -= logsumexp(lp)
        states[t] = np.random.choice(K, p=np.exp(lp))
    return states


def fit_hmm(Z_train, Z_full, seed, crisis_mask,
            n_iter=HMM_ITERATIONS, n_burnin=HMM_BURNIN):
    np.random.seed(seed)
    T, D = Z_train.shape
    med = np.median(Z_train[:, 0])
    s0 = (Z_train[:, 0] < med).astype(int)  # DD below median = panic init
    mu = np.array([
        Z_train[s0 == k].mean(axis=0) if (s0 == k).sum() > 1 else np.zeros(D)
        for k in range(K)
    ])
    Sigma = np.array([
        np.cov(Z_train[s0 == k].T) + 0.01 * np.eye(D) if (s0 == k).sum() > 2 else np.eye(D)
        for k in range(K)
    ])
    P = np.array([[0.9, 0.1], [0.1, 0.9]])

    m0 = np.full(D, HMM_PRIOR_M0)
    kappa0 = HMM_PRIOR_KAPPA0
    nu0 = D + HMM_PRIOR_NU0_OFF
    Psi0 = np.eye(D) * (nu0 - D - 1)
    alpha_dir = np.asarray(HMM_PRIOR_DIRICHLET_ALPHA, dtype=float)

    for it in range(n_iter):
        states = ffbs(Z_train, mu, Sigma, P)
        for k in range(K):
            idx = states == k
            nk = idx.sum()
            if nk < 2:
                continue
            xbar = Z_train[idx].mean(axis=0)
            S = (Z_train[idx] - xbar).T @ (Z_train[idx] - xbar)
            kn = kappa0 + nk
            mn = (kappa0 * m0 + nk * xbar) / kn
            nun = nu0 + nk
            Psin = Psi0 + S + kappa0 * nk / kn * np.outer(xbar - m0, xbar - m0)
            Sigma[k] = invwishart.rvs(df=nun, scale=Psin)
            mu[k] = np.random.multivariate_normal(mn, Sigma[k] / kn)
        for i in range(K):
            counts = np.array([
                ((states[:-1] == i) & (states[1:] == j)).sum() for j in range(K)
            ])
            P[i] = np.random.dirichlet(alpha_dir[i] + counts)

    # Sign correction: use data-driven signs only if the crisis mask contains
    # enough months for the 95th-percentile comparison to be reliable. With small
    # crisis samples (e.g. LTCM alone = 4 months) the data-driven procedure is
    # noisy and can invert the panic-state label. Fall back to theoretical fixed
    # signs (DD:-1, DISP:+1, REL_N:-1, CS:+1) based on the known crisis direction
    # of each feature. Assumes HMM_FEATURES order in config.py = [DD, DISP, REL_N, CS].
    MIN_CRISIS_MONTHS_FOR_DATA_DRIVEN = 20
    if crisis_mask is not None and crisis_mask.sum() >= MIN_CRISIS_MONTHS_FOR_DATA_DRIVEN:
        signs = np.array([
            1 if np.percentile(Z_train[crisis_mask, j], 95)
               >= np.percentile(Z_train[~crisis_mask, j], 95)
            else -1
            for j in range(D)
        ])
    else:
        # Fixed theoretical signs for features [DD, DISP, REL_N, CS]
        signs = np.array([-1, 1, -1, 1])

    scores = [sum(signs[j] * mu[k, j] for j in range(D)) for k in range(K)]
    panic = int(np.argmax(scores))

    pi = forward_filter(Z_full, mu, Sigma, P)
    if panic == 0:
        pi = 1 - pi
    return pi


# ── Portfolio builder ─────────────────────────────────────────────────────────

def build_ls(df_test, score_col):
    prev_lw, prev_sw = {}, {}
    monthly = []
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lw = (longs.set_index('permno')['me'] / longs['me'].sum()).to_dict()
        sw = (shorts.set_index('permno')['me'] / shorts['me'].sum()).to_dict()
        tl = sum(abs(lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(lw) | set(prev_lw)) / 2
        ts = sum(abs(sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(sw) | set(prev_sw)) / 2
        r_l = (longs['ret_fwd'] * longs['me']).sum() / longs['me'].sum()
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        monthly.append({'date': date, 'ret': r_l - r_s - FEE * (tl + ts)})
        prev_lw, prev_sw = lw, sw
    return (pd.DataFrame(monthly).set_index('date')['ret']
            if monthly else pd.Series(dtype=float))


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(r):
    if len(r) < 2 or r.std() == 0:
        return dict(n=len(r), sharpe=np.nan, ann_ret=np.nan, cum=np.nan,
                    mdd=np.nan, monthly_mean=np.nan)
    sharpe = r.mean() / r.std() * np.sqrt(12)
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    cum = (1 + r).prod() - 1
    cumser = (1 + r).cumprod()
    mdd = (cumser / cumser.cummax() - 1).min()
    return dict(n=len(r), sharpe=sharpe, ann_ret=ann_ret, cum=cum, mdd=mdd,
                monthly_mean=r.mean())


def decompose_subperiods(r, subperiods=SUBPERIODS):
    rows = []
    r_idx = pd.DatetimeIndex(r.index)
    for label, start, end in subperiods:
        mask = (r_idx >= start) & (r_idx <= end)
        rs = r[mask]
        if len(rs) == 0:
            continue
        m = compute_metrics(rs)
        rows.append({'subperiod': label, 'start': start, 'end': end, **m})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-end', required=True,
                    help='End of training window (exclusive), e.g. 2000-01-01')
    ap.add_argument('--test-end', required=True,
                    help='End of test window (exclusive), e.g. 2011-01-01')
    ap.add_argument('--hmm-seeds', type=int, default=200,
                    help='Number of HMM seeds (default 200 = production)')
    ap.add_argument('--xgb-seeds', type=int, default=50,
                    help='Number of XGB seeds (default 50 = production)')
    ap.add_argument('--tag', required=True,
                    help='Output file tag, e.g. smoke, prod_1990_1999')
    args = ap.parse_args()

    TRAIN_END = args.train_end
    TEST_END = args.test_end
    HMM_SEEDS = list(range(1, args.hmm_seeds + 1))
    XGB_SEEDS = list(range(1, args.xgb_seeds + 1))
    TAG = args.tag

    FEATURES = MOM_FEATURES + ['pi_filter']
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.time()
    print("=" * 70)
    print(f"  HISTORICAL OOS TEST: tag={TAG}")
    print(f"  Train: 1990-01-01 to {TRAIN_END} (exclusive)")
    print(f"  Test:  {TRAIN_END} to {TEST_END} (exclusive)")
    print(f"  HMM seeds: {len(HMM_SEEDS)}, XGB seeds: {len(XGB_SEEDS)}")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/5] Loading data ...")
    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    sub = (panel[['date'] + HMM_FEATURES]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date'))
    sub = sub[sub['date'] >= '1990-01-01'].reset_index(drop=True)

    stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
    stocks['date'] = pd.to_datetime(stocks['date'])
    stocks = stocks[
        stocks['shrcd'].isin([10, 11])
        & stocks['exchcd'].isin([1, 2, 3])
        & (stocks['prc'].abs() > 1)
    ].sort_values(['permno', 'date']).reset_index(drop=True)

    stocks['_lr'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
    stocks['_lr_s1'] = stocks.groupby('permno')['_lr'].shift(1)
    for lb in range(1, 13):
        rs = (stocks.groupby('permno', sort=False)['_lr_s1']
              .rolling(lb, min_periods=lb)
              .sum()
              .reset_index(level='permno', drop=True)
              .sort_index())
        stocks[f'mom_{lb}'] = np.expm1(rs)
    stocks.drop(columns=['_lr', '_lr_s1'], inplace=True)
    stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

    Z_full = sub[HMM_FEATURES].values.astype(float)
    Z_train = sub[sub['date'] < TRAIN_END][HMM_FEATURES].values.astype(float)
    print(f"  HMM train rows: {len(Z_train)}, full series: {len(Z_full)}")

    # ── Crisis mask for sign correction ──
    train_dates = sub[sub['date'] < TRAIN_END]['date']
    if (train_dates <= '1999-12-31').any() and (train_dates >= '2000-01-01').any():
        # Training includes dot-com crisis
        crisis_mask = ((train_dates >= '2000-03-01')
                       & (train_dates <= '2002-10-31')).values
    elif (train_dates >= '2000-01-01').any():
        # Training starts 2000+: dot-com always available
        crisis_mask = ((train_dates >= '2000-03-01')
                       & (train_dates <= '2002-10-31')).values
    else:
        # Training ends 1999 or earlier: use LTCM/Russia 1998
        crisis_mask = ((train_dates >= '1998-07-01')
                       & (train_dates <= '1998-10-31')).values
    print(f"  Crisis mask: {crisis_mask.sum()} months")

    # ── [2/5] HMM: average pi_filter across seeds ──
    print(f"\n[2/5] Running {len(HMM_SEEDS)} HMM seeds ...")
    t0 = time.time()
    pi_all = np.zeros(len(Z_full))
    for i, hs in enumerate(HMM_SEEDS, 1):
        pi_all += fit_hmm(Z_train, Z_full, seed=hs, crisis_mask=crisis_mask)
        if i % max(1, len(HMM_SEEDS) // 10) == 0 or i == len(HMM_SEEDS):
            elapsed = time.time() - t0
            est_total = elapsed / i * len(HMM_SEEDS)
            print(f"  HMM seed {i}/{len(HMM_SEEDS)} "
                  f"({elapsed:.0f}s elapsed, ~{est_total - elapsed:.0f}s remaining)")
    pi_all /= len(HMM_SEEDS)
    print(f"  HMM done: {time.time() - t0:.0f}s")

    # ── [3/5] Merge pi, build XGB ensemble ──
    print(f"\n[3/5] Running {len(XGB_SEEDS)} XGB seeds ...")
    pi_df = pd.DataFrame({'date': sub['date'], 'pi_filter': pi_all})
    stocks_w = stocks.merge(pi_df, on='date', how='left')
    stocks_w['pi_filter'] = stocks_w['pi_filter'].ffill()

    df = stocks_w.dropna(subset=['ret_fwd'] + FEATURES).copy()
    train_xgb = df[df['date'] < TRAIN_END]
    test_xgb = df[(df['date'] >= TRAIN_END) & (df['date'] < TEST_END)]

    X_tr = train_xgb[FEATURES].values.astype(float)
    X_te = test_xgb[FEATURES].values.astype(float)
    y_tr = train_xgb['ret_fwd'].values.astype(float)

    mask_tr = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
    mask_te = ~np.isnan(X_te).any(axis=1)
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
    X_te = X_te[mask_te]
    test_xgb = test_xgb[mask_te].copy()

    print(f"  XGB train rows: {len(X_tr):,} | test rows: {len(X_te):,}")

    t0 = time.time()
    preds = np.zeros(len(X_te))
    for i, xs in enumerate(XGB_SEEDS, 1):
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                           learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE, tree_method='hist',
                           random_state=xs, verbosity=0)
        xgb.fit(X_tr, y_tr)
        preds += xgb.predict(X_te)
        if i % max(1, len(XGB_SEEDS) // 5) == 0 or i == len(XGB_SEEDS):
            elapsed = time.time() - t0
            print(f"  XGB seed {i}/{len(XGB_SEEDS)} ({elapsed:.0f}s elapsed)")
    preds /= len(XGB_SEEDS)
    print(f"  XGB done: {time.time() - t0:.0f}s")

    test_xgb['score'] = preds

    # ── [4/5] Build L/S portfolio and decompose ──
    print("\n[4/5] Building L/S portfolio and decomposing sub-periods ...")
    r = build_ls(test_xgb, 'score')

    # Overall
    overall = compute_metrics(r)
    print(f"\n  OVERALL ({TRAIN_END} to {TEST_END}):")
    print(f"    N months:     {overall['n']}")
    print(f"    Sharpe:       {overall['sharpe']:.2f}")
    print(f"    Ann return:   {overall['ann_ret'] * 100:+.1f}%")
    print(f"    Cumulative:   {overall['cum'] * 100:+.1f}%")
    print(f"    Max DD:       {overall['mdd'] * 100:+.1f}%")

    # Sub-periods
    sub_df = decompose_subperiods(r)
    print("\n  SUB-PERIODS:")
    print(f"    {'Subperiod':<20} {'N':>4} {'Sharpe':>8} {'Cum':>8} {'MDD':>8}")
    for _, row in sub_df.iterrows():
        print(f"    {row['subperiod']:<20} {row['n']:>4} "
              f"{row['sharpe']:>8.2f} {row['cum'] * 100:>+7.1f}% "
              f"{row['mdd'] * 100:>+7.1f}%")

    # ── [5/5] Save outputs ──
    print("\n[5/5] Saving outputs ...")
    returns_path = os.path.join(RESULTS_DIR, f'oos_returns_{TAG}.csv')
    r.to_frame('ret').to_csv(returns_path)
    print(f"  Saved: {returns_path}")

    sub_path = os.path.join(RESULTS_DIR, f'oos_subperiods_{TAG}.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")

    summary = pd.DataFrame([{
        'tag': TAG,
        'train_end': TRAIN_END,
        'test_end': TEST_END,
        'hmm_seeds': len(HMM_SEEDS),
        'xgb_seeds': len(XGB_SEEDS),
        **overall,
    }])
    summary_path = os.path.join(RESULTS_DIR, f'oos_summary_{TAG}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min")
    print("Done.")


if __name__ == '__main__':
    main()
