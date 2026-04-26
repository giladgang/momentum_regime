"""
oos_pi_filter_only.py
=====================
Compute and save OOS pi_filter for a given training window. No XGB, no
portfolio build -- just the HMM forward-filtered probability of panic.

Purpose:
  (1) Diagnostic: was dot-com classified as panic by the OOS-trained HMM?
  (2) Enables panic-subtype decomposition (crash vs recovery) of M2 OOS
      returns using an OOS-estimated regime signal.

Uses 50 HMM seeds (~8 min) rather than the full 200 -- this is a diagnostic,
not a production number. The 200-seed average in the main OOS runs may differ
slightly but the regime classification is stable.

Usage:
    python scripts/oos_pi_filter_only.py --train-end 2000-01-01 --tag prod_1990_1999

Output:
    results/oos_pi_filter_<tag>.csv
"""

import argparse
import numpy as np
import pandas as pd
import time
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp

from config import HMM_FEATURES, HMM_ITERATIONS, HMM_BURNIN, RESULTS_DIR

# Import HMM helpers from the production script
from historical_oos_production import fit_hmm  # noqa: E402

warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--hmm-seeds', type=int, default=50)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    HMM_SEEDS = list(range(1, args.hmm_seeds + 1))
    TRAIN_END = args.train_end

    print("=" * 70)
    print(f"  OOS PI_FILTER: tag={args.tag}, train_end={TRAIN_END}")
    print(f"  HMM seeds: {len(HMM_SEEDS)}")
    print("=" * 70)

    panel = pd.read_parquet('data/panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    sub = (panel[['date'] + HMM_FEATURES]
           .dropna()
           .drop_duplicates('date')
           .sort_values('date'))
    sub = sub[sub['date'] >= '1990-01-01'].reset_index(drop=True)

    Z_full = sub[HMM_FEATURES].values.astype(float)
    Z_train = sub[sub['date'] < TRAIN_END][HMM_FEATURES].values.astype(float)
    print(f"  HMM train: {len(Z_train)} months, full series: {len(Z_full)}")

    train_dates = sub[sub['date'] < TRAIN_END]['date']
    if (train_dates >= '2000-01-01').any():
        crisis_mask = ((train_dates >= '2000-03-01')
                       & (train_dates <= '2002-10-31')).values
    else:
        crisis_mask = ((train_dates >= '1998-07-01')
                       & (train_dates <= '1998-10-31')).values
    print(f"  Crisis mask: {crisis_mask.sum()} months")

    t0 = time.time()
    pi_all = np.zeros(len(Z_full))
    for i, hs in enumerate(HMM_SEEDS, 1):
        pi_all += fit_hmm(Z_train, Z_full, seed=hs, crisis_mask=crisis_mask)
        if i % max(1, len(HMM_SEEDS) // 5) == 0 or i == len(HMM_SEEDS):
            elapsed = time.time() - t0
            est_total = elapsed / i * len(HMM_SEEDS)
            print(f"  Seed {i}/{len(HMM_SEEDS)} "
                  f"({elapsed:.0f}s, ~{est_total - elapsed:.0f}s remaining)")
    pi_all /= len(HMM_SEEDS)

    out = pd.DataFrame({'date': sub['date'], 'pi_filter': pi_all})
    out_path = os.path.join(RESULTS_DIR, f'oos_pi_filter_{args.tag}.csv')
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Quick diagnostic: pi_filter stats by sub-period
    SUBPERIODS = [
        ('Dot-com bust',   '2000-03-01', '2002-10-31'),
        ('2003-2007 bull', '2002-11-01', '2007-09-30'),
        ('GFC crash',      '2007-10-01', '2009-02-28'),
        ('GFC rebound',    '2009-03-01', '2009-12-31'),
        ('Post-GFC calm',  '2010-01-01', '2010-12-31'),
    ]
    out_idx = out.set_index(pd.to_datetime(out['date']))
    print("\n  Pi_filter sub-period stats:")
    print(f"  {'Subperiod':<18} {'N':>4} {'mean':>8} {'median':>8} {'% >= 0.5':>10}")
    for name, start, end in SUBPERIODS:
        mask = (out_idx.index >= start) & (out_idx.index <= end)
        pi_sp = out_idx.loc[mask, 'pi_filter']
        if len(pi_sp) == 0:
            continue
        print(f"  {name:<18} {len(pi_sp):>4} "
              f"{pi_sp.mean():>8.2f} {pi_sp.median():>8.2f} "
              f"{(pi_sp >= 0.5).mean() * 100:>9.0f}%")

    print(f"\nDone. Total: {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
