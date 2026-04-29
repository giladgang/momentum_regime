"""Phase 3: per-rule characterisation.

For each rule (a cluster of long-leg stock-months):
  - Typical input feature centroid (12 mom horizons + pi_filter)
  - Regime distribution (mean pi_filter, calm vs panic share)
  - Realised performance: mean monthly return, bootstrap 95% Sharpe CI
    on plurality months (months where >50% of long-leg picks belong to
    the rule). Mirrors the thesis 5.2.3 plurality-month convention.

Outputs:
  results/thesis/rule_path_centroids.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_helpers import block_bootstrap_sharpe

RES_DIR = 'results/thesis'
LABELS_CSV = f'{RES_DIR}/rule_path_labels.csv'
OUT_CSV = f'{RES_DIR}/rule_path_centroids.csv'


def main():
    print('Loading rule labels and panel...', flush=True)
    labels_df = pd.read_csv(LABELS_CSV, parse_dates=['date'])

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    mom_cols = [f'mom_{h}' for h in range(1, 13)]

    # Returns
    with open('results/thesis/fundamentals_returns.pkl', 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    # ---- Merge labels with stock-level features ----
    key = ['date', 'permno']
    feature_cols = mom_cols + ['pi_filter']
    merged = labels_df.merge(
        test[key + feature_cols],
        on=key,
        how='left',
    )
    assert merged[feature_cols].notna().all().all(), \
        'Some rule-labeled stock-months have no feature row in the test panel'

    # ---- Plurality-month assignment ----
    # For each month, count picks per rule; rule is "plurality" if it
    # has > 50% of that month's picks. Otherwise the month is "no-plurality".
    by_month_rule = (labels_df
                     .groupby(['date', 'rule_id'])
                     .size()
                     .reset_index(name='n'))
    by_month_total = (labels_df
                      .groupby('date')
                      .size()
                      .reset_index(name='total'))
    by_month_rule = by_month_rule.merge(by_month_total, on='date')
    by_month_rule['share'] = by_month_rule['n'] / by_month_rule['total']
    plurality = (by_month_rule[by_month_rule['share'] > 0.5]
                 [['date', 'rule_id']])

    rules = sorted(labels_df['rule_id'].unique())
    print(f'{len(rules)} rules; '
          f'{len(plurality)} of {labels_df["date"].nunique()} months have '
          f'a plurality rule', flush=True)

    rows = []
    for r in rules:
        sub = merged[merged['rule_id'] == r]
        n_stockmonths = len(sub)
        n_panic = int((sub['pi_filter'] > 0.5).sum())

        # Plurality-month performance
        rule_dates = plurality[plurality['rule_id'] == r]['date']
        rs = m2_ret.reindex(rule_dates).dropna().values
        if len(rs) > 0:
            ci = block_bootstrap_sharpe(rs, block_size=6, n_reps=5000, seed=42)
            mean_ret = float(rs.mean())
        else:
            ci = {'sharpe_point': np.nan,
                  'sharpe_lo95': np.nan,
                  'sharpe_hi95': np.nan,
                  'n_reps_valid': 0}
            mean_ret = np.nan

        row = {
            'rule_id': int(r),
            'n_stockmonths': n_stockmonths,
            'n_plurality_months': len(rule_dates),
            'mean_pi': round(float(sub['pi_filter'].mean()), 3),
            'panic_share': round(n_panic / n_stockmonths, 3),
            'plurality_mean_ret_pct': (round(100 * mean_ret, 3)
                                       if not np.isnan(mean_ret) else None),
            'plurality_sharpe': (round(ci['sharpe_point'], 3)
                                 if not np.isnan(ci['sharpe_point']) else None),
            'plurality_sharpe_lo95': (round(ci['sharpe_lo95'], 3)
                                      if not np.isnan(ci['sharpe_lo95']) else None),
            'plurality_sharpe_hi95': (round(ci['sharpe_hi95'], 3)
                                      if not np.isnan(ci['sharpe_hi95']) else None),
        }
        # Stock-level feature centroid
        for c in feature_cols:
            row[f'centroid_{c}'] = round(float(sub[c].mean()), 4)
        rows.append(row)
        print(f'  rule {r}: n_sm={n_stockmonths}, '
              f'plurality_months={len(rule_dates)}, '
              f'sharpe={row["plurality_sharpe"]} '
              f'CI=[{row["plurality_sharpe_lo95"]}, '
              f'{row["plurality_sharpe_hi95"]}]', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
