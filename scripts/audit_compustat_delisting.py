"""
audit_compustat_delisting.py
============================
Phase 1b investigation step: characterise delisting events in the existing
UK and Japan Compustat Global panels, so we can decide whether the
Shumway-equivalent treatment must:

  (a) re-pull the raw data with `secstat` / `dldte` / `dlrsn` from
      `comp.g_security`, then apply rules conditioning on those fields, OR
  (b) be done heuristically from the existing panels using
      "disappearance + price-drop" signals.

This script does NOT modify any data file. It only inspects the existing
`data/{uk,jp}_stock_panel.parquet` and prints diagnostics that inform the
design of `scripts/apply_shumway_intl.py`.

Output
------
Prints to stdout. Optional CSV at `results/intl_delisting_audit.csv` with
one row per region summarising the diagnostics.

Run
---
    python scripts/audit_compustat_delisting.py
    python scripts/audit_compustat_delisting.py --csv results/intl_delisting_audit.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)


def audit_region(region):
    """Return a dict of delisting-relevant diagnostics for one region."""
    panel_path = REPO / 'data' / f'{region.lower()}_stock_panel.parquet'
    if not panel_path.exists():
        print(f'  [SKIP] {panel_path} not present')
        return None
    df = pd.read_parquet(panel_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['secid', 'date']).reset_index(drop=True)
    panel_end = df['date'].max()

    print(f'\n=== {region} stock panel ===')
    print(f'  Total rows:          {len(df):,}')
    print(f'  Unique securities:   {df["secid"].nunique():,}')
    print(f'  Date range:          {df["date"].min().date()} → {panel_end.date()}')
    print(f'  Columns:             {list(df.columns)}')

    # 1. Securities whose LAST appearance is before panel end → "disappeared"
    last_dates = df.groupby('secid')['date'].max()
    # Allow a 3-month grace window before treating as disappearance, since
    # Compustat sometimes lags 1-2 months for the most recent securities.
    grace = pd.Timedelta(days=90)
    disappeared_secids = last_dates[last_dates < (panel_end - grace)].index
    n_disappeared = len(disappeared_secids)
    print(f'\n  Disappeared securities (last appearance > 90 days '
          f'before panel end): {n_disappeared:,} '
          f'({n_disappeared / df["secid"].nunique() * 100:.1f}%)')

    if n_disappeared == 0:
        return _summarise(region, df, n_disappeared, 0, 0, 0, panel_end)

    # 2. For disappeared securities, look at the LAST month's `ret`
    last_rows = (df[df['secid'].isin(disappeared_secids)]
                 .sort_values(['secid', 'date'])
                 .groupby('secid').tail(1))
    n_last_ret_missing = last_rows['ret'].isna().sum()
    print(f'  ... of which last-row `ret` is missing:    {n_last_ret_missing:,}')

    # 3. Of the disappeared+ret-missing rows, how many have a prior month
    # `prc_close` that's much higher than the last `prc_close` (price-drop
    # signal of distress delisting)?
    last_with_prev = []
    for secid in disappeared_secids:
        sub = df[df['secid'] == secid].sort_values('date')
        if len(sub) < 2:
            continue
        last = sub.iloc[-1]
        prev = sub.iloc[-2]
        if pd.isna(last['ret']) and prev['prc_close'] > 0:
            ratio = last['prc_close'] / prev['prc_close']
            last_with_prev.append({
                'secid': secid,
                'last_date': last['date'],
                'prev_close': prev['prc_close'],
                'last_close': last['prc_close'],
                'ratio': ratio,
                'computed_ret': ratio - 1.0,
            })
    last_with_prev = pd.DataFrame(last_with_prev)
    n_signal_rows = len(last_with_prev)
    print(f'  ... of which prev-month close is also available '
          f'(can compute manual ret): {n_signal_rows:,}')

    if n_signal_rows > 0:
        crash_count = int((last_with_prev['ratio'] < 0.5).sum())
        sev_count = int((last_with_prev['ratio'] < 0.7).sum())
        mod_count = int((last_with_prev['ratio'] < 1.0).sum())
        print(f'    drop > 50%   (ratio < 0.5):    {crash_count:,}')
        print(f'    drop > 30%   (ratio < 0.7):    {sev_count:,}')
        print(f'    drop > 0%    (ratio < 1.0):    {mod_count:,}')
        print(f'    distribution of computed_ret on these: '
              f'{last_with_prev["computed_ret"].describe().to_dict()}')

    # 4. How many disappeared securities have a final-month ret that's
    # already strongly negative (e.g. < -30%)? These are already captured
    # without imputation.
    n_already_negative = int(((last_rows['ret'] < -0.30)
                              & last_rows['ret'].notna()).sum())
    print(f'  Disappeared with last-month ret < -30% '
          f'(already captured, no imputation needed): {n_already_negative:,}')

    return _summarise(region, df, n_disappeared, n_last_ret_missing,
                      n_signal_rows, n_already_negative, panel_end)


def _summarise(region, df, n_disappeared, n_last_ret_missing,
               n_signal_rows, n_already_negative, panel_end):
    return {
        'region': region,
        'panel_rows': int(len(df)),
        'unique_securities': int(df['secid'].nunique()),
        'panel_end': str(panel_end.date()),
        'n_disappeared': int(n_disappeared),
        'pct_disappeared': float(n_disappeared / df['secid'].nunique() * 100)
            if df['secid'].nunique() else 0.0,
        'n_last_ret_missing': int(n_last_ret_missing),
        'n_with_price_drop_signal': int(n_signal_rows),
        'n_already_negative': int(n_already_negative),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None,
                        help='Optional CSV output path')
    parser.add_argument('--regions', nargs='+', default=['UK', 'JP'])
    args = parser.parse_args()

    rows = []
    for region in args.regions:
        out = audit_region(region)
        if out is not None:
            rows.append(out)

    if args.csv and rows:
        out_df = pd.DataFrame(rows)
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f'\nWrote: {out_path}')

    print('\n' + '=' * 70)
    print('  RECOMMENDATION FOR scripts/apply_shumway_intl.py')
    print('=' * 70)
    if not rows:
        print('  No regions audited — cannot recommend.')
        return
    total_signal = sum(r['n_with_price_drop_signal'] for r in rows)
    total_disappeared = sum(r['n_disappeared'] for r in rows)
    print(f'  Across regions: {total_disappeared:,} disappeared securities, '
          f'{total_signal:,} have a price-drop signal we can act on.')
    print()
    if total_signal > 0:
        print('  → Heuristic Shumway-equivalent IS feasible without re-pulling.')
        print('    Imputation rule: for disappeared+ret-missing rows where')
        print('    prc_close/prev_prc_close < 0.7, set ret_adj to that ratio−1.')
        print('    Otherwise ret_adj := ret (passthrough).')
    else:
        print('  → Existing panel has no actionable delisting signal.')
        print('    Need to re-pull from WRDS with secstat/dldte/dlrsn fields')
        print('    (modify scripts/import_intl_stocks.py first).')


if __name__ == '__main__':
    main()
