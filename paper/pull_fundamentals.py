"""Pull annual Compustat fundamentals (comp.funda) for the regime-conditional
banding study, link gvkey -> permno via CCM, stamp a 6-month reporting lag
(PIT), and save raw linked fields. Signal construction (B/M, operating and
cash-based profitability, asset growth) happens downstream in
paper/src/fundamentals.py.

Mirrors the connection + CCM-link + PIT pattern of scripts/pull_cashflow.py,
but writes only under paper/ (applied-study territory; thesis data/ untouched).

Fields: book equity (seq, ceq, txditc, pstkrv, pstkl, pstk, at, lt);
operating profitability (revt, cogs, xsga, xint); cash-based adjustments
(rect, invt, ap, xacc, xpp, che); investment (at, + lagged at in-script).

Usage:  .venv/bin/python -m paper.pull_fundamentals
Output: paper/results/data/funda_linked.parquet
"""
import os
import sys
from pathlib import Path

import pandas as pd
import wrds

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

OUT = 'paper/results/data/funda_linked.parquet'
FIELDS = ['seq', 'ceq', 'txditc', 'pstkrv', 'pstkl', 'pstk', 'at', 'lt',
          'revt', 'cogs', 'xsga', 'xint',
          'rect', 'invt', 'ap', 'xacc', 'xpp', 'che']


def _connect():
    pgpass = Path.home() / '.pgpass'
    _u, _p = 'giladgang', None
    for line in pgpass.read_text().splitlines():
        parts = line.strip().split(':')
        if len(parts) == 5 and parts[0].startswith('wrds-pgdata'):
            _u, _p = parts[3], parts[4]
            break
    return wrds.Connection(wrds_username=_u, wrds_password=_p)


def main():
    print('Connecting to WRDS ...', flush=True)
    db = _connect()

    cols = ', '.join(f'f.{c}' for c in FIELDS)
    print('Pulling comp.funda (annual fundamentals) ...', flush=True)
    funda = db.raw_sql(f"""
        SELECT f.gvkey, f.datadate, f.fyear, {cols}
        FROM comp.funda AS f
        INNER JOIN crsp.ccmxpf_lnkhist AS lnk
            ON  f.gvkey     = lnk.gvkey
            AND lnk.linktype IN ('LU', 'LC')
            AND lnk.linkprim IN ('P', 'C')
        WHERE f.datadate BETWEEN '1985-01-01' AND '2025-12-31'
            AND f.indfmt  = 'INDL'
            AND f.datafmt = 'STD'
            AND f.popsrc  = 'D'
            AND f.consol  = 'C'
        ORDER BY f.gvkey, f.datadate
    """, date_cols=['datadate'])
    # INNER JOIN can duplicate a firm-year across overlapping link spans;
    # collapse to one row per (gvkey, datadate).
    funda = funda.drop_duplicates(subset=['gvkey', 'datadate'], keep='first')
    print(f'  funda rows: {len(funda):,}', flush=True)

    print('Pulling ccm link table ...', flush=True)
    ccm = db.raw_sql("""
        SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """, date_cols=['linkdt', 'linkenddt'])
    db.close()

    # lagged assets for asset growth (within gvkey, ordered by datadate)
    funda = funda.sort_values(['gvkey', 'datadate']).reset_index(drop=True)
    funda['at_lag'] = funda.groupby('gvkey')['at'].shift(1)

    # 6-month reporting lag -> availability (month-end)
    funda['avail_date'] = ((funda['datadate'] + pd.DateOffset(months=6))
                           + pd.offsets.MonthEnd(0))

    # link gvkey -> permno with date validity
    ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
    ccm['permno'] = ccm['permno'].astype(int)
    linked = funda.merge(ccm[['gvkey', 'permno', 'linkdt', 'linkenddt']],
                         on='gvkey', how='inner')
    linked = linked[(linked['datadate'] >= linked['linkdt']) &
                    (linked['datadate'] <= linked['linkenddt'])].copy()

    # one row per (permno, avail_date): keep the most recent fiscal year
    linked = (linked.sort_values(['permno', 'avail_date', 'datadate'])
              .drop_duplicates(subset=['permno', 'avail_date'], keep='last')
              .reset_index(drop=True))

    keep = ['permno', 'gvkey', 'datadate', 'avail_date', 'fyear',
            'at_lag'] + FIELDS
    out = linked[keep]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    yr = out['datadate'].dt.year
    print(f'\nSaved: {OUT}  ({len(out):,} rows, '
          f'{out["permno"].nunique():,} permnos, '
          f'{yr.min()}-{yr.max()})', flush=True)


if __name__ == '__main__':
    main()
