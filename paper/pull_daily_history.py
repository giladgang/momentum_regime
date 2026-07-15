"""Backfill daily CRSP (dsf_v2) 1990-2009 for IVOL + pre-2011 spreads.

WRDS-GATED: run only after Gilad's explicit per-session approval.
Writes per-year parquets into experiments/results/spreads/ with the same
naming as the existing 2010-2025 files so downstream globs pick them up.

Usage: .venv/bin/python -m paper.pull_daily_history [--start 1990 --end 2009]
"""
import argparse
import os
import sys
from pathlib import Path

import wrds

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

OUT_DIR = 'experiments/results/spreads'
COLS = ('permno, dlycaldt, dlybid, dlyask, dlyhigh, dlylow, dlyprc, dlyret')


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
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1990)
    ap.add_argument('--end', type=int, default=2009)
    a = ap.parse_args()
    db = _connect()
    for y in range(a.start, a.end + 1):
        out = os.path.join(OUT_DIR, f'dsf_v2_{y}.parquet')
        if os.path.exists(out):
            print(f'[skip] {out} exists', flush=True)
            continue
        d = db.raw_sql(f"""
            SELECT {COLS}
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '{y}-01-01' AND '{y}-12-31'
        """, date_cols=['dlycaldt'])
        d.to_parquet(out, index=False)
        print(f'[pull] {y}: {len(d):,} rows -> {out}', flush=True)
    db.close()


if __name__ == '__main__':
    main()
