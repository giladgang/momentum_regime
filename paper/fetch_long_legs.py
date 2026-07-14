"""Fetch Ken French long-only factor legs for the long-only benchmark.

Marc's question (2026-07-14): a long-only book should be benchmarked against
the LONG legs of the factor sorts, not the self-financing long-short factors.
This module downloads the four 2x3 sort portfolio files from the French data
library (public; no WRDS) and builds the favorable (long) leg of each factor:

  size_long         mean of the three SMALL portfolios      (SMB long side)
  value_long        mean(SMALL HiBM,  BIG HiBM)              (HML long side)
  robust_long       mean(SMALL HiOP,  BIG HiOP)              (RMW long side)
  conservative_long mean(SMALL LoINV, BIG LoINV)             (CMA long side)
  winner_long       mean(SMALL HiPRIOR, BIG HiPRIOR)         (UMD long side)

CAPM's market factor (Mkt-RF) is already a long-only excess return, so it is
reused as-is from data/ff_factors.parquet.

pandas_datareader mis-parses 6_Portfolios_2x3 (its annual section trips the
reader), so we parse the monthly value-weighted block straight from the zip.

Output: paper/results/data/ff_long_legs.parquet  (decimal monthly returns,
month-end index). Run once:  .venv/bin/python -m paper.fetch_long_legs
"""
import io
import os
import re
import ssl
import sys
import urllib.request
import zipfile

import certifi
import pandas as pd

_SSL = ssl.create_default_context(cafile=certifi.where())

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from paper import config as C                                   # noqa: E402

_BASE = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
         '{}_CSV.zip')
# dataset -> (French file stem, suffix of the columns that form the long leg,
#             whether the leg is BOTH small+big corners or ALL small)
_SETS = {
    'size':         ('6_Portfolios_2x3',           'SMALL',   'all_small'),
    'value':        ('6_Portfolios_2x3',           'HiBM',    'corners'),
    'robust':       ('6_Portfolios_ME_OP_2x3',     'HiOP',    'corners'),
    'conservative': ('6_Portfolios_ME_INV_2x3',    'LoINV',   'corners'),
    'winner':       ('6_Portfolios_ME_Prior_12_2', 'HiPRIOR', 'corners'),
}


def _monthly_vw_block(stem):
    """Download a French 6-portfolio zip and return the first monthly
    value-weighted block as a DataFrame (percent units, month-end index)."""
    url = _BASE.format(stem)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    raw = urllib.request.urlopen(req, timeout=60, context=_SSL).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    txt = zf.read(zf.namelist()[0]).decode('latin-1')
    lines = txt.splitlines()
    # header = the first line naming the six portfolios (has SMALL and BIG)
    hidx = next(i for i, ln in enumerate(lines)
                if 'SMALL' in ln and 'BIG' in ln)
    cols = [c.strip() for c in lines[hidx].split(',') if c.strip()]
    rows = []
    for ln in lines[hidx + 1:]:
        m = re.match(r'\s*(\d{6})\s*,(.*)', ln)
        if not m:
            if rows:            # first contiguous monthly run ended -> stop
                break
            continue
        vals = [float(x) for x in m.group(2).split(',')]
        rows.append((m.group(1), *vals))
    df = pd.DataFrame(rows, columns=['yyyymm'] + cols)
    df['date'] = (pd.to_datetime(df['yyyymm'], format='%Y%m')
                  + pd.offsets.MonthEnd(0))
    return df.set_index('date').drop(columns='yyyymm').astype(float)


def build():
    legs = {}
    cache = {}
    for leg, (stem, suffix, mode) in _SETS.items():
        if stem not in cache:
            cache[stem] = _monthly_vw_block(stem)
        blk = cache[stem]
        if mode == 'all_small':
            # French labels the 3 small portfolios SMALL Lo.., ME1 .., SMALL Hi..
            cols = [c for c in blk.columns if c.startswith(('SMALL', 'ME1'))]
        else:
            cols = [c for c in blk.columns if c.replace(' ', '').endswith(suffix)]
        assert cols, f'{leg}: no columns matched {suffix} in {list(blk.columns)}'
        legs[leg] = blk[cols].mean(axis=1) / 100.0     # percent -> decimal
    out = pd.DataFrame(legs)
    os.makedirs(os.path.dirname(C.FF_LONG_LEGS), exist_ok=True)
    out.to_parquet(C.FF_LONG_LEGS)
    print('columns used per leg:')
    for leg, (stem, suffix, mode) in _SETS.items():
        blk = cache[stem]
        cols = ([c for c in blk.columns if c.startswith(('SMALL', 'ME1'))] if mode == 'all_small'
                else [c for c in blk.columns if c.replace(' ', '').endswith(suffix)])
        print(f'  {leg:13s} <- {stem:26s} {cols}')
    print(f'\nrange {out.index.min():%Y-%m} .. {out.index.max():%Y-%m}  '
          f'n={len(out)}  mean ann:')
    print(((1 + out).prod() ** (12 / len(out)) - 1).round(3).to_string())
    print('\nSaved:', C.FF_LONG_LEGS)
    return out


if __name__ == '__main__':
    build()
