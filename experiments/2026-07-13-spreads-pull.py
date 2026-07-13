"""S5 spread-engine data: dsf_v2 dailies (high/low/close/bid/ask/volume)
2010-12..2025-12 for every permno that ever enters the applied top-1000.
Gilad-authorized WRDS pull (2026-07-13). Resume-safe per year.
Output: experiments/results/spreads/dsf_v2_<year>.parquet
"""
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

OUT = 'experiments/results/spreads'
os.makedirs(OUT, exist_ok=True)
_CNF = os.path.join(OUT, 'openssl_classical.cnf')
if not os.path.exists(_CNF):
    with open(_CNF, 'w') as f:
        f.write('openssl_conf = openssl_init\n[openssl_init]\n'
                'ssl_conf = ssl_sect\n[ssl_sect]\n'
                'system_default = system_default_sect\n'
                '[system_default_sect]\nGroups = X25519:P-256\n')
os.environ['OPENSSL_CONF'] = os.path.abspath(_CNF)

import pandas as pd                                             # noqa: E402

permnos = set()
for f in glob.glob('paper/results/xsec/xsec_*.parquet'):
    permnos |= set(pd.read_parquet(f, columns=['permno'])['permno'])
permnos = sorted(int(p) for p in permnos)
print(f'[spreads] {len(permnos)} permnos', flush=True)

from sqlalchemy import create_engine, text                      # noqa: E402
eng = create_engine(
    'postgresql+psycopg2://giladgang@wrds-pgdata.wharton.upenn.edu:9737/wrds',
    connect_args={'connect_timeout': 30, 'sslmode': 'require'})

plist = ','.join(str(p) for p in permnos)
for year in range(2010, 2026):
    dst = f'{OUT}/dsf_v2_{year}.parquet'
    if os.path.exists(dst):
        print(f'[spreads] SKIP {year}', flush=True)
        continue
    lo = f'{year}-12-01' if year == 2010 else f'{year}-01-01'
    df = pd.read_sql(text(f"""
        SELECT permno, dlycaldt, dlyhigh, dlylow, dlyclose, dlyprc,
               dlybid, dlyask, dlyvol, dlyret
        FROM crsp.dsf_v2
        WHERE dlycaldt BETWEEN '{lo}' AND '{year}-12-31'
          AND permno IN ({plist})
        ORDER BY permno, dlycaldt
    """), eng, parse_dates=['dlycaldt'])
    df.to_parquet(dst, index=False)
    print(f'[spreads] {year}: {len(df):,} rows', flush=True)
print('[spreads] DONE', flush=True)
