"""
2026-07-13-import-ext2026.py
============================
Exploratory. S0 pull for the applied paper study
(docs/superpowers/specs/2026-07-13-applied-paper-design.md): refresh the WRDS
CIZ (v2) extension through the latest available month (~2026-05/06 given CRSP
lag). PULL ONLY — the integrated panel/stock build is ported into
paper/src/data_import.py per the spec. Writes ONLY experiments/results/ext2026/.

Same verified filters/conventions as 2026-07-09-import-ext2025.py, same query
start dates (so the ext2025 overlap can be row-identity checked), plus a new
verification: the re-pulled rows must match the 2025-07-09 pull on the common
window (catches CIZ restatements; reported, not hard-failed).

TLS note (diagnosed 2026-07-13): psycopg2's bundled OpenSSL 3.5 sends a
post-quantum-group ClientHello too large for this network path (VPN MTU) and
the handshake black-holes. Workaround: force classical groups via an
OPENSSL_CONF written by this script — set in the environment BEFORE any
sqlalchemy/psycopg2 import.

Usage:
    .venv/bin/python experiments/2026-07-13-import-ext2026.py --stage pull
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

OUT = 'experiments/results/ext2026'
OLD = 'experiments/results/ext2025'
END = '2026-07-31'          # upper bound; actual max set by CRSP availability

# ── OPENSSL_CONF workaround: must be in env before libssl loads ──────────────
_CNF = os.path.join(OUT, 'openssl_classical.cnf')
os.makedirs(OUT, exist_ok=True)
if not os.path.exists(_CNF):
    with open(_CNF, 'w') as f:
        f.write('openssl_conf = openssl_init\n[openssl_init]\n'
                'ssl_conf = ssl_sect\n[ssl_sect]\n'
                'system_default = system_default_sect\n'
                '[system_default_sect]\nGroups = X25519:P-256\n')
os.environ['OPENSSL_CONF'] = os.path.abspath(_CNF)

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

ENG_URL = "postgresql+psycopg2://giladgang@wrds-pgdata.wharton.upenn.edu:9737/wrds"

CIZ_FILTER = """
      sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM'
  AND usincflg = 'Y' AND issuertype IN ('ACOR', 'CORP')
  AND primaryexch IN ('N', 'A', 'Q') AND conditionaltype = 'RW'
  AND tradingstatusflg = 'A'
"""


def stage_pull():
    from sqlalchemy import create_engine, text
    eng = create_engine(ENG_URL, connect_args={'connect_timeout': 30,
                                               'sslmode': 'require'})

    print(f'[pull] msf_v2 monthly 2023-10..{END} ...', flush=True)
    msf = pd.read_sql(text(f"""
        SELECT permno, mthcaldt, mthret, mthretx, mthprc, mthcap, mthprevcap,
               shrout, primaryexch
        FROM crsp.msf_v2
        WHERE mthcaldt BETWEEN '2023-10-01' AND '{END}'
          AND {CIZ_FILTER}
        ORDER BY permno, mthcaldt
    """), eng, parse_dates=['mthcaldt'])
    msf.to_parquet(f'{OUT}/msf_v2_raw.parquet', index=False)
    print(f'  {len(msf):,} rows, {msf.permno.nunique():,} permnos, '
          f'{msf.mthcaldt.min().date()}..{msf.mthcaldt.max().date()}')

    print(f'[pull] dsf_v2 daily 2024-11..{END} (index universe, for VOL) ...',
          flush=True)
    dsf = pd.read_sql(text(f"""
        SELECT permno, dlycaldt, dlyret, dlyprevcap
        FROM crsp.dsf_v2
        WHERE dlycaldt BETWEEN '2024-11-01' AND '{END}'
          AND primaryexch IN ('N', 'A', 'Q')
        ORDER BY dlycaldt
    """), eng, parse_dates=['dlycaldt'])
    dsf.to_parquet(f'{OUT}/dsf_v2_raw.parquet', index=False)
    print(f'  {len(dsf):,} rows, {dsf.dlycaldt.min().date()}..'
          f'{dsf.dlycaldt.max().date()}')

    print('[pull] msf_v2 monthly INDEX universe (unfiltered) ...', flush=True)
    midx = pd.read_sql(text(f"""
        SELECT mthcaldt, mthret, mthprevcap
        FROM crsp.msf_v2
        WHERE mthcaldt BETWEEN '2023-10-01' AND '{END}'
          AND primaryexch IN ('N', 'A', 'Q')
        ORDER BY mthcaldt
    """), eng, parse_dates=['mthcaldt'])
    midx.to_parquet(f'{OUT}/msf_v2_index_raw.parquet', index=False)
    print(f'  {len(midx):,} rows')

    print('[pull] FRED series ...', flush=True)
    import pandas_datareader.data as web
    fred = web.DataReader(['BAA', 'AAA', 'VIXCLS', 'DGS10', 'DGS2'],
                          'fred', start='2023-01-01', end=END)
    fred.to_parquet(f'{OUT}/fred_raw.parquet')
    print(f'  {len(fred)} rows, cols={list(fred.columns)}, '
          f'last={fred.index.max().date()}')

    # ── verification 1: VW market return vs classic msi (2024 overlap) ──
    print('[verify] v2-derived VW market return vs classic msi (2024) ...')
    mi = midx[(midx.mthcaldt >= '2024-01-01')
              & (midx.mthcaldt <= '2024-12-31')].copy()
    mi['ym'] = mi.mthcaldt.dt.to_period('M')
    vwr = (mi.dropna(subset=['mthret', 'mthprevcap'])
           .groupby('ym')
           .apply(lambda g: np.average(g['mthret'], weights=g['mthprevcap']),
                  include_groups=False))
    msi = pd.read_sql(text("SELECT date, vwretd FROM crsp.msi "
                           "WHERE date BETWEEN '2024-01-01' AND '2024-12-31'"),
                      eng, parse_dates=['date'])
    msi['ym'] = msi.date.dt.to_period('M')
    both = pd.DataFrame({'v2': vwr}).join(msi.set_index('ym')['vwretd']).dropna()
    diff = (both['v2'] - both['vwretd']).abs()
    print(f'  max |diff| = {diff.max():.5f}  corr = '
          f'{both["v2"].corr(both["vwretd"]):.5f}')
    assert diff.max() < 0.001, 'VW market return mismatch vs classic msi'

    # ── verification 2: v2 mthret vs classic ret_adj (2024-06 overlap) ──
    print('[verify] v2 mthret vs classic ret_adj (2024-06, matched permnos) ...')
    classic = pd.read_parquet('data/crsp_msf_raw.parquet',
                              columns=['permno', 'date', 'ret_adj', 'me'])
    c6 = classic[classic.date.dt.to_period('M') == '2024-06']
    m = msf.copy()
    m['ym'] = m.mthcaldt.dt.to_period('M')
    v6 = m[m['ym'] == '2024-06'][['permno', 'mthret', 'mthcap']]
    j = c6.merge(v6, on='permno')
    rdiff = (j['ret_adj'] - j['mthret']).abs()
    print(f'  matched {len(j):,} permnos | ret max|diff|={rdiff.max():.4f} '
          f'p99={rdiff.quantile(0.99):.5f}')
    assert rdiff.quantile(0.99) < 0.002, 'v2 vs classic return mismatch'

    # ── verification 3: row identity vs the 2026-07-09 pull (restatements) ──
    print('[verify] re-pull vs ext2025 pull on common window ...')
    old = pd.read_parquet(f'{OLD}/msf_v2_raw.parquet',
                          columns=['permno', 'mthcaldt', 'mthret'])
    new = msf[msf.mthcaldt <= old.mthcaldt.max()][['permno', 'mthcaldt',
                                                   'mthret']]
    jj = old.merge(new, on=['permno', 'mthcaldt'], how='outer',
                   suffixes=('_old', '_new'), indicator=True)
    n_old = int((jj['_merge'] == 'left_only').sum())
    n_new = int((jj['_merge'] == 'right_only').sum())
    b = jj[jj['_merge'] == 'both'].dropna(subset=['mthret_old', 'mthret_new'])
    rmax = (b['mthret_old'] - b['mthret_new']).abs().max()
    print(f'  rows only-in-old={n_old:,} only-in-new={n_new:,} '
          f'max|ret diff| on both={rmax:.6f}')
    if n_old + n_new > 0 or rmax > 1e-8:
        print('  NOTE: CIZ restatements present — disclosed, new pull is primary')

    print('[pull] DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pull'], required=True)
    ap.parse_args()
    stage_pull()
