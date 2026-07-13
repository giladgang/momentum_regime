"""
2026-07-13-live2026-compustat.py
================================
Exploratory. Live-2026 appendix data for the applied paper
(docs/superpowers/specs/2026-07-13-applied-paper-design.md, decision
2026-07-13: Compustat-spliced extension reported as a separate live
out-of-sample series; primary sample stays CRSP-pure through 2025-11).

CRSP on this subscription is the annual cut (max 2025-12-31). Compustat is
current. This script:
  --stage pull     : comp.secm monthly 2023-01..2026-07 (US commons, USD,
                     NYSE/AMEX/NASDAQ: fic='USA', tpci='0', curcdm='USD',
                     exchg in (11,12,14)); crsp_a_ccm.ccmxpf_lnkhist link
                     table (LU/LC, P/C); comp.secd daily 2025-11..2026-07
                     (same filters; for the VOL feature); FRED refresh.
  --stage validate : splice-quality gates on the 2024-01..2025-12 overlap —
                     link coverage of the CRSP top-1000, per-row return match
                     (trt1m/100 vs CRSP ret_adj), me units (prccm*cshom vs
                     CRSP me), and Compustat-computed DISP/SKEW/N vs the
                     shipped panel columns. Prints a verdict per gate.

Output: experiments/results/live2026/*.parquet
TLS: psycopg2's bundled OpenSSL 3.5 needs classical groups on this network
(see 2026-07-13-import-ext2026.py); the OPENSSL_CONF is set before import.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

OUT = 'experiments/results/live2026'
EXT25 = 'experiments/results/ext2025'
END = '2026-07-31'

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

SEC_FILTER = """
      tpci = '0' AND fic = 'USA' AND exchg IN (11, 12, 14)
"""


def stage_pull():
    from sqlalchemy import create_engine, text
    eng = create_engine(ENG_URL, connect_args={'connect_timeout': 30,
                                               'sslmode': 'require'})

    print('[pull] comp.secm monthly 2023-01..' + END, flush=True)
    secm = pd.read_sql(text(f"""
        SELECT gvkey, iid, datadate, trt1m, prccm, cshom, exchg, tpci,
               fic, curcdm
        FROM comp.secm
        WHERE datadate BETWEEN '2023-01-01' AND '{END}'
          AND {SEC_FILTER} AND curcdm = 'USD'
        ORDER BY gvkey, iid, datadate
    """), eng, parse_dates=['datadate'])
    secm.to_parquet(f'{OUT}/secm_raw.parquet', index=False)
    print(f'  {len(secm):,} rows, {secm.gvkey.nunique():,} gvkeys, '
          f'{secm.datadate.min().date()}..{secm.datadate.max().date()}')

    print('[pull] ccmxpf_lnkhist link table', flush=True)
    lnk = pd.read_sql(text("""
        SELECT gvkey, liid, linktype, linkprim, lpermno, linkdt, linkenddt
        FROM crsp_a_ccm.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
          AND lpermno IS NOT NULL
    """), eng, parse_dates=['linkdt', 'linkenddt'])
    lnk.to_parquet(f'{OUT}/ccm_link.parquet', index=False)
    print(f'  {len(lnk):,} links, {lnk.gvkey.nunique():,} gvkeys')

    print('[pull] comp.secd daily 2025-11..' + END + ' (for VOL)', flush=True)
    secd = pd.read_sql(text(f"""
        SELECT gvkey, iid, datadate, prccd, ajexdi, trfd, cshoc
        FROM comp.secd
        WHERE datadate BETWEEN '2025-11-01' AND '{END}'
          AND {SEC_FILTER} AND curcdd = 'USD'
        ORDER BY datadate
    """), eng, parse_dates=['datadate'])
    secd.to_parquet(f'{OUT}/secd_raw.parquet', index=False)
    print(f'  {len(secd):,} rows, {secd.datadate.min().date()}..'
          f'{secd.datadate.max().date()}')

    print('[pull] FRED refresh', flush=True)
    import pandas_datareader.data as web
    fred = web.DataReader(['BAA', 'AAA', 'VIXCLS', 'DGS10', 'DGS2'],
                          'fred', start='2023-01-01', end=END)
    fred.to_parquet(f'{OUT}/fred_raw.parquet')
    print(f'  {len(fred)} rows, last={fred.index.max().date()}')
    print('[pull] DONE')


def _linked_secm():
    secm = pd.read_parquet(f'{OUT}/secm_raw.parquet')
    lnk = pd.read_parquet(f'{OUT}/ccm_link.parquet')
    lnk['linkenddt'] = lnk['linkenddt'].fillna(pd.Timestamp('2059-12-31'))
    m = secm.merge(lnk, left_on=['gvkey', 'iid'], right_on=['gvkey', 'liid'],
                   how='inner')
    m = m[(m['datadate'] >= m['linkdt']) & (m['datadate'] <= m['linkenddt'])]
    m = m.rename(columns={'lpermno': 'permno'})
    m['permno'] = m['permno'].astype('int64')
    m['ym'] = m['datadate'].dt.to_period('M')
    m['ret_c'] = m['trt1m'] / 100.0
    m['me_c'] = m['prccm'] * m['cshom'] / 1e6
    m = m.dropna(subset=['ret_c'])
    # one row per (permno, ym): keep the largest-cap issue (primary link dupes)
    m = (m.sort_values('me_c', ascending=False)
         .drop_duplicates(['permno', 'ym']))
    m['exchcd'] = m['exchg'].map({11: 1, 12: 2, 14: 3}).astype('Int64')
    return m[['permno', 'ym', 'datadate', 'ret_c', 'me_c', 'prccm', 'exchcd']]


def stage_validate():
    crsp = pd.read_parquet(f'{EXT25}/crsp_msf_ext.parquet',
                           columns=['permno', 'date', 'ret_adj', 'me'])
    crsp['ym'] = pd.to_datetime(crsp['date']).dt.to_period('M')
    comp = _linked_secm()

    win = comp[(comp['ym'] >= '2024-01') & (comp['ym'] <= '2025-12')]
    cw = crsp[(crsp['ym'] >= '2024-01') & (crsp['ym'] <= '2025-12')]
    j = cw.merge(win, on=['permno', 'ym'], how='left',
                 indicator=True)

    # gate 1: coverage of the CRSP top-1000 by matched Compustat rows
    cw1000 = cw.dropna(subset=['me']).copy()
    cw1000['rank'] = cw1000.groupby('ym')['me'].rank(ascending=False,
                                                     method='first')
    top = cw1000[cw1000['rank'] <= 1000][['permno', 'ym']]
    tj = top.merge(win[['permno', 'ym', 'ret_c']], on=['permno', 'ym'],
                   how='left')
    cov = tj['ret_c'].notna().mean()
    print(f'[gate1] top-1000 coverage by linked Compustat: {cov:.2%} '
          f"({'PASS' if cov > 0.95 else 'FAIL'})")

    # gate 2: return match on matched rows
    b = j[j['_merge'] == 'both'].dropna(subset=['ret_adj', 'ret_c'])
    d = (b['ret_adj'] - b['ret_c']).abs()
    corr = b['ret_adj'].corr(b['ret_c'])
    within = (d < 0.001).mean()
    print(f'[gate2] return match n={len(b):,}: corr={corr:.4f}, '
          f'median|diff|={d.median():.5f}, within 10bp={within:.2%} '
          f"({'PASS' if corr > 0.99 and within > 0.90 else 'FAIL'})")

    # gate 3: me units
    bb = j[j['_merge'] == 'both'].dropna(subset=['me', 'me_c'])
    ratio = (bb['me_c'] / bb['me']).median()
    print(f'[gate3] me_c/me median ratio = {ratio:.4f} '
          f"({'PASS' if 0.95 < ratio < 1.05 else 'FAIL'})")

    # gate 4: cross-sectional moments vs shipped panel (DISP, SKEW, N)
    panel = pd.read_parquet(f'{EXT25}/panel_ext.parquet')
    panel['ym'] = pd.to_datetime(panel['date']).dt.to_period('M')
    pw = panel[(panel['ym'] >= '2024-01') & (panel['ym'] <= '2025-11')]
    g = win.groupby('ym')
    disp_c = np.log(g['ret_c'].std())
    skew_c = g['ret_c'].skew()
    n_c = g['permno'].nunique()
    cmpr = pd.DataFrame({'DISP_c': disp_c, 'SKEW_c': skew_c, 'N_c': n_c}) \
        .join(pw.set_index('ym')[['DISP', 'SKEW']]).dropna()
    dcorr = cmpr['DISP_c'].corr(cmpr['DISP'])
    dbias = (cmpr['DISP_c'] - cmpr['DISP']).mean()
    scorr = cmpr['SKEW_c'].corr(cmpr['SKEW'])
    print(f'[gate4] DISP corr={dcorr:.3f} bias={dbias:+.3f} | '
          f'SKEW corr={scorr:.3f} | N_c mean={n_c.mean():.0f} '
          f"({'PASS' if dcorr > 0.9 else 'CHECK'})")
    print('[validate] DONE')


# ── build: panel + stock extension through 2026 (frozen ext2025 conventions) ──

# Frozen train (<2011) z stats — identical to 2026-07-09-import-ext2025.py.
Z_STATS = {
    'DD': (-0.053282, 0.079857), 'VOL': (-2.080973, 0.441349),
    'DISP': (-1.730973, 0.268857), 'REL_N': (-0.008644, 0.023818),
    'CS': (1.113004, 0.461027), 'LVIX': (2.953166, 0.346242),
    'TERM': (1.098251, 0.921577), 'SKEW': (3.813643, 4.168375),
}
WINSOR_COLS = ['DD', 'VOL', 'CS', 'DISP', 'REL_N']


def stage_build():
    panel25 = pd.read_parquet(f'{EXT25}/panel_ext.parquet')      # ..2025-11
    panel25['date'] = pd.to_datetime(panel25['date'])
    panel0 = pd.read_parquet('data/panel.parquet')               # winsor bounds
    msf25 = pd.read_parquet(f'{EXT25}/msf_v2_raw.parquet')       # ..2025-12
    msf25['ym'] = pd.to_datetime(msf25['mthcaldt']).dt.to_period('M')
    midx25 = pd.read_parquet(f'{EXT25}/msf_v2_index_raw.parquet')
    midx25['ym'] = pd.to_datetime(midx25['mthcaldt']).dt.to_period('M')
    dsf25 = pd.read_parquet(f'{EXT25}/dsf_v2_raw.parquet')
    stocks25 = pd.read_parquet(f'{EXT25}/crsp_msf_ext.parquet')  # ..2025-12
    comp = _linked_secm()
    comp = comp[comp['ym'] <= '2026-06']                         # July partial
    secd = pd.read_parquet(f'{OUT}/secd_raw.parquet')
    fred = pd.read_parquet(f'{OUT}/fred_raw.parquet')
    fred.index = pd.to_datetime(fred.index)

    new_yms = pd.period_range('2025-12', '2026-06', freq='M')
    comp_yms = pd.period_range('2026-01', '2026-06', freq='M')

    # ── monthly VW market return: Dec-2025 CRSP index universe, 2026 Compustat
    mi = midx25[midx25['ym'] == '2025-12'].dropna(subset=['mthret',
                                                          'mthprevcap'])
    vw_dec = float(np.average(mi['mthret'], weights=mi['mthprevcap']))
    cme = comp.set_index(['permno', 'ym'])['me_c']
    cprev = cme.groupby('permno').shift(1).rename('me_prev')
    cc = comp.set_index(['permno', 'ym']).join(cprev).reset_index()
    cc = cc.dropna(subset=['ret_c', 'me_prev'])
    vw_comp = (cc.groupby('ym')
               .apply(lambda g: np.average(g['ret_c'], weights=g['me_prev']),
                      include_groups=False))
    # seam check: Compustat VW vs CRSP VW on the 2024-01..2025-12 overlap
    mo = midx25[(midx25['ym'] >= '2024-01')].dropna(subset=['mthret',
                                                            'mthprevcap'])
    vw_crsp_o = (mo.groupby('ym')
                 .apply(lambda g: np.average(g['mthret'],
                                             weights=g['mthprevcap']),
                        include_groups=False))
    both = pd.DataFrame({'crsp': vw_crsp_o, 'comp': vw_comp}).dropna()
    print(f'[build] VW overlap n={len(both)} corr='
          f'{both.crsp.corr(both.comp):.4f} '
          f'max|diff|={(both.crsp - both.comp).abs().max():.5f}', flush=True)
    vwr = pd.concat([pd.Series({pd.Period("2025-12", "M"): vw_dec}),
                     vw_comp.reindex(comp_yms)])

    # ── P / DD ──
    P_hist = panel25.set_index(panel25['date'].dt.to_period('M'))['P']
    P_new = float(P_hist.iloc[-1]) * (1 + vwr).cumprod()
    P_all = pd.concat([P_hist, P_new])
    DD_new = ((P_all - P_all.rolling(12).max()) / P_all.rolling(12).max()
              ).loc[new_yms]

    # ── VOL: daily VW — Dec-2025 from CRSP dsf, 2026 from Compustat secd ──
    dsf25['ym'] = pd.to_datetime(dsf25['dlycaldt']).dt.to_period('M')
    dd = dsf25[dsf25['ym'] == '2025-12'].dropna(subset=['dlyret',
                                                        'dlyprevcap'])
    dvw_dec = (dd.groupby('dlycaldt')
               .apply(lambda g: np.average(g['dlyret'],
                                           weights=g['dlyprevcap']),
                      include_groups=False))
    sd = secd.dropna(subset=['prccd', 'ajexdi', 'cshoc']).copy()
    sd['trfd'] = sd['trfd'].fillna(1.0)
    sd = sd.sort_values(['gvkey', 'iid', 'datadate'])
    sd['pr'] = sd['prccd'] / sd['ajexdi'] * sd['trfd']
    g = sd.groupby(['gvkey', 'iid'])
    sd['dret'] = g['pr'].pct_change()
    sd['cap_prev'] = (sd['prccd'] * sd['cshoc']).groupby(
        [sd['gvkey'], sd['iid']]).shift(1)
    sd = sd.dropna(subset=['dret', 'cap_prev'])
    sd = sd[sd['dret'].abs() < 1.0]
    dvw_comp = (sd.groupby('datadate')
                .apply(lambda x: np.average(x['dret'], weights=x['cap_prev']),
                       include_groups=False))
    dvw_comp.index = pd.to_datetime(dvw_comp.index)
    # seam check on Dec-2025 (both vendors have dailies)
    dcheck = pd.DataFrame({
        'crsp': dvw_dec, 'comp': dvw_comp.reindex(dvw_dec.index)}).dropna()
    print(f'[build] daily VW Dec-2025 check n={len(dcheck)} corr='
          f'{dcheck.crsp.corr(dcheck.comp):.4f}', flush=True)
    dvw = pd.concat([dvw_dec,
                     dvw_comp[dvw_comp.index >= '2026-01-01']]).sort_index()
    VOL_new = np.log(dvw.groupby(dvw.index.to_period('M')).std(ddof=1)
                     * np.sqrt(252)).reindex(new_yms)

    # ── DISP / SKEW: Dec-2025 CRSP, 2026 Compustat ──
    m12 = msf25[msf25['ym'] == '2025-12']['mthret'].dropna()
    DISP_new = pd.concat([
        pd.Series({pd.Period('2025-12', 'M'): np.log(m12.std())}),
        np.log(comp[comp['ym'] >= '2026-01'].groupby('ym')['ret_c'].std())
    ]).reindex(new_yms)
    SKEW_new = pd.concat([
        pd.Series({pd.Period('2025-12', 'M'): m12.skew()}),
        comp[comp['ym'] >= '2026-01'].groupby('ym')['ret_c'].skew()
    ]).reindex(new_yms)

    # ── REL_N: CRSP count history; 2026 Compustat counts scaled for level
    #    continuity at the Dec-2025 seam (disclosed) ──
    stocks25['ym'] = pd.to_datetime(stocks25['date']).dt.to_period('M')
    n_crsp = stocks25.groupby('ym')['permno'].nunique()
    n_comp = comp.groupby('ym')['permno'].nunique()
    scale = n_crsp.loc[pd.Period('2025-12', 'M')] / \
        n_comp.loc[pd.Period('2025-12', 'M')]
    n_2026 = (n_comp.reindex(comp_yms) * scale).round()
    n_all = pd.concat([n_crsp, n_2026])
    n_all = n_all[~n_all.index.duplicated(keep='first')].sort_index()
    REL_N_new = np.log(n_all / n_all.rolling(12).mean()).reindex(new_yms)
    print(f'[build] REL_N seam: N_crsp(2025-12)={n_crsp.iloc[-1]} '
          f'N_comp(2025-12)={n_comp.loc[pd.Period("2025-12", "M")]} '
          f'scale={scale:.3f}', flush=True)

    # ── FRED features ──
    cs_m = (fred['BAA'] - fred['AAA']).dropna()
    CS_new = cs_m.groupby(cs_m.index.to_period('M')).last().reindex(new_yms)
    LVIX_new = np.log(fred['VIXCLS'].resample('ME').mean()
                      ).rename(lambda d: d.to_period('M')).reindex(new_yms)
    TERM_new = ((fred['DGS10'] - fred['DGS2']).resample('ME').mean()
                ).rename(lambda d: d.to_period('M')).reindex(new_yms)

    ext = pd.DataFrame({'ym': new_yms, 'vwretd': vwr.reindex(new_yms).values,
                        'P': P_new.reindex(new_yms).values,
                        'DD': DD_new.values, 'VOL': VOL_new.values,
                        'DISP': DISP_new.values, 'REL_N': REL_N_new.values,
                        'CS': CS_new.values, 'LVIX': LVIX_new.values,
                        'TERM': TERM_new.values, 'SKEW': SKEW_new.values})
    dates_map = {pd.Period('2025-12', 'M'):
                 msf25[msf25['ym'] == '2025-12']['mthcaldt'].max()}
    for ym in comp_yms:
        dates_map[ym] = comp[comp['ym'] == ym]['datadate'].max()
    ext['date'] = ext['ym'].map(dates_map)
    ext['ret_next'] = ext['vwretd'].shift(-1)

    for c in WINSOR_COLS:
        lo, hi = panel0[c].min(), panel0[c].max()
        ext[c] = ext[c].clip(lo, hi)
    for c, (mu, sd_) in Z_STATS.items():
        ext[f'{c}_z'] = (ext[c] - mu) / sd_

    print('[build] extension rows:')
    print(ext[['date', 'DD', 'VOL', 'DISP', 'REL_N', 'CS', 'LVIX', 'TERM',
               'SKEW']].round(4).to_string(index=False))

    keep = list(panel25.columns)
    ext_rows = ext.drop(columns=['ym'])
    for c in keep:
        if c not in ext_rows.columns:
            ext_rows[c] = np.nan
    ext_rows = ext_rows[keep]
    ext_rows = ext_rows[ext_rows['ret_next'].notna()]     # -> through 2026-05
    panel_live = pd.concat([panel25, ext_rows], ignore_index=True)
    panel_live.to_parquet(f'{OUT}/panel_live.parquet', index=False)
    print(f'[build] panel_live: {len(panel_live)} rows '
          f'({panel_live.date.min().date()}..{panel_live.date.max().date()})')

    # ── stock file: ext2025 stocks (+me backfill for 2025 from mthcap)
    #    + Compustat 2026 rows ──
    s25 = stocks25.copy()
    cap = msf25[['permno', 'ym', 'mthcap']].drop_duplicates(['permno', 'ym'])
    s25 = s25.merge(cap, on=['permno', 'ym'], how='left')
    s25['me'] = s25['me'].fillna(s25['mthcap'] / 1000.0)
    s25.drop(columns=['mthcap'], inplace=True)

    c26 = comp[comp['ym'] >= '2026-01'].copy()
    stock_new = pd.DataFrame({
        'permno': c26['permno'].astype('int64'),
        'date': c26['datadate'],
        'ret': pd.array(c26['ret_c'], dtype='Float64'),
        'prc': pd.array(c26['prccm'], dtype='Float64'),
        'exchcd': c26['exchcd'],
        'shrcd': pd.array([10] * len(c26), dtype='Int64'),
        'year_month': c26['ym'],
        'ret_adj': c26['ret_c'].astype(float),
        'me': c26['me_c'].astype(float),
    })
    for c in s25.columns:
        if c not in stock_new.columns:
            stock_new[c] = np.nan
    stock_new = stock_new[s25.columns]
    stocks_live = pd.concat([s25, stock_new], ignore_index=True)
    stocks_live = stocks_live.sort_values(['permno', 'date']
                                          ).reset_index(drop=True)
    stocks_live.to_parquet(f'{OUT}/stocks_live.parquet', index=False)
    print(f'[build] stocks_live: {len(stocks_live):,} rows '
          f'(+{len(stock_new):,} Compustat 2026), through '
          f'{stocks_live.date.max().date()}')
    print('[build] DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pull', 'validate', 'build'],
                    required=True)
    a = ap.parse_args()
    {'pull': stage_pull, 'validate': stage_validate,
     'build': stage_build}[a.stage]()
