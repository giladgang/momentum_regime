"""
Pull Compustat cash-flow features (oancfy, capxy) from WRDS, link to
permno, apply the same 4-month reporting lag as the existing fundamentals,
and save as a supplementary panel keyed on (permno, date) that the ablation
script can merge onto crsp_msf_raw.parquet.

Features produced:
    cfo_a    : trailing 12m operating cash flow / total assets
    fcf_a    : trailing 12m (operating CF - capex) / total assets
    accruals : (trailing 12m ibq - trailing 12m oancf) / atq  (Sloan 1996)

Output:
    data/cashflow_features.parquet  (permno, date, cfo_a, fcf_a, accruals)
"""

import numpy as np
import pandas as pd
import wrds

print("Connecting to WRDS ...")
db = wrds.Connection()

print("Pulling comp.fundq (cash flow + net income + assets) ...")
fundq = db.raw_sql("""
    SELECT f.gvkey, f.datadate, f.fqtr, f.fyearq,
           f.atq, f.ibq,
           f.oancfy,
           f.capxy
    FROM comp.fundq AS f
    INNER JOIN crsp.ccmxpf_lnkhist AS lnk
        ON  f.gvkey    = lnk.gvkey
        AND lnk.linktype  IN ('LU', 'LC')
        AND lnk.linkprim  IN ('P', 'C')
    WHERE f.datadate BETWEEN '1986-01-01' AND '2025-12-31'
        AND f.indfmt  = 'INDL'
        AND f.datafmt = 'STD'
        AND f.popsrc  = 'D'
        AND f.consol  = 'C'
        AND f.fqtr IS NOT NULL
    ORDER BY f.gvkey, f.datadate
""", date_cols=['datadate'])

print(f"  rows: {len(fundq):,}")

print("Pulling ccm link table ...")
ccm_link = db.raw_sql("""
    SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LU', 'LC')
      AND linkprim IN ('P', 'C')
""", date_cols=['linkdt', 'linkenddt'])
db.close()

# ── YTD → quarterly (within fiscal year) ──
fundq = fundq.sort_values(['gvkey', 'fyearq', 'fqtr']).reset_index(drop=True)
for ytd_col, q_col in [('oancfy', 'oancf_q'), ('capxy', 'capx_q')]:
    within_fy = fundq.groupby(['gvkey', 'fyearq'])[ytd_col].diff()
    fundq[q_col] = within_fy.fillna(fundq[ytd_col])  # Q1 YTD == Q1 quarterly

# ── Trailing 12m sums ──
fundq = fundq.sort_values(['gvkey', 'datadate']).reset_index(drop=True)
for q_col, ttm_col in [('oancf_q', 'oancf_ttm'), ('capx_q', 'capx_ttm'),
                        ('ibq',    'ibq_ttm')]:
    fundq[ttm_col] = (fundq.groupby('gvkey')[q_col]
                     .transform(lambda x: x.rolling(4, min_periods=4).sum()))

# ── Features ──
at_nonzero = fundq['atq'].replace(0, np.nan)
fundq['cfo_a']    = fundq['oancf_ttm'] / at_nonzero
fundq['fcf_a']    = (fundq['oancf_ttm'] - fundq['capx_ttm']) / at_nonzero
fundq['accruals'] = (fundq['ibq_ttm'] - fundq['oancf_ttm']) / at_nonzero

# ── 4-month reporting lag ──
fundq['avail_date'] = ((fundq['datadate'] + pd.DateOffset(months=4))
                       + pd.offsets.MonthEnd(0))

# ── Link gvkey → permno ──
ccm_link['linkenddt'] = ccm_link['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
ccm_link['permno'] = ccm_link['permno'].astype(int)

linked = fundq.merge(ccm_link[['gvkey', 'permno', 'linkdt', 'linkenddt']],
                     on='gvkey', how='inner')
linked = linked[(linked['datadate'] >= linked['linkdt']) &
                (linked['datadate'] <= linked['linkenddt'])].copy()

linked = (linked.sort_values(['permno', 'avail_date', 'datadate'])
         .drop_duplicates(subset=['permno', 'avail_date'], keep='last')
         .reset_index(drop=True))

# ── merge_asof onto the CRSP monthly dates ──
print("Merging onto CRSP monthly panel ...")
msf = pd.read_parquet('data/crsp_msf_raw.parquet', columns=['permno', 'date'])
msf['date'] = pd.to_datetime(msf['date'])
msf = msf.sort_values('date').reset_index(drop=True)

linked_sorted = (linked[['permno', 'avail_date', 'cfo_a', 'fcf_a', 'accruals']]
                .sort_values('avail_date').reset_index(drop=True))

merged = pd.merge_asof(
    msf, linked_sorted,
    left_on='date', right_on='avail_date',
    by='permno', direction='backward'
)

# ── Winsorize using training-period statistics (pre-2011) ──
train_mask = merged['date'] < '2011-01-01'
for col in ['cfo_a', 'fcf_a', 'accruals']:
    vals = merged[col].astype(float)
    lo = vals[train_mask].quantile(0.01)
    hi = vals[train_mask].quantile(0.99)
    merged[col] = vals.clip(lo, hi)
    n_valid = merged[col].notna().sum()
    print(f"  {col}: winsorize [{lo:+.3f}, {hi:+.3f}]  "
          f"coverage={n_valid/len(merged):.1%}")

out = merged[['permno', 'date', 'cfo_a', 'fcf_a', 'accruals']]
out.to_parquet('data/cashflow_features.parquet', index=False)
print(f"\nSaved: data/cashflow_features.parquet  ({len(out):,} rows)")
