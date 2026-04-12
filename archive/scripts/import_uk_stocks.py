"""
import_uk_stocks.py
===================
Downloads UK stock-level data from WRDS (Compustat Global + Global Security
Daily/Monthly) for all ordinary UK shares listed on the London Stock Exchange.

This mirrors the US pipeline (import_crsp_stocks.py + import_wrds.py) but uses
Compustat Global tables instead of CRSP + Compustat North America.

Data sources (all via WRDS):
  - comp.g_secd        : Global Security Daily (prices, returns, shares)
  - comp.g_fundq       : Global Fundamentals Quarterly (accounting data)
  - comp.g_security    : Security-level identifiers (isin, exchange, issue type)
  - comp.g_company     : Company-level identifiers (country, name)

Output
------
  data/uk_stock_panel.parquet   — monthly stock returns + fundamentals
  data/uk_market_panel.parquet  — monthly market-level features for HMM

Variable mapping (US → UK):
  permno  → gvkey + iid (unique security identifier)
  exchcd  → excntry = 'GBR' + LSE exchange codes
  shrcd   → tpci (issue type: '0' = common equity)
  ret     → computed from prccd (closing price, adjusted)
  prc     → prccd
  shrout  → cshoc (shares outstanding, company-level)
"""

import sys
import pandas as pd
import numpy as np
from datetime import date

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE = '1990-01-01'
END_DATE = date.today().strftime('%Y-%m-%d')
OUTPUT_STOCK = 'data/uk_stock_panel.parquet'
OUTPUT_MARKET = 'data/uk_market_panel.parquet'

# ── Connect to WRDS ──────────────────────────────────────────────────────────

try:
    import wrds
except ImportError:
    sys.exit("ERROR: wrds package not installed. Run: pip install wrds")

print("Connecting to WRDS ...")
try:
    db = wrds.Connection()
except Exception as e:
    sys.exit(f"ERROR: Could not connect to WRDS.\n{e}")

print(f"Connected. Pulling UK stock data: {START_DATE} -> {END_DATE}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Monthly stock returns from Compustat Global Security Daily
# ══════════════════════════════════════════════════════════════════════════════
# Compustat Global does not have a monthly file like CRSP's msf.
# We pull daily data and aggregate to monthly returns ourselves.
# Alternatively, comp.g_secm (monthly) exists but has limited coverage.
# We try g_secm first; if coverage is poor, fall back to g_secd.

print("\n[ 1/4 ] Pulling UK monthly security data ...")

# First try the monthly table (g_secm) which is cleaner
query_monthly = f"""
    SELECT
        s.gvkey,
        s.iid,
        s.datadate,
        s.prccd,        -- closing price
        s.prchd,        -- high price
        s.prcld,        -- low price
        s.cshtrd,       -- trading volume (shares)
        s.curcdd,       -- currency code
        sec.isin,
        sec.tpci,       -- issue type (0 = common equity)
        sec.exchg,      -- exchange code
        c.conm,         -- company name
        c.fic,          -- country of incorporation
        c.loc           -- country of headquarters
    FROM comp.g_secd AS s
    INNER JOIN comp.g_security AS sec
        ON s.gvkey = sec.gvkey AND s.iid = sec.iid
    INNER JOIN comp.g_company AS c
        ON s.gvkey = c.gvkey
    WHERE c.loc = 'GBR'                          -- UK-headquartered
      AND sec.excntry = 'GBR'                     -- traded in UK
      AND sec.tpci = '0'                          -- common equity only
      AND s.datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND s.prccd IS NOT NULL
      AND s.prccd > 0
      AND s.curcdd = 'GBP'                        -- GBP-denominated only
    ORDER BY s.gvkey, s.iid, s.datadate
"""

try:
    daily = db.raw_sql(query_monthly, date_cols=['datadate'])
    print(f"  Retrieved {len(daily):,} daily observations")
except Exception as e:
    db.close()
    sys.exit(f"ERROR: Security query failed.\n{e}")

# ── Aggregate daily to monthly ────────────────────────────────────────────────

daily = daily.sort_values(['gvkey', 'iid', 'datadate']).reset_index(drop=True)
daily['year_month'] = daily['datadate'].dt.to_period('M')

# Create a unique security identifier
daily['secid'] = daily['gvkey'].astype(str) + '_' + daily['iid'].astype(str)

# Monthly return = last price / first price - 1 (within each month)
def monthly_agg(grp):
    grp = grp.sort_values('datadate')
    return pd.Series({
        'date': grp['datadate'].iloc[-1],           # month-end date
        'prc': grp['prccd'].iloc[-1],                # end-of-month price
        'prc_open': grp['prccd'].iloc[0],            # start-of-month price
        'ret': grp['prccd'].iloc[-1] / grp['prccd'].iloc[0] - 1,
        'n_days': len(grp),                          # trading days
        'volume': grp['cshtrd'].sum(),
        'conm': grp['conm'].iloc[0],
        'isin': grp['isin'].iloc[0],
    })

print("  Aggregating daily to monthly ...")
msf = (daily.groupby(['secid', 'gvkey', 'iid', 'year_month'])
       .apply(monthly_agg, include_groups=False)
       .reset_index())

msf['date'] = pd.to_datetime(msf['date'])
msf = msf.sort_values(['secid', 'date']).reset_index(drop=True)

# Filter: require at least 10 trading days in the month
msf = msf[msf['n_days'] >= 10].copy()

# Filter: minimum price of 10 pence (equivalent to US $1 filter)
msf = msf[msf['prc'] >= 0.10].copy()

n_stocks = msf['secid'].nunique()
print(f"  Monthly panel: {len(msf):,} obs, {n_stocks:,} unique securities")
print(f"  Date range: {msf['date'].min().date()} -> {msf['date'].max().date()}")

# ── Market equity ─────────────────────────────────────────────────────────────

print("\n[ 2/4 ] Pulling shares outstanding ...")

query_shares = """
    SELECT gvkey, datadate, cshoc    -- shares outstanding (company level, millions)
    FROM comp.g_fundq
    WHERE gvkey IN (SELECT DISTINCT gvkey FROM comp.g_company WHERE loc = 'GBR')
      AND cshoc IS NOT NULL
    ORDER BY gvkey, datadate
"""
shares = db.raw_sql(query_shares, date_cols=['datadate'])
shares = shares.sort_values(['gvkey', 'datadate']).reset_index(drop=True)

# merge_asof: for each stock-month, take the most recent shares outstanding
msf_sorted = msf.sort_values('date').reset_index(drop=True)
shares_sorted = shares.sort_values('datadate').reset_index(drop=True)

msf_sorted = pd.merge_asof(
    msf_sorted,
    shares_sorted.rename(columns={'datadate': 'shares_date'}),
    left_on='date',
    right_on='shares_date',
    by='gvkey',
    direction='backward'
)

# ME = price * shares outstanding (cshoc is in millions, price is in GBP)
# Result: ME in millions of GBP
msf_sorted['me'] = msf_sorted['prc'] * msf_sorted['cshoc']
msf_sorted = msf_sorted[msf_sorted['me'] > 0].copy()

print(f"  Stocks with valid market equity: {msf_sorted['secid'].nunique():,}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Fundamentals from Compustat Global Quarterly
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 3/4 ] Pulling Compustat Global fundamentals ...")

query_fundq = f"""
    SELECT
        f.gvkey, f.datadate, f.fqtr, f.fyearq,
        f.atq,      -- total assets
        f.ceqq,     -- common equity (book value)
        f.ibq,      -- income before extraordinary items
        f.revtq,    -- total revenue
        f.cogsq,    -- cost of goods sold
        f.dlttq,    -- long-term debt
        f.dlcq      -- debt in current liabilities
    FROM comp.g_fundq AS f
    INNER JOIN comp.g_company AS c
        ON f.gvkey = c.gvkey
    WHERE c.loc = 'GBR'
      AND f.datadate BETWEEN '1986-01-01' AND '{END_DATE}'
      AND f.indfmt  = 'INDL'
      AND f.datafmt = 'HIST_STD'
      AND f.popsrc  = 'I'          -- international
      AND f.consol  = 'C'
      AND f.fqtr IS NOT NULL
      AND f.curcdq = 'GBP'         -- GBP reporting currency
    ORDER BY f.gvkey, f.datadate
"""

try:
    fundq = db.raw_sql(query_fundq, date_cols=['datadate'])
    print(f"  Retrieved {len(fundq):,} quarterly observations")
except Exception as e:
    db.close()
    sys.exit(f"ERROR: Fundamentals query failed.\n{e}")

db.close()
print("  WRDS connection closed.")

# ── Compute fundamental features (same as US pipeline) ────────────────────────

fundq = fundq.sort_values(['gvkey', 'datadate']).reset_index(drop=True)

# YoY changes: shift(4) = same quarter one year ago
fundq['earnings_growth'] = (
    fundq.groupby('gvkey')['ibq']
    .transform(lambda x: x / x.shift(4) - 1)
)
fundq['asset_growth'] = (
    fundq.groupby('gvkey')['atq']
    .transform(lambda x: x / x.shift(4) - 1)
)

# Ratios
fundq['roe'] = fundq['ibq'] / fundq['ceqq'].replace(0, np.nan)
fundq['leverage'] = (fundq['dlttq'].fillna(0) + fundq['dlcq'].fillna(0)) / \
                     fundq['atq'].replace(0, np.nan)
fundq['gross_profit_a'] = (fundq['revtq'] - fundq['cogsq'].fillna(0)) / \
                            fundq['atq'].replace(0, np.nan)

# Book equity must be positive
fundq['ceqq_pos'] = fundq['ceqq'].where(fundq['ceqq'] > 0, np.nan)

# 4-month reporting lag
fundq['avail_date'] = (fundq['datadate'] + pd.DateOffset(months=4)) + pd.offsets.MonthEnd(0)

# Merge fundamentals onto stock panel
fund_feat_cols = ['ceqq_pos', 'roe', 'earnings_growth', 'leverage',
                  'asset_growth', 'gross_profit_a']

fundq_sorted = (fundq[['gvkey', 'avail_date'] + fund_feat_cols]
                .sort_values('avail_date')
                .drop_duplicates(subset=['gvkey', 'avail_date'], keep='last')
                .reset_index(drop=True))

msf_final = msf_sorted.sort_values('date').reset_index(drop=True)

msf_final = pd.merge_asof(
    msf_final,
    fundq_sorted,
    left_on='date',
    right_on='avail_date',
    by='gvkey',
    direction='backward'
)

# Book-to-market
msf_final['bm'] = msf_final['ceqq_pos'] / msf_final['me'].replace(0, np.nan)

# Earnings growth: signed log transform
eg = msf_final['earnings_growth'].astype(float)
msf_final['earnings_growth'] = np.sign(eg) * np.log1p(np.abs(eg))

# Log market equity
msf_final['log_me'] = np.log(msf_final['me'].replace(0, np.nan))

# ── Momentum lookbacks ────────────────────────────────────────────────────────

print("\n  Computing momentum lookbacks ...")
msf_final = msf_final.sort_values(['secid', 'date']).reset_index(drop=True)

for k in range(1, 13):
    msf_final[f'mom_{k}'] = (
        msf_final.groupby('secid')['ret']
        .transform(lambda x: (1 + x).rolling(k).apply(np.prod, raw=True) - 1)
        .shift(1)  # skip current month
    )

# Forward return (target)
msf_final['ret_fwd'] = msf_final.groupby('secid')['ret'].shift(-1)

# ── Winsorize (training sample bounds only) ───────────────────────────────────

winsor_bounds = {
    'bm':              (0.01, 0.99),
    'roe':             (0.01, 0.99),
    'earnings_growth': (0.01, 0.99),
    'leverage':        (0.01, 0.99),
    'asset_growth':    (0.05, 0.95),
    'gross_profit_a':  (0.01, 0.99),
}
train_mask = msf_final['date'] < '2011-01-01'
for col, (lo_q, hi_q) in winsor_bounds.items():
    vals = msf_final[col].astype(float)
    lo = vals[train_mask].quantile(lo_q)
    hi = vals[train_mask].quantile(hi_q)
    msf_final[col] = vals.clip(lo, hi)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: Market-level features for HMM
# ══════════════════════════════════════════════════════════════════════════════

print("\n[ 4/4 ] Computing market-level HMM features ...")

# Value-weighted market return each month
msf_final['me_lag'] = msf_final.groupby('secid')['me'].shift(1)
monthly_mkt = msf_final.dropna(subset=['ret', 'me_lag']).copy()

def compute_market_month(grp):
    weights = grp['me_lag'] / grp['me_lag'].sum()
    mkt_ret = (grp['ret'] * weights).sum()
    disp = grp['ret'].std()
    n_stocks = len(grp)
    return pd.Series({
        'mkt_ret': mkt_ret,
        'disp': np.log(disp) if disp > 0 else np.nan,
        'n_stocks': n_stocks,
    })

market = (monthly_mkt.groupby('date')
          .apply(compute_market_month, include_groups=False)
          .reset_index())
market = market.sort_values('date').reset_index(drop=True)

# Cumulative market index for drawdown
market['cum_index'] = (1 + market['mkt_ret']).cumprod() * 100

# DD: drawdown from 12-month rolling peak
market['peak_12m'] = market['cum_index'].rolling(12, min_periods=1).max()
market['dd'] = (market['cum_index'] - market['peak_12m']) / market['peak_12m']

# VOL: log annualised volatility (use monthly returns, annualise)
market['vol'] = np.log(market['mkt_ret'].rolling(12, min_periods=6).std() * np.sqrt(12))

# DISP: already computed above (log cross-sectional std of returns)

# REL_N: relative market participation
market['n_avg_12m'] = market['n_stocks'].rolling(12, min_periods=6).mean()
market['rel_n'] = np.log(market['n_stocks'] / market['n_avg_12m'])

# ── Save ──────────────────────────────────────────────────────────────────────

import os
os.makedirs('data', exist_ok=True)

# Clean up intermediate columns
keep_cols = ['secid', 'gvkey', 'iid', 'date', 'year_month', 'conm', 'isin',
             'prc', 'ret', 'me', 'log_me',
             'bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth',
             'gross_profit_a',
             'ret_fwd'] + [f'mom_{k}' for k in range(1, 13)]
keep_cols = [c for c in keep_cols if c in msf_final.columns]
msf_out = msf_final[keep_cols].copy()
msf_out = msf_out.sort_values(['secid', 'date']).reset_index(drop=True)

msf_out.to_parquet(OUTPUT_STOCK, index=False)
print(f"\n  Saved stock panel: {OUTPUT_STOCK}")
print(f"    {len(msf_out):,} obs, {msf_out['secid'].nunique():,} securities")
print(f"    {msf_out['date'].min().date()} -> {msf_out['date'].max().date()}")

market_cols = ['date', 'mkt_ret', 'dd', 'vol', 'disp', 'rel_n', 'n_stocks']
market_out = market[market_cols].dropna().copy()
market_out.to_parquet(OUTPUT_MARKET, index=False)
print(f"\n  Saved market panel: {OUTPUT_MARKET}")
print(f"    {len(market_out)} months")
print(f"    {market_out['date'].min().date()} -> {market_out['date'].max().date()}")

print("\n" + "=" * 60)
print("  UK IMPORT COMPLETE")
print("=" * 60)
