"""
import_wrds.py
=============
Pulls raw data from WRDS (CRSP) and FRED, constructs monthly regime features,
cleans and standardizes them, and splits into train/test sets.

Features constructed
--------------------
DD     : Drawdown -- how far the market is below its 12-month peak. Always <= 0.
         Deep negative = market in significant decline.
VOL    : Log realized volatility -- annualized, from daily CRSP returns.
         High = turbulent market. Computed as log(std(r_d) * sqrt(252)).
DISP   : Log cross-sectional return dispersion -- std of individual stock returns
         each month. High = heterogeneous market stress, corr ~0.27 with VOL.
REL_N  : Relative market participation -- log(N_stocks / 12-month MA of N_stocks).
         Negative = accelerating delistings/exits, signals market distress.
CS     : Credit spread -- BAA minus AAA Moody's yield (%). Widens during stress.
         Kept for downstream models but no longer used in HMM.
LVIX   : Log VIX -- market-implied expected volatility from options. From 1990 only.
         Kept for downstream models but no longer used in HMM.
GDP_g  : Log quarterly GDP growth. Available only in the first month of each quarter.
         Negative = recession. Not forward-filled to avoid artificial autocorrelation.

Target variable
---------------
ret_next : Next month's value-weighted market return (vwretd shifted -1).
           Signal z_t observed at month-end t is used to predict ret_{t+1}.

Train/test split
----------------
Train : 1970-2010 (or 1990-2010 for VIX-dependent models)
Test  : 2011-2025
Standardization uses train mean/std only -- no leakage into test.
Expanding-window refit happens during HMM fitting, not here.

Outputs
-------
panel     : Full cleaned monthly dataframe
train_z   : Standardized train features (core 4: DD, VOL, DISP, REL_N)
test_z    : Standardized test features (core 4: DD, VOL, DISP, REL_N)
features_stress.png : Sanity check plot -- DD, VOL, DISP, REL_N
features_macro.png  : Sanity check plot -- CS, LVIX, GDP_g
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend: saves to file, no window

import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from scipy.stats import mstats

# ── Section 1: Pull raw data ──────────────────────────────────────────────────

print("[ 1/4 ] Pulling market & stock data from WRDS ...")
db = wrds.Connection()

# A) CRSP monthly market index
#    vwretd = value-weighted return including dividends
#    Used to build cumulative price index P_t and drawdown DD_t
msi = db.raw_sql("""
    SELECT date, vwretd
    FROM crsp.msi
    WHERE date BETWEEN '1970-01-01' AND '2025-12-31'
    ORDER BY date
""", date_cols=['date'])

msi['P'] = 100 * (1 + msi['vwretd']).cumprod()  # P_t = 100 * prod(1 + r)

# B) CRSP daily market returns
#    Used to compute monthly realized volatility RV_t
dsi = db.raw_sql("""
    SELECT date, vwretd
    FROM crsp.dsi
    WHERE date BETWEEN '1970-01-01' AND '2025-12-31'
    ORDER BY date
""", date_cols=['date'])

dsi['year_month'] = dsi['date'].dt.to_period('M')
rv = (
    dsi.groupby('year_month')['vwretd']
    .apply(lambda r: r.std() * np.sqrt(252))  # annualized realized vol (mean-centered)
    .reset_index()
    .rename(columns={'vwretd': 'RV'})
)

# C) Credit spread: BAA minus AAA Moody's yield (FRED)
#    Widens when investors demand more premium for holding risky debt
fred_data = web.DataReader(['BAA', 'AAA'], 'fred', start='1970-01-01', end='2025-12-31')
fred_data.index = fred_data.index.to_period('M')
fred_data['CS'] = fred_data['BAA'] - fred_data['AAA']
cs = fred_data.reset_index().rename(columns={'DATE': 'year_month'})

# D) VIX: market-implied expected volatility (FRED, daily → monthly average)
#    Only available from 1990. Log-transformed to match scale of VOL.
vix_daily = web.DataReader('VIXCLS', 'fred', start='1990-01-01', end='2025-12-31')
vix = vix_daily.resample('ME').mean()
vix.index = vix.index.to_period('M')
vix = vix.reset_index().rename(columns={'DATE': 'year_month', 'VIXCLS': 'VIX'})

# E) Real GDP (FRED, quarterly, chained 2017 dollars)
#    Forward-filled to monthly since GDP is released once per quarter
gdp = web.DataReader('GDPC1', 'fred', start='1970-01-01', end='2025-12-31')
gdp.index = gdp.index.to_period('Q')
gdp = gdp.reset_index().rename(columns={'DATE': 'quarter', 'GDPC1': 'GDP'})

# F) CRSP monthly stock file — individual stock returns and prices
#    Used for cross-sectional momentum portfolio construction
msf = db.raw_sql("""
    SELECT
        a.permno,
        a.date,
        a.ret,
        a.prc,
        a.shrout,
        b.exchcd,
        b.shrcd
    FROM crsp.msf AS a
    INNER JOIN crsp.msenames AS b
        ON  a.permno  = b.permno
        AND a.date   >= b.namedt
        AND a.date   <= b.nameendt
    WHERE
        a.date BETWEEN '1990-01-01' AND '2025-12-31'
        AND b.shrcd  IN (10, 11)
        AND b.exchcd IN (1, 2, 3)
    ORDER BY a.permno, a.date
""", date_cols=['date'])

# G) CRSP delisting returns — final return when a stock exits the market
#    dlret: actual return in the delisting month (merger proceeds, liquidation value, etc.)
#    dlstcd: delisting code — used to identify bankruptcies vs. mergers
#    Without this, stocks that go bankrupt show a missing last return, overstating momentum returns.
#    Standard treatment (Shumway 1997): use dlret if available, else -30% for performance delistings.
msedelist = db.raw_sql("""
    SELECT permno, dlstdt, dlret, dlstcd
    FROM crsp.msedelist
    WHERE dlstdt >= '1990-01-01'
""", date_cols=['dlstdt'])

db.close()

print("[ 2/4 ] Building market features ...")

# ── Section 2: Construct monthly features ────────────────────────────────────

# DD: drawdown over rolling 12-month window
#     DD_t = (P_t - max(P_{t-11}, ..., P_t)) / max(...)
msi['year_month'] = msi['date'].dt.to_period('M')
L = 12
msi['M_t'] = msi['P'].rolling(L).max()
msi['DD']  = (msi['P'] - msi['M_t']) / msi['M_t']

# VOL: log realized vol (log makes it more normally distributed)
rv['VOL'] = np.log(rv['RV'])

# LVIX: log VIX (same reasoning — raw VIX is right-skewed)
vix['LVIX'] = np.log(vix['VIX'])

# GDP_g: log quarter-over-quarter GDP growth.
# Only assign to the FIRST month of each quarter to avoid repeating the same
# value 3 times (which creates artificial autocorrelation in the HMM features).
# The other 2 months in each quarter will be NaN and are excluded from HMM training.
gdp = gdp.sort_values('quarter')
gdp['GDP_g'] = np.log(gdp['GDP'] / gdp['GDP'].shift(1))
gdp_monthly = (
    gdp[['quarter', 'GDP_g']]
    .assign(year_month=gdp['quarter'].apply(lambda q: q.asfreq('M', 'S')))
    [['year_month', 'GDP_g']]
)

# Validate no duplicate year_months in any source table before merging
# (duplicates would silently create extra rows via Cartesian product)
for name, df in [('rv', rv), ('cs', cs), ('vix', vix), ('gdp_monthly', gdp_monthly)]:
    n_dup = df['year_month'].duplicated().sum()
    assert n_dup == 0, f"{name} has {n_dup} duplicate year_month entries — fix before merging"

# Merge all features into one monthly panel
panel = (
    msi[['year_month', 'date', 'vwretd', 'P', 'DD']]
    .merge(rv[['year_month', 'VOL']],            on='year_month', how='left')
    .merge(cs[['year_month', 'CS']],             on='year_month', how='left')
    .merge(vix[['year_month', 'LVIX']],          on='year_month', how='left')
    .merge(gdp_monthly[['year_month', 'GDP_g']], on='year_month', how='left')
)

# DISP: cross-sectional return dispersion (log of std of individual stock returns)
# Requires stock-level returns (msf), so compute here after both msf and panel are available.
msf['year_month'] = msf['date'].dt.to_period('M')
disp = msf.groupby('year_month')['ret'].std().reset_index()
disp.columns = ['year_month', 'DISP']
disp['DISP'] = np.log(disp['DISP'])
panel = panel.merge(disp, on='year_month', how='left')

# REL_N: relative market participation — log(N_stocks / 12-month MA of N_stocks)
# Negative values indicate accelerating delistings, a sign of market distress.
n_stocks = msf.groupby('year_month')['permno'].nunique().reset_index()
n_stocks.columns = ['year_month', 'N_STOCKS']
n_stocks = n_stocks.sort_values('year_month')
n_stocks['N_STOCKS_MA12'] = n_stocks['N_STOCKS'].rolling(12).mean()
n_stocks['REL_N'] = np.log(n_stocks['N_STOCKS'] / n_stocks['N_STOCKS_MA12'])
panel = panel.merge(n_stocks[['year_month', 'REL_N']], on='year_month', how='left')

print(f"  DISP: {panel['DISP'].notna().sum()} non-null months")
print(f"  REL_N: {panel['REL_N'].notna().sum()} non-null months")

# ── Section 3: Data quality checks ───────────────────────────────────────────

# Stale CS: FRED sometimes forward-fills missing months with the prior value.
# 19 stale months is acceptable; a large cluster (>6 in a row) would be a problem.
stale_cs = cs['CS'].diff().eq(0).sum()
gaps = panel['date'].diff().dt.days.gt(35)
gap_msg = f"GAPS FOUND: {panel.loc[gaps, 'date'].tolist()}" if gaps.any() else "no gaps"
print(f"  Credit spread stale months: {stale_cs}  |  CRSP monthly continuity: {gap_msg}")

# ── Section 4: Winsorize core features ───────────────────────────────────────
# Clips extreme values at 1st/99th percentile to prevent outliers (e.g. 2008
# credit spread spike) from distorting HMM regime boundaries.
# Bounds computed on TRAIN data only (pre-2011) to avoid leaking test extremes.
train_panel = panel[panel['date'] < '2011-01-01']
for col in ['DD', 'VOL', 'CS', 'DISP', 'REL_N']:
    vals = train_panel[col].dropna()
    if len(vals) > 0:
        lo = vals.quantile(0.01)
        hi = vals.quantile(0.99)
        panel[col] = panel[col].astype(float).clip(lo, hi)

# ── Section 5: Align timing and drop incomplete rows ─────────────────────────
# ret_next is the target: market return in month t+1.
# Features at month t are used to predict ret_next (no look-ahead).
panel['ret_next'] = panel['vwretd'].shift(-1)
panel = panel.dropna(subset=['DD', 'VOL', 'CS', 'ret_next'])
# Note: LVIX (pre-1990), GDP_g (first quarter), DISP/REL_N (pre-1990) may still be NaN.

# Timing spot-check: ret_next in row t should equal vwretd in row t+1
ok = np.allclose(panel['ret_next'].iloc[:-1].values,
                 panel['vwretd'].iloc[1:].values, equal_nan=False)
assert ok, "Timing check FAILED: ret_next does not equal next month's vwretd — check panel alignment"
print(f"  Timing check (ret_next = next vwretd): OK")

# ── Section 6: Train/test split and standardization ──────────────────────────

train = panel[panel['date'] <  '2011-01-01']
test  = panel[panel['date'] >= '2011-01-01']

# Core 4 features used in HMM (selected via exhaustive search + validation).
# CS, LVIX kept for downstream models. GDP_g excluded due to quarterly lag.
features     = ['DD', 'VOL', 'DISP', 'REL_N']
features_ext = ['DD', 'VOL', 'DISP', 'REL_N', 'CS', 'LVIX']

# Compute mean/std on train only — apply same scale to test (no leakage)
z_mean = train[features].mean()
z_std  = train[features].std()
train_z = (train[features] - z_mean) / z_std
test_z  = (test[features]  - z_mean) / z_std

print(f"  Train: {train['date'].min().date()} -> {train['date'].max().date()}  ({len(train)} months)")
print(f"  Test:  {test['date'].min().date()} -> {test['date'].max().date()}  ({len(test)} months)")
null_pct = panel[features_ext].isnull().mean().mul(100).round(1)
print(f"  Null %: " + "  ".join(f"{c}={v}%" for c, v in null_pct.items()))

# ── Section 7: Sanity check plots ────────────────────────────────────────────
# Saved to file only — no interactive window.
# Open features_stress.png and features_macro.png to inspect.

crises = [
    ('1973-10-01', '1974-12-01', 'Oil shock'),
    ('1987-10-01', '1987-12-01', 'Black Monday'),
    ('2000-03-01', '2002-10-01', 'Dot-com'),
    ('2007-10-01', '2009-06-01', 'GFC'),
    ('2020-02-01', '2020-05-01', 'COVID'),
]

def add_crises(ax, label=False):
    for start, end, lbl in crises:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color='grey')
        if label:
            ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.95,
                    lbl, fontsize=7, color='grey', va='top')

# Plot 1: HMM features (DD, VOL, DISP, REL_N)
fig1, axes1 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for ax, (col, title, color) in zip(axes1, [
    ('DD',    'Drawdown DD_t',          'steelblue'),
    ('VOL',   'Log Realized Vol',       'darkorange'),
    ('DISP',  'Log Return Dispersion',  'purple'),
    ('REL_N', 'Relative Participation', 'seagreen'),
]):
    ax.plot(panel['date'], panel[col], color=color, linewidth=0.8)
    ax.set_ylabel(title, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
    add_crises(ax, label=(col == 'DD'))
axes1[-1].set_xlabel('Date')
fig1.suptitle('HMM Stress Indicators (grey = known crisis)', fontsize=11)
plt.tight_layout()
fig1.savefig('features_stress.png', dpi=150)
plt.close(fig1)

# Plot 2: Supplemental indicators (CS, LVIX, GDP_g)
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for ax, (col, title, color) in zip(axes2, [
    ('CS',    'Credit Spread (%)',      'crimson'),
    ('LVIX',  'Log VIX (from 1990)',    'purple'),
    ('GDP_g', 'GDP Growth (quarterly)', 'seagreen'),
]):
    ax.plot(panel['date'], panel[col], color=color, linewidth=0.8)
    ax.set_ylabel(title, fontsize=9)
    ax.axhline(0, color='black', linewidth=0.4, linestyle='--')
    add_crises(ax, label=(col == 'CS'))
axes2[-1].set_xlabel('Date')
fig2.suptitle('Supplemental Indicators (grey = known crisis)', fontsize=11)
plt.tight_layout()
fig2.savefig('features_macro.png', dpi=150)
plt.close(fig2)

print("  Plots saved: features_stress.png, features_macro.png")

# ── Section 8: Save panel to disk ────────────────────────────────────────────
# Standardize all HMM features using train stats only (no leakage).
# CS, LVIX use only non-null train rows for mean/std.
panel = panel.copy()

# Core 4 HMM features: already have z_mean/z_std from train
for col in features:
    panel[f'{col}_z'] = (panel[col] - z_mean[col]) / z_std[col]

# Supplemental features: compute train stats on non-null rows only
for col in ['CS', 'LVIX']:
    train_vals = train[col].dropna()
    col_mean   = train_vals.mean()
    col_std    = train_vals.std()
    panel[f'{col}_z'] = (panel[col] - col_mean) / col_std

# Save full panel including raw and standardized features.
# Load in hmm_model.py with: pd.read_parquet('data/panel.parquet')
panel.to_parquet('data/panel.parquet', index=False)
print(f"  Saved data/panel.parquet  ({len(panel)} rows, {len(panel.columns)} cols)")

# ── Section 9: Merge delisting returns and save stock-level data ──────────────

msf['date']   = pd.to_datetime(msf['date'])
msf['permno'] = msf['permno'].astype(int)

# Align delisting date to month-end to match msf date format
msedelist['date']   = pd.to_datetime(msedelist['dlstdt']) + pd.offsets.MonthEnd(0)
msedelist['permno'] = msedelist['permno'].astype(int)

# Merge dlret and dlstcd onto the stock panel by permno + month-end date
msf = msf.merge(
    msedelist[['permno', 'date', 'dlret', 'dlstcd']],
    on=['permno', 'date'],
    how='left'
)

# Construct ret_adj: adjusted return that accounts for delistings
# Performance delistings (dlstcd 500-584): bankruptcy/liquidation — use -30% if dlret missing
# Other delistings (merger, exchange move): use dlret directly if ret is missing
perf_delist = msf['dlstcd'].between(500, 584)
msf['ret_adj'] = msf['ret']
msf.loc[msf['ret'].isna() & msf['dlret'].notna(),          'ret_adj'] = msf['dlret']
msf.loc[msf['ret'].isna() & msf['dlret'].isna() & perf_delist, 'ret_adj'] = -0.30

msf = msf.sort_values(['permno', 'date']).reset_index(drop=True)

n_delist  = msf['dlret'].notna().sum()
n_imputed = (msf['ret'].isna() & msf['dlret'].isna() & perf_delist).sum()
# Note: crsp_msf_raw.parquet is written after fundamentals are merged (Section 11).

# ── Section 10: Pull Compustat quarterly fundamentals ─────────────────────────
# Features computed (all at fiscal quarter end, lagged 4 months for availability):
#   bm             : Book-to-market — ceqq / market_equity (value signal)
#   roe            : Return on equity — ibq / ceqq (profitability)
#   earnings_growth: YoY earnings growth — ibq / ibq.shift(4) - 1
#   leverage       : Debt ratio — (dlttq + dlcq) / atq
#   asset_growth   : YoY asset growth — atq / atq.shift(4) - 1 (investment signal)
#   gross_profit_a : Gross profitability / assets — (revtq - cogsq) / atq (Novy-Marx)
#
# Reporting lag: features from fiscal quarter ending in month Q are assumed
# available starting Q + 4 months.  This is conservative but standard.
#
# Link: gvkey (Compustat) → permno (CRSP) via crsp.ccmxpf_lnkhist,
#       using linktype IN ('LU','LC') and linkprim IN ('P','C').

print("[ 3/4 ] Pulling Compustat fundamentals from WRDS ...")
db = wrds.Connection()
fundq = db.raw_sql("""
    SELECT
        f.gvkey, f.datadate, f.fqtr, f.fyearq,
        f.atq,    -- total assets
        f.ceqq,   -- common equity (book value)
        f.ibq,    -- net income (income before extraordinary items)
        f.revtq,  -- total revenue
        f.cogsq,  -- cost of goods sold
        f.dlttq,  -- long-term debt
        f.dlcq    -- debt in current liabilities
    FROM comp.fundq AS f
    -- restrict to firms that have a CRSP link (keeps only exchange-listed US stocks)
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


print("[ 4/4 ] Merging fundamentals onto stock panel ...")
ccm_link = db.raw_sql("""
    SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype  IN ('LU', 'LC')
      AND linkprim  IN ('P', 'C')
""", date_cols=['linkdt', 'linkenddt'])

db.close()

# ── Section 11: Build per-stock fundamental features ─────────────────────────

# Step 1: Compute features within each firm (sort by gvkey + datadate)
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

# Ratios (set to NaN when denominator is zero or negative)
fundq['roe']           = fundq['ibq'] / fundq['ceqq'].replace(0, np.nan)
fundq['leverage']      = (fundq['dlttq'].fillna(0) + fundq['dlcq'].fillna(0)) / \
                          fundq['atq'].replace(0, np.nan)
fundq['gross_profit_a'] = (fundq['revtq'] - fundq['cogsq'].fillna(0)) / \
                           fundq['atq'].replace(0, np.nan)

# Book equity must be positive for B/M to make sense
fundq['ceqq_pos'] = fundq['ceqq'].where(fundq['ceqq'] > 0, np.nan)

# Step 2: Apply 4-month reporting lag — feature is available at datadate + 4M (month-end)
fundq['avail_date'] = (fundq['datadate'] + pd.DateOffset(months=4)) + pd.offsets.MonthEnd(0)

# Step 3: Link gvkey → permno (respect date ranges in link table)
#   linkenddt NaT means still active — replace with far future date
ccm_link['linkenddt'] = ccm_link['linkenddt'].fillna(pd.Timestamp('2099-12-31'))
ccm_link['permno'] = ccm_link['permno'].astype(int)

fundq_linked = fundq.merge(ccm_link[['gvkey', 'permno', 'linkdt', 'linkenddt']],
                            on='gvkey', how='inner')

# Keep only rows where the fiscal quarter data date is within the link's active window
fundq_linked = fundq_linked[
    (fundq_linked['datadate'] >= fundq_linked['linkdt']) &
    (fundq_linked['datadate'] <= fundq_linked['linkenddt'])
].copy()

# Drop duplicate permno-avail_date rows (keep most recent datadate per permno-month)
fundq_linked = (fundq_linked
    .sort_values(['permno', 'avail_date', 'datadate'])
    .drop_duplicates(subset=['permno', 'avail_date'], keep='last')
    .reset_index(drop=True)
)

fund_feat_cols = ['ceqq_pos', 'roe', 'earnings_growth', 'leverage',
                  'asset_growth', 'gross_profit_a']

# Step 4: For each (permno, month) in msf, find the most recently available quarter
#   merge_asof: for each msf date, take the latest avail_date <= date (per permno)
msf_sorted    = msf.sort_values('date').reset_index(drop=True)
fundq_sorted  = (fundq_linked[['permno', 'avail_date'] + fund_feat_cols]
                 .sort_values('avail_date')
                 .reset_index(drop=True))

msf_with_fund = pd.merge_asof(
    msf_sorted,
    fundq_sorted,
    left_on='date',
    right_on='avail_date',
    by='permno',
    direction='backward'   # most recent quarter whose avail_date <= current date
)

# Step 5: Compute book-to-market using CRSP market equity
#   ME = |prc| * shrout (shrout in thousands → ME in thousands, ceqq in millions)
#   Convert both to the same units: ceqq in millions, ME in millions
msf_with_fund['me'] = msf_with_fund['prc'].abs() * msf_with_fund['shrout'] / 1000
msf_with_fund['bm'] = msf_with_fund['ceqq_pos'] / msf_with_fund['me'].replace(0, np.nan)

# earnings_growth: signed log-transform before winsorizing
# sign(x) * log(1 + |x|) compresses extreme % changes (e.g. 0→positive earnings)
# while preserving sign and monotonicity
eg = msf_with_fund['earnings_growth'].astype(float)
msf_with_fund['earnings_growth'] = np.sign(eg) * np.log1p(np.abs(eg))

# Winsorize — tighter clip for asset_growth (95th) due to fat right tail,
# standard 1/99 for the rest
winsor_bounds = {
    'bm':             (0.01, 0.99),
    'roe':            (0.01, 0.99),
    'earnings_growth':(0.01, 0.99),
    'leverage':       (0.01, 0.99),
    'asset_growth':   (0.05, 0.95),   # tighter — young fast-growers skew right
    'gross_profit_a': (0.01, 0.99),
}
train_stock_mask = msf_with_fund['date'] < '2011-01-01'
for col, (lo_q, hi_q) in winsor_bounds.items():
    vals      = msf_with_fund[col].astype(float)
    lo = vals[train_stock_mask].quantile(lo_q)
    hi = vals[train_stock_mask].quantile(hi_q)
    msf_with_fund[col] = vals.clip(lo, hi)

msf_with_fund = msf_with_fund.sort_values(['permno', 'date']).reset_index(drop=True)
msf_with_fund.drop(columns=['ceqq_pos', 'avail_date'], inplace=True)

msf_with_fund.to_parquet('data/crsp_msf_raw.parquet', index=False)

print()
print("=" * 60)
print("  IMPORT COMPLETE")
print("=" * 60)
print(f"  Market panel   data/panel.parquet")
print(f"    {len(panel)} months  |  {panel['date'].min().date()} → {panel['date'].max().date()}")
print()
print(f"  Stock panel    data/crsp_msf_raw.parquet")
print(f"    {len(msf_with_fund):,} rows  |  {msf_with_fund['permno'].nunique():,} stocks")
print(f"    {msf_with_fund['date'].min().date()} → {msf_with_fund['date'].max().date()}")
print(f"    Delistings merged: {n_delist:,}  |  imputed at -30%: {n_imputed:,}")
fund_cov = msf_with_fund['bm'].notna().mean()
print(f"    Fundamental coverage (bm non-null): {fund_cov:.1%}")
print("=" * 60)
