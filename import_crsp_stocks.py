"""
import_crsp_stocks.py
====================
Downloads raw CRSP monthly stock-level data from WRDS for all U.S. common
stocks from 1990-01-01 through today. Saves the raw panel locally for later
momentum portfolio construction (Daniel & Moskowitz / Jegadeesh & Titman style).

No momentum signals are computed here — this is raw data only.

Output
------
data/crsp_msf_raw.parquet  (fallback: data/crsp_msf_raw.csv)

Variable reference for later momentum construction
---------------------------------------------------
permno   : Unique stock identifier (permanent number). Used to track a stock
           across name changes, ticker changes, and delistings.
date     : Month-end date. Used to align returns, sort into portfolios, and
           compute rolling past returns.
ret      : Monthly holding-period return (includes dividends). Core input to
           momentum signal: past 12-month cumulative return = prod(1+ret) - 1
           over months t-12 to t-2 (skipping last month to avoid reversal).
prc      : End-of-month price (negative if bid-ask midpoint, not last trade).
           abs(prc) * shrout = market cap. Used to size-weight portfolios and
           to filter out penny stocks (e.g. abs(prc) >= 5).
shrout   : Shares outstanding in thousands. Combined with prc to compute
           market cap, which is used for value-weighting and size screens.
exchcd   : Exchange code (1=NYSE, 2=AMEX, 3=NASDAQ). Used to apply NYSE-only
           breakpoints when sorting into momentum deciles — standard in the
           literature to avoid small-stock bias in cutoffs.
shrcd    : Share type code. Restricted to 10/11 (ordinary common shares).
           Excludes ADRs, REITs, closed-end funds, and foreign listings.
"""

import sys
import pandas as pd
import numpy as np
from datetime import date

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE  = '1990-01-01'
END_DATE    = date.today().strftime('%Y-%m-%d')
OUTPUT_PATH = 'data/crsp_msf_raw.parquet'
OUTPUT_CSV  = 'data/crsp_msf_raw.csv'   # fallback if pyarrow not available

# ── Connect to WRDS ───────────────────────────────────────────────────────────

try:
    import wrds
except ImportError:
    sys.exit("ERROR: wrds package not installed. Run: pip install wrds")

print(f"Connecting to WRDS ...")
try:
    db = wrds.Connection()
except Exception as e:
    sys.exit(f"ERROR: Could not connect to WRDS.\n{e}")

print(f"Connected. Pulling CRSP monthly stock data: {START_DATE} → {END_DATE}")

# ── Query ─────────────────────────────────────────────────────────────────────
# Join crsp.msf (returns/prices) to crsp.msenames (exchange/share codes).
# The names table is effective for a date range [namedt, nameendt], so we
# must match on permno AND require namedt <= date <= nameendt to get the
# correct exchcd/shrcd valid at the time of each observation.

query = f"""
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
        a.date BETWEEN '{START_DATE}' AND '{END_DATE}'
        AND b.shrcd  IN (10, 11)
        AND b.exchcd IN (1, 2, 3)
    ORDER BY
        a.permno, a.date
"""

print("Executing query (this may take 1-3 minutes for the full history) ...")

try:
    df = db.raw_sql(query, date_cols=['date'])
except Exception as e:
    db.close()
    sys.exit(f"ERROR: Query failed.\n{e}")

finally:
    db.close()
    print("WRDS connection closed.")

# ── Post-processing ───────────────────────────────────────────────────────────

# Ensure date is pandas datetime (raw_sql with date_cols should handle this,
# but being explicit protects against version differences)
df['date'] = pd.to_datetime(df['date'])

# Sort by stock then time
df = df.sort_values(['permno', 'date']).reset_index(drop=True)

# Cast types to save memory
df['permno'] = df['permno'].astype(int)
df['shrout'] = pd.to_numeric(df['shrout'], errors='coerce')
df['prc']    = pd.to_numeric(df['prc'],    errors='coerce')
df['ret']    = pd.to_numeric(df['ret'],    errors='coerce')
df['exchcd'] = df['exchcd'].astype('Int16')
df['shrcd']  = df['shrcd'].astype('Int16')

# ── Summary ───────────────────────────────────────────────────────────────────

n_rows    = len(df)
n_stocks  = df['permno'].nunique()
date_min  = df['date'].min().date()
date_max  = df['date'].max().date()
ret_null  = df['ret'].isna().sum()
prc_null  = df['prc'].isna().sum()

print(f"\n--- Data summary ---")
print(f"  Rows:            {n_rows:>10,}")
print(f"  Unique permnos:  {n_stocks:>10,}")
print(f"  Date range:      {date_min} → {date_max}")
print(f"  Missing ret:     {ret_null:>10,}  ({100*ret_null/n_rows:.1f}%)")
print(f"  Missing prc:     {prc_null:>10,}  ({100*prc_null/n_rows:.1f}%)")
print(f"\n--- First 5 rows ---")
print(df.head().to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────

import os
os.makedirs('data', exist_ok=True)

try:
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")
except Exception as e:
    print(f"WARNING: parquet save failed ({e}). Falling back to CSV ...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}  ({os.path.getsize(OUTPUT_CSV)/1e6:.1f} MB)")
