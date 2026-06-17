"""
_pull_daily_top500.py
=====================
One-off data pull (experiments/ helper, not part of the production pipeline).

Pulls CRSP daily returns (crsp.dsf) for the 1,105 permnos that ever enter the
Top-500-by-market-cap universe over 2010-2024, so we can compute the daily-return
volatility used in the S&P 500 Momentum risk-adjusted momentum value (Appendix A
of the S&P Momentum Indices methodology).

Connects with psycopg2/libpq directly so the password is read from ~/.pgpass
(the wrds wrapper falls back to an interactive username prompt under automation,
which has no stdin). SSL is required by WRDS.
"""
import os
import sys
import pandas as pd
import psycopg2

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERMNO_CSV = os.path.join(_ROOT, 'real_trade_analysis', 'data', '_top500_permnos_2010_2024.csv')
OUT = os.path.join(_ROOT, 'data', 'spmo_daily_top500.parquet')

# Connection params from ~/.pgpass (host:port:dbname:user:password). Password is
# supplied automatically by libpq from ~/.pgpass -- never passed in code.
parts = open(os.path.expanduser('~/.pgpass')).readline().strip().split(':')
host, port, dbname, user = parts[0], parts[1], parts[2], parts[3]

permnos = pd.read_csv(PERMNO_CSV)['permno'].astype(int).tolist()
print(f"connecting to {host}:{port}/{dbname} as {user} ...", flush=True)
conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user,
                        sslmode='require', connect_timeout=60)
print("connected. pulling crsp.dsf daily returns ...", flush=True)

# Single query; ~3-4M rows, small enough to fetch at once.
sql = """
    select permno, date, ret
    from crsp.dsf
    where permno in %(permnos)s
      and date between '2010-01-01' and '2024-12-31'
"""
df = pd.read_sql_query(sql, conn, params={'permnos': tuple(permnos)})
conn.close()

df['date'] = pd.to_datetime(df['date'])
df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
df = df.sort_values(['permno', 'date']).reset_index(drop=True)
df.to_parquet(OUT, index=False)

print(f"saved {OUT}", flush=True)
print(f"rows={len(df):,}  permnos={df['permno'].nunique():,}  "
      f"dates {df['date'].min().date()}..{df['date'].max().date()}  "
      f"ret non-null={df['ret'].notna().mean():.1%}", flush=True)
