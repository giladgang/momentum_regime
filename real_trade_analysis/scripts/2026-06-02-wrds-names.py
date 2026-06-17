"""
2026-06-02-wrds-names.py  (EXPLORATORY)
WRDS unlocked. Pull permno -> ticker + company name from crsp.stocknames for every
permno in our holdings tables; take the most recent name per permno. Save a crosswalk
and reprint the Jan-2024 + full-period SPMO top holdings WITH verified names.
Run with the sandbox disabled (network egress).
"""
import os, sys, numpy as np, pandas as pd, psycopg2
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'

# collect permnos from every holdings file we have
files=['holdings_spmo_top_jan2024.csv','holdings_spmo_top.csv','holdings_xgb_top.csv']
permnos=set()
for f in files:
    p=f'{R}/results/{f}'
    if os.path.exists(p): permnos|=set(pd.read_csv(p)['permno'].astype(int))
print(f"permnos to resolve: {len(permnos)}")

c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',
                   user='giladgang',sslmode='require',connect_timeout=30)
q="SELECT permno, ticker, comnam, namedt, nameenddt FROM crsp.stocknames WHERE permno IN ({})".format(
    ','.join(str(int(p)) for p in permnos))
nm=pd.read_sql(q,c); c.close()
nm['nameenddt']=pd.to_datetime(nm['nameenddt'])
# most-recent name per permno
latest=nm.sort_values('nameenddt').groupby('permno').tail(1).set_index('permno')
latest.to_csv(f'{R}/results/permno_name_crosswalk.csv')
print(f"resolved {latest.shape[0]} names -> saved permno_name_crosswalk.csv\n")

def show(fn,title,n=15):
    p=f'{R}/results/{fn}'
    if not os.path.exists(p): return
    d=pd.read_csv(p).head(n)
    print(f"  {title}")
    print(f"  {'#':>2} {'permno':>7} {'ticker':>7}  {'company':<28}{'avg wt':>8}{'%mo':>6}")
    for i,r in d.iterrows():
        pn=int(r['permno']); tk=latest['ticker'].get(pn,'?'); co=str(latest['comnam'].get(pn,'?'))[:27]
        print(f"  {i+1:>2} {pn:>7} {str(tk):>7}  {co:<28}{r['avg_wt']:>7.1%}{r['pct_months']*100:>5.0f}%")
    print()

show('holdings_spmo_top_jan2024.csv','SPMO replica - TOP 15, Jan-2024-onward')
show('holdings_spmo_top.csv','SPMO replica - TOP 15, full period 2011-2024')
show('holdings_xgb_top.csv','XGB long-only - TOP 15, full period 2011-2024')
