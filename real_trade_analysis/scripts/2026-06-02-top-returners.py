"""
2026-06-02-top-returners.py  (EXPLORATORY)
Top S&P 500 companies by realized return. Universe = Top-500 by market cap entering
the year; rank by calendar-year total return. Clean CRSP v2 (consistent units).
Marks which were also in the SPMO replica that year. Default year = 2024.
"""
import sys, numpy as np, pandas as pd, psycopg2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/giladgang/momentum_regime/real_trade_analysis'
YEAR=int(sys.argv[1]) if len(sys.argv)>1 else 2024
c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='giladgang',sslmode='require',connect_timeout=40)
elig="securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS' AND primaryexch IN ('N','A','Q')"
cap=pd.read_sql(f"SELECT permno, mthcap FROM crsp.msf_v2 WHERE EXTRACT(month FROM mthcaldt)=12 AND EXTRACT(year FROM mthcaldt)={YEAR-1} AND {elig} AND mthcap IS NOT NULL",c)
top500=set(cap.nlargest(500,'mthcap')['permno'])
rr=pd.read_sql(f"SELECT permno, mthcaldt AS date, mthret AS ret, ticker, issuernm FROM crsp.msf_v2 WHERE EXTRACT(year FROM mthcaldt)={YEAR} AND {elig} AND mthret IS NOT NULL",c); c.close()
rr['date']=pd.to_datetime(rr['date']); rr=rr[rr['permno'].isin(top500)]
nm=rr.sort_values('date').groupby('permno').agg(ticker=('ticker','last'),name=('issuernm','last'))
yr=rr.groupby('permno')['ret'].apply(lambda s:(1+s).prod()-1) if len(rr) else pd.Series(dtype=float)
res=pd.concat([yr.rename('ret'),nm],axis=1).dropna(subset=['ret']).sort_values('ret',ascending=False)
top=res.head(20)
print(f"Top-20 S&P 500 (Top-500 by cap) companies by {YEAR} return:")
for i,(p,r) in enumerate(top.iterrows(),1): print(f"  {i:>2} {str(r['ticker']):<7}{str(r['name'])[:30]:<31}{r['ret']:>+7.0%}")
res.to_csv(f'{R}/results/2026-06-02-top-returners-{YEAR}.csv')

fig,ax=plt.subplots(figsize=(11,7.5)); y=np.arange(len(top))[::-1]
ax.barh(y,top['ret']*100,color='crimson')
ax.set_yticks(y); ax.set_yticklabels([f"{t}  {n[:22]}" for t,n in zip(top['ticker'],top['name'])],fontsize=9)
for yi,v in zip(y,top['ret']*100): ax.text(v+3,yi,f'{v:.0f}%',va='center',fontsize=8)
ax.set_xlabel(f'{YEAR} total return (%)'); ax.set_title(f'Top-20 S&P 500 companies by {YEAR} return (Top-500-by-cap universe, CRSP v2)',fontsize=12)
ax.grid(True,axis='x',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-top-returners-{YEAR}.{e}',dpi=150,bbox_inches='tight')
print(f"\nsaved plots/2026-06-02-top-returners-{YEAR}.pdf + .png")
