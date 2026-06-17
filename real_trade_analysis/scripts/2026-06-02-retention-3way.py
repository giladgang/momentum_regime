"""
2026-06-02-retention-3way.py  (EXPLORATORY)
Year-over-year retention of the S&P 500 top-20% list, ranked three ways, 2010-2025:
  SIZE     = top-100 by market cap            (biggest companies)
  MOMENTUM = top-100 by risk-adj 12-1 return  (past winners; SPMO-style signal)
  RETURN   = top-100 by that year's return    (best performers)
Retention = % of last year's list still on this year's list. Single consistent CRSP v2
source. Shows size persists (~90%) while momentum/return are revolving doors (~20%).
"""
import numpy as np, pandas as pd, psycopg2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/giladgang/momentum_regime/real_trade_analysis'
c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='giladgang',sslmode='require',connect_timeout=60)
elig="securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS' AND primaryexch IN ('N','A','Q')"
df=pd.read_sql(f"SELECT permno, mthcaldt AS date, mthret AS ret, mthcap AS me FROM crsp.msf_v2 WHERE mthcaldt>='2007-12-01' AND mthcaldt<='2025-12-31' AND {elig}",c); c.close()
df['date']=pd.to_datetime(df['date']); df=df.sort_values(['permno','date'])
# risk-adj 12-1 momentum signal (skip most recent month): 12-mo cum return / 12-mo vol, both shifted 1
lr=np.log1p(df['ret'].clip(lower=-0.999))
df['mom']=np.expm1(lr.groupby(df['permno']).transform(lambda s:s.shift(1).rolling(12).sum()))
df['vol']=df.groupby('permno')['ret'].transform(lambda s:s.shift(1).rolling(12).std())
df['sig']=df['mom']/df['vol']
df['cyr']=np.expm1(lr.groupby([df['permno'],df['date'].dt.year]).transform('sum'))   # calendar-year return (same each month of a year)
df['year']=df['date'].dt.year
dec=df[df['date'].dt.month==12].copy()

def lists(y):
    g=dec[dec['year']==y].dropna(subset=['me'])
    univ=g.nlargest(500,'me'); k=max(1,round(0.2*len(univ)))
    return {'SIZE':set(univ.nlargest(k,'me')['permno']),
            'MOMENTUM':set(univ.dropna(subset=['sig']).nlargest(k,'sig')['permno']),
            'RETURN':set(univ.dropna(subset=['cyr']).nlargest(k,'cyr')['permno'])}
yrs=list(range(2009,2026)); L={y:lists(y) for y in yrs}
ser={k:[] for k in ['SIZE','MOMENTUM','RETURN']}; YR=[]
for i in range(1,len(yrs)):
    y=yrs[i]
    if y<2010: continue
    YR.append(y)
    for k in ser: ser[k].append(len(L[y][k]&L[yrs[i-1]][k])/max(1,len(L[y][k])))
df_out=pd.DataFrame(ser,index=YR); print((df_out*100).round(0).astype(int).to_string())
print("\nmean retention:", {k:f'{np.mean(v):.0%}' for k,v in ser.items()})

fig,ax=plt.subplots(figsize=(13,6.3))
COL={'SIZE':'seagreen','MOMENTUM':'darkorange','RETURN':'crimson'}
for k in ['SIZE','MOMENTUM','RETURN']:
    ax.plot(YR,np.array(ser[k])*100,marker='o',lw=2.4,color=COL[k],label=f'{k} (mean {np.mean(ser[k]):.0%})')
ax.set_ylabel('% of last year\'s top-20% list still on the list'); ax.set_ylim(0,100); ax.set_xticks(YR)
ax.set_title('S&P 500 top-20% list: year-over-year retention, ranked by SIZE vs MOMENTUM vs RETURN (2010-2025)\nbiggest companies persist (~90%); best performers / past winners are a revolving door (~20%)',fontsize=11)
ax.legend(fontsize=10,loc='center right'); ax.grid(True,alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-retention-3way.{e}',dpi=150,bbox_inches='tight')
df_out.to_csv(f'{R}/results/2026-06-02-retention-3way.csv'); print("saved retention-3way")
