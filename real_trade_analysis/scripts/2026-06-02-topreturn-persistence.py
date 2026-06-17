"""
2026-06-02-topreturn-persistence.py  (EXPLORATORY)
Do the best-performing companies repeat? Each year, rank the S&P 500 (Top-500 by cap
entering the year) by calendar-year total return and take the top-20% (top-100). Then
measure how many of LAST year's top-returners are STILL on this year's list (retention).
2010-2025 (most recent full year = 2025). Clean CRSP v2. Contrast with size churn.
"""
import numpy as np, pandas as pd, psycopg2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/giladgang/momentum_regime/real_trade_analysis'
c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='giladgang',sslmode='require',connect_timeout=60)
elig="securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS' AND primaryexch IN ('N','A','Q')"
df=pd.read_sql(f"SELECT permno, mthcaldt AS date, mthret AS ret, mthcap AS me FROM crsp.msf_v2 WHERE mthcaldt>='2008-12-01' AND mthcaldt<='2025-12-31' AND {elig}",c); c.close()
df['date']=pd.to_datetime(df['date']); df['year']=df['date'].dt.year
deccap=df[df['date'].dt.month==12].set_index(['year','permno'])['me']
ann=df.dropna(subset=['ret']).groupby(['year','permno'])['ret'].apply(lambda s:(1+s).prod()-1)

topret={}
for y in range(2009,2026):
    cap=deccap.loc[y-1].dropna() if (y-1) in deccap.index.get_level_values(0) else None
    if cap is None: continue
    univ=set(cap.nlargest(500).index)               # S&P-500 proxy entering year y
    r=ann.loc[y].reindex(list(univ)).dropna()
    topret[y]=set(r.nlargest(max(1,round(0.2*len(r)))).index)

yrs=sorted(topret); rows=[]
for i in range(1,len(yrs)):
    a,b=topret[yrs[i-1]],topret[yrs[i]]
    if yrs[i]<2010: continue
    rows.append({'year':yrs[i],'size':len(b),'retained':len(a&b),'pct_retained':len(a&b)/len(b)})
pe=pd.DataFrame(rows)
print("year  list_size  retained_from_prior  %retained")
for _,r in pe.iterrows(): print(f"  {int(r['year'])}    {int(r['size']):>4}        {int(r['retained']):>3}             {r['pct_retained']:>4.0%}")
print(f"\n  mean retention {pe['pct_retained'].mean():.0%}  (vs ~91% for top-by-SIZE, ~17% for top-by-MOMENTUM)")

fig,ax=plt.subplots(figsize=(13,6)); x=np.arange(len(pe))
ax.bar(x,pe['pct_retained']*100,color='steelblue',width=.7)
ax.axhline(pe['pct_retained'].mean()*100,color='navy',ls='--',lw=1,label=f"mean retention {pe['pct_retained'].mean():.0%}")
ax.axhline(91,color='seagreen',ls=':',lw=1.3,label='top-by-SIZE retention ~91%')
for i,r in pe.iterrows(): ax.text(i,r['pct_retained']*100+0.6,f"{int(r['retained'])}/{int(r['size'])}",ha='center',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(pe['year'].astype(int)); ax.set_ylabel('% of last year\'s top-20%-by-RETURN still on the list')
ax.set_title('Do the best-returning S&P 500 companies repeat? Year-over-year retention of the top-20% by return\n(near-zero persistence: last year\'s winners almost never repeat -- the opposite of size)',fontsize=11)
ax.legend(); ax.grid(True,axis='y',alpha=.25); ax.set_ylim(0,100); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-topreturn-persistence.{e}',dpi=150,bbox_inches='tight')
pe.to_csv(f'{R}/results/2026-06-02-topreturn-persistence.csv',index=False); print("saved topreturn-persistence")
