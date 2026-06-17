"""
2026-06-02-topquintile-churn.py  (EXPLORATORY)
How much does SPMO's selection universe change year to year? SPMO holds the TOP 20%
(top quintile) by risk-adjusted momentum. Each year-end we take that top-20% set and
measure how many names are NEW vs the prior year-end (membership turnover). High churn
= momentum leadership rotating; low churn = the same winners persisting. Real panel.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']

# top-20% set at each year's last available month
sets={}
for y,g in test.groupby(test['date'].dt.year):
    last=g[g['date']==g['date'].max()].dropna(subset=['spmo_signal'])
    n=max(1,round(0.2*len(last)))
    sets[y]=set(last.nlargest(n,'spmo_signal')['permno'])

yrs=sorted(sets); rows=[]
for i in range(1,len(yrs)):
    a,b=sets[yrs[i-1]],sets[yrs[i]]
    new=len(b-a); left=len(a-b)
    rows.append({'year':yrs[i],'size':len(b),'new':new,'pct_changed':new/len(b),'stayed':len(a&b)})
ch=pd.DataFrame(rows)
print("year  top20%_size  new_names  %changed  stayed")
for _,r in ch.iterrows():
    print(f"  {int(r['year'])}    {int(r['size']):>4}        {int(r['new']):>4}     {r['pct_changed']:>5.0%}   {int(r['stayed']):>4}")
print(f"\n  mean %changed {ch['pct_changed'].mean():.0%} | 2024 {ch[ch.year==2024]['pct_changed'].values}")

fig,ax=plt.subplots(figsize=(13,6))
x=np.arange(len(ch))
col=['crimson' if y==2024 else 'steelblue' for y in ch['year']]
ax.bar(x,ch['pct_changed']*100,color=col,width=0.7)
ax.axhline(ch['pct_changed'].mean()*100,color='gray',ls='--',lw=1,label=f"mean {ch['pct_changed'].mean():.0%}")
for i,r in ch.iterrows(): ax.text(i,r['pct_changed']*100+0.8,f"{int(r['new'])}/{int(r['size'])}",ha='center',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(ch['year'].astype(int),rotation=0)
ax.set_ylabel('% of the top-20% momentum list that is NEW vs prior year')
ax.set_title('Year-over-year churn of SPMO\'s top-20% (top-quintile) momentum universe\n(labels = new names / list size; high = leadership rotated, low = winners persisted; red=2024)',fontsize=11.5)
ax.legend(fontsize=9.5); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-topquintile-churn.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); ch.to_csv(f'{R}/results/2026-06-02-topquintile-churn.csv',index=False)
print("\nsaved plots/2026-06-02-topquintile-churn.pdf + .png")
