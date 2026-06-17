"""
2026-06-02-why-2024-chart.py  (EXPLORATORY)
Why was 2024 SPMO's standout year vs the S&P 500? Per year, decompose the linear
excess (SPMO - cap-weighted market) into the top-3 contributing NAMES vs the rest.
Shows 2024 is uniquely large AND uniquely concentrated in a few mega-caps. Real panel.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; ha=_load('ha','2026-05-31-holdings-analysis.py')
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT
xw=pd.read_csv(f'{R}/results/permno_name_crosswalk.csv').set_index('permno')['ticker'].to_dict()

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spath=ha.spmo_path(list(test.groupby('date')))

# per-year per-name contribution to excess: sum_t (w_spmo - w_mkt)*ret
contrib={}
for d,g in test.groupby('date'):
    g=g.dropna(subset=['me','ret_fwd'])
    if d not in spath or not spath[d]: continue
    wmkt=(g.set_index('permno')['me']/g['me'].sum()).to_dict(); r=g.set_index('permno')['ret_fwd'].to_dict(); wsp=spath[d]
    y=d.year; contrib.setdefault(y,{})
    for p in set(wsp)|set(wmkt):
        contrib[y][p]=contrib[y].get(p,0)+(wsp.get(p,0)-wmkt.get(p,0))*r.get(p,0)

years=sorted(contrib); top3,rest,labels=[],[],{}
for y in years:
    s=pd.Series(contrib[y]); t3=s.nlargest(3)
    top3.append(t3.sum()); rest.append(s.sum()-t3.sum())
    labels[y]='+'.join(xw.get(p,str(p)) for p in t3.index)
top3,rest=np.array(top3)*100,np.array(rest)*100

fig,ax=plt.subplots(figsize=(13,6.2))
x=np.arange(len(years))
ax.bar(x,rest,color='lightsteelblue',label='all other names')
ax.bar(x,top3,bottom=np.where(rest>0,rest,0),color='crimson',label='top-3 names that year')
for i,y in enumerate(years):
    tot=top3[i]+rest[i]
    ax.text(i,tot+0.4 if tot>=0 else tot-0.8,f'{tot:+.0f}',ha='center',fontsize=8,fontweight='bold' if y==2024 else 'normal')
ax.text(years.index(2024),top3[years.index(2024)]+rest[years.index(2024)]+2.0,labels[2024],ha='center',fontsize=9,color='crimson',fontweight='bold')
ax.axhline(0,color='k',lw=.7); ax.set_xticks(x); ax.set_xticklabels(years,rotation=0)
ax.set_ylabel('SPMO excess vs S&P 500 (%/yr, linear)')
ax.set_title('Why 2024 stood out: SPMO minus S&P 500 each year, split into top-3 names vs the rest\n2024 is both the largest excess AND almost entirely 3 mega-caps (NVDA, AVGO, META)',fontsize=11.5)
ax.legend(loc='upper left',fontsize=9.5); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-why-2024.{e}',dpi=150,bbox_inches='tight')
plt.close(fig)
print("per-year excess (linear), top-3 share, top-3 names:")
for i,y in enumerate(years):
    tot=top3[i]+rest[i]; print(f"  {y}: excess {tot:+5.1f}%  top3 {top3[i]:+5.1f}%  ({top3[i]/tot*100 if tot else 0:>4.0f}% of excess)  {labels[y]}")
print("\nsaved plots/2026-06-02-why-2024.pdf + .png")
