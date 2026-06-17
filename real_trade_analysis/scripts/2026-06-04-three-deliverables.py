"""
2026-06-04-three-deliverables.py  (EXPLORATORY)
(1) Annual SPMO minus S&P 500 return, bar per year (2011-2025).
(2) SPMO chosen companies + average weight, 2024 and 2025 (tables -> CSV + print).
(3) % of SPMO's chosen top-20% that agreed with the ORACLE (top-20% by realized next-6mo
    return) per year -- SPMO's selection hit-rate vs perfect foresight.
Real spliced panel. Names via permno crosswalk.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; ha=_load('ha','2026-05-31-holdings-analysis.py')
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT
xw=pd.read_csv(f'{R}/results/permno_name_crosswalk.csv').set_index('permno')
nm=lambda p: xw['ticker'].get(p,str(p)); co=lambda p: str(xw['comnam'].get(p,''))[:26]

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
test=test.sort_values(['permno','date'])
lr=np.log1p(test['ret_adj'].clip(lower=-0.999))
test['fwd6']=np.expm1(lr.groupby(test['permno']).transform(lambda s:s.rolling(6).sum().shift(-6)))

# ---------- (1) annual SPMO - S&P 500 ----------
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
ann=lambda s:s.dropna().groupby(s.dropna().index.year).agg(lambda x:(1+x).prod()-1)
sy,my=ann(spm),ann(mkt); yrs=sorted(set(sy.index)&set(my.index)); diff=(sy.reindex(yrs)-my.reindex(yrs))*100
fig,ax=plt.subplots(figsize=(13,6)); x=np.arange(len(yrs))
ax.bar(x,diff.values,color=['crimson' if v>=0 else 'steelblue' for v in diff.values],width=.7)
for i,v in enumerate(diff.values): ax.text(i,v+(0.6 if v>=0 else -1.4),f'{v:+.0f}',ha='center',fontsize=8,fontweight='bold' if yrs[i]==2024 else 'normal')
ax.axhline(0,color='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels(yrs); ax.set_ylabel('SPMO minus S&P 500 (pts/yr)')
ax.set_title('SPMO replica minus S&P 500, by year (2011-2025)\nflat-to-mixed for a decade, then a single huge year (2024)',fontsize=12)
ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-04-annual-spmo-minus-sp500.{e}',dpi=150,bbox_inches='tight');
plt.close(fig)
print("(1) annual SPMO-S&P (pts):"); print({int(y):round(float(diff[y]),1) for y in yrs})

# ---------- (2) SPMO holdings 2024 & 2025 ----------
spath=ha.spmo_path(list(test.groupby('date')))
def holdings(year,topn=15):
    ds=[d for d in spath if d.year==year and spath[d]]; acc={}
    for d in ds:
        for p,w in spath[d].items(): acc.setdefault(p,[]).append(w)
    rows=[{'permno':p,'ticker':nm(p),'company':co(p),'avg_wt':np.sum(v)/len(ds),'pct_months':len(v)/len(ds)} for p,v in acc.items()]
    return pd.DataFrame(rows).sort_values('avg_wt',ascending=False).head(topn)
for y in (2024,2025):
    h=holdings(y); h.to_csv(f'{R}/results/holdings_spmo_{y}.csv',index=False)
    print(f"\n(2) SPMO top-15 holdings {y} (avg weight):")
    print(f"  {'#':>2} {'ticker':>7} {'company':<27}{'avg wt':>8}{'%mo':>6}")
    for i,(_,r) in enumerate(h.iterrows(),1): print(f"  {i:>2} {str(r['ticker']):>7} {r['company']:<27}{r['avg_wt']:>7.1%}{r['pct_months']*100:>5.0f}%")

# ---------- (3) SPMO-oracle selection agreement per year ----------
rows=[]
for d,g in test.groupby('date'):
    if d.month not in (3,9): continue
    g=g.dropna(subset=['spmo_signal','fwd6','me']);
    if len(g)<50: continue
    k=max(1,round(0.2*len(g)))
    spsel=set(g.nlargest(k,'spmo_signal')['permno']); ora=set(g.nlargest(k,'fwd6')['permno'])
    rows.append({'year':d.year,'agree':len(spsel&ora)/k})
ag=pd.DataFrame(rows).groupby('year')['agree'].mean()
fig,ax=plt.subplots(figsize=(13,6)); x=np.arange(len(ag))
ax.bar(x,ag.values*100,color=['crimson' if y==2024 else 'seagreen' for y in ag.index],width=.7)
ax.axhline(ag.mean()*100,color='gray',ls='--',lw=1,label=f'mean {ag.mean():.0%}')
ax.axhline(20,color='navy',ls=':',lw=1.2,label='random-chance baseline (20%)')
for i,v in enumerate(ag.values): ax.text(i,v*100+0.8,f'{v*100:.0f}',ha='center',fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(ag.index.astype(int)); ax.set_ylabel('% of SPMO\'s picks that were also true top-20% (oracle)')
ax.set_title('How often did SPMO pick the actual winners? % of SPMO\'s chosen top-20% that agreed with the oracle\n(oracle = top-20% by realized next-6-month return; per year)',fontsize=11.5)
ax.legend(fontsize=9.5); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-04-spmo-oracle-agreement.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); ag.to_csv(f'{R}/results/2026-06-04-spmo-oracle-agreement.csv')
print("\n(3) SPMO-oracle agreement per year (%):"); print({int(y):round(float(v*100)) for y,v in ag.items()})
print(f"  mean {ag.mean():.0%}  (vs 20% random)")
print("\nsaved: plots/2026-06-04-annual-spmo-minus-sp500.*, plots/2026-06-04-spmo-oracle-agreement.*, results/holdings_spmo_2024.csv, holdings_spmo_2025.csv")
