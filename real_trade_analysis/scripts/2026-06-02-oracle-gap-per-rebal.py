"""
2026-06-02-oracle-gap-per-rebal.py  (EXPLORATORY)
At each SPMO decision point (semi-annual Mar/Sep rebalance), how much did the
perfect-foresight oracle beat SPMO over the following 6-month holding window?
Gap = oracle 6-mo return - SPMO 6-mo return, per decision. Same machinery as
2026-06-02-oracle-vs-spmo.py. Real panel.
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
test=test.sort_values(['permno','date'])
lr=np.log1p(test['ret_adj'].clip(lower=-0.999))
test['fwd6']=np.expm1(lr.groupby(test['permno']).transform(lambda s:s.rolling(6).sum().shift(-6)))
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
ora=sp.build_longonly(test.dropna(subset=['fwd6']),'fwd6','score',0.20,0.09)

# assign each month to its active decision (most recent Mar/Sep rebalance)
def decision(d):
    y=d.year
    if d.month>=9: return pd.Timestamp(y,9,1)
    if d.month>=3: return pd.Timestamp(y,3,1)
    return pd.Timestamp(y-1,9,1)
df=pd.concat([spm.rename('spmo'),ora.rename('oracle')],axis=1).dropna()
df['dp']=[decision(d) for d in df.index]
per=df.groupby('dp').apply(lambda g:pd.Series({'spmo':(1+g['spmo']).prod()-1,'oracle':(1+g['oracle']).prod()-1,'n':len(g)}),include_groups=False)
per=per[per['n']>=4]   # full ~6-month windows only
per['gap']=per['oracle']-per['spmo']
print("decision point   SPMO_6mo  oracle_6mo   gap (oracle-SPMO)")
for d,r in per.iterrows():
    print(f"  {d.strftime('%b-%Y'):<14}{r['spmo']:>+8.1%}{r['oracle']:>+11.1%}{r['gap']:>+13.1%}")
print(f"\n  mean gap per decision {per['gap'].mean():+.1%} | median {per['gap'].median():+.1%} | min {per['gap'].min():+.1%} | max {per['gap'].max():+.1%}")

fig,ax=plt.subplots(figsize=(14,6))
x=np.arange(len(per))
ax.bar(x,per['gap']*100,color='crimson',alpha=.85,width=0.75)
ax.axhline(per['gap'].mean()*100,color='gray',ls='--',lw=1,label=f"mean gap {per['gap'].mean():.0%}")
ax.set_xticks(x); ax.set_xticklabels([d.strftime('%b-%y') for d in per.index],rotation=90,fontsize=8)
ax.set_ylabel('Oracle minus SPMO return over the 6-mo window (pts)')
ax.set_title('How much perfect foresight would have beaten SPMO at each semi-annual decision\n(gap = perfect top-20% pick minus actual momentum pick, per rebalance)',fontsize=11.5)
ax.legend(fontsize=9.5); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-oracle-gap-per-rebal.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); per.to_csv(f'{R}/results/2026-06-02-oracle-gap-per-rebal.csv')
print("\nsaved plots/2026-06-02-oracle-gap-per-rebal.pdf + .png")
