"""
2026-06-02-oracle-vs-spmo-yearly.py  (EXPLORATORY)
Per-year stacked bar: bottom = SPMO actual annual return (what backward momentum got),
top = extra that the perfect-foresight oracle would have added (oracle - SPMO), so the
full bar height = the oracle's annual return. Oracle = same SPMO machinery but selects
the top-20% by REALIZED next-6-month return ("if we knew which stocks would have the
best momentum"). Real panel. Oracle ends mid-2025 (needs 6 months of future).
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

def annual(s):
    s=s.dropna(); return s.groupby(s.index.year).agg(lambda x:((1+x).prod()-1) if len(x)>=11 else np.nan).dropna()
sy,oy=annual(spm),annual(ora)
yrs=sorted(set(sy.index)&set(oy.index))
sy,oy=sy.reindex(yrs),oy.reindex(yrs)
extra=(oy-sy).clip(lower=0)
print("year   SPMO    oracle   oracle-SPMO")
for y in yrs: print(f"  {y}  {sy[y]:>+7.0%}{oy[y]:>+8.0%}{(oy[y]-sy[y]):>+11.0%}")

fig,ax=plt.subplots(figsize=(13.5,6.5))
x=np.arange(len(yrs))
ax.bar(x,sy.values*100,color='steelblue',label='SPMO actual (backward momentum)')
ax.bar(x,extra.values*100,bottom=sy.values*100,color='crimson',alpha=.8,label='extra if we knew the winners (oracle - SPMO)')
for i,y in enumerate(yrs):
    ax.text(i,oy[y]*100+2,f'{oy[y]*100:.0f}',ha='center',fontsize=8,color='crimson')
    ax.text(i,sy[y]*100/2,f'{sy[y]*100:.0f}',ha='center',fontsize=7,color='white',fontweight='bold')
ax.axhline(0,color='k',lw=.7); ax.set_xticks(x); ax.set_xticklabels(yrs,rotation=0)
ax.set_ylabel('Annual return (%)');
ax.set_title('Each year: SPMO actual return (blue) vs the perfect-foresight ceiling (full bar)\nlight red = return left on the table by picking on PAST instead of knowing the winners',fontsize=11.5)
ax.legend(loc='upper left',fontsize=9.5); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-oracle-vs-spmo-yearly.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print("\nsaved plots/2026-06-02-oracle-vs-spmo-yearly.pdf + .png")
