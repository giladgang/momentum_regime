"""
2026-06-02-spmo-rotation.py  (EXPLORATORY)
SPMO replica holdings rotation, Jan-2024 -> Dec-2025, from the real spliced panel
(_ext_crsp_real.parquet, legacy + CRSP v2). Charts (A) the weight trajectory of the
top mega-cap names (NVIDIA highlighted) and (B) top-10 concentration. No network.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; ha=_load('ha','2026-05-31-holdings-analysis.py')
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'
exp._CRSP_PATH=EXT; sp._CRSP=EXT

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spath=ha.spmo_path(list(test.groupby('date')))

xw=pd.read_csv(f'{R}/results/permno_name_crosswalk.csv').set_index('permno')['ticker'].to_dict()
dates=[d for d in sorted(spath) if d>=pd.Timestamp('2024-01-01')]
allp=set().union(*[set(spath[d]) for d in dates])
W=pd.DataFrame({p:[spath[d].get(p,0.0) for d in dates] for p in allp},index=pd.to_datetime(dates))
# top names by average weight over the window
top=W.mean().sort_values(ascending=False)
KEY=list(top.head(8).index)
top10c=W.apply(lambda r:np.sort(r.values)[::-1][:10].sum(),axis=1)
print("avg weight Jan-2024..Dec-2025, top names:")
for p in KEY: print(f"  {xw.get(p,p):<7} {top[p]:.1%}   (Jan24 {W[p].iloc[0]:.1%} -> Dec25 {W[p].iloc[-1]:.1%})")
print(f"NVIDIA(86580): start {W[86580].iloc[0]:.1%}  max {W[86580].max():.1%}  end {W[86580].iloc[-1]:.1%}")
print(f"top-10 concentration: start {top10c.iloc[0]:.0%}  end {top10c.iloc[-1]:.0%}")

fig,(a1,a2)=plt.subplots(1,2,figsize=(15,6.2))
cmap=plt.cm.tab10
for i,p in enumerate(KEY):
    lw=3.5 if p==86580 else 1.8; z=5 if p==86580 else 2
    a1.plot(W.index,W[p]*100,lw=lw,color=cmap(i),marker='o',ms=3,label=xw.get(p,str(p)),zorder=z)
a1.axhline(9,color='gray',ls=':',lw=1); a1.text(W.index[0],9.2,'9% cap',fontsize=8,color='gray')
a1.set_ylabel('Weight in SPMO replica (%)'); a1.set_title('Top mega-cap weights (NVIDIA bold)',fontsize=11)
a1.legend(loc='upper right',fontsize=8.5,ncol=2); a1.grid(True,alpha=.25)
a2.plot(top10c.index,top10c*100,lw=2.6,color='crimson',marker='o',ms=3.5)
a2.set_ylabel('Top-10 weight share (%)'); a2.set_title('SPMO replica concentration (top-10 share)',fontsize=11)
a2.grid(True,alpha=.25); a2.set_ylim(0,top10c.max()*110)
fig.suptitle('SPMO replica holdings rotation, Jan-2024 to Dec-2025 (real CRSP data)',fontsize=12.5); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-spmo-rotation.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print("\nsaved plots/2026-06-02-spmo-rotation.pdf + .png")
