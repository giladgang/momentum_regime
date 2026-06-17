"""
2026-06-05-spmo-replica-returns-2011-2024.py  (EXPLORATORY)
Cumulative return (growth of $1) of the SPMO replica, 2011-2024. Same validated machinery
(top-20% risk-adj momentum, FMC x score, 9% cap, Mar/Sep rebalance). Real panel, capped at
Dec-2024. Neutral/descriptive; factual CAGR + final-multiple annotation only.
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
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09).dropna()
spm=spm[spm.index<=pd.Timestamp('2024-12-31')]

cum=(1+spm).cumprod()
tot=cum.iloc[-1]-1; cagr=cum.iloc[-1]**(12/len(spm))-1
vol=spm.std()*np.sqrt(12); sharpe=spm.mean()/spm.std()*np.sqrt(12)
print(f"SPMO replica  {spm.index.min().date()} -> {spm.index.max().date()}  ({len(spm)} months)")
print(f"  total return {tot:+.0%}   $1 -> ${cum.iloc[-1]:.2f}   CAGR {cagr:.1%}   vol {vol:.1%}   Sharpe {sharpe:.2f}")

fig,ax=plt.subplots(figsize=(13,6.2))
ax.plot(cum.index,cum.values,color='steelblue',lw=2.4)
ax.fill_between(cum.index,cum.values,1,color='steelblue',alpha=.08)
ax.axhline(1,color='k',lw=.7)
ax.set_ylabel('Growth of $1 invested'); ax.grid(True,alpha=.25)
ax.set_title('SPMO replica cumulative return, 2011-2024',fontsize=12)
ax.text(0.012,0.96,f'$1 → ${cum.iloc[-1]:.2f}   (total {tot:+.0%}, CAGR {cagr:.1%})',
        transform=ax.transAxes,fontsize=10,va='top',color='steelblue')
plt.tight_layout()
OUT=f'{R}/plots/2026-06-05-spmo-replica-returns-2011-2024'
for e in ('pdf','png'): fig.savefig(f'{OUT}.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print(f"\nsaved {OUT}.pdf + .png")
