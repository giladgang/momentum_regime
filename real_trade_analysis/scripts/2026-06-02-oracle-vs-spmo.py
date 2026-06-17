"""
2026-06-02-oracle-vs-spmo.py  (EXPLORATORY)
SPMO (real, backward-looking momentum) vs a PERFECT-FORESIGHT "oracle" that uses the
identical machinery (top-quintile = top 20%, FMC x score weighting, 9% cap, semi-annual
rebalance) but SELECTS on the realized next-6-month return instead of past momentum.
The oracle is a cheating upper bound: "if we had known which top 20% to pick." The gap =
how much of the achievable cross-sectional return momentum leaves on the table. + market.
Real panel 2011-Dec-2025.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from src.utils import metrics
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy()
test['spmo_signal']=test['mom_value']/test['sigma_m']
# perfect-foresight signal: realized next-6-month return (the holding window between rebalances)
test=test.sort_values(['permno','date'])
lr=np.log1p(test['ret_adj'].clip(lower=-0.999))
test['fwd6']=np.expm1(lr.groupby(test['permno']).transform(lambda s:s.rolling(6).sum().shift(-6)))

spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)               # actual momentum
ora=sp.build_longonly(test.dropna(subset=['fwd6']),'fwd6','score',0.20,0.09)  # perfect foresight
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)

def cagr(r): r=r.dropna(); return (1+r).prod()**(12/len(r))-1
for nm,r in [('S&P 500 (market)',mkt),('SPMO (real momentum)',spm),('ORACLE (perfect top-20% pick)',ora)]:
    a,v,s,m=metrics(r.dropna()); print(f"  {nm:<32} CAGR {cagr(r):>7.1%}  vol {v:>5.1%}  Sharpe {s:>4.2f}")
print(f"\n  2024 returns:  market {((1+mkt[mkt.index.year==2024]).prod()-1):+.0%}  SPMO {((1+spm[spm.index.year==2024]).prod()-1):+.0%}  oracle {((1+ora[ora.index.year==2024]).prod()-1):+.0%}")

fig,ax=plt.subplots(figsize=(12.5,6.5))
for nm,r,c in [('S&P 500 (market)',mkt,'gray'),('SPMO (real, picks by PAST momentum)',spm,'steelblue'),('ORACLE: perfect foresight, picks the top-20% by FUTURE return',ora,'crimson')]:
    cc=(1+r.dropna()).cumprod()
    ax.plot(cc.index,cc.values,lw=2.3,color=c,label=f'{nm}  (CAGR {cagr(r):.0%})')
ax.set_yscale('log'); ax.set_ylabel('Growth of $1 (log scale)'); ax.grid(True,alpha=.25,which='both')
ax.set_title('SPMO vs the perfect-foresight ceiling: same rules (top 20%, FMC x score, semi-annual),\nbut the oracle picks the winners in advance. The gap = what momentum cannot see.',fontsize=11)
ax.legend(loc='upper left',fontsize=9.5); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-oracle-vs-spmo.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print("\nsaved plots/2026-06-02-oracle-vs-spmo.pdf + .png")
