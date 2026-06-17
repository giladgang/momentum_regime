"""
2026-06-04-capture-bars.py  (EXPLORATORY)
SPMO's down-capture and up-capture vs the S&P 500: of the market's down-months and up-months,
what % of the move does SPMO capture? 2011-2024. 100% = moves exactly with the market;
<100% on downs = cushions; >100% = amplifies. (down-capture/up-capture were table values in
the tail-profile; this plots them.)
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END
base=sp.add_spmo_inputs(exp.build_features()); t5=exp.apply_size_screen(base,500); t5=t5[t5['date']>=TRAIN_END].copy()
t5['spmo_signal']=t5['mom_value']/t5['sigma_m']
mkt=t5.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False).dropna()
spm=sp.build_longonly(t5,'spmo_signal','score',0.20,0.09).dropna()
a,b=spm.align(mkt,join='inner'); down=b<0; up=b>=0
dc=a[down].mean()/b[down].mean()*100; uc=a[up].mean()/b[up].mean()*100
print(f"down-capture {dc:.0f}%  up-capture {uc:.0f}%  (n_down={down.sum()}, n_up={up.sum()})")

fig,ax=plt.subplots(figsize=(8,6))
bars=ax.bar(['Down-capture\n(market-down months)','Up-capture\n(market-up months)'],[dc,uc],
            color=['steelblue','seagreen'],width=.55)
ax.axhline(100,color='crimson',ls='--',lw=1.6,label='S&P 500 = 100% (moves with market)')
for rect,v in zip(bars,[dc,uc]): ax.text(rect.get_x()+rect.get_width()/2,v+1.5,f'{v:.0f}%',ha='center',fontsize=13,fontweight='bold')
ax.set_ylim(0,120); ax.set_ylabel('% of the S&P 500 move captured')
ax.set_title('SPMO: how much of the S&P\'s down-moves and up-moves it captures (2011-2024)\ndown 85% (cushions a little) / up 96% -- a slightly defensive but market-like long-only book',fontsize=11)
ax.legend(loc='lower center',fontsize=9.5); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-04-capture-bars.{e}',dpi=150,bbox_inches='tight')
print("saved capture-bars")
