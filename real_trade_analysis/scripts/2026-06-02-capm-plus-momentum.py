"""
2026-06-02-capm-plus-momentum.py  (EXPLORATORY)
Would a CAPM(market) + momentum(SPMO) blend beat momentum alone? Blend w*SPMO +
(1-w)*market for w in [0,1]; report return, vol, Sharpe; find the max-Sharpe mix.
2011-Dec-2025, real panel. Answers: blend can lift Sharpe (diversification) but not
raw return (diluting with lower-return market). Both legs ~the same Sharpe, highly
correlated, so the gain is modest.
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
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
d=pd.concat([spm.rename('mom'),mkt.rename('mkt')],axis=1).dropna()
rf=0.0  # excess-of-zero Sharpe, consistent with prior charts
def stats(r): a=(1+r).prod()**(12/len(r))-1; v=r.std()*np.sqrt(12); return a,v,a/v
print(f"corr(momentum, market) = {d['mom'].corr(d['mkt']):.2f}\n")
ws=np.linspace(0,1,101); rows=[]
for w in ws:
    b=w*d['mom']+(1-w)*d['mkt']; a,v,s=stats(b); rows.append({'w':w,'ann':a,'vol':v,'sharpe':s})
fr=pd.DataFrame(rows); best=fr.loc[fr['sharpe'].idxmax()]
for lbl,w in [('Market (CAPM) only',0.0),('50/50 blend',0.5),(f'max-Sharpe (w*={best.w:.0%} mom)',best.w),('Momentum (SPMO) only',1.0)]:
    a,v,s=stats(w*d['mom']+(1-w)*d['mkt']); print(f"  {lbl:<28} ann {a:>6.1%}  vol {v:>5.1%}  Sharpe {s:>4.2f}")

fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5.8))
a1.plot(fr['w']*100,fr['sharpe'],color='navy',lw=2.4)
a1.axvline(best.w*100,color='crimson',ls='--',lw=1.2,label=f'max Sharpe at {best.w:.0%} momentum')
a1.scatter([0,100],[fr['sharpe'].iloc[0],fr['sharpe'].iloc[-1]],c=['gray','steelblue'],zorder=5)
a1.annotate('market',(0,fr['sharpe'].iloc[0]),textcoords='offset points',xytext=(8,-4),fontsize=9,color='gray')
a1.annotate('momentum',(100,fr['sharpe'].iloc[-1]),textcoords='offset points',xytext=(-60,-4),fontsize=9,color='steelblue')
a1.set_xlabel('% in momentum (rest in market)'); a1.set_ylabel('Sharpe'); a1.set_title('Sharpe of the market+momentum blend',fontsize=11); a1.legend(fontsize=9); a1.grid(True,alpha=.25)
a2.plot(fr['vol']*100,fr['ann']*100,color='purple',lw=2.2)
for w,nm,c in [(0,'market','gray'),(0.5,'50/50','green'),(best.w,'max-Sharpe','crimson'),(1,'momentum','steelblue')]:
    a,v,s=stats(w*d['mom']+(1-w)*d['mkt']); a2.scatter(v*100,a*100,c=c,zorder=5); a2.annotate(nm,(v*100,a*100),textcoords='offset points',xytext=(6,4),fontsize=9,color=c)
a2.set_xlabel('Volatility (%/yr)'); a2.set_ylabel('Return (%/yr)'); a2.set_title('Risk-return of blends (market -> momentum)',fontsize=11); a2.grid(True,alpha=.25)
fig.suptitle('CAPM (market) + momentum (SPMO) blend vs momentum alone, 2011-2025',fontsize=12.5); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-capm-plus-momentum.{e}',dpi=150,bbox_inches='tight')
fr.to_csv(f'{R}/results/2026-06-02-capm-plus-momentum.csv',index=False); print("\nsaved capm-plus-momentum")
