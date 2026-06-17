"""
2026-06-02-why-spmo-works.py  (EXPLORATORY)
Why did SPMO work in 2024 and not other years? SPMO buys past-momentum winners and
HOLDS them 6 months, so it pays only when those winners keep rising (trend persists)
and stalls/loses when they reverse (momentum crash). Per year we measure that
persistence directly: at each rebalance, the cap-weighted forward-6-month return of
the SELECTED winners minus the market over the same window ("did holding the winners
pay?"). Overlay SPMO's annual excess to show they move together. Real panel.
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

# persistence at each Mar/Sep rebalance: cap-wt fwd6 of selected winners - cap-wt fwd6 of market
rows=[]
for d,g in test.groupby('date'):
    if d.month not in (3,9): continue
    g=g.dropna(subset=['spmo_signal','me','fwd6'])
    if len(g)<50: continue
    n=max(1,round(0.2*len(g))); win=g.nlargest(n,'spmo_signal')
    bw=(win['me']/win['me'].sum()*win['fwd6']).sum(); mw=(g['me']/g['me'].sum()*g['fwd6']).sum()
    rows.append({'date':d,'persistence':bw-mw})
pp=pd.DataFrame(rows).set_index('date'); pp['year']=pp.index.year
persist=pp.groupby('year')['persistence'].mean()

# SPMO annual excess
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
ann=lambda s:s.groupby(s.index.year).apply(lambda x:(1+x).prod()-1)
exc=(ann(spm)-ann(mkt)).rename('spmo_excess')

both=pd.concat([persist.rename('persistence'),exc],axis=1).dropna()
print(both.round(3).to_string())
print(f"\n  corr(persistence, SPMO excess) = {both['persistence'].corr(both['spmo_excess']):+.2f}")

fig,ax=plt.subplots(figsize=(13.5,6.2))
x=np.arange(len(both)); yrs=both.index
col=['crimson' if y==2024 else ('seagreen' if both['persistence'].iloc[i]>0 else 'indianred') for i,y in enumerate(yrs)]
ax.bar(x,both['persistence']*100,color=col,width=0.7,label='did holding the winners pay? (winners fwd-6mo minus market)')
ax.axhline(0,color='k',lw=.8)
ax2=ax.twinx(); ax2.plot(x,both['spmo_excess']*100,color='navy',lw=2,marker='o',ms=4,label='SPMO excess vs market (right axis)')
ax.set_xticks(x); ax.set_xticklabels(yrs,rotation=0)
ax.set_ylabel('Winner persistence: winners minus market over next 6mo (pts)')
ax2.set_ylabel('SPMO annual excess vs market (pts)',color='navy'); ax2.tick_params(axis='y',colors='navy')
ax.set_title('Why SPMO works some years and not others: it pays when its winners KEEP winning\n(2024 = strong persistence, trend held; reversal years = winners crash and SPMO stalls)',fontsize=11.5)
ax.annotate('2024',xy=(list(yrs).index(2024),both['persistence'].loc[2024]*100),xytext=(0,8),textcoords='offset points',ha='center',color='crimson',fontweight='bold')
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,loc='upper left',fontsize=9)
ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-why-spmo-works.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print("\nsaved plots/2026-06-02-why-spmo-works.pdf + .png")
