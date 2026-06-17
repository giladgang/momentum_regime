"""
2026-06-04-top10-overlap-spmo-oracle.py  (EXPLORATORY)
Per year: what % of SPMO's top-10 holdings were ALSO the perfect-foresight oracle's
top-10 holdings. Same machinery for both (top-20% select, FMC x score weight, 9% cap,
Mar/Sep rebalance); the ONLY difference is the selection signal:
  SPMO   -> past momentum (mom_value / sigma_m)      [what we can actually see]
  ORACLE -> realized next-6-month return (fwd6)       [knowledge from the future]
At each rebalance we take each portfolio's 10 largest positions and measure overlap
(|SPMO10 ∩ ORACLE10| / 10). One bar per year = average overlap across that year's
rebalances. Oracle needs 6 months of future, so Sep-2025 rebalance is excluded
(2025 = March only). Real panel.
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

def top10(grp, signal_col):
    """The 10 largest positions of the top-20% FMC x score portfolio for one rebalance."""
    g=grp.dropna(subset=['me',signal_col,'ret_fwd']).copy()
    if len(g)<20: return None
    g['_ms']=sp.momentum_score(g[signal_col])
    n_sel=max(1,round(sp.TOP_FRAC*len(g)))
    sel=g.nlargest(n_sel,'_ms').copy()
    w=sp.cap_weights(sel['me']*sel['_ms'], sel['me'], sp.WCAP, sp.WCAP_MULT)
    sel['_w']=np.asarray(w)
    return set(sel.nlargest(min(10,len(sel)),'_w')['permno'])

rows=[]
for date,grp in test.groupby('date'):
    if date.month not in sp.REBAL_MONTHS: continue
    s10=top10(grp,'spmo_signal')
    o10=top10(grp,'fwd6')                     # oracle: needs fwd6 (future), None when unavailable
    if not s10 or not o10: continue
    rows.append({'date':date,'year':date.year,'overlap':len(s10&o10)/10.0,
                 'spmo':sorted(s10),'oracle':sorted(o10)})
reb=pd.DataFrame(rows)
yearly=reb.groupby('year')['overlap'].mean()*100
nreb=reb.groupby('year')['overlap'].size()

print("rebalance-level overlap (top-10 SPMO vs oracle):")
for _,r in reb.iterrows(): print(f"  {r['date'].date()}  {r['overlap']*100:>5.0f}%  ({int(r['overlap']*10)}/10 names shared)")
print("\nyearly average:")
for y in yearly.index: print(f"  {y}  {yearly[y]:>5.0f}%   ({nreb[y]} rebal)")

fig,ax=plt.subplots(figsize=(12.5,6.0))
x=np.arange(len(yearly))
bars=ax.bar(x,yearly.values,color='steelblue',width=0.62)
for i,y in enumerate(yearly.index):
    ax.text(i,yearly.values[i]+1.2,f'{yearly.values[i]:.0f}%',ha='center',fontsize=9.5,fontweight='bold')
    if nreb[y]==1: ax.text(i,2,'Mar only',ha='center',fontsize=7,color='white',rotation=90,va='bottom')
ax.set_xticks(x); ax.set_xticklabels(yearly.index,rotation=0)
ax.set_ylim(0,100); ax.set_ylabel('% of SPMO top-10 also in the oracle top-10')
ax.set_title("How often were SPMO's 10 biggest bets the RIGHT bets?\n"
             "Overlap between SPMO's top-10 holdings and the perfect-foresight oracle's top-10, per year",
             fontsize=12)
ax.grid(True,axis='y',alpha=.25); ax.axhline(yearly.mean(),color='crimson',lw=1.3,ls='--',
        label=f'period average  {yearly.mean():.0f}%'); ax.legend(loc='upper right',fontsize=9.5)
plt.tight_layout()
OUT=f'{R}/plots/2026-06-04-top10-overlap-spmo-oracle'
for e in ('pdf','png'): fig.savefig(f'{OUT}.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print(f"\nsaved {OUT}.pdf + .png")
