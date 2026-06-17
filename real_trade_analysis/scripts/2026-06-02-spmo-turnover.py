"""
2026-06-02-spmo-turnover.py  (EXPLORATORY)
SPMO replica turnover at each semi-annual (Mar/Sep) rebalance, 2011-2025, from the
real spliced panel. Turnover at a rebalance = 0.5*sum_p |w_new - w_prev_drifted|
(fraction of the book traded). Low turnover = winners persisted (held); high = the
momentum lineup rotated. No network.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; ha=_load('ha','2026-05-31-holdings-analysis.py')
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spath=ha.spmo_path(list(test.groupby('date')))
dates=sorted(spath)

# turnover at each rebalance month (Mar/Sep) vs the prior month's drifted weights
rows=[]
for i,d in enumerate(dates):
    if d.month not in (3,9) or i==0: continue
    w,wp=spath[d],spath[dates[i-1]]
    if not w or not wp: continue
    to=0.5*sum(abs(w.get(p,0)-wp.get(p,0)) for p in set(w)|set(wp))
    rows.append({'date':d,'turnover':to})
t=pd.DataFrame(rows).set_index('date')
print("semi-annual rebalance turnover (fraction of book traded):")
print((t['turnover']*100).round(0).astype(int).to_string())
print(f"\n  mean {t['turnover'].mean():.0%} | 2024 rebals {t[t.index.year==2024]['turnover'].mean():.0%} | 2025 rebals {t[t.index.year==2025]['turnover'].mean():.0%}")

fig,ax=plt.subplots(figsize=(13,5.5))
col=['crimson' if d.year==2024 else ('darkorange' if d.year==2025 else 'steelblue') for d in t.index]
ax.bar(range(len(t)),t['turnover']*100,color=col,width=0.7)
ax.set_xticks(range(len(t))); ax.set_xticklabels([d.strftime('%b-%y') for d in t.index],rotation=90,fontsize=8)
ax.axhline(t['turnover'].mean()*100,color='gray',ls='--',lw=1,label=f"mean {t['turnover'].mean():.0%}")
ax.set_ylabel('Turnover at rebalance (% of book traded)')
ax.set_title('SPMO replica: turnover at each semi-annual rebalance (Mar/Sep), 2011-2025\nlow = winners persisted and were held; high = the momentum lineup rotated  (red=2024, orange=2025)',fontsize=11)
ax.legend(fontsize=9); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-spmo-turnover.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); t.to_csv(f'{R}/results/2026-06-02-spmo-turnover.csv')
print("\nsaved plots/2026-06-02-spmo-turnover.pdf + .png")
