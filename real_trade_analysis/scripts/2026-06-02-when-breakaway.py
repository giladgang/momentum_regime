"""
2026-06-02-when-breakaway.py  (EXPLORATORY)
When did SPMO actually break away from the S&P 500? Plot the RUNNING cumulative
out-performance = cumulative SUM of monthly (SPMO - market) excess. Additive, so no
compounding illusion (where an early lead looks widest later) and no ETF data
artifact. Steep slope = where the lead was earned. Real panel, 2011-Dec-2025.
Panel A: full history. Panel B: 2024-2025 zoom with a marker at Mar-2025.
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
ex=(spm-mkt).dropna()
cum=ex.cumsum()*100   # additive cumulative out-performance, in points

g23=cum[cum.index<=pd.Timestamp('2023-12-31')].iloc[-1]
g24=cum[cum.index<=pd.Timestamp('2024-12-31')].iloc[-1]
g25=cum.iloc[-1]
print(f"cumulative out-performance (additive, pts):  end-2023 {g23:+.0f}  end-2024 {g24:+.0f}  end-2025 {g25:+.0f}")
print(f"  built during 2024: {g24-g23:+.0f} pts | during 2025: {g25-g24:+.0f} pts | all of 2011-2023: {g23:+.0f} pts")

fig,(a1,a2)=plt.subplots(1,2,figsize=(15,6))
a1.plot(cum.index,cum.values,color='crimson',lw=2.2)
a1.fill_between(cum.index,cum.values,0,color='crimson',alpha=.08)
a1.axhline(0,color='k',lw=.7); a1.axvspan(pd.Timestamp('2024-01-01'),cum.index.max(),color='red',alpha=.06)
a1.set_title('Full history 2011-2025: running SPMO out-performance\n(flat for a decade, then nearly all of it in 2024)',fontsize=11)
a1.set_ylabel('Cumulative out-performance vs S&P 500 (pts, summed monthly)'); a1.grid(True,alpha=.25)
a1.annotate(f'+{g24-g23:.0f} pts\nin 2024',xy=(pd.Timestamp('2024-09-01'),(g23+g24)/2),fontsize=10,color='crimson',fontweight='bold',ha='center')

z=cum[cum.index>=pd.Timestamp('2024-01-01')]
a2.plot(z.index,z.values,color='crimson',lw=2.4,marker='o',ms=3)
a2.axvline(pd.Timestamp('2025-03-15'),color='gray',ls='--',lw=1.3); a2.text(pd.Timestamp('2025-03-18'),z.min()+2,'"Mar-2025"',fontsize=9,color='gray')
a2.axhline(0,color='k',lw=.7); a2.set_title('Zoom 2024-2025: the lead opens in 2024, then flattens\n(by Mar-2025 the gap was already built)',fontsize=11)
a2.set_ylabel('Cumulative out-performance (pts)'); a2.grid(True,alpha=.25)
fig.suptitle('When did SPMO break away from the S&P 500? (real CRSP, additive excess, no compounding/no ETF artifact; ends Dec-2025)',fontsize=12); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-when-breakaway.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print("\nsaved plots/2026-06-02-when-breakaway.pdf + .png")
