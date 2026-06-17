"""
2026-06-04-annual-excess-spmo-vs-sp500.py  (EXPLORATORY)
Per-year bar chart of SPMO's annual excess return vs the S&P 500 (Top-500 cap-wt proxy).
Excess_year = compound(SPMO monthly) - compound(market monthly), same convention as the
SUMMARY headline (2024 +23%, 2025 +4%, 2011-2023 ~+0.3%/yr). Shows a flat decade then the
2024 blow-out and a small 2025 echo. Real panel, from panel start (>=11 months/yr) to 2025.
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

def annual(s):
    s=s.dropna(); return s.groupby(s.index.year).agg(lambda x:((1+x).prod()-1) if len(x)>=11 else np.nan).dropna()
sy,my=annual(spm),annual(mkt)
yrs=sorted(set(sy.index)&set(my.index))
sy,my=sy.reindex(yrs),my.reindex(yrs)
exc=(sy-my)*100

pre=exc[[y for y in yrs if y<=2023]]
print("year   SPMO    S&P500   excess")
for y in yrs: print(f"  {y}  {sy[y]:>+7.0%}{my[y]:>+8.0%}{exc[y]:>+8.1f} pts")
print(f"\n2011-2023 avg excess: {pre.mean():+.1f} pts/yr   2024: {exc[2024]:+.1f}   2025: {exc[2025]:+.1f}")

fig,ax=plt.subplots(figsize=(13,6.2))
x=np.arange(len(yrs))
colors=['crimson' if y in (2024,2025) else 'steelblue' for y in yrs]
ax.bar(x,exc.values,color=colors,width=0.66)
for i,y in enumerate(yrs):
    v=exc.values[i]
    ax.text(i, v+(0.6 if v>=0 else -0.6), f'{v:+.0f}', ha='center',
            va='bottom' if v>=0 else 'top', fontsize=9,
            fontweight='bold' if y in (2024,2025) else 'normal',
            color='crimson' if y in (2024,2025) else 'black')
ax.axhline(0,color='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels(yrs,rotation=0)
ax.set_ylabel('Annual excess return vs S&P 500 (percentage points)')
ax.set_title("SPMO annual excess return vs the S&P 500, 2011-2025", fontsize=12)
ax.grid(True,axis='y',alpha=.25)
plt.tight_layout()
OUT=f'{R}/plots/2026-06-04-annual-excess-spmo-vs-sp500'
for e in ('pdf','png'): fig.savefig(f'{OUT}.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print(f"\nsaved {OUT}.pdf + .png")
