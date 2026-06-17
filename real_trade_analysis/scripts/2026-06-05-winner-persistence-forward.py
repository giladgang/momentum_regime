"""
2026-06-05-winner-persistence-forward.py  (EXPLORATORY)
Forward winner-persistence per year: of each year's top-20%-by-return names (S&P 500
proxy, top-500 by entering cap), what % were STILL in the top-20% the FOLLOWING year.
Re-indexes the existing destination-year CSV (2026-06-02-topreturn-persistence.csv) onto
the SOURCE year, so the bar at year Y = persistence of Y's winners into Y+1. 2024 (-> 2025)
highlighted. No new DB pull. Neutral, descriptive (no interpretive title text).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/giladgang/momentum_regime/real_trade_analysis'

pe=pd.read_csv(f'{R}/results/2026-06-02-topreturn-persistence.csv')
# CSV 'year'=destination d, pct_retained = |W(d-1) & W(d)| / |W(d)|.
# Forward persistence of SOURCE year s := value at destination s+1.
pe['source']=pe['year']-1
fwd=pe.set_index('source')['pct_retained']*100
fwd=fwd.sort_index()
print("source year  -> next-year persistence (% of winners still in top-20%)")
for s in fwd.index: print(f"  {s} -> {s+1}   {fwd[s]:>4.0f}%")
print(f"\n  mean {fwd.mean():.0f}%   2024->2025 {fwd.get(2024,np.nan):.0f}%   2023->2024 {fwd.get(2023,np.nan):.0f}%")

yrs=list(fwd.index)
fig,ax=plt.subplots(figsize=(13,6.2)); x=np.arange(len(yrs))
colors=['crimson' if y==2024 else 'steelblue' for y in yrs]
ax.bar(x,fwd.values,color=colors,width=0.68)
ax.axhline(fwd.mean(),color='gray',ls='--',lw=1.1,label=f'sample mean {fwd.mean():.0f}%')
for i,y in enumerate(yrs):
    ax.text(i,fwd.values[i]+0.7,f'{fwd.values[i]:.0f}',ha='center',
            fontsize=9,fontweight='bold' if y==2024 else 'normal',
            color='crimson' if y==2024 else 'black')
ax.set_xticks(x); ax.set_xticklabels([f'{y}→{y+1}' for y in yrs],rotation=45,ha='right',fontsize=8.5)
ax.set_ylim(0,40); ax.set_ylabel("% of year's top-20% performers still top-20% the next year")
ax.set_title("Year-over-year persistence of S&P 500 top-20% performers (next-year retention)",fontsize=12)
ax.legend(loc='upper left',fontsize=9.5); ax.grid(True,axis='y',alpha=.25)
plt.tight_layout()
OUT=f'{R}/plots/2026-06-05-winner-persistence-forward'
for e in ('pdf','png'): fig.savefig(f'{OUT}.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print(f"\nsaved {OUT}.pdf + .png")
