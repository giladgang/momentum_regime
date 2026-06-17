"""
2026-06-02-why2024-heatmap.py  (EXPLORATORY)
Why did SPMO win only in 2024? A conditions x years heatmap of the four conditions SPMO
needs (all oriented so higher = better), plus the outcome. Color = percentile rank across
years (green = favorable). 2024 is the only column green on all four; other years miss one.
  1 Dispersion        = annual mean 90-10 cross-sectional spread (wide spread to capture)
  2 Winner persistence= % of last year's top-20%-by-return still on the list (winners hold)
  3 Winner magnitude  = SPMO's top-3 name contributions to excess (concentrated bets paid)
  4 Breadth           = rest-of-book contribution = excess - top3 (nothing dragged)
  OUTCOME             = SPMO excess vs market (linear)
Reads saved CSVs. No recompute.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R='/Users/giladgang/momentum_regime/real_trade_analysis'
disp=pd.read_csv(f'{R}/results/2026-06-02-sp500-dispersion-thru2025.csv',parse_dates=['date'])
disp_y=disp.groupby(disp['date'].dt.year)['p90_p10'].mean()*100
ret=pd.read_csv(f'{R}/results/2026-06-02-retention-3way.csv',index_col=0)['RETURN']*100
attr=pd.read_csv(f'{R}/results/2026-06-02-attribution-concentration.csv').set_index('year')
yrs=[y for y in range(2011,2026) if y in attr.index and y in ret.index and y in disp_y.index]
mag=attr.loc[yrs,'top3_contrib']*100
exc=attr.loc[yrs,'excess']*100
brd=exc-mag
M=pd.DataFrame({'1. Dispersion (spread to capture)':disp_y.loc[yrs].values,
               '2. Winner persistence (winners hold)':ret.loc[yrs].values,
               '3. Winner magnitude (top-3 bets paid)':mag.values,
               '4. Breadth (rest didn\'t drag)':brd.values,
               'OUTCOME: SPMO excess vs market':exc.values},index=yrs).T
fmt={'1. Dispersion (spread to capture)':'{:.0f}','2. Winner persistence (winners hold)':'{:.0f}%',
     '3. Winner magnitude (top-3 bets paid)':'{:+.0f}','4. Breadth (rest didn\'t drag)':'{:+.0f}',
     'OUTCOME: SPMO excess vs market':'{:+.0f}'}
pct=M.rank(axis=1,pct=True)   # color by percentile across years, per row (green=high=favorable)

fig,ax=plt.subplots(figsize=(15,5.2))
ax.imshow(pct.values,cmap='RdYlGn',aspect='auto',vmin=0,vmax=1)
ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs)
ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index,fontsize=10)
for i,row in enumerate(M.index):
    for j,y in enumerate(yrs):
        ax.text(j,i,fmt[row].format(M.loc[row,y]),ha='center',va='center',fontsize=8.5,
                color='black')
ax.axhline(3.5,color='black',lw=2.5)   # divider above OUTCOME row
j24=yrs.index(2024)
ax.add_patch(Rectangle((j24-0.5,-0.5),1,len(M),fill=False,edgecolor='blue',lw=3))
ax.text(j24,-0.85,'all four green',ha='center',color='blue',fontsize=9,fontweight='bold')
ax.set_title('Why SPMO won only in 2024: the one year all four conditions aligned (green = favorable vs other years)',fontsize=12,pad=22)
plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-why2024-heatmap.{e}',dpi=150,bbox_inches='tight')
M.to_csv(f'{R}/results/2026-06-02-why2024-heatmap.csv')
print(M.round(1).to_string()); print('\nsaved why2024-heatmap')
