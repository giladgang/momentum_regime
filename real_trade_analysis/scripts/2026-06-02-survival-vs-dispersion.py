"""
2026-06-02-survival-vs-dispersion.py  (EXPLORATORY)
Two series by year (2010-2025) on one chart:
  (1) Past-winners survival  = % of last year's top-20%-by-return that stay on the list
      (from 2026-06-02-retention-3way.csv; momentum survival shown too, nearly identical)
  (2) S&P 500 cross-sectional return dispersion = annual mean of the monthly 90-10 spread
      (from 2026-06-02-sp500-dispersion-thru2025.csv)
Do winners survive more or less when dispersion is high? Reports the correlation.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/Users/giladgang/momentum_regime/real_trade_analysis'
ret=pd.read_csv(f'{R}/results/2026-06-02-retention-3way.csv',index_col=0)   # cols SIZE/MOMENTUM/RETURN (fractions)
disp=pd.read_csv(f'{R}/results/2026-06-02-sp500-dispersion-thru2025.csv',parse_dates=['date'])
disp_y=disp.groupby(disp['date'].dt.year)['p90_p10'].mean()
yrs=[y for y in ret.index if y in disp_y.index]
surv_ret=ret.loc[yrs,'RETURN']*100; surv_mom=ret.loc[yrs,'MOMENTUM']*100; dsp=disp_y.loc[yrs]*100
cr=np.corrcoef(surv_ret.values,dsp.values)[0,1]; cm=np.corrcoef(surv_mom.values,dsp.values)[0,1]
print("year  winners_survival(return)  momentum_survival  dispersion(p90-p10)")
for y in yrs: print(f"  {y}        {surv_ret[y]:>4.0f}%               {surv_mom[y]:>4.0f}%            {dsp[y]:>5.1f}%")
print(f"\n  corr(winner-survival, dispersion) = {cr:+.2f}   corr(momentum-survival, dispersion) = {cm:+.2f}")

fig,ax=plt.subplots(figsize=(13.5,6.3))
x=np.arange(len(yrs))
ax.plot(x,surv_ret.values,color='crimson',lw=2.6,marker='o',ms=6,label='past-winner survival (% of last year\'s top-20% by return still on the list)')
ax.set_ylabel('Past-winner survival (%)',color='crimson'); ax.tick_params(axis='y',colors='crimson'); ax.set_ylim(0,40)
ax.set_xticks(x); ax.set_xticklabels(yrs)
for xi,v in zip(x,surv_ret.values): ax.text(xi,v+1.1,f'{v:.0f}',ha='center',fontsize=7.5,color='crimson')
ax2=ax.twinx()
ax2.plot(x,dsp.values,color='navy',lw=2.6,marker='s',ms=6,label='S&P cross-sectional return dispersion (annual mean 90-10 spread)')
ax2.set_ylabel('Return dispersion: monthly 90-10 spread (%)',color='navy'); ax2.tick_params(axis='y',colors='navy'); ax2.set_ylim(0,28)
ax.set_title(f'Past-winner survival and S&P return dispersion by year (2010-2025)\ncorr = {cr:+.2f}: dispersion and persistence are independent -- a wide spread does not mean winners persist',fontsize=11.5)
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,loc='upper left',fontsize=9)
ax.grid(True,alpha=.2); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-survival-vs-dispersion.{e}',dpi=150,bbox_inches='tight')
print("\nsaved survival-vs-dispersion")
