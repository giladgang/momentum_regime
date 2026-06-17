"""
2026-06-02-dispersion-chart.py  (EXPLORATORY)
Chart the S&P 500 (Top-500) cross-sectional monthly-return dispersion (90-10 spread)
2005-2025, with the CRSP-v2 extension (post Nov-2024) highlighted, a 12-month rolling
mean, and period-mean reference lines.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'
d=pd.read_csv(f'{R}/results/2026-06-02-sp500-dispersion-thru2025.csv',parse_dates=['date']).sort_values('date')
d=d[d['date']>=pd.Timestamp('2005-01-01')].reset_index(drop=True)
d['roll12']=d['p90_p10'].rolling(12,min_periods=6).mean()
SPLICE=pd.Timestamp('2024-11-29')

fig,ax=plt.subplots(figsize=(13,6))
ax.plot(d['date'],d['p90_p10']*100,color='lightsteelblue',lw=0.9,label='monthly 90-10 spread')
leg=d[d['date']<=SPLICE]; ext=d[d['date']>=SPLICE]
ax.plot(leg['date'],leg['roll12']*100,color='navy',lw=2.6,label='12-mo rolling mean (legacy CRSP)')
ax.plot(ext['date'],ext['roll12']*100,color='crimson',lw=2.6,label='12-mo rolling mean (CRSP v2 ext., to Dec-2025)')
ax.axvline(SPLICE,color='gray',ls=':',lw=1)
# period mean reference lines
for a,b,lab,col in [('2011-01','2015-12','2011-15 mean','green'),('2025-01','2025-12','2025 mean','crimson')]:
    m=d[(d['date']>=pd.Timestamp(a))&(d['date']<=pd.Timestamp(b))]['p90_p10'].mean()*100
    ax.axhline(m,color=col,ls='--',lw=1,alpha=.6); ax.text(d['date'].iloc[2],m+0.3,f'{lab} {m:.1f}%',color=col,fontsize=8.5)
ax.set_ylabel('Cross-sectional 90-10 return spread (%/month)'); ax.set_xlabel('')
ax.set_title('S&P 500 cross-sectional monthly-return dispersion, 2005-2025\n(high dispersion = momentum has more to differentiate on; CRSP v2 extends past Nov-2024)',fontsize=12)
ax.legend(loc='upper left',fontsize=9.5); ax.grid(True,alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-dispersion-thru2025.{e}',dpi=150,bbox_inches='tight')
plt.close(fig)
print("saved plots/2026-06-02-dispersion-thru2025.pdf + .png")
print(f"latest 12-mo rolling mean: {d['roll12'].iloc[-1]*100:.1f}%  | 2025 mean {d[d['date'].dt.year==2025]['p90_p10'].mean()*100:.1f}%  | 2024 {d[d['date'].dt.year==2024]['p90_p10'].mean()*100:.1f}%")
