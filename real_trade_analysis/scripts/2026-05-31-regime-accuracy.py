"""
2026-05-31-regime-accuracy.py
=============================
EXPLORATORY (real_trade_analysis). Two questions, no WRDS / no training needed:

(5) Is the HMM regime predictor (pi_filter) ACCURATE?
    - Crisis alignment: avg pi during known crises vs calm windows
    - Market concordance: market return / drawdown / vol conditional on panic-call
    - Persistence: regime run lengths & switch frequency (regimes should persist)
(6) Why did the XGB strategy lag in 2011-2015?
    - Regime composition by sub-period (how many panic months)
    - Strategy calm vs panic performance by sub-period

Reads panel_with_regimes.parquet + the saved XGB long-only (train1000/trade-S&P500)
returns. Writes a report CSV + a regime timeline plot.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'
p=pd.read_parquet(f'{_ROOT}/data/panel_with_regimes.parquet').copy()
p['date']=pd.to_datetime(p['date']); p=p.sort_values('date').reset_index(drop=True)
p['panic']=(p['pi_filter']>0.5).astype(int)

CRISES={'Dot-com (00-02)':('2000-03','2002-10'),'GFC (07-09)':('2007-10','2009-06'),
        'US downgrade/EU (11)':('2011-08','2011-10'),'China/oil (15-16)':('2015-08','2016-02'),
        'Q4 selloff (18)':('2018-10','2018-12'),'COVID (20)':('2020-02','2020-05'),
        '2022 bear':('2022-01','2022-10')}
CALM=[('Calm 03-06','2003-07','2006-12'),('Calm 13-14','2013-01','2014-12'),('Calm 17','2017-01','2017-12')]

def win(a,b):
    m=(p['date']>=a)&(p['date']<=b); return p[m]
print("="*72+"\n  (5a) CRISIS ALIGNMENT  (pi should be HIGH in crises, LOW in calm)\n"+"="*72)
print(f"  {'window':<24}{'avg pi_filter':>14}{'avg pi_smooth':>14}{'% panic':>9}")
for nm,(a,b) in CRISES.items():
    w=win(a,b); print(f"  {nm:<24}{w['pi_filter'].mean():>14.2f}{w['pi_smooth'].mean():>14.2f}{w['panic'].mean():>8.0%}")
print("  "+"-"*60)
for nm,a,b in CALM:
    w=win(a,b); print(f"  {nm:<24}{w['pi_filter'].mean():>14.2f}{w['pi_smooth'].mean():>14.2f}{w['panic'].mean():>8.0%}")

# (5b) market concordance
print("\n"+"="*72+"\n  (5b) MARKET STATE conditional on regime call (full sample)\n"+"="*72)
mret='vwretd' if 'vwretd' in p.columns else 'ret_next'
for lbl,sub in [('PANIC-called (pi>0.5)',p[p['panic']==1]),('CALM-called (pi<0.5)',p[p['panic']==0])]:
    print(f"  {lbl:<24} n={len(sub):>3}  mkt ret mean {sub[mret].mean():>+6.2%}  vol {sub[mret].std():>5.2%}  avg DD {sub['DD'].mean():>+6.2f}  avg VOL {sub['VOL'].mean():>5.2f}")
print(f"\n  concordance: corr(pi_filter, DD)={p['pi_filter'].corr(p['DD']):+.2f}  "
      f"corr(pi_filter, -mkt ret)={p['pi_filter'].corr(-p[mret]):+.2f}  corr(pi_filter, VOL)={p['pi_filter'].corr(p['VOL']):+.2f}")

# (5c) persistence
sw=(p['panic'].diff().abs()==1).sum()
runs=[]; cur=1
for i in range(1,len(p)):
    if p['panic'].iloc[i]==p['panic'].iloc[i-1]: cur+=1
    else: runs.append(cur); cur=1
runs.append(cur)
print("\n"+"="*72+"\n  (5c) PERSISTENCE  (regimes should persist, not flicker)\n"+"="*72)
print(f"  regime switches: {sw} over {len(p)} months  |  avg run length {np.mean(runs):.1f} mo  |  "
      f"panic episodes: {p['panic'].diff().eq(1).sum()}  |  overall % panic {p['panic'].mean():.0%}")

# (6) 2011-2015 weakness: regime composition + strategy perf by sub-period
print("\n"+"="*72+"\n  (6) REGIME COMPOSITION & STRATEGY PERF BY SUB-PERIOD\n"+"="*72)
xgb=pd.read_csv(f'{R}/results/2026-05-31-final-spmo-vs-xgb1000_returns.csv',index_col=0,parse_dates=True)['xgb_lo'].dropna()
pim=p.set_index('date')['pi_filter']; pim_p=pim.copy(); pim_p.index=pim.index.to_period('M')
def perf(r,a,b):
    ym=r.index.to_period('M'); rr=r[(ym>=pd.Period(a))&(ym<=pd.Period(b))]
    pp=pim_p.reindex(rr.index.to_period('M')).values
    sh=lambda x: x.mean()/x.std()*np.sqrt(12) if len(x)>1 and x.std()>0 else np.nan
    panic_share=(pp>=0.5).mean()
    return len(rr), sh(rr), sh(rr[pp<0.5]), sh(rr[pp>=0.5]), panic_share
print(f"  {'sub-period':<12}{'mo':>4}{'%panic':>8}{'XGB Sharpe':>12}{'calm':>7}{'panic':>7}")
for nm,a,b in [('2011-2015','2011-01','2015-12'),('2016-2020','2016-01','2020-12'),('2021-2024','2021-01','2024-11')]:
    n,full,calm,pan,ps=perf(xgb,a,b)
    print(f"  {nm:<12}{n:>4}{ps:>8.0%}{full:>12.2f}{calm:>7.2f}{pan:>7.2f}")

# timeline plot (test period)
fig,ax=plt.subplots(figsize=(13,4))
tp=p[p['date']>=pd.Timestamp('2011-01-01')]
ax.fill_between(tp['date'],tp['pi_filter'],color='crimson',alpha=0.4,label='pi_filter (panic prob)')
ax.axhline(0.5,color='k',lw=0.6,ls=':')
for nm,(a,b) in CRISES.items():
    if pd.Timestamp(a)>=pd.Timestamp('2011-01-01'):
        ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color='gray',alpha=0.18)
ax.set_ylim(0,1); ax.set_ylabel('pi_filter'); ax.set_title('HMM panic probability vs known stress windows (test 2011-2024; gray=crises)')
ax.legend(loc='upper left',fontsize=8); plt.tight_layout()
fig.savefig(f'{R}/plots/2026-05-31-regime-timeline.png',dpi=150); plt.close(fig)
print(f"\n  saved plot real_trade_analysis/plots/2026-05-31-regime-timeline.png")
