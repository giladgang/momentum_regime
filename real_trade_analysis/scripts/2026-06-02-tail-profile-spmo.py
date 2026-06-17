"""
2026-06-02-tail-profile-spmo.py  (EXPLORATORY)
Crisis / tail profile 2011-2024, Market (cap-wt) vs SPMO replica only (no XGB).
Underwater drawdown chart + tail table (MDD, worst month, down/up capture, down-beta).
Default legacy paths. Fast (no XGB).
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END

base=sp.add_spmo_inputs(exp.build_features()); t5=exp.apply_size_screen(base,500); t5=t5[t5['date']>=TRAIN_END].copy()
t5['spmo_signal']=t5['mom_value']/t5['sigma_m']
mkt=t5.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False).dropna()
spm=sp.build_longonly(t5,'spmo_signal','score',0.20,0.09).dropna()
S={'Market (cap-wt)':mkt,'SPMO replica':spm}

def dd(r): c=(1+r).cumprod(); return c/c.cummax()-1
def tail(r,mk):
    a,b=r.align(mk,join='inner'); down=b<0; up=b>=0
    return {'ann':(1+r).prod()**(12/len(r))-1,'vol':r.std()*np.sqrt(12),'MDD':dd(r).min(),'worst_mo':r.min(),
            'worst_12mo':(1+r).rolling(12).apply(lambda x:x.prod()-1).min(),
            'down_capture':a[down].mean()/b[down].mean(),'up_capture':a[up].mean()/b[up].mean(),
            'beta_down':np.polyfit(b[down],a[down],1)[0]}
print(f"  {'strategy':<18}{'ann':>6}{'vol':>6}{'Sharpe':>7}{'MDD':>7}{'worstMo':>8}{'worst12m':>9}{'downCap':>8}{'upCap':>7}{'betaDn':>7}")
for nm,r in S.items():
    t=tail(r,mkt); print(f"  {nm:<18}{t['ann']:>+6.0%}{t['vol']:>6.0%}{t['ann']/t['vol']:>7.2f}{t['MDD']:>+7.0%}{t['worst_mo']:>+8.0%}{t['worst_12mo']:>+9.0%}{t['down_capture']:>8.2f}{t['up_capture']:>7.2f}{t['beta_down']:>7.2f}")

fig,ax=plt.subplots(figsize=(13.5,6))
COL={'Market (cap-wt)':'gray','SPMO replica':'steelblue'}
for nm,r in S.items():
    d=dd(r)*100; ax.plot(d.index,d.values,lw=2.2,color=COL[nm],label=f"{nm}  (MDD {d.min():.0f}%)")
for a,b,lab in [('2015-08','2016-02','China/oil'),('2018-10','2018-12','Q4-18'),('2020-02','2020-04','COVID'),('2022-01','2022-10','2022 bear')]:
    ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color='black',alpha=.06); ax.text(pd.Timestamp(a),1.5,lab,fontsize=8)
ax.axhline(0,color='k',lw=.6); ax.set_ylabel('Drawdown (%)')
ax.set_title('Crisis / tail profile 2011-2024: SPMO vs the market\n(long-only momentum tracks the market down -- no crisis protection; MDD slightly deeper)',fontsize=11.5)
ax.legend(loc='lower left',fontsize=10); ax.grid(True,alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-tail-profile-spmo.{e}',dpi=150,bbox_inches='tight')
print("\nsaved tail-profile-spmo")
