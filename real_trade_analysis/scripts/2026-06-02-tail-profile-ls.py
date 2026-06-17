"""
2026-06-02-tail-profile-ls.py  (EXPLORATORY)
Crisis / tail profile of the HEADLINE long-short XGB (full universe, dollar-neutral,
NYSE P10/P90, VW, mom+pi_filter), 2011-2024, vs Market and SPMO. The L/S can SHORT
crashing winners, so unlike the long-only book it may protect (or profit) in panics.
Reports MDD, worst month, down-capture, down-beta; plots underwater drawdowns.
Default (legacy) paths. Full-universe XGB retrain.
"""
import importlib.util, sys, numpy as np, pandas as pd
from xgboost import XGBRegressor
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from src.utils import long_short_port
from config import TRAIN_END,N_ESTIMATORS,MAX_DEPTH,LEARNING_RATE,SUBSAMPLE,COLSAMPLE,XGB_N_JOBS
SEEDS=exp.SEEDS

full=exp.build_features()                                  # FULL universe, legacy
tr=full[full['date']<TRAIN_END]; te=full[full['date']>=TRAIN_END].copy()
Xtr,ytr=tr[exp.FEATURES].values.astype(float),tr['ret_fwd'].values.astype(float)
pred=np.zeros(len(te))
for i,seed in enumerate(SEEDS,1):
    m=XGBRegressor(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,learning_rate=LEARNING_RATE,subsample=SUBSAMPLE,
                   colsample_bytree=COLSAMPLE,tree_method='hist',random_state=seed,verbosity=0,n_jobs=XGB_N_JOBS)
    m.fit(Xtr,ytr); pred+=m.predict(te[exp.FEATURES].values.astype(float))
    if i%10==0: print(f"  seed {i}/{len(SEEDS)}",flush=True)
te['score']=pred/len(SEEDS)
ls=long_short_port(te,'score').dropna()
# market + SPMO from Top-500 for reference
base=sp.add_spmo_inputs(exp.build_features()); t5=exp.apply_size_screen(base,500); t5=t5[t5['date']>=TRAIN_END].copy()
t5['spmo_signal']=t5['mom_value']/t5['sigma_m']
mkt=t5.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False).dropna()
spm=sp.build_longonly(t5,'spmo_signal','score',0.20,0.09).dropna()
S={'Market (cap-wt)':mkt,'SPMO (long-only)':spm,'XGB long-SHORT (dollar-neutral)':ls}

def dd(r): c=(1+r).cumprod(); return c/c.cummax()-1
def tail(r,mk):
    a,b=r.align(mk,join='inner'); down=b<0; up=b>=0
    return {'ann':(1+r).prod()**(12/len(r))-1,'vol':r.std()*np.sqrt(12),'MDD':dd(r).min(),'worst_mo':r.min(),
            'down_capture':a[down].mean()/b[down].mean() if down.any() else np.nan,
            'corr_mkt':a.corr(b),'beta_down':np.polyfit(b[down],a[down],1)[0] if down.sum()>2 else np.nan,
            'panic_note':''}
print(f"\n  {'strategy':<32}{'ann':>6}{'vol':>6}{'Sharpe':>7}{'MDD':>7}{'worstMo':>8}{'downCap':>8}{'corr':>6}{'betaDn':>7}")
for nm,r in S.items():
    t=tail(r,mkt); print(f"  {nm:<32}{t['ann']:>+6.0%}{t['vol']:>6.0%}{t['ann']/t['vol']:>7.2f}{t['MDD']:>+7.0%}{t['worst_mo']:>+8.0%}{t['down_capture']:>8.2f}{t['corr_mkt']:>6.2f}{t['beta_down']:>7.2f}")
print("  (L/S: low/neg corr & down-beta = market-neutral; >0 down-capture in crashes would mean it PROFITS when market falls.)")
# crisis-window returns for the L/S
print("\n  L/S return during crisis windows:")
for a,b,lab in [('2015-08','2016-02','China/oil'),('2018-10','2018-12','Q4-18'),('2020-02','2020-04','COVID'),('2022-01','2022-10','2022 bear')]:
    w=ls[(ls.index>=a)&(ls.index<=b)]; wm=mkt[(mkt.index>=a)&(mkt.index<=b)]
    print(f"    {lab:<12} L/S {((1+w).prod()-1):>+6.0%}   market {((1+wm).prod()-1):>+6.0%}")

fig,ax=plt.subplots(figsize=(13.5,6))
COL={'Market (cap-wt)':'gray','SPMO (long-only)':'steelblue','XGB long-SHORT (dollar-neutral)':'crimson'}
for nm,r in S.items():
    d=dd(r)*100; ax.plot(d.index,d.values,lw=2,color=COL[nm],label=f"{nm}  (MDD {d.min():.0f}%)")
for a,b,lab in [('2015-08','2016-02','China/oil'),('2018-10','2018-12','Q4-18'),('2020-02','2020-04','COVID'),('2022-01','2022-10','2022 bear')]:
    ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color='black',alpha=.06); ax.text(pd.Timestamp(a),2,lab,fontsize=7.5)
ax.axhline(0,color='k',lw=.6); ax.set_ylabel('Drawdown (%)')
ax.set_title('Crisis / tail profile 2011-2024: the long-SHORT XGB vs long-only momentum vs market\n(dollar-neutral L/S decouples from the market -- shallow drawdowns, can profit in panics)',fontsize=11.5)
ax.legend(loc='lower left',fontsize=9.5); ax.grid(True,alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-tail-profile-ls.{e}',dpi=150,bbox_inches='tight')
print("\nsaved tail-profile-ls")
