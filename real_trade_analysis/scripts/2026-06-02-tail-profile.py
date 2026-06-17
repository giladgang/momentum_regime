"""
2026-06-02-tail-profile.py  (EXPLORATORY)
Crisis / tail-risk profile, 2011-2024 (clean legacy panel; covers COVID, 2022 bear,
Q4-2018, 2015-16). Compares Market (cap-wt) vs SPMO replica vs regime-XGB long-only
(train Top-1000 <2011, trade S&P 500 top-decile VW). Reports max drawdown, worst month,
worst 12-mo, and up/down capture (does the strategy dampen or amplify losses?). Plots an
underwater (drawdown) chart so each crisis trough is visible. Uses DEFAULT (legacy) paths.
"""
import importlib.util, sys, numpy as np, pandas as pd
from xgboost import XGBRegressor
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; lo=_load('lo','2026-05-31-xgb-longonly-sp500.py')
from config import TRAIN_END,N_ESTIMATORS,MAX_DEPTH,LEARNING_RATE,SUBSAMPLE,COLSAMPLE,XGB_N_JOBS
SEEDS=exp.SEEDS

base=sp.add_spmo_inputs(exp.build_features())          # DEFAULT legacy paths (clean, ends ~Nov-2024)
df500=exp.apply_size_screen(base,500); test=df500[df500['date']>=TRAIN_END].copy()
test['spmo_signal']=test['mom_value']/test['sigma_m']
train=exp.apply_size_screen(base,1000); train=train[train['date']<TRAIN_END]
Xtr,ytr=train[exp.FEATURES].values.astype(float),train['ret_fwd'].values.astype(float)
pred=np.zeros(len(test))
for i,seed in enumerate(SEEDS,1):
    m=XGBRegressor(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,learning_rate=LEARNING_RATE,subsample=SUBSAMPLE,
                   colsample_bytree=COLSAMPLE,tree_method='hist',random_state=seed,verbosity=0,n_jobs=XGB_N_JOBS)
    m.fit(Xtr,ytr); pred+=m.predict(test[exp.FEATURES].values.astype(float))
    if i%10==0: print(f"  seed {i}/{len(SEEDS)}",flush=True)
test['score']=pred/len(SEEDS)
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
xgb=lo.long_only_port(test,'score')
S={'Market (cap-wt)':mkt.dropna(),'SPMO replica':spm.dropna(),'Regime-XGB long-only':xgb.dropna()}

def dd(r): c=(1+r).cumprod(); return c/c.cummax()-1
def tail(r,mk):
    d=dd(r); j=r.align(mk,join='inner'); a,b=j
    down=b<0; up=b>=0
    dc=a[down].mean()/b[down].mean() if down.any() else np.nan   # down-capture
    uc=a[up].mean()/b[up].mean() if up.any() else np.nan
    roll12=(1+r).rolling(12).apply(lambda x:x.prod()-1)
    return {'ann':(1+r).prod()**(12/len(r))-1,'vol':r.std()*np.sqrt(12),'sharpe':(1+r).prod()**(12/len(r))/1-1,
            'MDD':d.min(),'worst_mo':r.min(),'worst_12mo':roll12.min(),'down_capture':dc,'up_capture':uc,
            'beta_down':np.polyfit(b[down],a[down],1)[0] if down.sum()>2 else np.nan}
print(f"\n  {'strategy':<24}{'ann':>6}{'vol':>6}{'MDD':>7}{'worstMo':>8}{'worst12m':>9}{'downCap':>8}{'upCap':>7}{'betaDn':>7}")
for nm,r in S.items():
    t=tail(r,mkt); sh=t['ann']/t['vol']
    print(f"  {nm:<24}{t['ann']:>+6.0%}{t['vol']:>6.0%}{t['MDD']:>+7.0%}{t['worst_mo']:>+8.0%}{t['worst_12mo']:>+9.0%}{t['down_capture']:>8.2f}{t['up_capture']:>7.2f}{t['beta_down']:>7.2f}")
print("  (down_capture>1 = amplifies market losses; <1 = dampens. beta_down = sensitivity in down months.)")

fig,ax=plt.subplots(figsize=(13.5,6))
COL={'Market (cap-wt)':'gray','SPMO replica':'steelblue','Regime-XGB long-only':'crimson'}
for nm,r in S.items():
    d=dd(r)*100; ax.plot(d.index,d.values,lw=2,color=COL[nm],label=f"{nm}  (MDD {d.min():.0f}%)")
for a,b,lab in [('2015-08','2016-02','China/oil'),('2018-10','2018-12','Q4-18'),('2020-02','2020-04','COVID'),('2022-01','2022-10','2022 bear')]:
    ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color='black',alpha=.06); ax.text(pd.Timestamp(a),3,lab,fontsize=7.5,rotation=0)
ax.axhline(0,color='k',lw=.6); ax.set_ylabel('Drawdown (%)'); ax.set_ylim(top=8)
ax.set_title('Crisis / tail profile 2011-2024: drawdowns of Market vs SPMO vs regime-XGB\n(long-only momentum gives NO crisis protection; XGB is deepest -- higher beta, no hedge)',fontsize=11.5)
ax.legend(loc='lower left',fontsize=9.5); ax.grid(True,alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-tail-profile.{e}',dpi=150,bbox_inches='tight')
print("\nsaved tail-profile")
