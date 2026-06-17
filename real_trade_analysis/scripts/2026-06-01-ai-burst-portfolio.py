"""
2026-06-01-ai-burst-portfolio.py  (EXPLORATORY / STYLIZED -- NOT a forecast)

Portfolio stage of the AI-burst dynamic sim, TWO scenarios (train XGB once):
  (A) SUSTAINED deflation: AI top-20 crash -50% over 6mo, then flat. No recovery.
  (B) V-SHAPED:            AI crash -50% over 6mo, then recover ~+100% over 6mo.
Rest of Top-500 does +0.6%/mo + noise in both. HMM (DD_z & DISP_z ramp; CS_z &
REL_N_z benign) flags panic throughout (validated; dispersion-driven). Strategies:
XGB long-short, XGB long-only, SPMO replica, VOO. Plus leg-composition diagnostics
(is the XGB LONG or SHORT the crashing AI?). All knobs assumed; output qualitative.
"""
import importlib.util, os, sys, numpy as np, pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from xgboost import XGBRegressor
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{_ROOT}/real_trade_analysis/scripts/{f}')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
exp=_load('exp','2026-05-30-xgb-russell1000.py'); sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); lo=_load('lo','2026-05-31-xgb-longonly-sp500.py')
from src.utils import long_short_port, metrics
from config import (TRAIN_END,N_ESTIMATORS,MAX_DEPTH,LEARNING_RATE,SUBSAMPLE,COLSAMPLE,XGB_N_JOBS,HMM_FEATURES)
R=f'{_ROOT}/real_trade_analysis'; SEEDS=exp.SEEDS; K=2; NM=14; CRASH_MO=6; MOM=[f'mom_{i}' for i in range(1,13)]
rng=np.random.default_rng(42)
_CRSP0=exp._CRSP_PATH; _REG0=exp._REGIME_PATH

# ---- synthetic regime pi_filter (validated machinery; corr=1.000) ----
def forward_filter(Z,mu,Sig,P,pan):
    n=len(Z); le=np.column_stack([multivariate_normal.logpdf(Z,mean=mu[k],cov=Sig[k],allow_singular=True) for k in range(K)])
    la=np.zeros((n,K)); la[0]=np.log(.5)+le[0]
    for t in range(1,n):
        for k in range(K): la[t,k]=le[t,k]+logsumexp(la[t-1]+np.log(P[:,k]))
    la-=logsumexp(la,axis=1,keepdims=True); return np.exp(la)[:,pan]
c=np.load(f'{_ROOT}/data/mcmc_draws.npz'); mu=c['mu_draws'].mean(0); Sg=c['Sigma_draws'].mean(0); P=c['P_draws'].mean(0); pan=int(c['panic_state'])
pn=pd.read_parquet(_REG0); pn['date']=pd.to_datetime(pn['date']); pn=pn.dropna(subset=HMM_FEATURES).sort_values('date'); Zh=pn[HMM_FEATURES].values.astype(float)
def ramp(peak,n=NM): up=np.linspace(peak*0.3,peak,CRASH_MO); dn=np.linspace(peak,0.0,n-CRASH_MO+1)[1:]; return np.concatenate([up,dn])
Zs=np.zeros((NM,4)); Zs[:,0]=ramp(2.5); Zs[:,1]=ramp(2.0)
pi_syn=forward_filter(np.vstack([Zh,Zs]),mu,Sg,P,pan)[-NM:]

# ---- train XGB once on ACTUAL Top-1000 <2011 ----
exp._CRSP_PATH=_CRSP0; exp._REGIME_PATH=_REG0
feat0=exp.build_features(); train=exp.apply_size_screen(feat0,1000); train=train[train['date']<TRAIN_END]
Xtr,ytr=train[exp.FEATURES].values.astype(float),train['ret_fwd'].values.astype(float)
models=[]
for i,seed in enumerate(SEEDS,1):
    m=XGBRegressor(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,learning_rate=LEARNING_RATE,subsample=SUBSAMPLE,
                   colsample_bytree=COLSAMPLE,tree_method='hist',random_state=seed,verbosity=0,n_jobs=XGB_N_JOBS)
    m.fit(Xtr,ytr); models.append(m)
    if i%10==0: print(f"  trained seed {i}/{len(SEEDS)}",flush=True)

# ---- last-month universe + AI basket ----
cols=['permno','date','ret_adj','shrcd','exchcd','prc','me']
raw=pd.read_parquet(_CRSP0,columns=cols); raw['date']=pd.to_datetime(raw['date']); last=raw['date'].max()
elig=raw[(raw['date']==last)&raw['shrcd'].isin([10,11])&raw['exchcd'].isin([1,2,3])&(raw['prc'].abs()>1)].copy(); elig=elig[elig['me'].notna()]
top500=elig.nlargest(500,'me'); ai=set(top500.nlargest(20,'me')['permno'])
sdates=[last+pd.offsets.MonthEnd(i) for i in range(1,NM+1)]
ai_mo=0.5**(1/CRASH_MO)-1; rec_mo=2.0**(1/CRASH_MO)-1   # -50% then back to ~par

def ai_path(kind):
    if kind=='sustained': return [ai_mo]*CRASH_MO+[0.0]*(NM-CRASH_MO)
    return [ai_mo]*CRASH_MO+[rec_mo]*CRASH_MO+[0.0]*(NM-2*CRASH_MO)   # V-shaped

def run(kind):
    path=ai_path(kind); rows=[]
    for _,s in top500.iterrows():
        p=s['permno']; me=s['me']; prc=abs(s['prc']); isai=p in ai
        for i,d in enumerate(sdates):
            r=(path[i] if isai else 0.006)+rng.normal(0,0.03 if isai else 0.02)
            me*=(1+r); prc=max(1.01,prc*(1+r)); rows.append({'permno':p,'date':d,'ret_adj':r,'shrcd':s['shrcd'],'exchcd':s['exchcd'],'prc':prc,'me':me})
    ext=pd.concat([raw,pd.DataFrame(rows)],ignore_index=True).sort_values(['permno','date']).reset_index(drop=True)
    reg=pd.concat([pd.read_parquet(_REG0)[['date','pi_filter']].assign(date=lambda x:pd.to_datetime(x['date'])),
                   pd.DataFrame({'date':sdates,'pi_filter':pi_syn})],ignore_index=True)
    ext.to_parquet(f'{R}/data/_ext_crsp.parquet'); reg.to_parquet(f'{R}/data/_ext_regime.parquet')
    exp._CRSP_PATH=f'{R}/data/_ext_crsp.parquet'; exp._REGIME_PATH=f'{R}/data/_ext_regime.parquet'; sp._CRSP=f'{R}/data/_ext_crsp.parquet'
    feats=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(feats,500)
    test=test[test['date'].isin(sdates)].copy()
    pred=np.zeros(len(test))
    for m in models: pred+=m.predict(test[exp.FEATURES].values.astype(float))
    test['score']=pred/len(models); test['spmo_signal']=test['mom_value']/test['sigma_m']; test['is_ai']=test['permno'].isin(ai)
    xls=long_short_port(test,'score'); xlo=lo.long_only_port(test,'score'); spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
    voo=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
    return test,{'XGB long-short':xls,'XGB long-only':xlo,'SPMO replica':spm,'VOO (S&P 500 cap-wt)':voo}

def diag(test):
    """During the burst (first CRASH_MO months): is the XGB long or short the AI names?"""
    bd=set(sdates[:CRASH_MO]); rows=[]
    for d,g in test.groupby('date'):
        if d not in bd: continue
        g=g.dropna(subset=['score']+MOM); ny=g[g['exchcd']==1]['score']
        if len(ny)<10: continue
        hi,loq=ny.quantile(.9),ny.quantile(.1); L=g[g['score']>=hi]; S=g[g['score']<=loq]
        rows.append({'long_ai%':L['is_ai'].mean(),'short_ai%':S['is_ai'].mean(),
                     'long_mom1':L['mom_1'].mean(),'long_mom12':L['mom_12'].mean(),
                     'short_mom1':S['mom_1'].mean(),'short_mom12':S['mom_12'].mean()})
    return pd.DataFrame(rows).mean()

print("\n"+"="*70)
res={}
for kind in ['sustained','V-shaped']:
    test,series=run(kind); res[kind]=series
    print(f"\n  [{kind.upper()}]   strategy            cum.ret   ann    minMo")
    for nm,r in series.items():
        r=r.dropna(); print(f"      {nm:<22}{(1+r).prod()-1:>+8.1%}{r.mean()*12:>+7.1%}{r.min():>+8.1%}")
    d=diag(test)
    print(f"      burst positioning: LONG leg AI-share {d['long_ai%']:.0%} (mom_1 {d['long_mom1']:+.2f}, mom_12 {d['long_mom12']:+.2f}) | "
          f"SHORT leg AI-share {d['short_ai%']:.0%} (mom_1 {d['short_mom1']:+.2f}, mom_12 {d['short_mom12']:+.2f})")

# ---- chart: both scenarios side by side ----
COL={'XGB long-short':'crimson','XGB long-only':'darkorange','SPMO replica':'steelblue','VOO (S&P 500 cap-wt)':'gray'}
fig,axes=plt.subplots(1,2,figsize=(15,6.2),sharey=False)
for ax,kind in zip(axes,['sustained','V-shaped']):
    for nm,r in res[kind].items():
        cc=(1+r.dropna().sort_index()).cumprod(); ax.plot(range(len(cc)),(cc.values-1)*100,marker='o',ms=3.5,lw=2.2,color=COL[nm],label=f'{nm} ({(cc.iloc[-1]-1)*100:+.0f}%)')
    ax.axhline(0,color='k',lw=.7,ls=':'); ax.axvspan(0,CRASH_MO-1,color='red',alpha=.07)
    ax.set_xlabel('Synthetic month'); ax.set_title(f'{kind.upper()}  (AI -50% over 6mo'+(', no recovery)' if kind=='sustained' else ', then recover)'),fontsize=11)
    ax.legend(loc='best',fontsize=8.5); ax.grid(True,alpha=.25)
axes[0].set_ylabel('Cumulative return (%)')
fig.suptitle('STYLIZED what-if: AI bubble bursts, rest of S&P 500 okay -- regime-momentum vs SPMO vs market',fontsize=12.5)
plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-01-ai-burst-sim.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print("\nsaved plots/2026-06-01-ai-burst-sim.pdf + .png")
