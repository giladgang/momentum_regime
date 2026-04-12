"""
run_ultimate_tests.py
=====================
Steps 1-5 of the ultimate feature validation.
Called by ultimate_feature_test.py after HMM quality gates pass.

Step 1: Heavy Ensemble (20 HMM + 50 XGB)
Step 2: Sub-period Consistency
Step 3: Stability Test (20 experiments)
Step 4: Leave-one-year-out
Step 5: Factor Model Alphas
"""

import numpy as np, pandas as pd, pickle, warnings, time, sys, os
from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from xgboost import XGBRegressor
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRADING_FEE, TRAIN_END, N_ESTIMATORS, MAX_DEPTH,
                    LEARNING_RATE, SUBSAMPLE, COLSAMPLE, MOM_FEATURES,
                    HMM_ITERATIONS, HMM_BURNIN, HMM_FEATURES)

FEATS = HMM_FEATURES
REDUCED = MOM_FEATURES + ['pi_filter']
FEE = TRADING_FEE
K = 2

# ── Load data ──
panel = pd.read_parquet('data/panel.parquet')
panel['date'] = pd.to_datetime(panel['date'])
with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
try:
    ff = pd.read_parquet('data/ff_factors.parquet')
except:
    ff = None

sub = panel[['date'] + FEATS].dropna().reset_index(drop=True)
sub = sub[sub['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
dates_all = sub['date'].values
Z_full = sub[FEATS].values.astype(float)
Z_tr = Z_full[(sub['date'] < TRAIN_END).values]
train_dates = sub[sub['date'] < TRAIN_END]['date'].values

# ── HMM functions ──
def log_emission(Z, mu, Sigma):
    return np.column_stack([multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True) for k in range(K)])

def ffbs(Z, mu, Sigma, P):
    n=len(Z); le=log_emission(Z,mu,Sigma); la=np.zeros((n,K)); la[0]=np.log(0.5)+le[0]
    for t in range(1,n):
        for k in range(K): la[t,k]=le[t,k]+logsumexp(la[t-1]+np.log(P[:,k]+1e-300))
    la-=logsumexp(la,axis=1,keepdims=True); a=np.exp(la); s=np.zeros(n,dtype=int)
    s[n-1]=np.random.choice(K,p=a[n-1])
    for t in range(n-2,-1,-1):
        p=a[t]*P[:,s[t+1]]; p/=p.sum(); s[t]=np.random.choice(K,p=p)
    return s

def forward_filter(Z, mu, Sigma, P):
    n=len(Z); le=log_emission(Z,mu,Sigma); la=np.zeros((n,K)); la[0]=np.log(0.5)+le[0]
    for t in range(1,n):
        for k in range(K): la[t,k]=le[t,k]+logsumexp(la[t-1]+np.log(P[:,k]+1e-300))
    la-=logsumexp(la,axis=1,keepdims=True); return np.exp(la)

def fit_hmm(seed=42):
    np.random.seed(seed); D=Z_tr.shape[1]
    m0=np.zeros(D); k0=0.01; nu0=D+2; Psi0=np.eye(D)*(nu0-D-1)
    adir=np.array([[9.,1.],[1.,9.]])
    states=(Z_tr[:,0]<np.median(Z_tr[:,0])).astype(int)
    mu=np.zeros((K,D)); Sig=np.array([np.eye(D)]*K)
    for k in range(K):
        idx=states==k
        if idx.sum()>D+1: mu[k]=Z_tr[idx].mean(0); Sig[k]=np.cov(Z_tr[idx].T)+1e-6*np.eye(D)
    P=np.array([[.95,.05],[.05,.95]])
    for _ in range(HMM_ITERATIONS):
        states=ffbs(Z_tr,mu,Sig,P)
        for k in range(K):
            Zk=Z_tr[states==k]; nk=len(Zk)
            if nk<D+2: continue
            xb=Zk.mean(0); Sk=(Zk-xb).T@(Zk-xb)
            kn=k0+nk; mn=(k0*m0+nk*xb)/kn; nun=nu0+nk
            Pn=Psi0+Sk+(k0*nk/kn)*np.outer(xb-m0,xb-m0)
            try: Sig[k]=invwishart.rvs(df=nun,scale=Pn); mu[k]=np.random.multivariate_normal(mn,Sig[k]/kn)
            except: pass
        for i in range(K):
            c=np.array([np.sum((states[:-1]==i)&(states[1:]==j)) for j in range(K)],dtype=float)
            P[i]=np.random.dirichlet(adir[i]+c)
    crisis_windows=[('2000-03-01','2002-10-01'),('2007-10-01','2009-06-01')]
    crisis_mask=np.zeros(len(Z_tr),dtype=bool)
    for s,e in crisis_windows:
        crisis_mask|=((train_dates>=np.datetime64(s))&(train_dates<=np.datetime64(e)))
    signs=np.zeros(D)
    if crisis_mask.sum()>5:
        for j in range(D):
            signs[j]=1.0 if np.percentile(Z_tr[crisis_mask,j],95)>=np.percentile(Z_tr[~crisis_mask,j],95) else -1.0
    score0=np.sum(signs*mu[0]); score1=np.sum(signs*mu[1])
    ps=1 if score1>=score0 else 0
    return forward_filter(Z_full,mu,Sig,P)[:,ps]

def long_short_port(df_test, score_col, fee=FEE):
    monthly,plw,psw=[],{},{}
    for date,grp in df_test.groupby('date'):
        nyse=grp[grp['exchcd']==1][score_col].dropna()
        if len(nyse)<10: continue
        lo,hi=nyse.quantile(0.10),nyse.quantile(0.90)
        L=grp[grp[score_col]>=hi]; S=grp[grp[score_col]<=lo]
        if L['me'].sum()==0 or S['me'].sum()==0: continue
        lme=L['me'].sum(); nlw=(L.set_index('permno')['me']/lme).to_dict()
        rl=(L['ret_fwd']*L['me']).sum()/lme
        sme=S['me'].sum(); nsw=(S.set_index('permno')['me']/sme).to_dict()
        rs=(S['ret_fwd']*S['me']).sum()/sme
        tl=sum(abs(nlw.get(p,0)-plw.get(p,0)) for p in set(nlw)|set(plw))/2
        ts=sum(abs(nsw.get(p,0)-psw.get(p,0)) for p in set(nsw)|set(psw))/2
        monthly.append({'date':date,'ret':rl-rs-fee*(tl+ts)})
        plw,psw=nlw,nsw
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def compute_sharpe(r):
    r=pd.Series(r).dropna()
    if len(r)<6 or r.std()==0: return 0
    return r.mean()/r.std()*np.sqrt(12)

def run_pipeline(pi_filter, xgb_seeds):
    pi_df = pd.DataFrame({'date':dates_all,'pi_filter':pi_filter})
    tr = train.drop(columns=['pi_filter'],errors='ignore').merge(pi_df,on='date',how='left')
    tr['pi_filter'] = tr['pi_filter'].ffill().fillna(0.5)
    te = test.drop(columns=['pi_filter'],errors='ignore').merge(pi_df,on='date',how='left')
    te['pi_filter'] = te['pi_filter'].ffill().fillna(0.5)
    X_tr = tr[REDUCED].values.astype(float)
    X_te = te[REDUCED].values.astype(float)
    y_tr = tr['ret_fwd'].values.astype(float)
    preds = np.zeros(len(X_te))
    for xs in xgb_seeds:
        xgb = XGBRegressor(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,
                           learning_rate=LEARNING_RATE,subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE,tree_method='hist',
                           random_state=xs,verbosity=0)
        xgb.fit(X_tr,y_tr)
        preds += xgb.predict(X_te)
    preds /= len(xgb_seeds)
    te['score'] = preds
    return long_short_port(te,'score'), te

t0 = time.time()

# Pre-compute 20 HMM pi_filters
HMM_SEEDS = [42,2201,1337,314,789,999,123,456,7777,55,
             101,202,303,404,505,606,707,808,909,1010]
XGB_SEEDS = list(range(1,51))

print("[ 1/5 ] Heavy Ensemble (20 HMM + 50 XGB) ...")
print("  Computing 20 HMM pi_filters ...", end=' ', flush=True)
all_pis = {}
for s in HMM_SEEDS:
    all_pis[s] = fit_hmm(seed=s)
pi_20 = np.mean([all_pis[s] for s in HMM_SEEDS], axis=0)
print("done")

print("  Training 50 XGB seeds ...", end=' ', flush=True)
r_heavy, te_heavy = run_pipeline(pi_20, XGB_SEEDS)
print("done")

ann_ret=(1+r_heavy).prod()**(12/len(r_heavy))-1
ann_vol=r_heavy.std()*np.sqrt(12)
sharpe=compute_sharpe(r_heavy)
cum=(1+r_heavy).cumprod()
mdd=((cum-cum.cummax())/cum.cummax()).min()
print(f"\n  Sharpe: {sharpe:.3f}  Return: {ann_ret:.1%}  Vol: {ann_vol:.1%}  MDD: {mdd:.1%}")

# Step 2
print("\n[ 2/5 ] Sub-period consistency ...")
all_pos = True
for pname,start,end in [('2011-2015','2011-01-01','2016-01-01'),
                         ('2016-2020','2016-01-01','2021-01-01'),
                         ('2021-2025','2021-01-01','2026-01-01')]:
    r_sub=r_heavy[(r_heavy.index>=start)&(r_heavy.index<end)]
    if len(r_sub)>6:
        sh=compute_sharpe(r_sub)
        ar=(1+r_sub).prod()**(12/len(r_sub))-1
        if sh<=0: all_pos=False
        print(f"  {pname}: Sharpe={sh:.3f}  Return={ar:.1%}")
print(f"  All positive: {'YES' if all_pos else 'NO'}")

# Step 3
print("\n[ 3/5 ] Stability (20 experiments) ...")
np.random.seed(99999)
stab_sharpes=[]
for exp in range(20):
    hs=list(np.random.choice(HMM_SEEDS,size=3,replace=False))
    xs=XGB_SEEDS[exp]
    pi_exp=np.mean([all_pis[s] for s in hs],axis=0)
    r_exp,_=run_pipeline(pi_exp,[xs])
    stab_sharpes.append(compute_sharpe(r_exp))
    if (exp+1)%5==0: print(f"    {exp+1}/20 done")
print(f"  Mean={np.mean(stab_sharpes):.3f}  Std={np.std(stab_sharpes):.3f}  Min={np.min(stab_sharpes):.3f}  Max={np.max(stab_sharpes):.3f}")
print(f"  All>0: {all(s>0 for s in stab_sharpes)}")

# Step 4
print("\n[ 4/5 ] Leave-one-year-out ...")
years=sorted(r_heavy.index.year.unique())
all_loo=True
for yr in years:
    r_loo=r_heavy[r_heavy.index.year!=yr]
    sh_loo=compute_sharpe(r_loo)
    if sh_loo<=0: all_loo=False
    print(f"  Drop {yr}: Sharpe={sh_loo:.3f}")
print(f"  All positive: {'YES' if all_loo else 'NO'}")

# Step 5
print("\n[ 5/5 ] Factor model alphas ...")
if ff is not None:
    r_df=r_heavy.to_frame('ret')
    r_df.index=r_df.index+pd.offsets.MonthEnd(0)
    merged=r_df.join(ff[['Mkt-RF','SMB','HML','RMW','CMA','RF','UMD']],how='inner')
    y=merged['ret'].values
    for mname,cols in [('CAPM',['Mkt-RF']),('FF3',['Mkt-RF','SMB','HML']),
                       ('Carhart4',['Mkt-RF','SMB','HML','UMD']),
                       ('FF5',['Mkt-RF','SMB','HML','RMW','CMA']),
                       ('FF5+Mom',['Mkt-RF','SMB','HML','RMW','CMA','UMD'])]:
        X=sm.add_constant(merged[cols].values)
        res=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':6})
        a=res.params[0]*12; t=res.tvalues[0]; p=res.pvalues[0]
        stars='***' if p<0.01 else '**' if p<0.05 else '*' if p<0.10 else ''
        print(f"  {mname:<10s}: alpha={a:>6.1%}  t={t:>5.2f}{stars}")

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY: {FEATS}")
print(f"{'='*70}")
print(f"  Heavy Ensemble Sharpe:  {sharpe:.3f}")
print(f"  Sub-periods positive:   {'YES' if all_pos else 'NO'}")
print(f"  Stability:              {np.mean(stab_sharpes):.3f} +/- {np.std(stab_sharpes):.3f}")
print(f"  Leave-one-year:         {'All positive' if all_loo else 'HAS NEGATIVE'}")
print(f"  Total time:             {(time.time()-t0)/60:.1f} minutes")
print(f"{'='*70}")
