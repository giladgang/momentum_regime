import importlib.util, sys, numpy as np, pandas as pd
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT
base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
eqw=test.dropna(subset=['ret_fwd']).groupby('date')['ret_fwd'].mean()
excess=(spm-mkt).dropna(); breadth=(mkt-eqw)
disp=pd.read_csv(f'{R}/results/2026-06-02-sp500-dispersion-thru2025.csv',parse_dates=['date']).set_index('date')['p90_p10']
df=pd.concat([excess.rename('ex'),breadth.rename('breadth'),disp.rename('disp')],axis=1).dropna()
df['disp_lag']=df['disp'].shift(1); df['breadth_lag']=df['breadth'].shift(1); df=df.dropna()
df['year']=df.index.year

# Diagnose shape early
print(f"df.shape after inner join: {df.shape}")
print(f"excess index head: {excess.index[:3].tolist()}")
print(f"disp index head: {disp.index[:3].tolist()}")

if df.shape[0] < 100:
    print("WARNING: df too short — attempting Period alignment")
    excess2 = excess.copy(); excess2.index = excess2.index.to_period('M')
    breadth2 = breadth.copy(); breadth2.index = breadth2.index.to_period('M')
    disp2 = disp.copy(); disp2.index = disp2.index.to_period('M')
    df = pd.concat([excess2.rename('ex'), breadth2.rename('breadth'), disp2.rename('disp')], axis=1).dropna()
    df['disp_lag'] = df['disp'].shift(1); df['breadth_lag'] = df['breadth'].shift(1); df = df.dropna()
    df['year'] = df.index.year
    print(f"df.shape after Period alignment: {df.shape}")

assert df['breadth_lag'].notna().all() and (df['year']>=2024).sum()>=20, "build problem"

# 1. rolling 36m annualized excess; is post-2024 a statistical outlier?
roll=df['ex'].rolling(36).apply(lambda x:(1+x).prod()**(12/len(x))-1)
pre=df.loc[df['year']<=2023,'ex']; postv=df.loc[df['year']>=2024,'ex']
post_mean=postv.mean()*12
z=(postv.mean()-pre.mean())/(pre.std()/np.sqrt(len(postv)))
print(f"post-2024 mean excess {post_mean:+.1%}/yr ; pre mean {pre.mean()*12:+.1%}/yr ; z(post vs pre) {z:+.2f}")

# 2. regime conditioning: mean annualized excess by lagged-tercile
for v in ['disp_lag','breadth_lag']:
    try:
        t=pd.qcut(df[v],3,labels=['low','mid','high'])
    except ValueError:
        t=pd.qcut(df[v],3,labels=['low','mid','high'],duplicates='drop')
    g=(df.groupby(t,observed=True)['ex'].mean()*12)
    print(f"\nmean ann excess by {v} tercile:\n{g.round(3).to_string()}")

# 3. block-bootstrap CI on post-2024 monthly excess
post=postv.values; n=len(post); B=5000; bl=6; rng=np.random.default_rng(7)
def bmean(x):
    out=[]
    for _ in range(B):
        idx=[]
        while len(idx)<n:
            s=rng.integers(0,n); idx+=list(range(s,min(s+bl,n)))
        out.append(np.mean(np.take(x,idx[:n])))
    return np.percentile(out,[2.5,50,97.5])*12
lo,md,hi=bmean(post)
print(f"\npost-2024 excess {md:+.1%}/yr  95% block-bootstrap CI [{lo:+.1%}, {hi:+.1%}]  (n={n} months, block={bl})")
print("NOTE: post-2024 spans ~1-2 independent episodes (2024); CANNOT establish a durable premium in-sample.")
pd.DataFrame({'roll36_excess':roll}).dropna().to_csv(f'{R}/results/2026-06-02-durability.csv')
print("saved results/2026-06-02-durability.csv")
