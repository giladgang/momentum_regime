"""
2026-06-06-hmm-nfci-pass2.py  (EXPLORATORY)

Pass-2 (downstream L/S Sharpe) on a focused shortlist, to test whether NFCI actually
IMPROVES the strategy vs the production HMM feature set (not just in-sample fit, which
Pass-1 already showed it does). Reuses the production harness's fit_hmm / long_short_port /
compute_sharpe and the exact Pass-2 recipe (avg pi over 5 HMM seeds -> merge into stock
panel -> XGB ensemble over 10 seeds on MOM_FEATURES+pi_filter -> L/S portfolio Sharpe;
LR shown for reference). Non-destructive: separate output file, no config/panel/prod-CSV edits.

Shortlist:
  DD+CS+DISP+REL_N   <- PRODUCTION baseline
  DD+DISP+REL_N+NFCI <- swap CS -> NFCI (the key test)
  DD+REL_N+CS+NFCI   <- best 4-feature combo by Pass-1 log-likelihood
  DD+CS+NFCI         <- best 3-feature
  DD+NFCI            <- best 2-feature
"""
import os, sys, time, pickle, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT='/Users/giladgang/momentum_regime'; os.chdir(ROOT); sys.path.insert(0, ROOT)
spec=importlib.util.spec_from_file_location('hfs', f'{ROOT}/scripts/hmm_feature_selection.py')
hfs=importlib.util.module_from_spec(spec); spec.loader.exec_module(hfs)
from config import (TRAIN_END, MOM_FEATURES, N_ESTIMATORS, MAX_DEPTH,
                    LEARNING_RATE, SUBSAMPLE, COLSAMPLE)
from xgboost import XGBRegressor
REDUCED = MOM_FEATURES + ['pi_filter']
HMM_SEEDS = hfs.PASS2_HMM_SEEDS; XGB_SEEDS = hfs.PASS2_XGB_SEEDS
OUT=f'{ROOT}/results/cv/hmm_nfci_pass2.csv'

# ---- panel with NFCI (same build as pass-1 experiment) ----
panel=pd.read_parquet(f'{ROOT}/data/panel.parquet')
panel['date']=pd.to_datetime(panel['date']); panel['year_month']=panel['date'].dt.to_period('M')
tm=panel['date']<TRAIN_END
extra=pd.read_pickle(f'{ROOT}/data/extra_hmm_features.pkl'); panel=panel.merge(extra,on='year_month',how='left')
for col in ['ADR','SKEW','TERM']:
    if col in panel.columns and f'{col}_z' not in panel.columns:
        v=panel[col].astype(float); mu,sd=v[tm].dropna().mean(),v[tm].dropna().std()
        if sd>0: panel[f'{col}_z']=(v-mu)/sd
nf=pd.read_csv(f'{ROOT}/data/nfci_fred.csv'); nf.columns=['date','NFCI']
nf['date']=pd.to_datetime(nf['date']); nf=nf.dropna(subset=['NFCI']); nf['year_month']=nf['date'].dt.to_period('M')
nfm=nf.sort_values('date').groupby('year_month')['NFCI'].last().reset_index()
panel=panel.merge(nfm,on='year_month',how='left')
v=panel['NFCI'].astype(float); mu,sd=v[tm].dropna().mean(),v[tm].dropna().std(); panel['NFCI_z']=(v-mu)/sd

# ---- stock data ----
print("loading cs_artefacts ...", flush=True)
art=pickle.load(open(f'{ROOT}/artefacts/cs_artefacts_data.pkl','rb'))
train_stocks=art['train'].copy(); test_stocks=art['test'].copy()

SHORTLIST=[(['DD_z','CS_z','DISP_z','REL_N_z'],'PRODUCTION'),
           (['DD_z','DISP_z','REL_N_z','NFCI_z'],'swap CS->NFCI'),
           (['DD_z','REL_N_z','CS_z','NFCI_z'],'best 4-feat'),
           (['DD_z','CS_z','NFCI_z'],'best 3-feat'),
           (['DD_z','NFCI_z'],'best 2-feat')]

def eval_combo(feat):
    sub=panel[['date']+feat].dropna().reset_index(drop=True)
    sub=sub[sub['date']>='1990-01-01'].sort_values('date').reset_index(drop=True)
    tr=sub[sub['date']<TRAIN_END]; Z_tr=tr[feat].values.astype(float); Z_full=sub[feat].values.astype(float)
    pi=np.zeros(len(Z_full))
    for s in HMM_SEEDS: pi+=hfs.fit_hmm(Z_tr,Z_full,tr['date'].values,seed=s)['pi_filter']
    pi/=len(HMM_SEEDS)
    pidf=pd.DataFrame({'date':sub['date'].values,'pi_filter':pi})
    trs=train_stocks.drop(columns=['pi_filter'],errors='ignore').merge(pidf,on='date',how='left')
    tes=test_stocks.drop(columns=['pi_filter'],errors='ignore').merge(pidf,on='date',how='left')
    trs['pi_filter']=trs['pi_filter'].ffill().fillna(0.5); tes['pi_filter']=tes['pi_filter'].ffill().fillna(0.5)
    Xtr=trs[REDUCED].values.astype(float); Xte=tes[REDUCED].values.astype(float); ytr=trs['ret_fwd'].values.astype(float)
    pred=np.zeros(len(Xte))
    for s in XGB_SEEDS:
        m=XGBRegressor(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,learning_rate=LEARNING_RATE,
                       subsample=SUBSAMPLE,colsample_bytree=COLSAMPLE,tree_method='hist',random_state=s,verbosity=0)
        m.fit(Xtr,ytr); pred+=m.predict(Xte)
    pred/=len(XGB_SEEDS); tes=tes.copy(); tes['score_xgb']=pred
    return hfs.compute_sharpe(hfs.long_short_port(tes,'score_xgb'))

rows=[]; t0=time.time()
print(f"Pass-2 on {len(SHORTLIST)} combos ({len(HMM_SEEDS)} HMM x {len(XGB_SEEDS)} XGB seeds)\n",flush=True)
for feat,label in SHORTLIST:
    t=time.time(); sh=eval_combo(feat); name="+".join(f.replace("_z","") for f in feat)
    rows.append({'name':name,'label':label,'n_feat':len(feat),'has_nfci':'NFCI_z' in feat,'sharpe_xgb':sh})
    print(f"  {name:<26s} [{label:<14s}]  XGB L/S Sharpe = {sh:.3f}   [{time.time()-t:.0f}s, tot {time.time()-t0:.0f}s]",flush=True)
res=pd.DataFrame(rows).sort_values('sharpe_xgb',ascending=False); res.to_csv(OUT,index=False)
print(f"\nsaved {OUT}")
print("\n"+"="*64+"\n  PASS-2 VERDICT (downstream L/S Sharpe)\n"+"="*64)
prod=res[res['label']=='PRODUCTION']['sharpe_xgb'].iloc[0]
for _,r in res.iterrows():
    d=r['sharpe_xgb']-prod
    print(f"  {'*' if r['has_nfci'] else ' '} {r['name']:<26s} {r['sharpe_xgb']:.3f}  ({d:+.3f} vs prod){'  <- PRODUCTION' if r['label']=='PRODUCTION' else ''}")
print("="*64)
