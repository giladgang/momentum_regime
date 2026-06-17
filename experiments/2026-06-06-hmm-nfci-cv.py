"""
2026-06-06-hmm-nfci-cv.py  (EXPLORATORY)

The METHODOLOGICALLY CLEAN test of NFCI: train-only cross-validation. Selecting features
by 2011-2024 test Sharpe (Pass-2) is test-set tuning. This instead uses the production CV
harness's expanding-window folds (validation windows 1997-2007, entirely inside the
1990-2010 training era -> never touches the 2011-2024 test set) and its exact functions
(zscore_train / fit_pi_filter / eval_xgb_ensemble). Reuses scripts/hmm_cv.py verbatim;
only adds NFCI to the panel. Non-destructive: separate output file.

Per (combo, fold, hmm_seed): fit HMM on the fold's train partition, forward-filter pi,
train XGB ensemble on train months, score L/S Sharpe on the fold's validation window.
Reported = mean validation Sharpe across folds and HMM seeds. Compares production
DD+CS+DISP+REL_N against NFCI variants. 3 HMM x 5 XGB seeds (the CV defaults).
"""
import os, sys, time, importlib.util, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT='/Users/giladgang/momentum_regime'; os.chdir(ROOT); sys.path.insert(0, ROOT)
spec=importlib.util.spec_from_file_location('hcv', f'{ROOT}/scripts/hmm_cv.py')
hcv=importlib.util.module_from_spec(spec); spec.loader.exec_module(hcv)
from config import TRADING_FEE

HMM_SEEDS=[0,1,2]; XGB_SEEDS=list(range(5)); N_ITER=2000; N_BURN=500; FEE=TRADING_FEE
OUT=f'{ROOT}/results/cv/hmm_nfci_cv.csv'
COMBOS=[(('DD','CS','DISP','REL_N'),'PRODUCTION'),
        (('DD','DISP','REL_N','NFCI'),'swap CS->NFCI'),
        (('DD','REL_N','CS','NFCI'),'best-4 (Pass2 test-winner)'),
        (('DD','CS','NFCI'),'DD+CS+NFCI'),
        (('DD','NFCI'),'DD+NFCI')]

# ---- HMM panel (+ NFCI) ----
panel=hcv.load_hmm_panel()                      # date + CANDIDATES
panel['year_month']=panel['date'].dt.to_period('M')
nf=pd.read_csv(f'{ROOT}/data/nfci_fred.csv'); nf.columns=['date','NFCI']
nf['date']=pd.to_datetime(nf['date']); nf=nf.dropna(subset=['NFCI']); nf['year_month']=nf['date'].dt.to_period('M')
nfm=nf.sort_values('date').groupby('year_month')['NFCI'].last().reset_index()
panel=panel.merge(nfm,on='year_month',how='left').drop(columns=['year_month'])
print(f"panel rows {len(panel)} | NFCI non-null {panel['NFCI'].notna().sum()} | "
      f"NFCI span {panel.loc[panel.NFCI.notna(),'date'].min().date()}..{panel.loc[panel.NFCI.notna(),'date'].max().date()}", flush=True)

# ---- stock panel (built fresh, pre-2011 incl.) ----
print("building stock panel ...", flush=True)
t0=time.time(); stocks=hcv.load_stock_panel(); print(f"  stocks {len(stocks):,} rows in {time.time()-t0:.0f}s", flush=True)

rows=[]; tA=time.time()
for combo,label in COMBOS:
    fold_means=[]
    for (fid,tre,vs,ve) in hcv.FOLDS:
        seed_sh=[]
        for s in HMM_SEEDS:
            pz=hcv.zscore_train(panel, list(combo), tre)
            pidf=hcv.fit_pi_filter(pz, [f+'_z' for f in combo], tre, s, N_ITER, N_BURN)
            sh,nmo=hcv.eval_xgb_ensemble(stocks, pidf, tre, vs, ve, XGB_SEEDS, FEE)
            seed_sh.append(sh)
        fm=np.nanmean(seed_sh); fold_means.append(fm)
        print(f"  {('+'.join(combo)):<22s} fold{fid}({vs[:4]}-{ve[:4]}) CV Sharpe={fm:+.3f}  [{time.time()-tA:.0f}s]", flush=True)
    mean_cv=np.nanmean(fold_means); std_cv=np.nanstd(fold_means)
    rows.append({'name':'+'.join(combo),'label':label,'n_feat':len(combo),'has_nfci':'NFCI' in combo,
                 'cv_sharpe_mean':mean_cv,'cv_sharpe_std':std_cv,
                 **{f'fold{hcv.FOLDS[i][0]}':fold_means[i] for i in range(len(fold_means))}})
    print(f"  => {('+'.join(combo)):<22s} mean CV Sharpe {mean_cv:+.3f} (fold std {std_cv:.3f})\n", flush=True)

res=pd.DataFrame(rows).sort_values('cv_sharpe_mean',ascending=False); res.to_csv(OUT,index=False)
print(f"saved {OUT}")
print("\n"+"="*70+"\n  CLEAN CV VERDICT  (train-only folds 1997-2007; no test-set tuning)\n"+"="*70)
prod=res[res['label']=='PRODUCTION']['cv_sharpe_mean'].iloc[0]
for _,r in res.iterrows():
    print(f"  {'*' if r['has_nfci'] else ' '} {r['name']:<22s} mean CV Sharpe {r['cv_sharpe_mean']:+.3f} "
          f"(fold std {r['cv_sharpe_std']:.3f})  {r['cv_sharpe_mean']-prod:+.3f} vs prod"
          f"{'  <- PRODUCTION' if r['label']=='PRODUCTION' else ''}")
print("="*70)
