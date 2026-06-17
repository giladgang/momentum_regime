"""
2026-06-06-hmm-nfci-feature-search.py  (EXPLORATORY)

Does the Chicago Fed NFCI help the regime HMM? Adds NFCI_z to the production 9-feature
candidate pool (-> 10 features) and re-runs the Pass-1 HMM quality screen (exhaustive
combos, DD always included, 1-4 features, 1 seed each), EXACTLY reusing the production
harness's fit_hmm + quality gates + crisis/calm validation periods from
scripts/hmm_feature_selection.py. Non-destructive: writes to a separate output file and
does NOT modify config.py, panel.parquet, or the production pass1 CSV.

NFCI: weekly FRED series (data/nfci_fred.csv), collapsed to end-of-month (point-in-time),
merged on year_month, z-scored on the 1990-2010 train period (same convention as the
other HMM features). The production combo DD+CS+DISP+REL_N is inside this same screen,
so the NFCI combos can be compared apples-to-apples.

Quick screen first (Pass 1). If NFCI-containing combos clear the gates, Pass 2/3
(downstream Sharpe) is the natural follow-up.
"""
import os, sys, time, importlib.util
import numpy as np, pandas as pd
ROOT='/Users/giladgang/momentum_regime'; os.chdir(ROOT); sys.path.insert(0, ROOT)

# import the production harness (import-safe: guarded by __main__) for fit_hmm + config
spec=importlib.util.spec_from_file_location('hfs', f'{ROOT}/scripts/hmm_feature_selection.py')
hfs=importlib.util.module_from_spec(spec); spec.loader.exec_module(hfs)
from config import TRAIN_END
from itertools import combinations

POOL=['DD_z','VOL_z','DISP_z','REL_N_z','CS_z','LVIX_z','ADR_z','SKEW_z','TERM_z','NFCI_z']
REQUIRED='DD_z'; FEATURE_COUNTS=[1,2,3,4]
GATES=hfs.QUALITY_GATES; VPER=hfs.VALIDATION_PERIODS; SEED=hfs.PASS1_HMM_SEEDS[0]
OUT=f'{ROOT}/results/cv/hmm_nfci_feature_search_pass1.csv'

# ---- data: replicate harness macro-load, then inject NFCI ----
print("Loading panel + extra features ...")
panel=pd.read_parquet(f'{ROOT}/data/panel.parquet')
panel['date']=pd.to_datetime(panel['date']); panel['year_month']=panel['date'].dt.to_period('M')
train_mask=panel['date']<TRAIN_END
extra=pd.read_pickle(f'{ROOT}/data/extra_hmm_features.pkl')
panel=panel.merge(extra, on='year_month', how='left')
for col in ['ADR','SKEW','TERM']:
    if col in panel.columns and f'{col}_z' not in panel.columns:
        v=panel[col].astype(float); mu,sd=v[train_mask].dropna().mean(),v[train_mask].dropna().std()
        if sd>0: panel[f'{col}_z']=(v-mu)/sd

# NFCI: weekly -> end-of-month -> merge -> z-score on train
nf=pd.read_csv(f'{ROOT}/data/nfci_fred.csv')
nf.columns=['date','NFCI']; nf['date']=pd.to_datetime(nf['date']); nf=nf.dropna(subset=['NFCI'])
nf['year_month']=nf['date'].dt.to_period('M')
nfm=nf.sort_values('date').groupby('year_month')['NFCI'].last().reset_index()
panel=panel.merge(nfm, on='year_month', how='left')
v=panel['NFCI'].astype(float); mu,sd=v[train_mask].dropna().mean(),v[train_mask].dropna().std()
panel['NFCI_z']=(v-mu)/sd
print(f"  NFCI merged: {panel['NFCI'].notna().sum()} months non-null, "
      f"train z mean {panel.loc[train_mask,'NFCI_z'].mean():.2f} sd {panel.loc[train_mask,'NFCI_z'].std():.2f}")

available=[f for f in POOL if f in panel.columns and panel[f].notna().sum()>200]
print(f"  Features available ({len(available)}): {available}")
assert 'NFCI_z' in available, "NFCI_z not available!"

# ---- all combos: DD always included, 1-4 features ----
others=[f for f in available if f!=REQUIRED]
combos=[]
for n in FEATURE_COUNTS:
    if n==1: combos.append((REQUIRED,))
    else:
        for c in combinations(others, n-1): combos.append((REQUIRED,)+c)
nfci_combos=[c for c in combos if 'NFCI_z' in c]
print(f"  Total combos: {len(combos)}  ({len(nfci_combos)} contain NFCI)\n")

# ---- Pass-1 screen (mirrors hfs.run_pass1 exactly) ----
results=[]; t0=time.time()
for i,combo in enumerate(combos):
    feat=list(combo); name="+".join(f.replace("_z","") for f in feat)
    sub=panel[['date']+feat].dropna().reset_index(drop=True)
    sub=sub[sub['date']>='1990-01-01'].sort_values('date').reset_index(drop=True)
    if len(sub)<200: continue
    tr=sub[sub['date']<TRAIN_END]; te=sub[sub['date']>=TRAIN_END]
    if len(tr)<150 or len(te)<50: continue
    Z_tr=tr[feat].values.astype(float); Z_full=sub[feat].values.astype(float)
    try: res=hfs.fit_hmm(Z_tr, Z_full, tr['date'].values, seed=SEED)
    except Exception: continue
    pi=res['pi_filter']; ad=sub['date'].values; pr={}; all_pass=True
    for pn,ps,pe,pt,th in VPER:
        m=(ad>=np.datetime64(ps))&(ad<=np.datetime64(pe))
        if m.sum()==0: pr[pn]=np.nan; continue
        mp=pi[m].mean(); pr[pn]=mp
        if pt=='crisis' and mp<th: all_pass=False
        if pt=='calm' and mp>th: all_pass=False
    req_sig=min(GATES['n_sig'], len(feat))
    mcmc_ok=(res['min_ess']>=GATES['min_ess'] and res['n_sig']>=req_sig)
    passed=mcmc_ok and all_pass
    results.append({'name':name,'combo':feat,'n_feat':len(feat),'has_nfci':'NFCI_z' in feat,
                    'passed':passed,'min_ess':res['min_ess'],'n_sig':res['n_sig'],'ll':res['ll'],
                    **{f'pi_{k}':v for k,v in pr.items()}})
    if (i+1)%10==0 or passed:
        npass=sum(1 for r in results if r['passed'])
        ps=" ".join(f"{k}={v:.2f}" for k,v in pr.items() if not (isinstance(v,float) and np.isnan(v)))
        print(f"  [{i+1:>3d}/{len(combos)}] {name:<32s} ESS={res['min_ess']:>6.0f} sig={res['n_sig']} {ps} "
              f"{'PASS' if passed else 'fail'} [{time.time()-t0:.0f}s, {npass} passed]")

df=pd.DataFrame(results); df.to_csv(OUT, index=False)
print(f"\nsaved {OUT}")

# ---- focused NFCI summary ----
print("\n"+"="*78)
print("  NFCI VERDICT")
print("="*78)
passed=df[df['passed']]
print(f"  combos tested {len(df)} | passed {len(passed)} | of which contain NFCI: {passed['has_nfci'].sum()}")
ddn=df[df['name']=='DD+NFCI']
if len(ddn): r=ddn.iloc[0]; print(f"  DD+NFCI alone: ESS={r['min_ess']:.0f} n_sig={r['n_sig']} passed={r['passed']}")
prod=df[df['combo'].apply(lambda c: set(c)=={'DD_z','CS_z','DISP_z','REL_N_z'})]
if len(prod): r=prod.iloc[0]; print(f"  PRODUCTION DD+CS+DISP+REL_N: ESS={r['min_ess']:.0f} n_sig={r['n_sig']} ll={r['ll']:.0f} passed={r['passed']}")
print("\n  top passed combos by log-likelihood (NFCI marked *):")
for _,r in passed.sort_values('ll',ascending=False).head(12).iterrows():
    print(f"    {'*' if r['has_nfci'] else ' '} {r['name']:<34s} ESS={r['min_ess']:>6.0f} sig={r['n_sig']} ll={r['ll']:.0f}")
print("="*78)
