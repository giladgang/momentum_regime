"""
2026-06-01-cluster4-longonly-zcurve.py  (EXPLORATORY)
Z-curve (thesis style) of the LONG-ONLY XGB (train Top-1000, trade S&P 500) long
picks during the 21 cluster-4 (deep-crisis) months. For each cluster-4 month:
standardize each momentum horizon across the Top-500 universe, take the long
basket (score >= NYSE P90), average its z at each horizon. Thin = per-month, bold
= centroid. Compares to the production full-universe L/S cluster-4 long centroid.
"""
import importlib.util, os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from xgboost import XGBRegressor
_ROOT='/Users/giladgang/momentum_regime'; sys.path.insert(0,_ROOT)
_s=importlib.util.spec_from_file_location("exp",f"{_ROOT}/real_trade_analysis/scripts/2026-05-30-xgb-russell1000.py")
exp=importlib.util.module_from_spec(_s); _s.loader.exec_module(exp)
from config import (TRAIN_END,N_ESTIMATORS,MAX_DEPTH,LEARNING_RATE,SUBSAMPLE,COLSAMPLE,XGB_N_JOBS)
H=list(range(1,13)); MC=[f'mom_{h}' for h in H]; SEEDS=exp.SEEDS; R=f'{_ROOT}/real_trade_analysis'

df=exp.apply_size_screen(exp.build_features(),500)
tr=df[df['date']<TRAIN_END]; te=df[df['date']>=TRAIN_END].copy()
pred=np.zeros(len(te))
for i,seed in enumerate(SEEDS,1):
    m=XGBRegressor(n_estimators=N_ESTIMATORS,max_depth=MAX_DEPTH,learning_rate=LEARNING_RATE,subsample=SUBSAMPLE,
                   colsample_bytree=COLSAMPLE,tree_method='hist',random_state=seed,verbosity=0,n_jobs=XGB_N_JOBS)
    m.fit(tr[exp.FEATURES].values.astype(float),tr['ret_fwd'].values.astype(float)); pred+=m.predict(te[exp.FEATURES].values.astype(float))
    if i%10==0: print(f"  seed {i}/{len(SEEDS)}",flush=True)
te['score']=pred/len(SEEDS)

lab=pd.read_csv(f'{_ROOT}/results/thesis/zscore_l2_k4_labels.csv',parse_dates=['date'])
c4_dates=set(lab[lab['cluster']==3]['date'])  # thesis Cluster 4 = code 3
curves=[]
for date,grp in te.groupby('date'):
    if date not in c4_dates: continue
    g=grp.dropna(subset=MC+['score']).copy()
    nyse=g[g['exchcd']==1]['score'].dropna()
    if len(nyse)<10: continue
    hi=nyse.quantile(0.90); longs=g[g['score']>=hi]
    if len(longs)<3: continue
    zc=[((longs[c]-g[c].mean())/g[c].std()).mean() for c in MC]  # avg standardized momentum of longs
    curves.append(zc)
curves=np.array(curves); cent=curves.mean(axis=0)
print(f"\nLong-only XGB z-curve, cluster-4 months (n={len(curves)}):")
print("centroid:",np.round(cent,2))

fig,ax=plt.subplots(figsize=(9,6.5))
for row in curves: ax.plot(H,row,color='steelblue',lw=0.8,alpha=0.4)
ax.plot(H,cent,color='crimson',lw=3,marker='o',label=f'Long-only XGB long-pick centroid (mean {cent.mean():+.2f}σ)')
# overlay production full-universe L/S cluster-4 long centroid for comparison
prod=pd.read_csv(f'{_ROOT}/results/thesis/zscore_l2_k4_centroids.csv')
pc=prod[prod['cluster']==3][MC].values.flatten()
ax.plot(H,pc,color='gray',lw=2,ls='--',marker='x',label=f'Production full-universe L/S centroid (mean {pc.mean():+.2f}σ)')
ax.axhline(0,color='black',lw=0.7,ls=':'); ax.set_xticks(H)
ax.set_xlabel('Momentum horizon (months)'); ax.set_ylabel('Cross-sectional z-score (σ from mean)')
ax.set_title(f'Cluster 4 (deep crisis) months: LONG-ONLY XGB long-pick z-curves (n={len(curves)})',fontsize=12)
ax.legend(loc='best',fontsize=9.5); ax.grid(True,alpha=0.25); plt.tight_layout()
for ext in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-01-cluster4-longonly-zcurve.{ext}',dpi=150,bbox_inches='tight')
plt.close(fig); print("saved cluster4-longonly-zcurve.pdf + .png")
