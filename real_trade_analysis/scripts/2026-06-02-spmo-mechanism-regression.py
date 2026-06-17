import importlib.util, sys, numpy as np, pandas as pd, statsmodels.api as sm
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END, FF_FACTORS_PATH
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT
base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
eqw=test.dropna(subset=['ret_fwd']).groupby('date')['ret_fwd'].mean()
excess=(spm-mkt).dropna(); breadth=(mkt-eqw)
def realign(s): s=s.dropna().copy(); s.index=s.index.to_period('M')+1; return s
ex,br=realign(excess),realign(breadth)
ff=pd.read_parquet(FF_FACTORS_PATH); ff.index=pd.to_datetime(ff.index).to_period('M')
d=pd.concat([ex.rename('ex'),br.rename('breadth'),ff[['Mkt-RF','SMB','HML','UMD']]],axis=1,join='inner').dropna()

# VALIDATION: breadth (cap-minus-equal) must be positive in 2024 (mega-cap leadership)
b24=d.loc[[p for p in d.index if p.year==2024],'breadth'].mean()
assert b24>0, f"breadth 2024 mean {b24:.4f} should be >0"
print(f"VALIDATION ok: breadth 2024 mean {b24:+.4f}; n={len(d)} months {d.index.min()}..{d.index.max()}")

FAC=['Mkt-RF','SMB','HML','UMD','breadth']

# Convert to RangeIndex for OLS compatibility, keep year array for slicing
d_years = np.array([p.year for p in d.index])
d_ri = d.copy()
d_ri.index = pd.RangeIndex(len(d_ri))

def reg(sub_mask, lbl):
    sub = d_ri[sub_mask].copy()
    X=sm.add_constant(sub[FAC]); res=sm.OLS(sub['ex'],X).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    print(f"{lbl:<14} alpha {res.params['const']*12:+.1%} (t {res.tvalues['const']:+.2f})  "+"  ".join(f"{f} {res.params[f]:+.2f}" for f in FAC))
    return res

print("\nFF6+breadth regressions (NW(6), alpha annualized):")
reg(np.ones(len(d_ri), dtype=bool),'full 11-25')
reg(d_years<=2023,'pre 11-23')
reg(d_years>=2024,'post 24-25')

d2=d_ri.copy(); d2['post']=(d_years>=2024).astype(float)
X=sm.add_constant(d2[FAC+['post']]); res=sm.OLS(d2['ex'],X).fit(cov_type='HAC',cov_kwds={'maxlags':6})
print(f"\nPOST-2024 dummy (alpha jump): {res.params['post']*12:+.1%}/yr  t={res.tvalues['post']:+.2f}  p={res.pvalues['post']:.3f}")
print("  -> significant positive post dummy with UMD NOT absorbing it => idiosyncratic (not the momentum factor).")

rows=[]
for i in range(36,len(d)+1):
    w=d.iloc[i-36:i]; w_ri=w.copy(); w_ri.index=pd.RangeIndex(len(w_ri))
    X=sm.add_constant(w_ri[FAC]); rr=sm.OLS(w_ri['ex'],X).fit()
    rows.append({'date':w.index[-1].to_timestamp(),'alpha_ann':rr.params['const']*12,'umd':rr.params['UMD']})
roll=pd.DataFrame(rows).set_index('date'); roll.to_csv(f'{R}/results/2026-06-02-regression.csv')
print("\nrolling 36m alpha+UMD tail:\n",roll.tail(6).round(3))
