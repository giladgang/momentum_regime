"""
2026-06-01-spmo-top-jan2024.py  (EXPLORATORY)
Replicated-SPMO top holdings averaged over Jan-2024-onward. Runs the SPMO replica
over the full test period (so the semi-annual rebalance/drift state is correct
entering 2024), then averages weights only over months >= 2024-01. No XGB needed.
"""
import importlib.util, os, sys, numpy as np, pandas as pd
_ROOT='/Users/giladgang/momentum_regime'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{_ROOT}/real_trade_analysis/scripts/{f}')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
ha=_load('ha','2026-05-31-holdings-analysis.py')
from config import TRAIN_END
R=f'{_ROOT}/real_trade_analysis'

base=sp.add_spmo_inputs(exp.build_features())
test=exp.apply_size_screen(base,500); test=test[test['date']>=TRAIN_END].copy()
test['spmo_signal']=test['mom_value']/test['sigma_m']
spmo_p=ha.spmo_path(list(test.groupby('date')))

CUT=pd.Timestamp('2024-01-01')
months=[d for d in sorted(spmo_p) if d>=CUT and spmo_p[d]]
print(f"averaging over {len(months)} months: {months[0].date()} .. {months[-1].date()}")
acc={}
for d in months:
    for p,w in spmo_p[d].items(): acc.setdefault(p,[]).append(w)
nM=len(months)
pf=pd.DataFrame([{'permno':p,'avg_wt':np.sum(v)/nM,'pct_months':len(v)/nM,'avg_wt_when_held':np.mean(v)} for p,v in acc.items()])
pf=pf.sort_values('avg_wt',ascending=False).reset_index(drop=True)
pf.to_csv(f'{R}/results/holdings_spmo_top_jan2024.csv',index=False)

NAMES={86580:'NVIDIA',84788:'Amazon',13407:'Meta',50876:'Eli Lilly',93002:'Broadcom',
       10107:'Microsoft',14542:'Alphabet (A)',90319:'Alphabet (C)',55976:'Walmart',12060:'GE',
       14593:'Apple*',11850:'Exxon Mobil*',12490:'IBM*',93436:'Tesla*'}
print("\n  TOP-15 replicated-SPMO holdings, Jan-2024-onward (avg weight):")
print(f"  {'#':>2} {'permno':>7} {'company':<16}{'avg wt':>8}{'%mo held':>9}{'wt|held':>9}")
for i,r in pf.head(15).iterrows():
    nm=NAMES.get(int(r['permno']),'(permno only)')
    print(f"  {i+1:>2} {int(r['permno']):>7} {nm:<16}{r['avg_wt']:>7.1%}{r['pct_months']:>9.0%}{r['avg_wt_when_held']:>9.1%}")
print("\n  saved results/holdings_spmo_top_jan2024.csv  (* = standard CRSP permno, WRDS-unverified)")
