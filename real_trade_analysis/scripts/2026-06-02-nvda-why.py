"""
2026-06-02-nvda-why.py  (EXPLORATORY)
Why did NVIDIA fall out of the SPMO replica in 2025? Decompose its risk-adjusted
momentum signal (spmo_signal = mom_value / sigma_m) vs the top-quintile cutoff at
each semi-annual rebalance, and show its 2025 monthly returns. Real spliced panel.
"""
import importlib.util, sys, numpy as np, pandas as pd
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT
NV=86580

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=pd.Timestamp('2023-09-01')].copy()
test['spmo_signal']=test['mom_value']/test['sigma_m']

print("  NVIDIA risk-adjusted momentum vs the SPMO top-quintile cutoff, at each rebalance (Mar/Sep):")
print(f"  {'rebal':<10}{'NVDA signal':>12}{'q80 cutoff':>12}{'in top20%?':>11}{'NVDA %ile':>11}{'mom_value':>11}{'sigma_m':>10}")
for d,g in test.groupby('date'):
    if d.month not in (3,9): continue
    gg=g.dropna(subset=['spmo_signal','me'])
    if NV not in set(gg['permno']):
        print(f"  {d.date()!s:<10}  NVDA not in Top-500 eligible"); continue
    cut=gg['spmo_signal'].quantile(0.80)
    row=gg[gg['permno']==NV].iloc[0]; sig=row['spmo_signal']
    pct=(gg['spmo_signal']<sig).mean()
    print(f"  {d.date()!s:<10}{sig:>12.2f}{cut:>12.2f}{'YES' if sig>=cut else 'NO':>11}{pct:>10.0%}{row['mom_value']:>11.1%}{row['sigma_m']:>10.3f}")

print("\n  NVIDIA monthly returns 2025 (volatility that lifted sigma_m, denominator):")
nv=base[(base['permno']==NV)&(base['date']>=pd.Timestamp('2025-01-01'))].sort_values('date')
for _,r in nv.iterrows(): print(f"    {r['date'].date()}  ret {r['ret_adj']:>+7.1%}")
print(f"    2025 cumulative: {((1+nv['ret_adj']).prod()-1):+.1%}  | monthly sigma {nv['ret_adj'].std():.3f}")
