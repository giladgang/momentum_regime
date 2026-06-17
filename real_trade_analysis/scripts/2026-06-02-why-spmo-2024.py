"""
2026-06-02-why-spmo-2024.py  (EXPLORATORY)
Investigate WHY SPMO suddenly beat the market in 2024-2025 vs the 2011-2023 wash.
Tests three hypotheses on the real spliced panel (legacy + CRSP v2 -> Dec-2025):

  H1 mega-cap leadership: SPMO (cap x score, concentrated in the biggest momentum
     names) wins when a few mega-caps lead -> proxy = cap-weight minus equal-weight
     Top-500 return. Correlate annual SPMO excess with it.
  H2 idiosyncratic, not the momentum factor: compare annual SPMO excess to the UMD
     factor return (FF). If UMD is flat while SPMO wins -> not the factor.
  H3 exact name attribution: SPMO and market weights both sum to 1, so
     sum_p (w_spmo - w_mkt) * ret = SPMO excess, decomposable per name. Which names
     drove 2024 vs the lagging years (2016, 2021)?
"""
import importlib.util, sys, numpy as np, pandas as pd
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; ha=_load('ha','2026-05-31-holdings-analysis.py')
from config import TRAIN_END, FF_FACTORS_PATH
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT
xw=pd.read_csv(f'{R}/results/permno_name_crosswalk.csv').set_index('permno')['ticker'].to_dict()

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spath=ha.spmo_path(list(test.groupby('date')))

# market cap-weight + equal-weight returns and SPMO weights per month
rows=[]; contrib={}   # contrib[year][permno] = summed excess contribution
for d,g in test.groupby('date'):
    g=g.dropna(subset=['me','ret_fwd']);
    if d not in spath or not spath[d]: continue
    wmkt=(g.set_index('permno')['me']/g['me'].sum()).to_dict()
    r=g.set_index('permno')['ret_fwd'].to_dict()
    wsp=spath[d]
    mkt=sum(wmkt[p]*r[p] for p in wmkt); eqw=g['ret_fwd'].mean()
    spr=sum(wsp[p]*r.get(p,0) for p in wsp)
    rows.append({'date':d,'spmo':spr,'mkt':mkt,'eqw':eqw})
    y=d.year; contrib.setdefault(y,{})
    for p in set(wsp)|set(wmkt):
        contrib[y][p]=contrib[y].get(p,0)+(wsp.get(p,0)-wmkt.get(p,0))*r.get(p,0)
df=pd.DataFrame(rows).set_index('date')

# annual aggregation
ann=df.copy(); ann.index=ann.index.to_period('Y')
A=(1+ann).groupby(ann.index).prod()-1; A['excess']=A['spmo']-A['mkt']; A['megacap_lead']=A['mkt']-A['eqw']
ff=pd.read_parquet(FF_FACTORS_PATH); ff.index=pd.to_datetime(ff.index).to_period('M')
umd=ff['UMD']; umd_y=(1+umd).groupby(umd.index.year).prod()-1   # FF parquet already in decimal
A['UMD']=[umd_y.get(int(str(y)),np.nan) for y in A.index]

print("  year   SPMO    market  excess | megacap_lead  UMD")
for y in A.index:
    print(f"  {str(y):<6}{A.loc[y,'spmo']:>+7.1%}{A.loc[y,'mkt']:>+8.1%}{A.loc[y,'excess']:>+8.1%} | {A.loc[y,'megacap_lead']:>+8.1%}    {A.loc[y,'UMD']:>+6.1%}")

pre=A[A.index<=pd.Period('2023')]; post=A[A.index>=pd.Period('2024')]
print(f"\n  avg excess  2011-2023: {pre['excess'].mean():+.1%}/yr   2024-2025: {post['excess'].mean():+.1%}/yr")
print(f"  corr(annual excess, megacap_lead) = {A['excess'].corr(A['megacap_lead']):+.2f}")
print(f"  corr(annual excess, UMD)          = {A['excess'].corr(A['UMD']):+.2f}")
print(f"  megacap_lead 2011-23 avg {pre['megacap_lead'].mean():+.1%}  vs 2024-25 {post['megacap_lead'].mean():+.1%}")
print(f"  UMD          2011-23 avg {pre['UMD'].mean():+.1%}  vs 2024-25 {post['UMD'].mean():+.1%}")

def top_contrib(y,k=6):
    s=pd.Series(contrib[y]).sort_values(ascending=False)
    pos=s.head(k); neg=s.tail(3)
    print(f"\n  {y}: SPMO excess {A.loc[pd.Period(str(y)),'excess']:+.1%}.  Top + contributors / worst -:")
    for p,v in pos.items(): print(f"      {xw.get(p,str(p)):<7} {v:>+6.1%}")
    print("      ...")
    for p,v in neg.items(): print(f"      {xw.get(p,str(p)):<7} {v:>+6.1%}")
for y in [2024,2025,2017,2021,2016]:
    if y in contrib: top_contrib(y)

# ---- excess concentration: how few names drive the excess, pre vs post ----
rows=[]
for y,cd in contrib.items():
    s=pd.Series(cd); tot=s.sum(); pos=s[s>0]
    top3=s.sort_values(ascending=False).head(3).sum()
    hhi=((pos/pos.sum())**2).sum() if pos.sum()>0 else np.nan
    rows.append({'year':int(str(y)),'excess':tot,'top3_contrib':top3,
                 'top3_share_of_excess':top3/tot if tot!=0 else np.nan,'pos_HHI':hhi})
conc=pd.DataFrame(rows).sort_values('year')
conc.to_csv(f'{R}/results/2026-06-02-attribution-concentration.csv',index=False)
print("\n  excess concentration by year (top-3 share of excess; HHI of positive contribs):")
print(conc.to_string(index=False))
print(f"\n  avg top-3 share of excess   PRE(<=2023) {conc[conc.year<=2023]['top3_share_of_excess'].mean():+.0%}   POST(>=2024) {conc[conc.year>=2024]['top3_share_of_excess'].mean():+.0%}")
# VALIDATION: per-name contributions must sum to the linear monthly excess sum
# (contrib accumulates sum_t (w_spmo-w_mkt)*ret_t, i.e. arithmetic sum of monthly spreads;
#  A['excess'] uses compounded annual returns so a compounding wedge is expected and correct)
lin_excess={y:(df[df.index.year==y]['spmo']-df[df.index.year==y]['mkt']).sum() for y in contrib}
for y in [2024,2021]:
    s=sum(contrib[y].values()); l=lin_excess[y]
    assert abs(s-l)<1e-8, f"{y}: attribution {s:.6f} != linear monthly excess {l:.6f}"
print("VALIDATION ok: attribution sums to linear monthly excess (2024, 2021)")
# save 2024 per-name top contributors for the chart
c24=pd.Series(contrib[2024]).sort_values(ascending=False)
c24.index=[xw.get(p,str(p)) for p in c24.index]
c24.head(10).to_csv(f'{R}/results/2026-06-02-attr2024-top.csv',header=['excess_contrib'])
print("\n  2024 top-10 name contributions to excess:\n",(c24.head(10)*100).round(2).to_string())
