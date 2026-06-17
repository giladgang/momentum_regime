"""
2026-06-02-spmo-vs-sp500-thru2025.py  (EXPLORATORY)
Extend the SPMO replica vs S&P 500 (cap-weight Top-500) head-to-head through Dec-2025
using REAL CRSP v2 data (msf_v2, CIZ) spliced onto the legacy panel (ends Nov-2024).
SPMO needs no regime signal, so this is fully real. Charts cumulative growth 2011-2025
and prints annual returns incl. the new 2025 row. Also tracks NVIDIA's SPMO weight.
Sandbox off (network).
"""
import importlib.util, os, sys, numpy as np, pandas as pd, psycopg2
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'
sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; ha=_load('ha','2026-05-31-holdings-analysis.py')
from src.utils import metrics
from config import TRAIN_END
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# ---- pull v2 monthly Dec-2024..Dec-2025, map CIZ -> legacy schema ----
c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='giladgang',sslmode='require',connect_timeout=40)
v=pd.read_sql("""SELECT permno, mthcaldt AS date, mthret AS ret_adj, mthcap AS me, mthprc AS prc, primaryexch
                 FROM crsp.msf_v2 WHERE mthcaldt > '2024-11-30' AND mthcaldt <= '2025-12-31'
                   AND securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS'
                   AND primaryexch IN ('N','A','Q') AND mthret IS NOT NULL AND mthcap IS NOT NULL""",c); c.close()
v['date']=pd.to_datetime(v['date']); v['exchcd']=v['primaryexch'].map({'N':1,'A':2,'Q':3}); v['shrcd']=10
v=v[['permno','date','ret_adj','shrcd','exchcd','prc','me']]

# ---- splice onto legacy CRSP ----
leg=pd.read_parquet(exp._CRSP_PATH,columns=['permno','date','ret_adj','shrcd','exchcd','prc','me'])
leg['date']=pd.to_datetime(leg['date'])
ext=pd.concat([leg,v],ignore_index=True).drop_duplicates(['permno','date'],keep='first').sort_values(['permno','date']).reset_index(drop=True)
ext.to_parquet(f'{R}/data/_ext_crsp_real.parquet')
print(f"legacy {leg['date'].max().date()} + v2 -> {ext['date'].max().date()} ({v['date'].nunique()} new months)")

# ---- recompute features + SPMO inputs on the spliced series ----
exp._CRSP_PATH=f'{R}/data/_ext_crsp_real.parquet'; sp._CRSP=f'{R}/data/_ext_crsp_real.parquet'
base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']

# ---- SPMO replica + S&P 500 cap-weight benchmark ----
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09)
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
spm,mkt=spm.dropna(),mkt.dropna()

def ann_table(r):
    ym=r.copy(); ym.index=ym.index.to_period('Y'); return (1+ym).groupby(ym.index).prod()-1
print("\n  annual returns:  year    SPMO     S&P500")
at_s,at_m=ann_table(spm),ann_table(mkt)
for y in sorted(set(at_s.index)|set(at_m.index)):
    print(f"           {str(y):>8}{at_s.get(y,np.nan):>+8.1%}{at_m.get(y,np.nan):>+8.1%}")
for lbl,r in [('SPMO replica',spm),('S&P 500',mkt)]:
    a,vv,sh,md=metrics(r); print(f"  {lbl:<14} full 2011-2025: ann {a:+.1%}  vol {vv:.1%}  Sharpe {sh:.2f}  MDD {md:.1%}")

# ---- NVIDIA weight in SPMO over time ----
spath=ha.spmo_path(list(test.groupby('date')))
nvda=[(d,w.get(86580,0.0)) for d,w in sorted(spath.items())]
nv=pd.Series(dict(nvda)); print(f"\n  NVIDIA SPMO weight: Jan-2024 {nv.get(pd.Timestamp('2024-01-31'),0):.1%} -> Dec-2025 {nv.iloc[-1]:.1%} (max {nv.max():.1%})")

# ---- chart ----
SPL=pd.Timestamp('2024-11-29')
fig,(a1,a2)=plt.subplots(1,2,figsize=(15,6.2))
for ax,start in [(a1,'2011-01-01'),(a2,'2020-01-01')]:
    for nm,r,col in [('SPMO replica',spm,'steelblue'),('S&P 500 (cap-wt)',mkt,'gray')]:
        rr=r[r.index>=pd.Timestamp(start)]; cc=(1+rr).cumprod()/(1+rr).cumprod().iloc[0]
        ax.plot(cc.index,cc.values,lw=2.3,color=col,label=f'{nm} ({(cc.iloc[-1]-1)*100:+.0f}%)')
    ax.axvline(SPL,color='crimson',ls=':',lw=1.2); ax.axvspan(SPL,spm.index.max(),color='crimson',alpha=.05)
    ax.set_yscale('log'); ax.set_title(f'from {start[:4]}  (red = real CRSP-v2 extension, post Nov-2024)',fontsize=10.5)
    ax.legend(loc='upper left',fontsize=9); ax.grid(True,alpha=.25,which='both')
a1.set_ylabel('Growth of $1 (log scale)')
fig.suptitle('SPMO replica vs S&P 500, extended through Dec-2025 with real CRSP data',fontsize=12.5); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-spmo-vs-sp500-thru2025.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print("\nsaved plots/2026-06-02-spmo-vs-sp500-thru2025.pdf + .png")
