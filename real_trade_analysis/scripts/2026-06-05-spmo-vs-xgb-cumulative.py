"""
2026-06-05-spmo-vs-xgb-cumulative.py  (EXPLORATORY)
SPMO replica vs XGB long-only (trained Top-1000, traded S&P-500), growth of $1 over the
common full-test window 2011-2024. SPMO from the validated replica machinery; XGB LO monthly
returns reused from 2026-05-31-final-spmo-vs-xgb1000_returns.csv (no retrain). Both long-only.
Neutral/descriptive; factual CAGR + Sharpe annotation only.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09).dropna()
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)  # S&P 500 = cap-wt Top-500

xgb=pd.read_csv(f'{R}/results/2026-05-31-final-spmo-vs-xgb1000_returns.csv',parse_dates=['date']).set_index('date')['xgb_lo']
idx=spm.index.intersection(xgb.index).intersection(mkt.index)   # common full-test window (2011-2024)
spm,xgb,mkt=spm.reindex(idx),xgb.reindex(idx),mkt.reindex(idx)
def stat(r): c=(1+r).cumprod(); return c, c.iloc[-1]**(12/len(r))-1, r.mean()/r.std()*np.sqrt(12)
cs,cagr_s,sh_s=stat(spm); cx,cagr_x,sh_x=stat(xgb); cm,cagr_m,sh_m=stat(mkt)
print(f"window {idx.min().date()} -> {idx.max().date()}  ({len(idx)} months)")
print(f"  S&P 500       $1->${cm.iloc[-1]:.2f}  CAGR {cagr_m:.1%}  Sharpe {sh_m:.2f}")
print(f"  SPMO replica  $1->${cs.iloc[-1]:.2f}  CAGR {cagr_s:.1%}  Sharpe {sh_s:.2f}")
print(f"  XGB long-only $1->${cx.iloc[-1]:.2f}  CAGR {cagr_x:.1%}  Sharpe {sh_x:.2f}")

fig,ax=plt.subplots(figsize=(13,6.4))
ax.plot(cm.index,cm.values,color='gray',lw=2.0,label=f'S&P 500 (cap-wt)  (CAGR {cagr_m:.1%}, Sharpe {sh_m:.2f})')
ax.plot(cs.index,cs.values,color='steelblue',lw=2.4,label=f'SPMO replica   (CAGR {cagr_s:.1%}, Sharpe {sh_s:.2f})')
ax.plot(cx.index,cx.values,color='darkorange',lw=2.4,label=f'XGB long-only  (CAGR {cagr_x:.1%}, Sharpe {sh_x:.2f})')
ax.axhline(1,color='k',lw=.7); ax.set_ylabel('Growth of $1 invested'); ax.grid(True,alpha=.25)
ax.set_title('SPMO replica vs XGB long-only vs S&P 500, cumulative return 2011-2024',fontsize=12)
ax.legend(loc='upper left',fontsize=10)
fig.text(0.5,0.01,'SPMO and XGB are long-only on the S&P 500 (Top-500 proxy), same trading universe.  XGB trained on the broader Top-1000.  '
         'S&P 500 = cap-weighted Top-500.  Common window 2011-2024 (167 months).',
         ha='center',fontsize=8.5,color='dimgray')
plt.tight_layout(rect=[0,0.035,1,1])
OUT=f'{R}/plots/2026-06-05-spmo-vs-xgb-cumulative'
for e in ('pdf','png'): fig.savefig(f'{OUT}.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); print(f"\nsaved {OUT}.pdf + .png")
