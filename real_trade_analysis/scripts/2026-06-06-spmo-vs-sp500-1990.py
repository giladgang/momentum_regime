"""
2026-06-06-spmo-vs-sp500-1990.py  (EXPLORATORY)
SPMO replica vs S&P 500 (cap-wt Top-500) over the FULL available panel 1990 -> Dec-2025
(growth of $1, log scale). The replica is a mechanical rules-based index (top-20% risk-adj
momentum, FMC x score, 9% cap, Mar/Sep rebalance), so a full-history backtest is valid (no
model training). NOTE: the real SPMO ETF only launched Nov-2015; pre-2015 is a backtest.
No TRAIN_END filter -> uses every month the momentum signal exists (~1991 on).
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)   # full panel, NO train_end filter
test['spmo_signal']=test['mom_value']/test['sigma_m']
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09).dropna()
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)

idx=spm.index.intersection(mkt.index)
spm,mkt=spm.reindex(idx),mkt.reindex(idx)
def stat(r): c=(1+r).cumprod(); return c, c.iloc[-1]**(12/len(r))-1, r.mean()/r.std()*np.sqrt(12)
cs,cagr_s,sh_s=stat(spm); cm,cagr_m,sh_m=stat(mkt)
print(f"window {idx.min().date()} -> {idx.max().date()}  ({len(idx)} months)")
print(f"  S&P 500       $1->${cm.iloc[-1]:.1f}  CAGR {cagr_m:.1%}  Sharpe {sh_m:.2f}")
print(f"  SPMO replica  $1->${cs.iloc[-1]:.1f}  CAGR {cagr_s:.1%}  Sharpe {sh_s:.2f}")

fig,ax=plt.subplots(figsize=(13.5,6.6))
ax.plot(cm.index,cm.values,color='gray',lw=2.0,label=f'S&P 500 (cap-wt)   $1→${cm.iloc[-1]:.0f}  (CAGR {cagr_m:.1%}, Sharpe {sh_m:.2f})')
ax.plot(cs.index,cs.values,color='steelblue',lw=2.3,label=f'SPMO replica      $1→${cs.iloc[-1]:.0f}  (CAGR {cagr_s:.1%}, Sharpe {sh_s:.2f})')
ax.axvline(pd.Timestamp('2015-11-30'),color='crimson',ls='--',lw=1.1)
ax.text(pd.Timestamp('2015-12-15'),cs.iloc[-1]*0.9,'real SPMO ETF\nlaunch (Nov-2015)',fontsize=8,color='crimson',va='top')
ax.set_yscale('log'); ax.set_ylabel('Growth of $1 invested (log scale)'); ax.grid(True,alpha=.25,which='both')
ax.set_title('SPMO replica vs S&P 500, growth of $1 (log scale), 1990-2025',fontsize=12.5)
ax.legend(loc='upper left',fontsize=10)
fig.text(0.5,0.01,'S&P 500 = cap-weighted Top-500 (S&P 500 proxy).  SPMO replica is a mechanical rules-based backtest '
         '(top-20% risk-adj momentum, FMC x score, 9% cap, Mar/Sep rebalance); pre-2015 is a backtest, the real ETF launched Nov-2015.',
         ha='center',fontsize=8,color='dimgray')
plt.tight_layout(rect=[0,0.03,1,1])
OUT=f'{R}/plots/2026-06-06-spmo-vs-sp500-1990'
for e in ('pdf','png'): fig.savefig(f'{OUT}.{e}',dpi=160,bbox_inches='tight')
plt.close(fig); print(f"\nsaved {OUT}.pdf + .png")
