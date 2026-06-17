"""
2026-06-06-for-gadi-from-1995.py  (EXPLORATORY)
Re-based for-gadi performance graphs starting 1995 (instead of 1990/2011). Two figures,
both SPMO replica vs S&P 500 (cap-wt Top-500), real panel 1995 -> Dec-2025:
  (A) growth of $1 (log scale)
  (B) cumulative additive out-performance = running sum of monthly (SPMO - S&P), in points
SPMO replica = mechanical rules-based index (top-20% risk-adj momentum, FMC x score, 9% cap,
Mar/Sep rebalance), so the full backtest is valid; real ETF only launched Nov-2015.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT
START='1995-01-01'

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)   # features use full history
test=test[test['date']>=START].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spm=sp.build_longonly(test,'spmo_signal','score',0.20,0.09).dropna()
mkt=test.dropna(subset=['me','ret_fwd']).groupby('date').apply(lambda g:(g['me']/g['me'].sum()*g['ret_fwd']).sum(),include_groups=False)
idx=spm.index.intersection(mkt.index); spm,mkt=spm.reindex(idx),mkt.reindex(idx)

def cagr(r): return (1+r).prod()**(12/len(r))-1
def sh(r): return r.mean()/r.std()*np.sqrt(12)
cs,cm=(1+spm).cumprod(),(1+mkt).cumprod()
print(f"window {idx.min().date()} -> {idx.max().date()} ({len(idx)} mo)")
print(f"  S&P500       $1->${cm.iloc[-1]:.1f} CAGR {cagr(mkt):.1%} Sharpe {sh(mkt):.2f}")
print(f"  SPMO replica $1->${cs.iloc[-1]:.1f} CAGR {cagr(spm):.1%} Sharpe {sh(spm):.2f}")

FOOT=('S&P 500 = cap-weighted Top-500 (S&P 500 proxy). SPMO replica is a mechanical rules-based backtest '
      '(top-20% risk-adj momentum, FMC x score, 9% cap, Mar/Sep rebalance); pre-2015 is a backtest, the real ETF launched Nov-2015.')

# (A) growth of $1, log
figA,axA=plt.subplots(figsize=(13.5,6.6))
axA.plot(cm.index,cm.values,color='gray',lw=2.0,label=f'S&P 500 (cap-wt)   $1→${cm.iloc[-1]:.0f}  (CAGR {cagr(mkt):.1%}, Sharpe {sh(mkt):.2f})')
axA.plot(cs.index,cs.values,color='steelblue',lw=2.3,label=f'SPMO replica      $1→${cs.iloc[-1]:.0f}  (CAGR {cagr(spm):.1%}, Sharpe {sh(spm):.2f})')
axA.axvline(pd.Timestamp('2015-11-30'),color='crimson',ls='--',lw=1.1)
axA.text(pd.Timestamp('2015-12-15'),cs.iloc[-1]*0.78,'real SPMO ETF\nlaunch (Nov-2015)',fontsize=8,color='crimson',va='top')
axA.set_yscale('log'); axA.set_ylabel('Growth of $1 invested (log scale)'); axA.grid(True,alpha=.25,which='both')
axA.set_title('SPMO replica vs S&P 500, growth of $1 (log scale), 1995-2025',fontsize=12.5)
axA.legend(loc='upper left',fontsize=10)
figA.text(0.5,0.01,FOOT,ha='center',fontsize=8,color='dimgray'); plt.tight_layout(rect=[0,0.03,1,1])
OUTA=f'{R}/plots/2026-06-06-spmo-vs-sp500-1995'
for e in ('pdf','png'): figA.savefig(f'{OUTA}.{e}',dpi=160,bbox_inches='tight')
plt.close(figA)

# (B) cumulative additive out-performance
ex=(spm-mkt).dropna(); cum=ex.cumsum()*100
g_end=cum.iloc[-1]
figB,axB=plt.subplots(figsize=(13.5,6.4))
axB.plot(cum.index,cum.values,color='crimson',lw=2.2); axB.fill_between(cum.index,cum.values,0,color='crimson',alpha=.08)
axB.axhline(0,color='k',lw=.7); axB.grid(True,alpha=.25)
axB.set_ylabel('Cumulative out-performance vs S&P 500 (pts, summed monthly)')
axB.set_title('Cumulative SPMO out-performance vs S&P 500 (running sum of monthly excess), 1995-2025',fontsize=12)
figB.text(0.5,0.01,FOOT,ha='center',fontsize=8,color='dimgray'); plt.tight_layout(rect=[0,0.03,1,1])
OUTB=f'{R}/plots/2026-06-06-when-lead-earned-1995'
for e in ('pdf','png'): figB.savefig(f'{OUTB}.{e}',dpi=160,bbox_inches='tight')
plt.close(figB)
print(f"\nsaved {OUTA}.* and {OUTB}.*")
