"""
2026-06-02-topsize-churn.py  (EXPLORATORY)
Year-over-year membership churn of the LARGEST companies in the S&P 500 (Top-500),
ranked by MARKET CAP (size), not momentum. Each year-end we take the top-20% by market
cap (the same threshold SPMO uses) and measure how many names are NEW vs the prior
year-end. Contrast with the ~80% churn of the momentum-ranked top-20%: size is sticky,
momentum reshuffles. Also shows the top-10 mega-cap churn. Real panel.
"""
import importlib.util, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT

base=exp.build_features(); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy()

def yearend_sets(frac=None,topn=None):
    s={}
    for y,g in test.groupby(test['date'].dt.year):
        last=g[g['date']==g['date'].max()].dropna(subset=['me'])
        n=topn if topn else max(1,round(frac*len(last)))
        s[y]=set(last.nlargest(n,'me')['permno'])
    return s
q20=yearend_sets(frac=0.20); t10=yearend_sets(topn=10)

def churn(sets):
    yrs=sorted(sets); out=[]
    for i in range(1,len(yrs)):
        a,b=sets[yrs[i-1]],sets[yrs[i]]
        out.append({'year':yrs[i],'size':len(b),'new':len(b-a),'pct':len(b-a)/len(b),'stayed':len(a&b)})
    return pd.DataFrame(out)
c20,c10=churn(q20),churn(t10)
print("TOP-20% BY MARKET CAP -- year-over-year churn:")
print("year  size  new  %changed  stayed")
for _,r in c20.iterrows(): print(f"  {int(r['year'])}  {int(r['size']):>4} {int(r['new']):>4}    {r['pct']:>5.0%}   {int(r['stayed']):>4}")
print(f"  mean %changed {c20['pct'].mean():.0%}   (vs ~83% for momentum-ranked top-20%)")
print(f"\nTOP-10 MEGA-CAPS churn (new names/yr): {dict(zip(c10['year'].astype(int),c10['new']))}")

fig,ax=plt.subplots(figsize=(13,6))
x=np.arange(len(c20))
ax.bar(x,c20['pct']*100,color=['crimson' if y==2024 else 'seagreen' for y in c20['year']],width=0.7,label='top-20% by market cap (size)')
ax.axhline(c20['pct'].mean()*100,color='seagreen',ls='--',lw=1,label=f"size mean {c20['pct'].mean():.0%}")
ax.axhline(83,color='gray',ls=':',lw=1.4,label='momentum top-20% mean 83% (for contrast)')
for i,r in c20.iterrows(): ax.text(i,r['pct']*100+0.6,f"{int(r['new'])}/{int(r['size'])}",ha='center',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(c20['year'].astype(int),rotation=0)
ax.set_ylabel('% of the top-20%-by-SIZE list that is NEW vs prior year'); ax.set_ylim(0,90)
ax.set_title('Year-over-year churn of the S&P 500\'s LARGEST companies (top-20% by market cap)\nsize is sticky (low churn) vs momentum which reshuffles ~80%/yr; red=2024',fontsize=11.5)
ax.legend(fontsize=9.5); ax.grid(True,axis='y',alpha=.25); plt.tight_layout()
for e in ('pdf','png'): fig.savefig(f'{R}/plots/2026-06-02-topsize-churn.{e}',dpi=150,bbox_inches='tight')
plt.close(fig); c20.to_csv(f'{R}/results/2026-06-02-topsize-churn.csv',index=False)
print("\nsaved plots/2026-06-02-topsize-churn.pdf + .png")
