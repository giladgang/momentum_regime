"""
2026-06-02-dispersion-extend.py  (EXPLORATORY)
Extend the S&P 500 (Top-500) cross-sectional monthly-return dispersion series from
CRSP v2 (crsp.msf_v2, CIZ format), which runs through Dec-2025 (legacy ends Nov-2024).
Methodology matches the existing series: per month, eligible common stock on a major
exchange, Top-500 by market cap, dispersion = xs_std / p90-p10 / IQR of monthly return.
Validates against the existing series on overlap months before splicing. Sandbox off.
"""
import os, numpy as np, pandas as pd, psycopg2
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'
EXIST=f'{R}/results/2026-06-01-sp500-dispersion.csv'

c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',user='giladgang',sslmode='require',connect_timeout=40)
q="""SELECT permno, mthcaldt AS date, mthret AS ret, mthcap AS me, mthprc AS prc,
            primaryexch, ticker, issuernm
     FROM crsp.msf_v2
     WHERE mthcaldt >= '2024-06-30' AND mthcaldt <= '2025-12-31'
       AND securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS'
       AND primaryexch IN ('N','A','Q') AND mthret IS NOT NULL AND mthcap IS NOT NULL"""
df=pd.read_sql(q,c); c.close()
df['date']=pd.to_datetime(df['date'])
df=df[df['prc'].abs()>1].copy()
print(f"pulled {len(df):,} rows, {df['date'].nunique()} months {df['date'].min().date()}..{df['date'].max().date()}")

def disp_month(g):
    g=g.nlargest(500,'me'); r=g['ret'].astype(float)
    return pd.Series({'xs_std':r.std(),'p90_p10':r.quantile(.9)-r.quantile(.1),
                      'iqr':r.quantile(.75)-r.quantile(.25),'n':len(r)})
out=df.groupby('date').apply(disp_month,include_groups=False).reset_index()
out['year']=out['date'].dt.year

# ---- validate against existing series on overlap months ----
ex=pd.read_csv(EXIST,parse_dates=['date'])
ovl=ex[ex['date'].isin(out['date'])][['date','xs_std','p90_p10','n']].merge(
    out[['date','xs_std','p90_p10','n']],on='date',suffixes=('_legacy','_v2'))
print("\n  OVERLAP VALIDATION (legacy panel vs CRSP v2):")
print(f"  {'date':<12}{'p90p10 legacy':>14}{'p90p10 v2':>11}{'n_leg':>7}{'n_v2':>6}")
for _,r in ovl.iterrows():
    print(f"  {r['date'].date()!s:<12}{r['p90_p10_legacy']:>14.4f}{r['p90_p10_v2']:>11.4f}{int(r['n_legacy']):>7}{int(r['n_v2']):>6}")

# sanity: largest names Dec-2025
dec=df[df['date']==df['date'].max()].nlargest(10,'me')[['ticker','issuernm','me']]
print("\n  Dec-2025 Top-10 by market cap (sanity):")
for _,r in dec.iterrows(): print(f"    {str(r['ticker']):<7}{str(r['issuernm'])[:30]:<31}{r['me']/1e6:>8.0f}B")

# ---- splice: append the NEW months (after the existing series end) ----
new=out[out['date']>ex['date'].max()].copy()
print(f"\n  appending {len(new)} new months: {new['date'].min().date()}..{new['date'].max().date()}")
ext=pd.concat([ex,new[['date','xs_std','p90_p10','iqr','n','year']]],ignore_index=True).sort_values('date')
ext.to_csv(f'{R}/results/2026-06-02-sp500-dispersion-thru2025.csv',index=False)

print("\n  2024-12 .. 2025-12 dispersion (p90-p10):")
for _,r in new.iterrows(): print(f"    {r['date'].date()}  p90p10={r['p90_p10']:.4f}  xs_std={r['xs_std']:.4f}  n={int(r['n'])}")
print(f"\n  2025 mean p90-p10 = {new[new['year']==2025]['p90_p10'].mean():.4f}  vs 2024 full-year {ex[ex['date'].dt.year==2024]['p90_p10'].mean():.4f}  vs 2011-15 mean {ex[(ex['date'].dt.year>=2011)&(ex['date'].dt.year<=2015)]['p90_p10'].mean():.4f}")
print(f"  saved results/2026-06-02-sp500-dispersion-thru2025.csv")
