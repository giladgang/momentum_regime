"""
2026-06-05-spmo-top10-successful-2024.py  (EXPLORATORY)
The 10 most successful SPMO-replica holdings in 2024, ranked by contribution to the
replica's 2024 return (sum over 2024 month-ends of weight x next-month return, the same
weight/return convention the attribution work uses). Reports company name, average 2024
weight, months held, the stock's own 2024 return (held period), and its return contribution.
No sector / WRDS needed.
"""
import importlib.util, sys, numpy as np, pandas as pd
_ROOT='/Users/giladgang/momentum_regime'; R=f'{_ROOT}/real_trade_analysis'; sys.path.insert(0,_ROOT)
def _load(n,f):
    s=importlib.util.spec_from_file_location(n,f'{R}/scripts/{f}'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
sp=_load('sp','2026-05-31-spmo-replica-vs-xgb.py'); exp=sp.exp; ha=_load('ha','2026-05-31-holdings-analysis.py')
from config import TRAIN_END
EXT=f'{R}/data/_ext_crsp_real.parquet'; exp._CRSP_PATH=EXT; sp._CRSP=EXT

cw=pd.read_csv(f'{R}/results/permno_name_crosswalk.csv',parse_dates=['namedt','nameenddt'])
cw=cw.sort_values('nameenddt').drop_duplicates('permno',keep='last')   # most recent name per permno
TIC=cw.set_index('permno')['ticker'].to_dict(); NAME=cw.set_index('permno')['comnam'].to_dict()

base=sp.add_spmo_inputs(exp.build_features()); test=exp.apply_size_screen(base,500)
test=test[test['date']>=TRAIN_END].copy(); test['spmo_signal']=test['mom_value']/test['sigma_m']
spath=ha.spmo_path(list(test.groupby('date')))
rmap={d:g.dropna(subset=['ret_fwd']).set_index('permno')['ret_fwd'].to_dict() for d,g in test.groupby('date')}

contrib={}; wsum={}; wn={}; rprod={}
for d in sorted(spath):
    if d.year!=2024 or not spath[d]: continue
    r=rmap.get(d,{})
    for p,wt in spath[d].items():
        rt=r.get(p,0.0)
        contrib[p]=contrib.get(p,0)+wt*rt
        wsum[p]=wsum.get(p,0)+wt; wn[p]=wn.get(p,0)+1; rprod[p]=rprod.get(p,1.0)*(1+rt)
pf=pd.DataFrame([{'permno':p,'ticker':TIC.get(p,str(p)),'company':NAME.get(p,'(unknown)'),
                 'avg_wt':wsum[p]/wn[p],'mo_held':wn[p],'stock_ret_2024':rprod[p]-1,
                 'contrib_2024':contrib[p]} for p in contrib]).sort_values('contrib_2024',ascending=False).reset_index(drop=True)
top10=pf.head(10).copy()

# GICS sector for the top-10 (WRDS Compustat via CRSP link)
GICS={10:'Energy',15:'Materials',20:'Industrials',25:'Consumer Discretionary',30:'Consumer Staples',
      35:'Health Care',40:'Financials',45:'Information Technology',50:'Communication Services',
      55:'Utilities',60:'Real Estate'}
def sectors(permnos):
    import psycopg2
    c=psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu',port=9737,dbname='wrds',
                       user='giladgang',sslmode='require',connect_timeout=60)
    cur=c.cursor()
    cur.execute("""SELECT DISTINCT ON (l.lpermno) l.lpermno, co.gsector
        FROM crsp.ccmxpf_lnkhist l JOIN comp.company co ON l.gvkey=co.gvkey
        WHERE l.linktype IN ('LU','LC') AND l.linkprim IN ('P','C','J') AND l.lpermno = ANY(%s)
        AND (l.linkenddt>=DATE '2024-01-01' OR l.linkenddt IS NULL) AND l.linkdt<=DATE '2024-12-31'
        ORDER BY l.lpermno, l.linkprim""",([int(x) for x in permnos],))
    out={int(p):GICS.get(int(g),f'GICS {g}') for p,g in cur.fetchall() if g is not None}
    c.close(); return out
sec=sectors(top10['permno'].tolist())
top10['sector']=top10['permno'].map(sec).fillna('(n/a)')
print(f"SPMO replica 2024 return (sum of all contribs, linear): {pf['contrib_2024'].sum():+.1%}")
print(f"top-10 share of that total: {top10['contrib_2024'].sum()/pf['contrib_2024'].sum():.0%}\n")
print(f"{'#':>2}  {'ticker':<6}{'company':<26}{'sector':<24}{'avg wt':>8}{'mo':>4}{'2024 ret':>10}{'contrib':>9}")
for i,r in top10.iterrows():
    print(f"{i+1:>2}  {r['ticker']:<6}{r['company'][:25]:<26}{r['sector'][:23]:<24}{r['avg_wt']:>7.1%}{int(r['mo_held']):>4}{r['stock_ret_2024']:>+9.0%}{r['contrib_2024']:>+9.1%}")
top10[['ticker','company','sector','avg_wt','mo_held','stock_ret_2024','contrib_2024']].to_csv(f'{R}/results/2026-06-05-spmo-top10-successful-2024.csv',index=False)
print(f"\nsaved results/2026-06-05-spmo-top10-successful-2024.csv")

# ---- formatted table figure for the deck ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
DISP={'NVDA':'NVIDIA','AVGO':'Broadcom','META':'Meta Platforms','AMZN':'Amazon','WMT':'Walmart',
      'JPM':'JPMorgan Chase','GOOG':'Alphabet','GE':'General Electric','LLY':'Eli Lilly','COST':'Costco'}
cols=['#','Ticker','Company','Sector','Avg wt','2024 return','Contribution']
cells=[[str(i+1),r['ticker'],DISP.get(r['ticker'],r['company'].title()),r['sector'],
        f"{r['avg_wt']:.1%}",f"{r['stock_ret_2024']:+.0%}",f"{r['contrib_2024']:+.1%}"]
       for i,r in top10.iterrows()]
fig,ax=plt.subplots(figsize=(12.5,4.7)); ax.axis('off')
tbl=ax.table(cellText=cells,colLabels=cols,cellLoc='center',loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1,1.55)
tbl.auto_set_column_width(col=list(range(len(cols))))
for (row,col),cell in tbl.get_celld().items():
    cell.set_edgecolor('white')
    if row==0:
        cell.set_facecolor('#2f4b7c'); cell.set_text_props(color='white',fontweight='bold')
    else:
        cell.set_facecolor('#eef2f7' if row%2 else '#ffffff')
        if col==6: cell.set_text_props(fontweight='bold',color='#b30000')
ax.set_title('SPMO replica: 10 most successful holdings, 2024',fontsize=14,pad=14,fontweight='bold')
fig.text(0.5,0.035,'Ranked by contribution to the replica’s 2024 return (avg weight × return).  '
         'These 10 = 70% of SPMO’s +41.5% (linear) 2024 return.  Sectors: GICS (WRDS Compustat).  Weights are 2024 averages.',
         ha='center',fontsize=8,color='dimgray')
plt.tight_layout(rect=[0,0.05,1,1])
OUT=f'{R}/plots/2026-06-05-spmo-top10-successful-2024'
for e in ('pdf','png'): fig.savefig(f'{OUT}.{e}',dpi=170,bbox_inches='tight')
plt.close(fig); print(f"saved {OUT}.pdf + .png")
