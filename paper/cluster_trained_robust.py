"""Robustness of the one flagged cell: profitability trained-per-cluster band
(+0.137, p=0.015). Seed x block, OOS start year, cluster K. If it flickers
like the momentum-GP result, it's a fluke."""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/giladgang/momentum_regime'); os.chdir('/Users/giladgang/momentum_regime')
from paper import config as C
from paper import execution as X
from paper import banding_study as B
from paper.src import metrics as M
from paper.src import signals
from paper.src import clusters as CL
from paper.gp_bands import _active
from paper.cluster_bands import _pit_month_labels
from paper.cluster_trained import _walk_trained, VECS, E_BASE

sp_pack = B.load_spreads()
stocks = pd.read_parquet(C.STOCKS_PARQUET, columns=['permno','date']+CL.MOMS)
stocks['date'] = pd.to_datetime(stocks['date']); stock_z = CL.build_stock_zcurves(stocks)

def build(kK):
    CL.K = kK  # override cluster count
    global VECS_K
    VECS_K = list(itertools.product([10,20,40], repeat=kK))
    x = signals.build('profitability'); x = x[x['date']>=C.SWEEP_START].reset_index(drop=True)
    book_z = CL.book_zcurve(stock_z, x); mlab = _pit_month_labels(x, book_z)
    x = x.assign(cluster=x['date'].map(mlab).fillna(0).astype(int))
    b,bled = X.simulate(x, ('benchmark', None), score_col='score')
    bench = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])
    def act(pol): return _active(x, pol, sp_pack, bench)
    va = {str(v): act(('cluster_band', v)) for v in VECS_K}
    sa = {e: act(('nmv_band',(e,e))) for e in E_BASE}
    return va, sa

def walk_reg(va, sa, start):
    # replicate _walk_trained but return trained_reg + static
    return _walk_trained(va, sa, start)

print("=== base (K=4, start 2001) ===")
va, sa = build(4)
st = walk_reg(va, sa, 2001)
z = pd.Series(0.0, index=st['static'].index)
for a in ['trained_argmax','trained_reg']:
    d=(st[a]-st['static']).dropna()
    print(f"  {a}: dIR {M.ir(st[a],z)-M.ir(st['static'],z):+.3f} n={len(d)}")

print("\n=== trained_reg CI across seed x block ===")
d=(st['trained_reg']-st['static']).dropna(); excl=0;tot=0
for block in [6,12,24]:
    for seed in range(10):
        lo,hi=X.paired_block_bootstrap(st['trained_reg'].reindex(d.index).astype(float), st['static'].reindex(d.index).astype(float), block=block, seed=seed)
        tot+=1; excl+=(lo>0 or hi<0)
    lo,hi=X.paired_block_bootstrap(st['trained_reg'].reindex(d.index).astype(float), st['static'].reindex(d.index).astype(float), block=block, seed=0)
    print(f"  block={block} seed0 CIx12 [{lo*12:+.3f},{hi*12:+.3f}]")
print(f"  -> excl 0 in {excl}/{tot}")

print("\n=== OOS start-year sensitivity (trained_reg dIR) ===")
for s0 in [1998,2001,2004,2007]:
    st2 = walk_reg(va, sa, s0); z2=pd.Series(0.0,index=st2['static'].index)
    d2=(st2['trained_reg']-st2['static']).dropna()
    lo,hi=X.paired_block_bootstrap(st2['trained_reg'].reindex(d2.index).astype(float), st2['static'].reindex(d2.index).astype(float))
    print(f"  start={s0}: dIR {M.ir(st2['trained_reg'],z2)-M.ir(st2['static'],z2):+.3f} CIx12 [{lo*12:+.3f},{hi*12:+.3f}] n={len(d2)}")

print("\n=== cluster-K sensitivity (trained_reg dIR, start 2001) ===")
for kK in [3,4,5]:
    vk, sk = build(kK); stk = walk_reg(vk, sk, 2001); zk=pd.Series(0.0,index=stk['static'].index)
    d3=(stk['trained_reg']-stk['static']).dropna()
    lo,hi=X.paired_block_bootstrap(stk['trained_reg'].reindex(d3.index).astype(float), stk['static'].reindex(d3.index).astype(float))
    print(f"  K={kK}: dIR {M.ir(stk['trained_reg'],zk)-M.ir(stk['static'],zk):+.3f} CIx12 [{lo*12:+.3f},{hi*12:+.3f}]")
CL.K = 4
