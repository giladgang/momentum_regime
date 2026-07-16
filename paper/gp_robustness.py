"""Robustness battery for the load-bearing momentum GP-band result.
Stress the marginal CI: bootstrap seed/block, OOS start year, ratio value,
E_base grid. Plus a multiple-testing (Benjamini-Hochberg) correction across
all 7 strategies' theory-vs-static deltas.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/Users/giladgang/momentum_regime')
os.chdir('/Users/giladgang/momentum_regime')
from paper import config as C
from paper import execution as X
from paper import banding_study as B
from paper.src import metrics as M
from paper.src import signals
from paper.gp_bands import _active, _regime_ratio, _walk_two, E_BASE, MONTHLY

sp_pack = B.load_spreads()
panel_vw = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
panel_vw['date'] = pd.to_datetime(panel_vw['date'])


def stitched(strategy, start, forced_ratio=None):
    x = signals.build(strategy)
    x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
    ratio = forced_ratio if forced_ratio is not None else \
        _regime_ratio(x, sp_pack, panel_vw, f'{start}-01-01')
    b, bled = X.simulate(x, ('benchmark', None), score_col='score')
    bench_net = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])
    sa, ta = {}, {}
    for e in E_BASE:
        sa[f's{e}'] = _active(x, ('nmv_band', (e, e)), sp_pack, bench_net)
        ep = float(np.clip(e * ratio, 5.0, 60.0))
        ta[f't{e}'] = _active(x, ('nmv_band', (e, ep)), sp_pack, bench_net)
    mo = _active(x, MONTHLY, sp_pack, bench_net)
    st = _walk_two(sa, ta, mo, start)
    return st, ratio


def delta_ir(st):
    z = pd.Series(0.0, index=st['static'].index)
    return M.ir(st['theory'], z) - M.ir(st['static'], z)


print("=== MOMENTUM base (start 2001) ===")
st, ratio = stitched('momentum', 2001)
d = (st['theory'] - st['static']).dropna()
print(f"ratio={ratio:.3f}  dIR={delta_ir(st):+.3f}  n={len(d)}")

print("\n=== CI across bootstrap seed x block (does it keep excluding 0?) ===")
excl = 0; tot = 0
for block in [6, 12, 24]:
    for seed in range(10):
        lo, hi = X.paired_block_bootstrap(
            st['theory'].reindex(d.index).astype(float),
            st['static'].reindex(d.index).astype(float),
            block=block, seed=seed)
        tot += 1; excl += (lo > 0 or hi < 0)
    lo, hi = X.paired_block_bootstrap(
        st['theory'].reindex(d.index).astype(float),
        st['static'].reindex(d.index).astype(float), block=block, seed=0)
    print(f"  block={block}: seed0 CI x12 [{lo*12:+.3f},{hi*12:+.3f}]")
print(f"  -> excludes 0 in {excl}/{tot} seed x block combos")

print("\n=== OOS start-year sensitivity ===")
for s0 in [1998, 2001, 2004, 2007]:
    st2, r2 = stitched('momentum', s0)
    dd = (st2['theory'] - st2['static']).dropna()
    lo, hi = X.paired_block_bootstrap(st2['theory'].reindex(dd.index).astype(float),
                                      st2['static'].reindex(dd.index).astype(float))
    print(f"  start={s0}: ratio={r2:.2f} dIR={delta_ir(st2):+.3f} "
          f"CIx12[{lo*12:+.3f},{hi*12:+.3f}] n={len(dd)}")

print("\n=== ratio sensitivity (1.0 = no regime = static) ===")
for fr in [0.5, 0.6, 0.68, 0.8, 0.9, 1.0]:
    st3, _ = stitched('momentum', 2001, forced_ratio=fr)
    print(f"  ratio={fr:.2f}: dIR={delta_ir(st3):+.3f}")

print("\n=== multiple-testing (BH-FDR across 7 strategies) ===")
gp = pd.read_csv('paper/results/banding_study/gp_bands.csv')
# bootstrap two-sided p per strategy from stitched series
pvals = {}
for s in gp['strategy']:
    s0 = 2001 if s != 'xgb' else 2013
    stx, _ = stitched(s, s0)
    dd = (stx['theory'] - stx['static']).dropna().values
    # sign-flip / block bootstrap p: fraction of resampled means <= 0
    rng = np.random.default_rng(0)
    n = len(dd); block = 12; means = []
    for _ in range(2000):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n)); idx += list(range(start, start+block))
        means.append(dd[[i % n for i in idx[:n]]].mean())
    means = np.array(means)
    frac_le0 = (means <= 0).mean()
    pvals[s] = 2 * min(frac_le0, 1 - frac_le0)
pv = pd.Series(pvals).sort_values()
m = len(pv); bh = pv.rank(method='first') / m * 0.05  # BH threshold at q=0.05
sig = pv <= bh
print(pd.DataFrame({'p_boot': pv.round(3), 'BH_thresh': bh.round(3),
                    'sig_FDR5%': sig}).to_string())
print(f"  -> {int(sig.sum())}/7 survive BH-FDR at q=0.05")
