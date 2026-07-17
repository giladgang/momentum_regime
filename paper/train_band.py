"""Trained per-stock band on (momentum score, regime score), walk-forward.

E_it = clip(E_base * exp(a*mom_z_it + b*(pi_t-0.5)), 5, 100), where mom_z_it is
the stock's cross-sectional momentum z-score and pi_t the HMM regime score.
The band parameters (E_base, a, b) are FIT on an expanding window (1992->) to
maximize TRAIN net-of-cost active IR, then applied out-of-sample. Distinct from
var_band / ml_band (which trained a PER-MONTH band on market factors): here each
stock's band depends on its OWN momentum score (per-stock, via stock_var_band).

Three walk-forward arms, all selected on prior data only:
  static : best a=b=0 candidate (uniform band) -- the honest baseline
  argmax : best candidate overall (overfit-prone; exhibits the inflation)
  reg    : prefer a static candidate within 1 Sharpe-SE of the best (the honest
           trained band -- 1-SE regularization toward static, the anti-overfit
           guard that this session showed is essential)

Reports OOS net-of-cost active IR for each arm and reg-minus-static with an
IR-difference block-bootstrap CI, for all 7 strategies. Also net cost bp/yr.

Usage: .venv/bin/python -m paper.train_band
Output: paper/results/banding_study/train_band.{md,csv}
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)
warnings.filterwarnings('ignore')

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper import banding_study as B                            # noqa: E402
from paper.src import signals                                   # noqa: E402

E_BASE = [15, 20, 30, 40]
A_GRID = [-0.5, -0.25, 0.0, 0.25, 0.5]           # momentum-score slope
B_GRID = [-0.5, 0.0, 0.5]                        # regime-score slope
_RNG = np.random.RandomState(0)


def _ir_slice(a):
    a = a.dropna()
    return a.mean() / a.std() * np.sqrt(12) if len(a) > 12 and a.std() > 0 \
        else -np.inf


def _walk(cand_active, is_static, start_oos):
    """cand_active: {key: OOS-able active-return Series}. Select per year on
    prior-data IR. Returns stitched static / argmax / reg active series."""
    A = pd.DataFrame(cand_active)
    yrs = [y for y in sorted({d.year for d in A.index}) if y >= start_oos]
    static_cols = [c for c in A.columns if is_static[c]]
    rec = {'static': [], 'argmax': [], 'reg': []}
    for y in yrs:
        tr, te = A[A.index.year < y], A[A.index.year == y]
        if len(tr) < 60 or len(te) == 0:
            continue
        irs = {c: _ir_slice(tr[c]) for c in A.columns}
        best = max(irs.values())
        ny = max(len(tr) / 12, 1)
        se = np.sqrt((1 + 0.5 * best ** 2) / ny)
        within = [c for c in A.columns if irs[c] >= best - se]
        sw = [c for c in within if is_static[c]]
        rec['static'].append(te[max(static_cols, key=lambda c: irs[c])])
        rec['argmax'].append(te[max(A.columns, key=lambda c: irs[c])])
        rec['reg'].append(te[max(sw or within, key=lambda c: irs[c])])
    return {k: pd.concat(v).sort_index() for k, v in rec.items() if v}


def _ir_diff_boot(a, b, n=2000, block=12, seed=0):
    """CI + p of the IR DIFFERENCE ir(a)-ir(b) via paired moving-block bootstrap
    (the correct statistic; not a mean-CI)."""
    d = pd.DataFrame({'a': a, 'b': b}).dropna()
    n_m = len(d)
    if n_m < 24:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    base = _ir_slice(d['a']) - _ir_slice(d['b'])
    diffs = []
    nb = n_m // block + 1
    for _ in range(n):
        starts = rng.randint(0, n_m - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n_m]
        s = d.iloc[idx]
        diffs.append(_ir_slice(s['a']) - _ir_slice(s['b']))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
    return (float(lo), float(hi), float(p))


def main():
    sp_pack = B.load_spreads()
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = 2001 if s != 'xgb' else 2013
        # per-stock momentum z-score within month; regime score pi
        x['mom_z'] = x.groupby('date')['mom_12'].transform(
            lambda v: (v - v.mean()) / v.std() if v.std() > 0 else 0.0
        ).fillna(0.0)
        pim = x.groupby('date')['pi'].first()
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bench = B._net(b['gross'], B.price_ledger(bled, sp_pack)[0])

        cand_active, is_static, cand_cost = {}, {}, {}
        for eb, a, bb in itertools.product(E_BASE, A_GRID, B_GRID):
            key = f'{eb}|{a}|{bb}'
            eband = np.clip(eb * np.exp(a * x['mom_z'].values
                                        + bb * (x['pi'].values - 0.5)), 5, 100)
            xx = x.assign(eband=eband)
            r, led = X.simulate(xx, ('stock_var_band', None), score_col='score')
            cost = B.price_ledger(led, sp_pack)[0].reindex(r.index).fillna(0.0)
            net = B._net(r['gross'], cost)
            cand_active[key] = (net - bench.reindex(net.index)).dropna()
            is_static[key] = (a == 0.0 and bb == 0.0)
            cand_cost[key] = float(cost.mean() * 12 * 1e4)

        stw = _walk(cand_active, is_static, start)
        z = pd.Series(0.0, index=stw['static'].index)
        ir_static = _ir_slice(stw['static'])
        ir_argmax = _ir_slice(stw['argmax'])
        ir_reg = _ir_slice(stw['reg'])
        lo, hi, p = _ir_diff_boot(stw['reg'].reindex(z.index),
                                  stw['static'].reindex(z.index))
        rows.append({
            'strategy': s, 'n_oos_mo': len(z),
            'ir_static': round(ir_static, 3), 'ir_argmax_overfit': round(ir_argmax, 3),
            'ir_reg_trained': round(ir_reg, 3),
            'reg_minus_static': round(ir_reg - ir_static, 3),
            'argmax_minus_static': round(ir_argmax - ir_static, 3),
            'ci_lo': round(lo, 3), 'ci_hi': round(hi, 3), 'p_boot': round(p, 3),
            'excl0': bool(lo > 0 or hi < 0)})
        print(f"[tb] {s}: static IR {ir_static:+.3f} | trained(reg) "
              f"{ir_reg:+.3f} (d{ir_reg-ir_static:+.3f}, p={p:.2f}) | "
              f"argmax(overfit) {ir_argmax:+.3f} (d{ir_argmax-ir_static:+.3f})",
              flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.BANDING_DIR, 'train_band.csv'), index=False)
    pd.set_option('display.width', 260)
    npos = int((R['reg_minus_static'] > 0).sum())
    nsig = int(((R['reg_minus_static'] > 0) & R['excl0']).sum())
    lines = [
        '# Trained per-stock band on (momentum score, regime score), walk-forward\n',
        'E_it = clip(E_base*exp(a*mom_z_it + b*(pi_t-.5)),5,100); (E_base,a,b) fit'
        ' expanding-window 1992-> on TRAIN net-of-cost active IR. reg = 1-SE'
        ' regularized toward static (honest trained band); argmax = unregularized'
        ' (shows overfit inflation). OOS from 2001 (xgb 2013). CI on the IR'
        ' DIFFERENCE (correct statistic).\n',
        R[['strategy', 'ir_static', 'ir_reg_trained', 'reg_minus_static',
           'ci_lo', 'ci_hi', 'p_boot', 'excl0', 'ir_argmax_overfit',
           'argmax_minus_static']].to_string(index=False), '',
        f'Trained(reg) beats static: {npos}/7. Positive AND CI excludes 0: '
        f'{nsig}/7. The argmax-minus-static column vs reg-minus-static shows how'
        ' much the raw (unregularized) trained band overfits.']
    with open(os.path.join(C.BANDING_DIR, 'train_band.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
