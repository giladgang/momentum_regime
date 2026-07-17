"""Continuous per-stock band shaped by momentum (+pi), expanding window.

Momentum predicts trading cost: detrended half-spread is U-shaped in momentum
(losers 1.21, mid 0.94, winners 1.10 -> 29% dispersion, vs 5% for clusters).
Constantinides/Janecek-Shreve: optimal band width ~ spread^(1/3). So hold wide-
spread names longer (band them wider) and churn cheap names freely.

Per-stock band: E_it = clip(E_base * (pred_spread_it/med_t)^kappa
                            * exp(beta*(pi_t-0.5)), 5, 100).
pred_spread_it is PIT: each year, the mean detrended log half-spread by momentum
decile is fit on data STRICTLY BEFORE that year (expanding window), then mapped
to the current stocks by their momentum decile. kappa=1/3 is the theory value;
kappa=0 recovers a uniform band (sanity check).

Cost is the outcome. The shaped-band frontier (turnover, cost) is compared to
the uniform-band frontier AT MATCHED TURNOVER: does momentum-shaping cut cost at
equal trading? All 7 strategies, vs the momentum-spread ceiling.

Usage: .venv/bin/python -m paper.mom_pi_band
Output: paper/results/banding_study/mom_pi_band.{md,csv}
"""
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
from paper.cluster_cost_study import _cost_to                   # noqa: E402

E_BASE = [10, 15, 20, 30, 40, 60, 100]
KAPPA = [0.0, 1.0 / 3, 2.0 / 3]                  # 0 = uniform; 1/3 = Constantinides
BETA = [0.0, 0.5]                                # pi tilt (0 = momentum only)
NDEC = 10


def _pit_mom_spread(x, sp):
    """Per-stock-month predicted RELATIVE spread from a PIT expanding-window
    momentum-decile -> detrended-spread map. Returns a Series aligned to x.index
    of pred_spread/median (>=~1 for wide, <1 for cheap), 1.0 where unknown."""
    d = x[['date', 'permno', 'mom_12']].copy()
    d['ym'] = d['date'].dt.to_period('M')
    d = d.merge(sp, on=['permno', 'ym'], how='left')
    d['yr'] = d['date'].dt.year
    d['dec'] = d.groupby('date')['mom_12'].transform(
        lambda s: pd.qcut(s, NDEC, labels=False, duplicates='drop'))
    d['lhs'] = np.log(d['hs'].astype(float))
    d['lhs_d'] = d['lhs'] - d.groupby('yr')['lhs'].transform('mean')
    pred = pd.Series(1.0, index=x.index)
    for y in sorted(d['yr'].unique()):
        past = d[(d['yr'] < y) & d['lhs_d'].notna()]
        if len(past) < 2000:
            continue
        # relative spread by momentum decile (exp of mean detrended log-spread)
        m = np.exp(past.groupby('dec')['lhs_d'].mean())
        m = m / m.mean()
        cur = d[d['yr'] == y]
        pred.loc[cur.index] = cur['dec'].map(m).fillna(1.0).values
    return pred.clip(0.3, 3.0)


def _frontier(x, sp_pack, eband_series_fn, e_grid):
    """(turnover, cost) points for a band whose per-stock eband is
    E_base * shape (shape from eband_series_fn, a Series of multipliers)."""
    pts = []
    shape = eband_series_fn
    for eb in e_grid:
        xx = x.assign(eband=np.clip(eb * shape.values, 5, 100))
        r, led = X.simulate(xx, ('stock_var_band', None), score_col='score')
        c, to, _ = _cost_to(r, led, sp_pack)
        pts.append((to, c))
    return np.array(sorted(pts))


def main():
    sp_pack = B.load_spreads()
    sp = sp_pack[0]
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[(x['date'] >= C.SWEEP_START)
              & (x['date'] <= '2024-11-30')].reset_index(drop=True)
        pi = x['date'].map(x.groupby('date')['pi'].first())
        pred = _pit_mom_spread(x, sp)                 # PIT momentum->rel spread

        # uniform frontier (kappa=0, beta=0): shape == 1 everywhere
        uni = _frontier(x, sp_pack, pd.Series(1.0, index=x.index), E_BASE)
        uni_to, uni_cost = uni[:, 0], uni[:, 1]
        c0 = float(uni_cost.max())                    # ~monthly cost (E=10)

        for kappa in KAPPA:
            for beta in BETA:
                if kappa == 0.0 and beta == 0.0:
                    continue                          # == uniform, skip
                shape = (pred ** kappa) * np.exp(beta * (pi - 0.5))
                fr = _frontier(x, sp_pack, shape, E_BASE)
                # compare to uniform at matched turnover: cost saving
                best = None
                for to1, c1 in fr:
                    if to1 < uni_to.min() - 1e-9:
                        continue
                    uc = float(np.interp(to1, uni_to, uni_cost))
                    gap = uc - c1                     # positive = shaped cheaper
                    if best is None or gap > best[0]:
                        best = (gap, to1, c1, uc)
                gap, to_b, c_b, uc_b = best
                rows.append({
                    'strategy': s, 'kappa': round(kappa, 2), 'beta': beta,
                    'cost_base_bpyr': round(c0, 2),
                    'shaped_minus_uniform_bpyr': round(gap, 4),
                    'shaped_minus_uniform_pct': round(gap / c0 * 100, 2)
                    if c0 else 0.0,
                    'at_turnover_yr': round(to_b * 12, 3),
                    'shaped_cost_bpyr': round(c_b, 3),
                    'uniform_cost_bpyr': round(uc_b, 3)})
        bs = max((r for r in rows if r['strategy'] == s),
                 key=lambda r: r['shaped_minus_uniform_pct'])
        print(f"[mp] {s}: base {c0:.1f}bp/yr | best momentum-shaped band "
              f"(k={bs['kappa']},b={bs['beta']}) beats uniform@matched-turnover "
              f"by {bs['shaped_minus_uniform_bpyr']:+.3f}bp/yr = "
              f"{bs['shaped_minus_uniform_pct']:+.2f}%", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.BANDING_DIR, 'mom_pi_band.csv'), index=False)
    pd.set_option('display.width', 260)
    best_per = R.loc[R.groupby('strategy')['shaped_minus_uniform_pct'].idxmax()]
    lines = [
        '# Continuous per-stock band shaped by momentum (+pi), expanding window\n',
        'E_it = E_base*(pred_spread)^kappa*exp(beta*(pi-.5)); pred_spread from a'
        ' PIT expanding-window momentum-decile->spread map. kappa=1/3 ='
        ' Constantinides. Cost outcome; shaped vs uniform frontier at MATCHED'
        ' turnover. 2011..2024 (full sweep 1992+ where available).\n',
        '## Best shaped-minus-uniform per strategy\n',
        best_per[['strategy', 'kappa', 'beta', 'cost_base_bpyr',
                  'shaped_minus_uniform_bpyr', 'shaped_minus_uniform_pct',
                  'at_turnover_yr']].to_string(index=False), '',
        f'mean best shaped-minus-uniform: '
        f'{best_per["shaped_minus_uniform_bpyr"].mean():+.3f} bp/yr. '
        'Momentum-spread dispersion is 29% (vs 5% clusters), so the ceiling here'
        ' is higher -- this is the real test of whether that channel is'
        ' exploitable by a tradable, expanding-window band.', '',
        '## Full grid: paper/results/banding_study/mom_pi_band.csv']
    with open(os.path.join(C.BANDING_DIR, 'mom_pi_band.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
