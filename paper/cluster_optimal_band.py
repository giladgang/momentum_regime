"""Jointly OPTIMAL per-cluster band: fit (E_0,E_1,E_2,E_3) together to minimize
cost at matched turnover, vs the best uniform band. Thesis k=4 labels, cost as
outcome, 2011-01..2024-11.

This is the STRONGEST case for cluster-conditioning: perfect-hindsight thesis
labels AND a brute-force-optimized per-cluster band. If even this cannot beat a
uniform band at matched turnover by more than the ~2.9% spread-timing ceiling,
no real-time or heuristic version can.

Cost and turnover both fall as bands widen, so the meaningful control is MATCHED
turnover: at equal trading, does an optimal per-cluster allocation reach lower
cost than uniform (by trading in cheaper clusters)? For each per-cluster combo
(turnover t, cost c) we compare c to the uniform band's cost at the same t. The
largest advantage is the best the optimized per-cluster band can do.

Usage: .venv/bin/python -m paper.cluster_optimal_band
Output: paper/results/banding_study/cluster_optimal.{md,csv}
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import execution as X                                # noqa: E402
from paper import banding_study as B                            # noqa: E402
from paper.src import signals                                   # noqa: E402
from paper.cluster_cost_study import _load_labels, _cost_to, END  # noqa: E402

LEVELS = [10, 20, 35, 60, 100]                   # per-cluster band grid (5^4=625)
UNIF_FINE = [10, 12, 15, 18, 22, 26, 30, 35, 40, 50, 60, 80, 100]


def main():
    sp_pack = B.load_spreads()
    L = _load_labels()
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[(x['date'] >= C.SWEEP_START) & (x['date'] <= END)].copy()
        x['cluster'] = x['date'].map(L)
        x = x[x['cluster'].notna()].reset_index(drop=True)
        x['cluster'] = x['cluster'].astype(int)

        r0, l0 = X.simulate(x, ('nmv_band', (10, 10)), score_col='score')
        c0, to0, _ = _cost_to(r0, l0, sp_pack)

        # uniform frontier (turnover asc, cost asc)
        uni = []
        for E in UNIF_FINE:
            ru, lu = X.simulate(x, ('nmv_band', (E, E)), score_col='score')
            cu, tou, _ = _cost_to(ru, lu, sp_pack)
            uni.append((tou, cu))
        uni = np.array(sorted(uni))
        uni_to, uni_cost = uni[:, 0], uni[:, 1]

        # brute-force joint per-cluster band; compare each to uniform at same turnover
        best = None                                   # (gap_bp, vec, to, c, uc)
        zero_gap = None
        for vec in itertools.product(LEVELS, repeat=4):
            r1, l1 = X.simulate(x, ('cluster_band', tuple(float(v) for v in vec)),
                                score_col='score')
            c1, to1, _ = _cost_to(r1, l1, sp_pack)
            if to1 < uni_to.min() - 1e-9:             # outside uniform range: skip
                continue
            uc = float(np.interp(to1, uni_to, uni_cost))
            gap = uc - c1                             # positive = cluster cheaper
            if vec == (10, 10, 10, 10):
                zero_gap = gap                        # sanity: must be ~0
            if best is None or gap > best[0]:
                best = (gap, vec, to1, c1, uc)
        assert zero_gap is None or abs(zero_gap) < 1e-6, \
            f'{s}: all-10 combo gap {zero_gap} != 0 (baseline mismatch)'

        gap_bp, vec, to_b, c_b, uc_b = best
        rows.append({
            'strategy': s, 'cost_base_bpyr': round(c0, 2),
            'opt_vec_E0_E1_E2_E3': str(vec),
            'max_opt_minus_uniform_bpyr': round(gap_bp, 4),
            'max_opt_minus_uniform_pct': round(gap_bp / c0 * 100, 2) if c0 else 0,
            'at_turnover_yr': round(to_b * 12, 3),
            'opt_cost_bpyr': round(c_b, 3), 'uniform_cost_bpyr': round(uc_b, 3)})
        print(f"[opt] {s}: base {c0:.1f}bp/yr | best OPTIMAL per-cluster band "
              f"{vec} beats uniform@matched-turnover by {gap_bp:+.3f}bp/yr = "
              f"{gap_bp/c0*100:+.2f}% (ceiling ~2.9%)", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.BANDING_DIR, 'cluster_optimal.csv'), index=False)
    pd.set_option('display.width', 260)
    lines = [
        '# Jointly OPTIMAL per-cluster band vs uniform (cost, matched turnover)\n',
        'Brute-force (E_0..E_3) in {10,20,35,60,100}^4 per strategy; each combo'
        ' compared to the uniform band at the SAME turnover. Thesis k=4 labels,'
        ' 2011-2024, in-sample (perfect-hindsight) UPPER BOUND -- the strongest'
        ' possible case for cluster-conditioning.\n',
        R.to_string(index=False), '',
        'max_opt_minus_uniform_pct = the MOST an optimally-fit per-cluster band'
        ' beats a uniform band at equal turnover, as % of cost. <= ~2.9% for all'
        ' => optimizing the per-cluster band adds nothing beyond uniform banding'
        ' even with perfect hindsight; the spread-timing ceiling binds regardless'
        ' of how the band is fit. Any material exceedance => ceiling falsified.',
        '', 'opt_vec = the winning (E_C0, E_C1, E_C2, E_C3).']
    with open(os.path.join(C.BANDING_DIR, 'cluster_optimal.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
