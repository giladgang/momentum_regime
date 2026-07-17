"""Direct COST saved by widening the no-trade band in each thesis cluster.

Cost is the outcome, measured directly (DNMV/NMV effective half-spread,
cost = sum|dw|*hs). Not IR (162x noise problem). Uses the THESIS k=4 cluster
labels (results/thesis/cluster_k4_member_dates.csv), all four clusters tested
one at a time, 2011-01..2024-11 (thesis label coverage).

For each strategy x cluster c x band level E:
  - widen the band to E ONLY in cluster-c months (baseline E=10 elsewhere),
  - measure cost saved vs baseline (raw bp/yr AND per treated-month),
  - measure turnover removed,
  - compare to the UNIFORM band that removes the SAME turnover (matched-turnover
    efficiency: does cluster-timing beat plain banding at equal turnover?),
  - all against the ~2.9% spread-timing ceiling (detrended cross-cluster spread
    ratio 1.052 -> cost timing capped at ~2.9%).

In-sample upper bound: thesis labels are a full-sample fit, so results are the
MAXIMUM a cluster-timed band could save with perfect hindsight of the cluster.

Usage: .venv/bin/python -m paper.cluster_cost_study
Output: paper/results/banding_study/cluster_cost.{md,csv}
"""
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

THESIS_LABELS = 'results/thesis/cluster_k4_member_dates.csv'
END = pd.Timestamp('2024-11-30')                 # thesis label coverage end
LEVELS = [10, 15, 20, 25, 30, 40, 60, 100]       # 10 = no widening (baseline)
UNIF_FINE = [10, 12, 15, 18, 22, 26, 30, 35, 40, 50, 60, 80, 100]
CLUSTER_NAME = {0: 'C0-winner', 1: 'C1', 2: 'C2', 3: 'C3-deeploser'}


def _cost_to(r, led, sp_pack):
    """(cost bp/yr, turnover/yr, net %/yr) for one simulated book."""
    cost, _, _ = B.price_ledger(led, sp_pack)
    cost = cost.reindex(r.index).fillna(0.0)
    net = B._net(r['gross'], cost)
    return (float(cost.mean() * 12 * 1e4), float(r['turnover'].mean() * 12),
            float(net.mean() * 12 * 100))


def _load_labels():
    lab = pd.read_csv(THESIS_LABELS, parse_dates=['date'])
    lab['date'] = lab['date'] + pd.offsets.MonthEnd(0)
    return lab.set_index('date')['cluster'].astype(int)


def main():
    sp_pack = B.load_spreads()
    L = _load_labels()
    rows = []
    for s in C.BS_STRATEGIES:
        x = signals.build(s)
        x = x[(x['date'] >= C.SWEEP_START) & (x['date'] <= END)].copy()
        # attach thesis cluster label; keep only labelled months
        x['cluster'] = x['date'].map(L)
        x = x[x['cluster'].notna()].reset_index(drop=True)
        x['cluster'] = x['cluster'].astype(int)
        months = x.groupby('date')['cluster'].first()
        share = {c: float((months == c).mean()) for c in range(4)}

        # baseline: no-op band every month
        r0, l0 = X.simulate(x, ('nmv_band', (10, 10)), score_col='score')
        c0, to0, n0 = _cost_to(r0, l0, sp_pack)
        # sanity: cluster_band with all-10 must equal baseline
        rc, lc = X.simulate(x, ('cluster_band', (10, 10, 10, 10)),
                            score_col='score')
        assert np.allclose(rc['gross'].values, r0['gross'].values), \
            'cluster_band(10,10,10,10) != baseline'

        # uniform frontier: (turnover removed, cost saved) over a fine E grid
        uni = []
        for E in UNIF_FINE:
            ru, lu = X.simulate(x, ('nmv_band', (E, E)), score_col='score')
            cu, tou, _ = _cost_to(ru, lu, sp_pack)
            uni.append((to0 - tou, c0 - cu))       # (turnover removed, cost saved)
        uni = np.array(sorted(uni))                # sort by turnover removed
        uni_x, uni_y = uni[:, 0], uni[:, 1]

        for c in range(4):
            n_c = int((months == c).sum())
            for E in LEVELS:
                arg = tuple(float(E if i == c else 10) for i in range(4))
                r1, l1 = X.simulate(x, ('cluster_band', arg), score_col='score')
                c1, to1, n1 = _cost_to(r1, l1, sp_pack)
                to_rm = to0 - to1
                cost_saved = c0 - c1
                # uniform cost saved at the SAME turnover removed (matched)
                uni_matched = (float(np.interp(to_rm, uni_x, uni_y))
                               if to_rm > 1e-9 else 0.0)
                rows.append({
                    'strategy': s, 'cluster': CLUSTER_NAME[c], 'E': E,
                    'share_mo': round(share[c], 3), 'n_mo_c': n_c,
                    'cost_base_bpyr': round(c0, 2),
                    'cost_saved_bpyr': round(cost_saved, 3),
                    'cost_saved_pct': round(cost_saved / c0 * 100, 2) if c0 else 0,
                    'saved_per_treated_mo_bp': round(
                        cost_saved / max(n_c, 1) * 12 * 100 / 12, 4),
                    'turnover_removed': round(to_rm, 4),
                    'uniform_saved_matched_bpyr': round(uni_matched, 3),
                    'cluster_minus_uniform_bpyr': round(cost_saved - uni_matched, 3),
                    'net_base_pctyr': round(n0, 3),
                    'net_delta_pctyr': round(n1 - n0, 3)})
        best = max((r for r in rows if r['strategy'] == s),
                   key=lambda r: r['cost_saved_pct'])
        print(f"[cc] {s}: base {c0:.1f}bp/yr | best single-cluster widen: "
              f"{best['cluster']} E={best['E']} saves {best['cost_saved_pct']:.1f}%"
              f" ({best['cost_saved_bpyr']:.2f}bp/yr); vs uniform-matched "
              f"{best['cluster_minus_uniform_bpyr']:+.3f}bp/yr", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(C.BANDING_DIR, 'cluster_cost.csv'), index=False)

    # summary: best cluster-minus-uniform per strategy, and vs 2.9% ceiling
    pd.set_option('display.width', 260)
    summ = (R.groupby('strategy')
            .apply(lambda d: pd.Series({
                'cost_base_bpyr': d['cost_base_bpyr'].iloc[0],
                'best_cluster_saved_pct': d['cost_saved_pct'].max(),
                'max_cluster_minus_uniform_bpyr':
                    d['cluster_minus_uniform_bpyr'].max(),
                'max_cluster_minus_uniform_pct': round(
                    d['cluster_minus_uniform_bpyr'].max()
                    / d['cost_base_bpyr'].iloc[0] * 100, 2)}))
            .reset_index())
    lines = [
        '# Direct cost saved by widening the band in each THESIS cluster\n',
        'Thesis k=4 labels, all clusters one at a time, 2011-01..2024-11. Cost is'
        ' the outcome (DNMV half-spread). Baseline = no-op band (E=10) every'
        ' month. cluster_minus_uniform = cluster saving MINUS the uniform band'
        ' that removes the same turnover (efficiency test). Ceiling: detrended'
        ' cross-cluster spread ratio 1.052 => cluster-timing capped at ~2.9%.\n',
        '## Per-strategy summary\n',
        summ.round(3).to_string(index=False), '',
        'If max_cluster_minus_uniform_pct <= ~2.9 for all strategies, cluster'
        ' timing adds nothing beyond plain banding at equal turnover, exactly as'
        ' the spread-timing ceiling predicts. If any exceeds it materially, the'
        ' ceiling is wrong (falsification).', '',
        '## Full grid: paper/results/banding_study/cluster_cost.csv'
        ' (7 strat x 4 clusters x 8 levels)']
    with open(os.path.join(C.BANDING_DIR, 'cluster_cost.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
