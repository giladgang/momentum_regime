"""S6: regime-conditional NMV banding (spec: docs/superpowers/specs/
2026-07-14-s6-regime-banding-design.md — expectations registered pre-run).

Grid of (E_calm, E_panic) buy/hold spreads over the main-model walk xsecs.
Enter at the decile, hold until rank falls out of E, count floats.
GATE: nmv_band(10,10) must reproduce walk_returns.csv gross exactly.

Usage: .venv/bin/python -m paper.banding   (-> paper/results/s6_banding/)
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
from paper.src import metrics as M                              # noqa: E402

S6 = os.path.join(C.RESULTS, 's6_banding')
E_GRID = [10, 15, 20, 30, 40]


def main():
    os.makedirs(S6, exist_ok=True)
    x = X.load_xsec()
    walk = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    walk = walk[walk.rule == 'rule_r'].set_index('date')
    bench = walk['bench_ret']

    # ── GATE: (10,10) == monthly == the walk, bit-exact ──
    base, _ = X.simulate(x, ('nmv_band', (10, 10)))
    err = (base['gross'] - walk['strat_ret'].reindex(base.index)).abs().max()
    assert err < 1e-10, f'GATE FAIL: nmv_band(10,10) vs walk max err {err}'
    print(f'[gate] nmv_band(10,10) reproduces the walk (max err {err:.1e})')
    benchmark, _ = X.simulate(x, ('benchmark', None))
    net_bench_to = benchmark['turnover']

    # ── 5x5 grid ──
    results = {}
    for ec in E_GRID:
        for ep in E_GRID:
            r, _ = X.simulate(x, ('nmv_band', (ec, ep)))
            results[(ec, ep)] = r
            print(f'[sim] E_calm={ec} E_panic={ep}: '
                  f'TO {r.turnover.mean():.2%}/mo  n {r.n_names.mean():.0f}',
                  flush=True)

    rows, net_act10 = [], {}
    for (ec, ep), r in results.items():
        act_g = r['gross'] - bench.reindex(r.index)
        row = {'e_calm': ec, 'e_panic': ep,
               'kind': 'diag' if ec == ep else
               ('tight_panic' if ep < ec else 'tight_calm'),
               'to_mo': r['turnover'].mean(),
               'n_names': r['n_names'].mean(),
               'gross_ir': M.ir(r['gross'], bench)}
        for c in X.FLAT_BPS:
            net = r['gross'] - r['traded'] * c / 1e4
            net_b = bench - net_bench_to.reindex(bench.index) * 2 * c / 1e4
            row[f'net_ir_{c}bp'] = M.ir(net, net_b)
            if c == 10:
                net_act10[(ec, ep)] = M.active(net, net_b)
        row['breakeven_bp'] = (act_g.mean() /
                               (r['traded'].mean()
                                - 2 * net_bench_to.mean()) * 1e4
                               if r['traded'].mean() > 2 * net_bench_to.mean()
                               else np.inf)
        # regime split of gross active + turnover
        pi = r['pi'] >= 0.5
        row['act_calm_mo'] = act_g[~pi].mean()
        row['act_panic_mo'] = act_g[pi].mean()
        row['to_calm_mo'] = r['turnover'][~pi].mean()
        row['to_panic_mo'] = r['turnover'][pi].mean()
        rows.append(row)
    cells = pd.DataFrame(rows)
    cells.round(4).to_csv(os.path.join(S6, 'cells.csv'), index=False)

    pivot = cells.pivot(index='e_calm', columns='e_panic',
                        values='net_ir_10bp').round(3)
    pivot.to_csv(os.path.join(S6, 'grid_net_ir.csv'))

    # ── best cells + paired bootstrap ──
    diag = cells[cells.kind == 'diag']
    offd = cells[cells.kind != 'diag']
    bd = diag.loc[diag['net_ir_10bp'].idxmax()]
    bo = offd.loc[offd['net_ir_10bp'].idxmax()]
    kd = (int(bd.e_calm), int(bd.e_panic))
    ko = (int(bo.e_calm), int(bo.e_panic))
    ci_diag = X.paired_block_bootstrap(net_act10[ko], net_act10[kd])
    ci_mon = X.paired_block_bootstrap(net_act10[ko], net_act10[(10, 10)])

    # ── expectations (registered 2026-07-14, pre-run) ──
    e1 = bo.e_panic < bo.e_calm
    e2 = bo.net_ir_10bp - bd.net_ir_10bp >= 0.03
    # E3: vs matched-panic diagonal (ep, ep): calm turnover falls,
    # panic active within noise (|d| < its own monthly std/sqrt(n))
    md = cells[(cells.e_calm == bo.e_panic) & (cells.e_panic == bo.e_panic)].iloc[0]
    d_act_p = bo.act_panic_mo - md.act_panic_mo
    n_pan = int((results[ko]['pi'] >= 0.5).sum())
    noise = (results[ko]['gross'] - bench.reindex(results[ko].index))[
        results[ko]['pi'] >= 0.5].std() / np.sqrt(n_pan)
    e3 = (bo.to_calm_mo < md.to_calm_mo) and (abs(d_act_p) < noise)
    e4_zero = ci_diag[0] <= 0 <= ci_diag[1]

    lines = [
        '# S6: regime-conditional NMV banding — results\n',
        f'Gate: nmv_band(10,10) == walk, max err {err:.1e}. '
        f'Main model only; grid E_calm x E_panic {E_GRID}; '
        'net = gross - traded*bps; benchmark pays reconstitution.\n',
        '## Net IR @10bp (rows E_calm, cols E_panic; diag = unconditional)\n',
        pivot.to_string(), '',
        f'Best diagonal: ({kd[0]},{kd[1]})  net IR@10bp {bd.net_ir_10bp:.3f}, '
        f'TO {bd.to_mo:.2%}/mo, breakeven {bd.breakeven_bp:.0f}bp',
        f'Best off-diag: ({ko[0]},{ko[1]})  net IR@10bp {bo.net_ir_10bp:.3f}, '
        f'TO {bo.to_mo:.2%}/mo, breakeven {bo.breakeven_bp:.0f}bp\n',
        f'Paired CI (best offdiag - best diag, net active@10bp, x12): '
        f'[{ci_diag[0]*12:+.3f},{ci_diag[1]*12:+.3f}]',
        f'Paired CI (best offdiag - monthly): '
        f'[{ci_mon[0]*12:+.3f},{ci_mon[1]*12:+.3f}]\n',
        '## Regime split (best offdiag vs matched-panic diagonal '
        f'({int(md.e_calm)},{int(md.e_panic)}))\n',
        f'calm TO {bo.to_calm_mo:.2%} vs {md.to_calm_mo:.2%} | '
        f'panic TO {bo.to_panic_mo:.2%} vs {md.to_panic_mo:.2%} | '
        f'calm act {bo.act_calm_mo:+.3%} vs {md.act_calm_mo:+.3%} | '
        f'panic act {bo.act_panic_mo:+.3%} vs {md.act_panic_mo:+.3%} '
        f'(d {d_act_p:+.3%}, noise {noise:.3%})\n',
        '## Registered expectations\n',
        f'E1 direction (best offdiag tighter in panic): '
        f'{"HIT" if e1 else "MISS"} (best offdiag = {ko})',
        f'E2 value-add >= +0.03 net IR@10bp over best diag: '
        f'{"HIT" if e2 else "MISS"} '
        f'({bo.net_ir_10bp:.3f} vs {bd.net_ir_10bp:.3f}, '
        f'd={bo.net_ir_10bp - bd.net_ir_10bp:+.3f})',
        f'E3 mechanism (calm TO falls, panic act within noise vs '
        f'({int(md.e_calm)},{int(md.e_panic)})): {"HIT" if e3 else "MISS"}',
        f'E4 CI vs best diag includes zero: {"YES" if e4_zero else "NO"} '
        f'-> claim is {"leads the frontier" if e4_zero else "dominates"}\n',
    ]
    text = '\n'.join(lines)
    with open(os.path.join(S6, 'report.md'), 'w') as f:
        f.write(text + '\n')
    print(text)
    print('Saved:', S6)


if __name__ == '__main__':
    main()
