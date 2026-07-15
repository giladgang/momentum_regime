"""Walk-forward (rolling) band selection — the robust OOS test.

For each strategy, simulate all 125 band policies once over 1992-2025 and
capture each band's monthly net-of-measured-cost ACTIVE return. Then, for
each evaluation year Y >= START_OOS, RE-SELECT the best static band (best
diagonal by net IR on months < Y) and the best regime band (best
non-diagonal) using ONLY prior data, apply those bands during year Y, and
stitch the realized year-Y active returns into a fully out-of-sample series.

Reports, per strategy, the stitched-OOS net IR of monthly / static / regime,
the regime-vs-static delta, and a paired block-bootstrap CI. This replaces
the single 2011 split with ~25 annually-reselected OOS years (6 non-XGB
strategies); XGB (no pre-2011 scores) starts OOS in 2013 and is flagged.

Usage: .venv/bin/python -m paper.walkforward_banding
Output: paper/results/banding_study/walkforward.md, walkforward.csv
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
from paper.src import metrics as M                              # noqa: E402
from paper.src import signals                                   # noqa: E402

MONTHLY = ('nmv_band', (10, 10))       # tightest band = full monthly rebalance
START_OOS = 2001                        # >= ~9y training for non-XGB


def _band_active(x, sp_pack):
    """{policy_str: monthly net-of-cost active series} for all 125 bands."""
    b, bled = X.simulate(x, ('benchmark', None), score_col='score')
    bcost, _, _ = B.price_ledger(bled, sp_pack)
    bench_net = B._net(b['gross'], bcost)
    out = {}
    for pol in B.cell_grid():
        r, led = X.simulate(x, pol, score_col='score')
        cost, _, _ = B.price_ledger(led, sp_pack)
        net = B._net(r['gross'], cost)
        out[str(pol[1])] = (net - bench_net.reindex(net.index)).astype(float)
    return out, {str(p[1]): B.family(p) for p in B.cell_grid()}


def _complexity(col):
    """distinct band widths in the policy tuple: static=1, 2-state=2, 3=3."""
    return len(set(eval(col)))


def _walk(active, fam, start_oos):
    """Stitch annually-reselected OOS active. Two fits per year:
    - regime_argmax: raw argmax net-IR over the 120 non-diagonal bands.
    - regime_reg: 1-SE complexity-regularized -- among ALL bands within one
      Sharpe-SE of the best training IR, pick the SIMPLEST (fewest distinct
      widths), tie-broken by IR. Adopts regime-conditioning only when it
      clearly beats the simpler band, neutralizing the 24x search asymmetry.
    """
    idx = active[str(MONTHLY[1])].index
    yrs = [y for y in sorted({d.year for d in idx}) if y >= start_oos]
    rec = {'monthly': [], 'static': [], 'regime_argmax': [], 'regime_reg': []}
    picks = []
    A = pd.DataFrame(active)
    diag = [c for c in A.columns if fam[c] == 'diag']
    nond = [c for c in A.columns if fam[c] != 'diag']
    for y in yrs:
        tr, te = A[A.index.year < y], A[A.index.year == y]
        if len(tr) < 60 or len(te) == 0:
            continue

        def ir(col):
            a = tr[col].dropna()
            return (a.mean() / a.std() * np.sqrt(12)
                    if len(a) > 12 and a.std() > 0 else -np.inf)
        irs = {c: ir(c) for c in A.columns}
        sb = max(diag, key=lambda c: irs[c])            # static argmax
        rb = max(nond, key=lambda c: irs[c])            # regime argmax
        # 1-SE complexity-regularized pick over ALL bands
        best = irs[max(irs, key=lambda c: irs[c])]
        ny = max(len(tr) / 12, 1)
        se = np.sqrt((1 + 0.5 * best ** 2) / ny)        # Lo (2002) Sharpe SE
        within = [c for c in A.columns if irs[c] >= best - se]
        rr = min(within, key=lambda c: (_complexity(c), -irs[c]))
        rec['monthly'].append(te[str(MONTHLY[1])])
        rec['static'].append(te[sb])
        rec['regime_argmax'].append(te[rb])
        rec['regime_reg'].append(te[rr])
        picks.append({'year': y, 'static': sb, 'regime_argmax': rb,
                      'regime_reg': rr, 'reg_complexity': _complexity(rr)})
    stitched = {k: pd.concat(v).sort_index() for k, v in rec.items() if v}
    return stitched, pd.DataFrame(picks)


def main():
    sp_pack = B.load_spreads()
    rows = []
    for s in C.BS_STRATEGIES:
        try:
            x = signals.build(s)
        except Exception as e:                                  # noqa: BLE001
            print(f'[skip] {s}: {e!r}', flush=True)
            continue
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = START_OOS if s != 'xgb' else 2013
        active, fam = _band_active(x, sp_pack)
        stitched, picks = _walk(active, fam, start)
        if 'regime_reg' not in stitched:
            print(f'[skip] {s}: insufficient OOS window', flush=True)
            continue
        mo, st = stitched['monthly'], stitched['static']
        ra, rr = stitched['regime_argmax'], stitched['regime_reg']
        z = pd.Series(0.0, index=st.index)

        def _ci(a):
            d = (a - st).dropna()
            lo, hi = X.paired_block_bootstrap(
                a.reindex(d.index).astype(float),
                st.reindex(d.index).astype(float))
            return lo, hi
        la, ha = _ci(ra)
        lr, hr = _ci(rr)
        rows.append({
            'strategy': s, 'oos_start': start, 'oos_months': len(st),
            'monthly_ir': M.ir(mo, z), 'static_ir': M.ir(st, z),
            'regime_argmax_ir': M.ir(ra, z), 'regime_reg_ir': M.ir(rr, z),
            'argmax_minus_static': M.ir(ra, z) - M.ir(st, z),
            'reg_minus_static': M.ir(rr, z) - M.ir(st, z),
            'argmax_excl0': bool(la > 0 or ha < 0),
            'reg_excl0': bool(lr > 0 or hr < 0),
            'reg_ci_x12': f'[{lr*12:+.3f},{hr*12:+.3f}]',
            'reg_pct_3state': (float((picks['reg_complexity'] == 3).mean())
                               if len(picks) else np.nan),
            'n_reselections': len(picks)})
        print(f'[wf] {s}: monthly {rows[-1]["monthly_ir"]:.3f} | static '
              f'{rows[-1]["static_ir"]:.3f} | regime_argmax '
              f'{rows[-1]["regime_argmax_ir"]:.3f} (d{rows[-1]["argmax_minus_static"]:+.3f}) '
              f'| regime_reg {rows[-1]["regime_reg_ir"]:.3f} '
              f'(d{rows[-1]["reg_minus_static"]:+.3f}, excl0 '
              f'{rows[-1]["reg_excl0"]})', flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(C.BANDING_DIR, exist_ok=True)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'walkforward.csv'),
                      index=False)
    n_argmax = int((R['argmax_minus_static'] > 0).sum())
    n_reg = int((R['reg_minus_static'] > 0).sum())
    lines = [
        '# Walk-forward (rolling annual re-selection) banding results\n',
        f'Expanding window, band re-selected EVERY YEAR on all prior months '
        f'(non-XGB OOS from {START_OOS}; XGB from 2013). Stitched true-OOS '
        f'net-of-measured-cost active return; supersedes the single 2011 '
        f'split. Two fits: regime_argmax (raw argmax over 120 bands) vs '
        f'regime_reg (1-SE complexity-regularized -- adopt conditioning only '
        f'when it clearly beats the simpler band).\n',
        R.round(3).to_string(index=False), '',
        f'Regime beats static (stitched-OOS IR): argmax fit {n_argmax}/{len(R)}'
        f', REGULARIZED fit {n_reg}/{len(R)} '
        f'({int(R["reg_excl0"].sum())}/{len(R)} with bootstrap CI excluding 0). '
        f'The regularized column is the honest number: it neutralizes the '
        f'24x search asymmetry between the 120-cell regime grid and the '
        f'5-cell static grid.',
    ]
    with open(os.path.join(C.BANDING_DIR, 'walkforward.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
