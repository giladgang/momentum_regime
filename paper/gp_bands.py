"""Principled (Garleanu-Pedersen / Constantinides) regime banding — no grid.

Instead of grid-searching the regime band widths (which overfits: 120 regime
cells vs 5 static), DERIVE the panic/calm band-width ratio from theory. For
proportional costs the optimal no-trade half-width scales as
    w ~ (spread / (gamma * variance))^(1/3),
so the parameter-free panic/calm ratio is
    ratio = (spread_panic/spread_calm * var_calm/var_panic)^(1/3).
Only ONE global scale (E_base) is fit -- the SAME one-parameter freedom as the
static band. If the theory-scaled regime band beats static out-of-sample under
this matched search, the regime effect is real (not search inflation).

Walk-forward: expanding window, band re-selected each year on prior data;
ratio estimated on prior data too (no look-ahead). Compares monthly / static /
theory-regime, stitched true-OOS, with a paired block-bootstrap CI.

Usage: .venv/bin/python -m paper.gp_bands
Output: paper/results/banding_study/gp_bands.md, gp_bands.csv
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

E_BASE = [10, 15, 20, 30, 40]           # single global scale (matched search)
MONTHLY = ('nmv_band', (10, 10))
START_OOS = 2001


def _active(x, policy, sp_pack, bench_net):
    r, led = X.simulate(x, policy, score_col='score')
    cost, _, _ = B.price_ledger(led, sp_pack)
    net = B._net(r['gross'], cost)
    return (net - bench_net.reindex(net.index)).astype(float)


def _regime_ratio(x, sp_pack, panel_vw, cutoff):
    """Theory panic/calm band ratio from prior-data spread & variance."""
    _, month_med, _ = sp_pack
    m = x[['date', 'pi']].drop_duplicates().copy()
    m = m[m['date'] < cutoff]
    m['ym'] = m['date'].dt.to_period('M')
    m['hs'] = m['ym'].map(month_med)
    m = m.merge(panel_vw, on='date', how='left').dropna(subset=['vwretd'])
    p = m['pi'] >= 0.5
    if p.sum() < 6 or (~p).sum() < 6:
        return 1.0
    spread_ratio = m.loc[p, 'hs'].mean() / m.loc[~p, 'hs'].mean()
    var_ratio = m.loc[~p, 'vwretd'].var() / m.loc[p, 'vwretd'].var()
    return float((spread_ratio * var_ratio) ** (1.0 / 3.0))


def _walk_two(static_act, theory_act, monthly_act, start_oos):
    idx = monthly_act.index
    yrs = [y for y in sorted({d.year for d in idx}) if y >= start_oos]
    S = pd.DataFrame(static_act)
    T = pd.DataFrame(theory_act)
    rec = {'monthly': [], 'static': [], 'theory': []}
    for y in yrs:
        trS, teS = S[S.index.year < y], S[S.index.year == y]
        trT, teT = T[T.index.year < y], T[T.index.year == y]
        if len(trS) < 60 or len(teS) == 0:
            continue

        def ir(fr, col):
            a = fr[col].dropna()
            return (a.mean() / a.std() * np.sqrt(12)
                    if len(a) > 12 and a.std() > 0 else -np.inf)
        sb = max(S.columns, key=lambda c: ir(trS, c))
        tb = max(T.columns, key=lambda c: ir(trT, c))
        rec['monthly'].append(monthly_act[monthly_act.index.year == y])
        rec['static'].append(teS[sb])
        rec['theory'].append(teT[tb])
    return {k: pd.concat(v).sort_index() for k, v in rec.items() if v}


def main():
    sp_pack = B.load_spreads()
    panel_vw = pd.read_parquet(C.PANEL_PARQUET, columns=['date', 'vwretd'])
    panel_vw['date'] = pd.to_datetime(panel_vw['date'])
    rows = []
    for s in C.BS_STRATEGIES:
        try:
            x = signals.build(s)
        except Exception as e:                                  # noqa: BLE001
            print(f'[skip] {s}: {e!r}', flush=True)
            continue
        x = x[x['date'] >= C.SWEEP_START].reset_index(drop=True)
        start = START_OOS if s != 'xgb' else 2013
        ratio = _regime_ratio(x, sp_pack, panel_vw, f'{start}-01-01')
        b, bled = X.simulate(x, ('benchmark', None), score_col='score')
        bcost, _, _ = B.price_ledger(bled, sp_pack)
        bench_net = B._net(b['gross'], bcost)

        static_act, theory_act = {}, {}
        for e in E_BASE:
            static_act[f's{e}'] = _active(x, ('nmv_band', (e, e)),
                                          sp_pack, bench_net)
            ep = float(np.clip(e * ratio, 5.0, 60.0))
            theory_act[f't{e}'] = _active(x, ('nmv_band', (e, ep)),
                                          sp_pack, bench_net)
        monthly_act = _active(x, MONTHLY, sp_pack, bench_net)
        st = _walk_two(static_act, theory_act, monthly_act, start)
        if 'theory' not in st:
            print(f'[skip] {s}: short OOS', flush=True)
            continue
        mo, sa, th = st['monthly'], st['static'], st['theory']
        z = pd.Series(0.0, index=sa.index)
        d = (th - sa).dropna()
        lo, hi = X.paired_block_bootstrap(th.reindex(d.index).astype(float),
                                          sa.reindex(d.index).astype(float))
        rows.append({
            'strategy': s, 'panic_ratio': ratio, 'oos_months': len(sa),
            'monthly_ir': M.ir(mo, z), 'static_ir': M.ir(sa, z),
            'theory_ir': M.ir(th, z),
            'theory_minus_static': M.ir(th, z) - M.ir(sa, z),
            'ci_x12': f'[{lo*12:+.3f},{hi*12:+.3f}]',
            'excl0': bool(lo > 0 or hi < 0)})
        print(f'[gp] {s}: ratio {ratio:.2f} ({"tighter" if ratio<1 else "wider"}'
              f'-in-panic) | static {rows[-1]["static_ir"]:.3f} | theory '
              f'{rows[-1]["theory_ir"]:.3f} (d{rows[-1]["theory_minus_static"]:+.3f}'
              f', excl0 {rows[-1]["excl0"]})', flush=True)
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(C.BANDING_DIR, 'gp_bands.csv'), index=False)
    npos = int((R['theory_minus_static'] > 0).sum())
    lines = [
        '# Garleanu-Pedersen / Constantinides analytic regime bands (no grid)\n',
        'Panic/calm band ratio derived from prior-data (spread, variance); '
        'only ONE scale E_base fit, matched to the static band. Walk-forward, '
        'expanding, OOS from 2001 (xgb 2013).\n',
        R.round(3).to_string(index=False), '',
        f'Theory-regime beats static OOS for {npos}/{len(R)} strategies; '
        f'{int(R["excl0"].sum())}/{len(R)} with bootstrap CI excluding 0. '
        f'ratio<1 => theory says trade MORE in panic (variance spike beats the '
        f'modest spread spike); ratio>1 => trade less.',
    ]
    with open(os.path.join(C.BANDING_DIR, 'gp_bands.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
