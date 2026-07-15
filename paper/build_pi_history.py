"""Monthly DD-only regime pi, 1991-2025.

2011-2025: reuse the applied walk's pi verbatim (from the xsec files via
execution.load_xsec) so every published applied number is untouched.
1991-2010: backfill with the same recipe the walk uses - expanding annual
refits of the 1-feature DD HMM (train < Jan-Y, infer year Y), mean pi over
SEL_HMM_SEEDS, N_ITER/N_BURNIN from config. Seeds logged in the output.

Usage: .venv/bin/python -m paper.build_pi_history [--start 1991 --end 2010]
Output: paper/results/data/pi_monthly.parquet  (date, pi, src)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper.src import hmm                                       # noqa: E402


def assemble(back, walk):
    back = back.assign(src='backfill')
    walk = walk.assign(src='walk')
    out = pd.concat([back[~back['date'].isin(set(walk['date']))], walk])
    return out.sort_values('date').reset_index(drop=True)


def _backfill(years, workers):
    panel = pd.read_parquet(C.PANEL_PARQUET)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = (panel.dropna(subset=['DD_z'])
             .sort_values('date').reset_index(drop=True))
    # DD-anchored sign rule (panic = deeper-drawdown state): the
    # selection.run_cell convention for DD-only combos. crisis_signs is NOT
    # usable here - CRISIS_WINDOWS start in 2000, so 1991-2000 training
    # windows contain no crisis months.
    signs = np.array([-1.0])
    rows = []
    for y in years:
        # full history from 1970 (no PANEL_START floor - early years need it)
        sub = panel[panel['date'] < f'{y + 1}-01-01'].reset_index(drop=True)
        tr = sub['date'] < f'{y}-01-01'
        Z_tr = sub.loc[tr, ['DD_z']].values.astype(float)
        Z_full = sub[['DD_z']].values.astype(float)
        pi = hmm.fit_pi(Z_tr, Z_full, signs, C.SEL_HMM_SEEDS, workers,
                        n_iter=C.N_ITER, n_burnin=C.N_BURNIN)
        yr = (sub['date'].dt.year == y).values
        rows.append(pd.DataFrame({'date': sub.loc[yr, 'date'],
                                  'pi': np.asarray(pi)[yr]}))
        print(f'[pi] {y}: panic months '
              f'{(rows[-1]["pi"] >= 0.5).sum()}/12', flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1991)
    ap.add_argument('--end', type=int, default=2010)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()

    from paper import execution as X
    x = X.load_xsec()
    walk = (x.groupby('date', as_index=False)['pi'].first())

    back = _backfill(range(a.start, a.end + 1), a.workers)
    out = assemble(back, walk)
    os.makedirs(C.DATA_OUT, exist_ok=True)
    out.to_parquet(C.PI_MONTHLY, index=False)
    print(f'Saved {C.PI_MONTHLY}: {len(out)} months '
          f'{out.date.min().date()} -> {out.date.max().date()} | '
          f'panic share {(out.pi >= 0.5).mean():.2%} | '
          f'seeds {C.SEL_HMM_SEEDS}')


if __name__ == '__main__':
    main()
