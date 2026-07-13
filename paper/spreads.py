"""S5 spread engine: stock-month half-spreads + spread-priced verdict.

Primary spread = monthly median closing quoted half-spread ((ask-bid)/2/mid)
from dsf_v2; cross-check = Corwin-Schultz high-low estimator. Winsor
[1, 200] bps, missing -> cap-decile month median (registered conventions).

Usage: .venv/bin/python -m paper.spreads
Outputs: paper/results/s5/half_spreads.parquet, spread_stats.csv,
         spread_verdict.csv
"""
import glob
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from paper import config as C                                   # noqa: E402
from paper import execution as E                                # noqa: E402
from paper.src import metrics as M                              # noqa: E402

S5 = os.path.join(C.RESULTS, 's5')
SPREAD_PARQUET = os.path.join(S5, 'half_spreads.parquet')


def build_spreads():
    if os.path.exists(SPREAD_PARQUET):
        return pd.read_parquet(SPREAD_PARQUET)
    frames = []
    for f in sorted(glob.glob('experiments/results/spreads/dsf_v2_*.parquet')):
        d = pd.read_parquet(f, columns=['permno', 'dlycaldt', 'dlybid',
                                        'dlyask', 'dlyhigh', 'dlylow',
                                        'dlyprc'])
        d['ym'] = pd.to_datetime(d['dlycaldt']).dt.to_period('M')
        mid = (d['dlyask'] + d['dlybid']) / 2
        d['qhs'] = ((d['dlyask'] - d['dlybid']) / 2 / mid).where(
            (d['dlyask'] > d['dlybid']) & (mid > 0))
        # Corwin-Schultz on adjacent-day pairs
        d = d.sort_values(['permno', 'dlycaldt'])
        hl = np.log(d['dlyhigh'] / d['dlylow']) ** 2
        h2 = np.maximum(d['dlyhigh'], d.groupby('permno')['dlyhigh'].shift(1))
        l2 = np.minimum(d['dlylow'], d.groupby('permno')['dlylow'].shift(1))
        same = d['permno'].eq(d['permno'].shift(1))
        beta = (hl + hl.groupby(d['permno']).shift(1)).where(same)
        gamma = (np.log(h2 / l2) ** 2).where(same)
        k = 3 - 2 * np.sqrt(2)
        alpha = ((np.sqrt(2 * beta) - np.sqrt(beta)) / k
                 - np.sqrt(gamma / k))
        cs = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)
        d['cs_hs'] = cs / 2
        frames.append(d.groupby(['permno', 'ym'])
                      .agg(qhs=('qhs', 'median'), cs_hs=('cs_hs', 'mean'))
                      .reset_index())
        print(f'[spreads] {f}: ok', flush=True)
    sp = (pd.concat(frames)
          .groupby(['permno', 'ym']).mean().reset_index())
    for c in ('qhs', 'cs_hs'):
        sp[c] = sp[c].clip(1e-4, 0.02)
    sp['hs'] = sp['qhs'].fillna(sp['cs_hs'])
    sp.to_parquet(SPREAD_PARQUET, index=False)
    return sp


def main():
    os.makedirs(S5, exist_ok=True)
    sp = build_spreads()
    x = E.load_xsec()
    x['ym'] = x['date'].dt.to_period('M')
    # cap-decile month median fallback
    x['cap_dec'] = x.groupby('date')['me'].transform(
        lambda s: pd.qcut(s.rank(method='first'), 10, labels=False))
    xs = x.merge(sp[['permno', 'ym', 'hs']], on=['permno', 'ym'], how='left')
    fb = xs.groupby(['ym', 'cap_dec'])['hs'].transform('median')
    xs['hs'] = xs['hs'].fillna(fb).fillna(xs.groupby('ym')['hs']
                                          .transform('median'))
    hs_map = xs.set_index(['date', 'permno'])['hs']
    print(f"[spreads] coverage: {xs['hs'].notna().mean():.2%} | "
          f"median hs {xs['hs'].median() * 1e4:.1f}bp")
    pan = xs.groupby('date').first()['pi'] >= 0.5
    med = xs.groupby('date')['hs'].median() * 1e4
    stats = pd.DataFrame({
        'regime': ['calm', 'panic'],
        'median_hs_bp': [float(med[~pan].median()), float(med[pan].median())],
        'p90_hs_bp': [float((xs.groupby('date')['hs'].quantile(0.9) * 1e4)
                            [~pan].median()),
                      float((xs.groupby('date')['hs'].quantile(0.9) * 1e4)
                            [pan].median())]})
    stats.round(1).to_csv(os.path.join(S5, 'spread_stats.csv'), index=False)
    print(stats.round(1).to_string(index=False))

    br = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    bench = br[br.rule == 'rule_r'].set_index('date')['bench_ret']
    _, led_bench = E.simulate(x, ('benchmark', None))

    def spread_cost(ledger):
        m = ledger.merge(hs_map.rename('hs'), left_on=['date', 'permno'],
                         right_index=True, how='left')
        m['hs'] = m['hs'].fillna(xs['hs'].median())
        return (m['dw'].abs() * m['hs']).groupby(m['date']).sum()

    cost_bench = spread_cost(led_bench).reindex(bench.index).fillna(0.0)
    net_bench = bench - cost_bench

    finalists = [('pi_band(30,15)', ('pi_band', (30, 15)), 'score_pi'),
                 ('panic_no_sell', ('panic_no_sell', None), 'score_pi'),
                 ('calm_bench', ('calm_bench', None), 'score_pi'),
                 ('pi_monthly', ('monthly', None), 'score_pi'),
                 ('mom_band40', ('band', 40), 'mom_12'),
                 ('mom_band30', ('band', 30), 'mom_12'),
                 ('mom_monthly', ('monthly', None), 'mom_12')]
    rows, nets = [], {}
    for key, pol, sc in finalists:
        if pol[0] == 'regime_patient':
            pol[1]['state'] = {'on': False}
        r, led = E.simulate(x, pol, score_col=sc, cap=0.05)
        cost = spread_cost(led).reindex(r.index).fillna(0.0)
        net = r['gross'] - cost
        nets[key] = M.active(net, net_bench)
        rows.append({'strategy': key, 'to_mo': r['turnover'].mean(),
                     'gross_ir': M.ir(r['gross'], bench),
                     'cost_bp_mo': float(cost.mean() * 1e4),
                     'net_ir_spread': M.ir(net, net_bench),
                     'net_ann_active': float(nets[key].mean() * 12)})
    tab = pd.DataFrame(rows)
    cis = []
    for key in tab['strategy']:
        lo, hi = E.paired_block_bootstrap(nets[key], nets['mom_band40'])
        cis.append(f'[{lo * 12:+.3f},{hi * 12:+.3f}]')
    tab['d_vs_mom_band40_CI'] = cis
    tab = tab.round(3)
    tab.to_csv(os.path.join(S5, 'spread_verdict.csv'), index=False)
    print(tab.to_string(index=False))
    print('\nSaved ->', os.path.join(S5, 'spread_verdict.csv'))


if __name__ == '__main__':
    main()
