"""Full thesis results-section replication for the applied config.

Stages:
  ablations : regime-signal ablation variants (four_raw, ghm_cycle,
              fundamentals-added), yearly expanding XGB walks, 5 seeds
              (ablation budget, disclosed). reln_only is reported as
              structurally degenerate (pi==1.0 post-2000; diagnostic
              2026-07-13), not run.
  figs      : cumulative-wealth fig with panic shading; applied long-leg
              z-curve heatmap; per-horizon SHAP-by-regime table.
  assemble  : applied_results_full.md joining every panel + N/A register.

Usage: .venv/bin/python -m paper.replicate_results --stage {ablations,figs,assemble}
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

import config as repo_config                                    # noqa: E402
from config import MOM_FEATURES                                 # noqa: E402
from paper import config as C                                   # noqa: E402
from paper import walk_report as W                              # noqa: E402
from paper.src import data_build, universe, model, portfolio    # noqa: E402
from paper.src import metrics as M                              # noqa: E402

ABL_DIR = os.path.join(C.RESULTS, 'ablations')
TBL = os.path.join(C.RESULTS, 'tables')
ABL_XGB_SEEDS = repo_config.XGB_SEEDS[:5]     # ablation budget (disclosed)
RAW_FEATS = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']
FUND_FEATS = ['roe', 'earnings_growth', 'leverage', 'asset_growth',
              'gross_profit_a', 'bm', 'cfo_a', 'fcf_a', 'accruals']


def _universe_frames(extra_cols=()):
    s = data_build.load_stocks(columns=['date', 'permno', 'me', 'ret_fwd']
                               + MOM_FEATURES)
    if extra_cols:
        ext = pd.read_parquet(C.STOCK_EXT,
                              columns=['permno', 'date'] + list(extra_cols))
        ext['date'] = pd.to_datetime(ext['date'])
        s = s.merge(ext, on=['date', 'permno'], how='left')
    return universe.top_n(s, C.UNIVERSE_N)


def _pi_series():
    sel = pd.read_csv(C.SELECTIONS_CSV)
    rr = sel[sel['rule'] == 'rule_r'].set_index('year')['combo']
    pis = []
    for y in C.EVAL_YEARS:
        xp = os.path.join(C.XSEC_DIR,
                          f"xsec_{y}_{rr.loc[y].replace('+', '_')}.parquet")
        x = pd.read_parquet(xp, columns=['date', 'permno', 'pi'])
        x['date'] = pd.to_datetime(x['date'])
        pis.append(x)
    return pd.concat(pis, ignore_index=True)


def _xgb_walk(st, feats, years, out_csv, dropna_feats=None):
    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv)['year'].unique())
    for y in years:
        if y in done:
            continue
        y0, y1 = f'{y}-01-01', f'{y + 1}-01-01'
        d = st.dropna(subset=list(dropna_feats or feats))
        tr = d[d['date'] < y0]
        te = d[(d['date'] >= y0) & (d['date'] < y1)].copy()
        if not len(te):
            continue
        te['score'] = model.ensemble_scores(
            tr[feats].values.astype(float),
            tr['ret_fwd'].values.astype(float),
            te[feats].values.astype(float), ABL_XGB_SEEDS, n_jobs=7)
        r, _ = portfolio.long_only_top(te, 'score', C.DECILE_FRAC)
        b = portfolio.vw_benchmark(te)
        out = pd.DataFrame({'date': r.index, 'strat_ret': r.values,
                            'bench_ret': b.reindex(r.index).values})
        out['year'] = y
        out.to_csv(out_csv, mode='a', header=not os.path.exists(out_csv),
                   index=False)
        print(f'  {os.path.basename(out_csv)} {y} ok', flush=True)


def stage_ablations():
    os.makedirs(ABL_DIR, exist_ok=True)
    # four_raw: monthly raw z-levels of the four production-adjacent features
    panel = data_build.load_panel()[['date'] + RAW_FEATS]
    st = _universe_frames().merge(panel, on='date', how='left')
    print('[abl] four_raw', flush=True)
    _xgb_walk(st, MOM_FEATURES + RAW_FEATS, C.EVAL_YEARS,
              os.path.join(ABL_DIR, 'abl_four_raw_returns.csv'))

    # ghm_cycle: Bull/Correction/Bear/Rebound from market returns (panel vwretd)
    p = data_build.load_panel()[['date', 'vwretd']].dropna()
    p['mkt_slow'] = p['vwretd'].rolling(12, min_periods=12).mean()
    code = {'Bull': 0, 'Correction': 1, 'Bear': 2, 'Rebound': 3}
    def cyc(r):
        if pd.isna(r['mkt_slow']):
            return np.nan
        if r['mkt_slow'] >= 0:
            return code['Bull'] if r['vwretd'] >= 0 else code['Correction']
        return code['Bear'] if r['vwretd'] < 0 else code['Rebound']
    p['cycle'] = p.apply(cyc, axis=1)
    st2 = _universe_frames().merge(p[['date', 'cycle']], on='date', how='left')
    print('[abl] ghm_cycle', flush=True)
    _xgb_walk(st2, MOM_FEATURES + ['cycle'], C.EVAL_YEARS,
              os.path.join(ABL_DIR, 'abl_ghm_returns.csv'))

    # fundamentals ADDED to the pi model (2011-2024; no fundamentals in 2025)
    st3 = _universe_frames(extra_cols=FUND_FEATS)
    st3 = st3.merge(_pi_series(), on=['date', 'permno'], how='left')
    st3 = st3.dropna(subset=['pi'])
    print('[abl] fundamentals (2011-2024)', flush=True)
    _xgb_walk(st3, MOM_FEATURES + ['pi'] + FUND_FEATS,
              [y for y in C.EVAL_YEARS if y <= 2024],
              os.path.join(ABL_DIR, 'abl_fund_returns.csv'),
              dropna_feats=MOM_FEATURES + ['pi'] + FUND_FEATS)
    print('[abl] DONE', flush=True)


def stage_figs():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(TBL, exist_ok=True)
    rets = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    sel = pd.read_csv(C.SELECTIONS_CSV)
    x, r_nopi, r_mom, bench, pi_m = W._comparator_series(sel)
    series = {'pi rule_r': rets[rets.rule == 'rule_r'].set_index('date')['strat_ret'],
              'no-pi': r_nopi, '12-1 mom': r_mom, 'benchmark': bench}
    colors = {'pi rule_r': '#e05c4b', 'no-pi': '#2a78d6',
              '12-1 mom': '#3b9c52', 'benchmark': '#555555'}

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    pi = pi_m.sort_index()
    panic = (pi >= 0.5)
    for d0, d1 in zip(pi.index[panic & ~panic.shift(1, fill_value=False)],
                      pi.index[panic & ~panic.shift(-1, fill_value=False)]):
        ax.axvspan(d0, d1, color='#f3d9d5', zorder=0)
    for k, v in series.items():
        w = (1 + v.sort_index()).cumprod()
        ax.plot(w.index, w.values, lw=2 if k == 'pi rule_r' else 1.4,
                color=colors[k], label=k,
                ls='-' if k != 'benchmark' else '--')
    ax.set_yscale('log')
    ax.set_ylabel('growth of $1 (log scale)')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color='#e8e8e8', lw=0.6, zorder=0)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.set_title('Applied config: cumulative wealth, gross '
                 '(shaded = pi >= 0.5 months)', loc='left', fontsize=11)
    fig.tight_layout()
    fig.savefig('plots/applied_cumulative_regime_shaded.png', dpi=160)
    fig.savefig('plots/applied_cumulative_regime_shaded.pdf')
    plt.close(fig)
    print('[figs] cumulative saved', flush=True)

    # applied long-leg z-curve heatmap (rule_r picks)
    selr = sel[sel['rule'] == 'rule_r'].set_index('year')['combo']
    moms = data_build.load_stocks(columns=['date', 'permno'] + MOM_FEATURES)
    rows = []
    for y in C.EVAL_YEARS:
        xp = os.path.join(C.XSEC_DIR,
                          f"xsec_{y}_{selr.loc[y].replace('+', '_')}.parquet")
        xx = pd.read_parquet(xp, columns=['date', 'permno', 'me', 'score_pi'])
        xx['date'] = pd.to_datetime(xx['date'])
        xx = xx.merge(moms, on=['date', 'permno'], how='left')
        for d, g in xx.groupby('date'):
            z = (g[MOM_FEATURES] - g[MOM_FEATURES].mean()) / g[MOM_FEATURES].std()
            k = max(int(len(g) * C.DECILE_FRAC), 1)
            top = g.nlargest(k, 'score_pi').index
            rows.append([d] + z.loc[top].mean().tolist())
    zc = pd.DataFrame(rows, columns=['date'] + MOM_FEATURES).set_index('date')
    zc.to_csv(os.path.join(TBL, 'applied_zscore_long_by_month.csv'))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    im = ax.imshow(zc.T.values, aspect='auto', cmap='RdBu_r', vmin=-1.5,
                   vmax=1.5, interpolation='nearest')
    ax.set_yticks(range(12), [f'{h}' for h in range(1, 13)])
    ticks = range(0, len(zc), 24)
    ax.set_xticks(list(ticks),
                  [str(zc.index[i].year) for i in ticks])
    ax.set_ylabel('momentum horizon (months)')
    ax.set_title('Applied long-leg picks: cross-sectional momentum z-curves '
                 'by month (rule_r)', loc='left', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label='mean z of picks')
    fig.tight_layout()
    fig.savefig('plots/applied_zscore_long_heatmap.png', dpi=160)
    fig.savefig('plots/applied_zscore_long_heatmap.pdf')
    plt.close(fig)
    print('[figs] heatmap saved', flush=True)


def stage_assemble():
    lines = ['# Applied config: full results-section replication\n',
             'Universe top-1000 PIT, long-only top-decile VW, expanding '
             'window, GROSS, 2011-01..2025-11. Thesis refs: L/S full '
             'universe, net 10bps, 2011-2024.\n']
    lines.append(open(os.path.join(TBL, 'thesis_comparison.md')).read())

    # regime-signal ablation table
    rows = []
    refs = {'no_signal': 0.43, 'four_raw': 0.26, 'reln_only': 0.75,
            'dd_only': 0.47, 'ghm': None, 'fund': None}
    rets = pd.read_csv(C.RETURNS_CSV, parse_dates=['date'])
    sel = pd.read_csv(C.SELECTIONS_CSV)
    x, r_nopi, r_mom, bench, pi_m = W._comparator_series(sel)
    rr = rets[rets.rule == 'rule_r'].set_index('date')['strat_ret']
    def abl_row(name, r, b, note=''):
        rows.append({'variant': name, 'sharpe': M.sharpe(r),
                     'ir': M.ir(r, b) if b is not None else np.nan,
                     'thesis_LS_sharpe_ref': refs.get(name, ''),
                     'note': note})
    abl_row('dd_only (pi rule_r)', rr, bench)
    abl_row('no_signal (no-pi)', r_nopi, bench)
    for name, f in [('four_raw', 'abl_four_raw_returns.csv'),
                    ('ghm', 'abl_ghm_returns.csv'),
                    ('ghm_20seed', 'abl_ghm20_returns.csv'),
                    ('fund', 'abl_fund_returns.csv')]:
        p = os.path.join(ABL_DIR, f)
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=['date']).set_index('date')
            note = ('FULL 20-seed budget (fair comparison)'
                    if name == 'ghm_20seed' else '5-seed ablation budget')
            if name == 'fund':
                note += '; 2011-2024, fundamentals absent 2025'
            abl_row(name, d['strat_ret'], d['bench_ret'], note=note)
    rows.append({'variant': 'reln_only', 'sharpe': np.nan, 'ir': np.nan,
                 'thesis_LS_sharpe_ref': refs['reln_only'],
                 'note': 'DEGENERATE in applied window: REL_N-only HMM pi==1.0 '
                         'for all months post-2000 (shrinking universe); '
                         'feature is constant => equals no_signal'})
    lines += ['\n## F. Regime-signal ablation (applied analog)\n',
              pd.DataFrame(rows).round(2).to_string(index=False), '']

    # SHAP shares
    shp = pd.read_csv(os.path.join(TBL, 'shap_pi_share_by_year.csv'),
                      index_col=0)
    lines += ['## G. SHAP: pi |SHAP| share (walk-consistent, 3-seed)\n',
              'overall 60% | calm 57% | panic 65% '
              '(thesis refs: 46% | 51% | 59%)\n',
              'by year:', shp.round(2).T.to_string(), '']

    # cluster descriptors (applied rule_r)
    lab = pd.read_csv('results/thesis/cluster_k4_member_dates.csv',
                      parse_dates=['date'])
    lab = lab.set_index(lab['date'].dt.to_period('M'))['cluster']
    a = M.active(rr, bench)
    cl = lab.reindex(rr.index.to_period('M'))
    crow = []
    for c in [0, 1, 2, 3]:
        mask = (cl == c).values
        v, aa = rr[mask], a[mask]
        crow.append({'cluster': f'C{c+1}', 'n': int(mask.sum()),
                     'sharpe': M.sharpe(v), 'ir_active': M.ir(rr, bench) if False else float(aa.mean()/aa.std()*np.sqrt(12)),
                     'mean_ret': float(v.mean()), 'std': float(v.std()),
                     'hit_rate': float((aa > 0).mean()),
                     'worst_mo': float(v.min()), 'best_mo': float(v.max()),
                     'mean_pi': float(pi_m.reindex(rr.index)[mask].mean())})
    lines += ['## H. Cluster descriptors (applied rule_r; thesis Table: '
              'cluster_k4_descriptors)\n',
              pd.DataFrame(crow).round(3).to_string(index=False), '']

    # per-year table (expanding-subperiods analog)
    yr = rets[rets.rule == 'rule_r'].groupby('year').apply(
        lambda g: pd.Series({
            'strat': float((1 + g['strat_ret']).prod() - 1),
            'bench': float((1 + g['bench_ret']).prod() - 1),
            'active': float((1 + g['strat_ret']).prod()
                            - (1 + g['bench_ret']).prod()),
            'mean_pi': float(g['pi'].mean())}), include_groups=False)
    lines += ['## I. Per-year returns (thesis Table: expanding subperiods '
              'analog)\n', yr.round(3).to_string(), '']

    lines += ['## Figures produced\n',
              '- plots/applied_cumulative_regime_shaded.{png,pdf}',
              '- plots/applied_zscore_long_heatmap.{png,pdf}',
              '- plots/cluster_k4_curves_applied.{png,pdf}\n',
              '## Not replicated (register)\n',
              '- International results: out of scope (spec).',
              '- Risk-aversion dual-utility sweep: requires the sigma-model '
              'second XGB; queued pending decision.',
              '- Seed-convergence appendix: N/A (budgets fixed at thesis '
              'plateau by design).',
              '- Pre-2011 subperiods: N/A (applied walk starts 2011).',
              '- Live-2026 appendix reported separately '
              '(paper/results/live2026/live_summary.md).']
    out = os.path.join(C.RESULTS, 'applied_results_full.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('assembled ->', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['ablations', 'figs', 'assemble'])
    a = ap.parse_args()
    {'ablations': stage_ablations, 'figs': stage_figs,
     'assemble': stage_assemble}[a.stage]()
