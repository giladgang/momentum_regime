"""
Render thesis tables from `results/PRODUCTION_METRICS.json`.

This is the single-source-of-truth renderer: every table generated here is
compiled FROM the canonical metrics store, not from a parallel computation.

This complements (and will progressively replace) the per-script `.tex` writes
in analysis scripts. The principle: scripts compute and persist *numbers* (to
the canonical store via `_canonical_metrics.set_metric`); this renderer turns
those numbers into formatted LaTeX tables on demand.

Run after `scripts/build_metrics.py`:

    python scripts/build_thesis_tables.py             # render all
    python scripts/build_thesis_tables.py performance # render one

Wired into `run_pipeline.py` step 21 alongside `build_metrics.py`.

Adding a new table: register a new render function in TABLES below.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
JSON_PATH = ROOT / 'results/PRODUCTION_METRICS.json'
TABLES_DIR = ROOT / 'tables'


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def _load():
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f'{JSON_PATH} not found. Run scripts/build_metrics.py first.'
        )
    with open(JSON_PATH) as f:
        return json.load(f)


def _g(metrics, section, key, default=''):
    """Get the .value field of metrics[section][key], or default."""
    entry = metrics.get(section, {}).get(key)
    if entry is None:
        return default
    return entry.get('value', default)


def _g_full(metrics, section, key):
    return metrics.get(section, {}).get(key, {})


def _pct(v, places=1):
    if v == '' or v is None:
        return '--'
    return f'{v:.{places}f}\\%'


def _signed_pct(v, places=1):
    if v == '' or v is None:
        return '--'
    return f'$-${abs(v):.{places}f}\\%' if v < 0 else f'{v:.{places}f}\\%'


def _num(v, places=2):
    if v == '' or v is None:
        return '--'
    return f'{v:.{places}f}'


def _signed_num(v, places=2):
    if v == '' or v is None:
        return '--'
    return f'$-${abs(v):.{places}f}' if v < 0 else f'{v:.{places}f}'


def _stars(s):
    if not s:
        return ''
    return f'$^{{{"*" * len(s)}}}$'


def _t_with_stars(metrics, section, key, places=2):
    e = _g_full(metrics, section, key)
    v = e.get('value')
    if v is None:
        return '--'
    s = e.get('stars', '')
    if v < 0:
        body = f'$-${abs(v):.{places}f}'
    else:
        body = f'{v:.{places}f}'
    return body + (_stars(s) if s else '')


# ───────────────────────────────────────────────────────────────────────────
# Per-table render functions
# ───────────────────────────────────────────────────────────────────────────

def render_table_performance(metrics):
    rows = [
        ('Market',              'market'),
        ('Fixed 12-mo mom',     'fixed_12'),
        ('Fixed 1-mo mom',      'fixed_1'),
        ('M0: Formula',         'm0'),
        ('M1: LR',              'm1'),
        ('M2: XGB (mom+$\\pi$)', 'm2'),
    ]
    sec = 'm2_perf'
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{l r r r r r r r}',
        r'\toprule',
        r' & Ann.\ Ret & Ann.\ Vol & Sharpe & Max DD & $\beta$ & NW $t$ & Final \$ \\',
        r'\midrule',
    ]
    # Market row (no NW t)
    label, key = rows[0]
    cells = [
        label,
        _pct(_g(metrics, sec, f'{key}_ann_ret')),
        _pct(_g(metrics, sec, f'{key}_ann_vol')),
        _num(_g(metrics, sec, f'{key}_sharpe')),
        _signed_pct(_g(metrics, sec, f'{key}_max_dd')),
        _num(_g(metrics, sec, f'{key}_beta')),
        '--',
        _num(_g(metrics, sec, f'{key}_final_dollar'), places=1),
    ]
    lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
    # Other rows
    for label, key in rows[1:]:
        beta = _g(metrics, sec, f'{key}_beta')
        beta_s = _signed_num(beta) if beta != '' and beta < 0 else _num(beta)
        cells = [
            label,
            _pct(_g(metrics, sec, f'{key}_ann_ret')),
            _pct(_g(metrics, sec, f'{key}_ann_vol')),
            _num(_g(metrics, sec, f'{key}_sharpe')),
            _signed_pct(_g(metrics, sec, f'{key}_max_dd')),
            beta_s,
            _t_with_stars(metrics, sec, f'{key}_nw_t'),
            _num(_g(metrics, sec, f'{key}_final_dollar'), places=1),
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Out-of-sample portfolio performance.}',
        r'\label{tab:performance}',
        r'',
        r'\medskip',
        r'\small',
        r'\textbf{Notes:} Long-short (top/bottom decile, NYSE breakpoints), value-weighted, net of 10\,bps one-way costs. $\beta$ is the market beta. NW $t$ uses Newey--West standard errors (6 lags). Test period: January 2011 to November 2025 (167 months). $^{*}$\,$p<0.10$; $^{**}$\,$p<0.05$; $^{***}$\,$p<0.01$.',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


def render_table_factor_alphas(metrics):
    rows = [
        ('CAPM',    'capm',    ['Mkt-RF']),
        ('FF3',     'ff3',     ['Mkt-RF', 'SMB', 'HML']),
        ('Carhart', 'carhart', ['Mkt-RF', 'SMB', 'HML', 'UMD']),
        ('FF5',     'ff5',     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']),
        ('FF6',     'ff6',     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']),
    ]
    sec = 'factor_alphas'
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{l r r r r r r r r}',
        r'\toprule',
        r'Model & $\alpha$ (\%) & $t(\alpha)$ & Mkt-RF & SMB & HML & UMD & RMW & CMA \\',
        r'\midrule',
    ]
    for label, key, _ in rows:
        e = _g_full(metrics, sec, f'{key}_alpha')
        alpha = e.get('value', '')
        t = e.get('t', '')
        # The factor loadings (Mkt-RF, SMB, ...) aren't in canonical yet; leave as
        # placeholder until a parser populates them.
        cells = [
            label,
            _num(alpha, places=1) if alpha != '' else '--',
            _num(t, places=2) if t != '' else '--',
        ] + [' '] * 6
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r"\caption{Factor model regressions for M2 (XGBoost, mom+$\pi$). $\alpha$ is annualised. $t$-statistics use Newey--West standard errors (6 lags). Factor loadings shown in the wide table generated by scripts/new_ls_analyses.py.}",
        r'\label{tab:factor_alphas}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


def render_table_regime_sharpe(metrics):
    rows = [
        ('Market',              'market'),
        ('Fixed 12-mo mom',     'fixed_12'),
        ('Fixed 1-mo mom',      'fixed_1'),
        ('M0: Formula',         'm0'),
        ('M1: LR',              'm1'),
        ('M2: XGB (mom+$\\pi$)', 'm2'),
    ]
    sec = 'regime_sharpe'
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\begin{tabular}{l r r r}',
        r'\toprule',
        r' & Full & Calm & Panic \\',
        r'\midrule',
    ]
    for i, (label, key) in enumerate(rows):
        if i == 1:
            lines.append(r'\midrule')
        cells = [
            label,
            _signed_num(_g(metrics, sec, f'{key}_full')),
            _signed_num(_g(metrics, sec, f'{key}_calm')),
            _signed_num(_g(metrics, sec, f'{key}_panic')),
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Regime-conditional Sharpe ratios. Calm/Panic split by HMM filtered probability $\pi_t^{\text{filter}} \lessgtr 0.5$.}',
        r'\label{tab:regime_sharpe}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


def render_table_panic_subtypes(metrics):
    rows = [
        ('Calm',           'calm'),
        ('Panic: Crash',   'panic_crash'),
        ('Panic: Recovery', 'panic_recovery'),
    ]
    sec = 'panic_subtypes'
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{l r r r r}',
        r'\toprule',
        r' & Ann.\ Ret & Ann.\ Vol & Sharpe & Months \\',
        r'\midrule',
    ]
    for label, key in rows:
        ar = _g(metrics, sec, f'{key}_ann_ret')
        av = _g(metrics, sec, f'{key}_ann_vol')
        sh = _g(metrics, sec, f'{key}_sharpe')
        mo = _g(metrics, sec, f'{key}_months')
        cells = [
            label,
            _signed_pct(ar),
            _pct(av),
            _signed_num(sh),
            str(int(mo)) if mo != '' else '--',
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Long-short performance by panic sub-type. Panic months are split by concurrent market return: crash (negative) vs recovery (positive). Net of 10\,bps one-way transaction costs.}',
        r'\label{tab:panic_subtypes}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


def render_table_hmm_separation(metrics):
    sec = 'hmm_separation'
    feats = [k.split('_')[0] for k in metrics.get(sec, {}) if k.endswith('_delta')]
    feats = sorted(set(feats))
    if not feats:
        return None
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\begin{tabular}{l r r r@{\hspace{1.5em}} c}',
        r'\toprule',
        r' & $\hat{\mu}_{\text{calm}}$ & $\hat{\mu}_{\text{panic}}$ & \multicolumn{1}{c}{$\Delta$} & 95\% CI \\',
        r'\midrule',
    ]
    for f in feats:
        calm  = _g(metrics, sec, f'{f}_calm')
        panic = _g(metrics, sec, f'{f}_panic')
        delta_e = _g_full(metrics, sec, f'{f}_delta')
        delta = delta_e.get('value', '')
        ci_lo = delta_e.get('ci_lo', '')
        ci_hi = delta_e.get('ci_hi', '')

        def _fmt(v):
            if v == '' or v is None: return '--'
            return f'$+${v:.3f}' if v >= 0 else f'$-${abs(v):.3f}'
        label = f.upper().replace('REL_N', r'REL\_N') + r'$_z$'
        cells = [
            label,
            _num(calm, places=3) if calm != '' and calm >= 0 else _signed_num(calm, places=3),
            _fmt(panic),
            _fmt(delta),
            f'[{_fmt(ci_lo)},\\,{_fmt(ci_hi)}]',
        ]
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{HMM posterior regime-mean separation.}',
        r'\label{tab:hmm_separation}',
        r'',
        r'\medskip',
        r'\small',
        r'\textbf{Notes:} Posterior mean estimates of regime-conditional feature means from the Bayesian two-state HMM, estimated via Gibbs sampling on the 1990--2010 training period. All features are standardised. $\Delta = \hat{\mu}_{\text{panic}} - \hat{\mu}_{\text{calm}}$. The 95\% CI is the Bayesian posterior credible interval for $\Delta$; zero lies outside all intervals, confirming significant regime separation. DD = market drawdown; DISP = log cross-sectional return dispersion; REL\_N = relative market participation; CS = credit spread (BAA$-$AAA).',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


# ───────────────────────────────────────────────────────────────────────────
# Registry
# ───────────────────────────────────────────────────────────────────────────

def render_table_international_results(metrics):
    """International robustness table: UK + JP x {Local market, Fixed 12-mo, M2}
    with Sharpe / Annualised return / Maximum drawdown. Data flows from
    parse_intl_summary in build_metrics.py."""
    rows = [
        ('Local market',     'market'),
        ('Fixed 12-mo mom',  'fixed_12'),
        ('M2 (XGB)',         'm2'),
    ]
    sec = 'international'
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{International cross-sectional model performance, 2011--2025 out-of-sample test period. M2 uses the US specification (12 momentum horizons, 50-seed XGBoost ensemble, decile-spread long-short construction, value-weighted by market equity) with locally-fitted HMM (200 production seeds, DD+VOL+REL\_N feature set selected per region). Long and short legs are formed from unconditional decile breakpoints across all listed stocks (no NYSE-equivalent tier exists for either market). Net of 10 bps one-way transaction cost.}',
        r'\label{tab:international_results}',
        r'\small',
        r'\begin{tabular}{l c c c c c c}',
        r'\toprule',
        r' & \multicolumn{3}{c}{UK} & \multicolumn{3}{c}{JP} \\',
        r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}',
        r'Strategy & Sharpe & Ann.\ Ret & Max DD & Sharpe & Ann.\ Ret & Max DD \\',
        r'\midrule',
    ]
    # Intl metrics are stored as raw decimals (e.g., ann_ret=0.16 for 16%) -- multiply by
    # 100 before passing to the percent helpers, which expect already-percentage values.
    def _to_pct(v):
        return v * 100 if isinstance(v, (int, float)) else v

    # Bold the M2 row's UK columns (UK regional is the headline finding); JP M2 not bolded.
    for i, (label, key) in enumerate(rows):
        cells = [label]
        for region in ['uk', 'jp']:
            uk_bold = (key == 'm2') and (region == 'uk')
            sh_raw = _g(metrics, sec, f'{region}_regional_{key}_sharpe')
            ar_raw = _g(metrics, sec, f'{region}_regional_{key}_ann_ret')
            dd_raw = _g(metrics, sec, f'{region}_regional_{key}_max_dd')
            sh = _num(sh_raw, places=2)
            ar = _signed_pct(_to_pct(ar_raw), places=1)
            dd = _signed_pct(_to_pct(dd_raw), places=1)
            if uk_bold:
                sh = rf'\textbf{{{sh}}}'
                ar = rf'\textbf{{{ar}}}'
                dd = rf'\textbf{{{dd}}}'
            cells += [sh, ar, dd]
        lines.append(' & '.join(cells) + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        '',
    ]
    return '\n'.join(lines)


TABLES = {
    'performance':     ('table_performance.tex', render_table_performance),
    'factor_alphas':   ('table_factor_alphas.tex', render_table_factor_alphas),
    'regime_sharpe':   ('table_regime_sharpe.tex', render_table_regime_sharpe),
    'panic_subtypes':  ('table_panic_subtypes.tex', render_table_panic_subtypes),
    'hmm_separation':  ('table_hmm_separation.tex', render_table_hmm_separation),
    'international_results': ('table_international_results.tex', render_table_international_results),
}


def main():
    metrics = _load()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rendered = 0
    for name, (filename, fn) in TABLES.items():
        if only and only != name:
            continue
        out = fn(metrics)
        if out is None:
            print(f'  [skip] {name}: no metrics')
            continue
        path = TABLES_DIR / filename
        # Write to a .canonical sibling so we don't clobber the script-written
        # table on the first integration. After review, rename to the canonical
        # filename.
        sibling = path.with_suffix('.canonical.tex')
        sibling.write_text(out)
        rendered += 1
        print(f'  ✓ {name}  →  {sibling.relative_to(ROOT)}')
    print(f'\nRendered {rendered} table(s) from canonical metrics.')
    print(f'Compare against script-written {tuple(filename for filename,_ in TABLES.values())} to verify equivalence,')
    print(f'then rename .canonical.tex → .tex to make canonical the single source.')


if __name__ == '__main__':
    main()
