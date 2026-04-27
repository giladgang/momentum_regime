"""
Generate latex/canonical_macros.tex from results/PRODUCTION_METRICS.json.

Each metric becomes a LaTeX macro. Use these in thesis prose for any number
that should be auto-updated when scripts re-run. Example:

    \\input{canonical_macros}   % at the top of your .tex
    The strategy delivers a Sharpe ratio of \\msharpe with a Newey-West
    t-statistic of \\mtnw, both well above conventional thresholds.

Macro naming: \\<section><key> with underscores stripped. e.g.
    {"m2_perf": {"sharpe": ...}}            -> \\mperfsharpe
    {"factor_alphas": {"capm": {...}}}      -> \\factoralphascapm
"""
import json
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results', 'PRODUCTION_METRICS.json')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'latex', 'canonical_macros.tex')


def _slug(s):
    return s.replace('_', '').replace('-', '')


def _fmt(value, fmt_hint=None):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if fmt_hint == 'pct':
            return f'{value*100:.1f}\\%'
        if abs(value) < 0.01:
            return f'{value:.4f}'
        if abs(value) < 1:
            return f'{value:.3f}'
        if abs(value) < 100:
            return f'{value:.2f}'
        return f'{value:.0f}'
    return str(value)


def main():
    if not os.path.exists(PATH):
        print(f'No PRODUCTION_METRICS.json at {PATH}; nothing to emit.')
        return

    with open(PATH) as f:
        data = json.load(f)

    lines = [
        '% Auto-generated from results/PRODUCTION_METRICS.json',
        '% by scripts/build_canonical_macros.py — DO NOT HAND EDIT.',
        '%',
        '% Single source of truth for thesis numbers. Use these macros in',
        '% thesis prose so values stay consistent with PRODUCTION_METRICS.json.',
        '',
    ]

    for section, items in sorted(data.items()):
        if section.startswith('_') or not isinstance(items, dict):
            continue
        lines.append(f'% --- {section} ---')
        for key, entry in sorted(items.items()):
            if not isinstance(entry, dict):
                continue
            value = entry.get('value')
            if value is None:
                continue
            macro_name = '\\m' + _slug(section) + _slug(key)
            fmt_hint = entry.get('fmt')
            lines.append(f'\\newcommand{{{macro_name}}}{{{_fmt(value, fmt_hint)}}}')
            # Optional CI / t-stat / stars companions
            for suffix in ('cilo', 'cihi', 't', 'p', 'stars'):
                src_key = {'cilo': 'ci_lo', 'cihi': 'ci_hi'}.get(suffix, suffix)
                if src_key in entry and entry[src_key] is not None:
                    lines.append(
                        f'\\newcommand{{{macro_name}{suffix}}}'
                        f'{{{_fmt(entry[src_key])}}}'
                    )
        lines.append('')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    n_macros = sum(1 for L in lines if L.startswith('\\newcommand'))
    print(f'Saved: {OUT} ({n_macros} macros)')


if __name__ == '__main__':
    main()
