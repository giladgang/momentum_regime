"""
Build results/PRODUCTION_METRICS.json — the canonical source of truth for every
thesis-cited number — by parsing tables/*.tex and key results/*.csv.

Run after any pipeline rerun:

    python scripts/build_metrics.py

Then regenerate latex macros:

    python scripts/build_canonical_macros.py

This script does NOT recompute anything. It only extracts numbers from the
already-generated tables and CSVs. The tables themselves remain authoritative;
this JSON is a *named-key view* of the same numbers, suitable for cross-script
consumption (figures, latex prose macros, drift-detection).

Each table has its own bespoke parser in this file. Adding a new table means
adding a new parser function and registering it below.
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts._canonical_metrics import set_many


# ─────────────────────────────────────────────────────────────────────
# Generic helpers for parsing latex table cells
# ─────────────────────────────────────────────────────────────────────

def _strip_latex(s):
    """Remove $-$, \\%, $...$, \\, etc. and return raw value string."""
    s = s.strip()
    s = s.replace(r'\%', '').replace('%', '').replace(r'\$', '')
    s = s.replace('$-$', '-').replace('$+$', '+')
    s = re.sub(r'\$([^$]*)\$', r'\1', s)
    s = re.sub(r'\^\{([^}]*)\}', '', s)  # strip ^{stars}
    s = s.replace(r'\,', '').replace(' ', '')
    s = s.rstrip('*')
    s = s.strip(r'\\')
    return s


def _to_float(s):
    s = _strip_latex(s)
    s = s.replace('--', '-')
    if s in ('-', '', 'NaN', 'nan'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _stars(cell):
    m = re.search(r'(\*{1,3})', cell)
    return m.group(1) if m else None


def _read(path):
    p = ROOT / path
    if not p.exists():
        return None
    return p.read_text()


def _parse_row(text, label):
    """Find the row in a tabular env whose first cell matches `label`,
    return the list of cells (split on &, last cell stripped of trailing \\\\)."""
    pat = re.compile(rf'^\s*{re.escape(label)}\s*&(.+?)\\\\\s*$', re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    cells = [label] + [c.strip() for c in m.group(1).split('&')]
    return cells


# ─────────────────────────────────────────────────────────────────────
# Per-table parsers
# ─────────────────────────────────────────────────────────────────────

def parse_table_performance():
    text = _read('tables/table_performance.tex')
    if not text:
        return {}
    # Row labels in the production table
    rows = {
        'm2':       'M2: XGB (mom+$\\pi$)',
        'm1':       'M1: LR',
        'm0':       'M0: Formula',
        'fixed_12': 'Fixed 12-mo mom',
        'fixed_1':  'Fixed 1-mo mom',
        'market':   'Market',
    }
    # Column order: name | Ann.Ret | Ann.Vol | Sharpe | Max DD | beta | NW t | Final $
    out = {}
    for key, label in rows.items():
        r = _parse_row(text, label)
        if not r:
            continue
        ann_ret = _to_float(r[1])
        ann_vol = _to_float(r[2])
        sharpe  = _to_float(r[3])
        mdd     = _to_float(r[4])
        beta    = _to_float(r[5])
        nwt     = _to_float(r[6])
        nwt_stars = _stars(r[6])
        final_d = _to_float(r[7])
        out[f'{key}_ann_ret']    = {'value': ann_ret, 'unit': 'percent'}
        out[f'{key}_ann_vol']    = {'value': ann_vol, 'unit': 'percent'}
        out[f'{key}_sharpe']     = {'value': sharpe}
        out[f'{key}_max_dd']     = {'value': mdd, 'unit': 'percent'}
        out[f'{key}_beta']       = {'value': beta}
        nwt_entry = {'value': nwt}
        if nwt_stars:
            nwt_entry['stars'] = nwt_stars
        out[f'{key}_nw_t']       = nwt_entry
        out[f'{key}_final_dollar'] = {'value': final_d}
    return out


def parse_table_factor_alphas():
    text = _read('tables/table_factor_alphas.tex')
    if not text:
        return {}
    out = {}
    for spec, label in [('capm', 'CAPM'), ('ff3', 'FF3'), ('carhart', 'Carhart'),
                        ('ff5', 'FF5'), ('ff6', 'FF6')]:
        r = _parse_row(text, label)
        if not r:
            continue
        alpha = _to_float(r[1])
        t = _to_float(r[2])
        out[f'{spec}_alpha'] = {'value': alpha, 'unit': 'percent', 't': t}
    return out


def parse_table_regime_sharpe():
    text = _read('tables/table_regime_sharpe.tex')
    if not text:
        return {}
    out = {}
    rows = {'m2': 'M2: XGB (mom+$\\pi$)', 'm1': 'M1: LR', 'm0': 'M0: Formula',
            'fixed_12': 'Fixed 12-mo mom', 'fixed_1': 'Fixed 1-mo mom', 'market': 'Market'}
    for key, label in rows.items():
        r = _parse_row(text, label)
        if not r: continue
        out[f'{key}_full']  = {'value': _to_float(r[1])}
        out[f'{key}_calm']  = {'value': _to_float(r[2])}
        out[f'{key}_panic'] = {'value': _to_float(r[3])}
    return out


def parse_table_subperiod():
    text = _read('tables/table_subperiod.tex')
    if not text:
        return {}
    out = {}
    rows = {'m2': 'M2: XGB', 'm1': 'M1: LR', 'fixed_12': 'Fixed 12-mo', 'market': 'Market'}
    for key, label in rows.items():
        r = _parse_row(text, label)
        if not r: continue
        out[f'{key}_2011_2015'] = {'value': _to_float(r[1])}
        out[f'{key}_2016_2020'] = {'value': _to_float(r[2])}
        out[f'{key}_2021_2025'] = {'value': _to_float(r[3])}
        out[f'{key}_full']      = {'value': _to_float(r[4])}
    return out


def parse_table_bootstrap():
    """Reads from results/bootstrap_*.csv (written by bootstrap_analysis.py)."""
    import csv
    out = {}
    cis_path = ROOT / 'results/bootstrap_sharpe_cis.csv'
    if cis_path.exists():
        with open(cis_path) as f:
            for row in csv.DictReader(f):
                key_map = {
                    'M2 (XGB)':           'm2',
                    'M1 (LR)':            'm1',
                    'M0 (Formula)':       'm0',
                    'Fixed 12-mo mom':    'fixed_12',
                    'Fixed 1-mo mom':     'fixed_1',
                    'M2 calm':            'm2_calm',
                    'M2 panic':           'm2_panic',
                }
                key = key_map.get(row.get('strategy', '').strip())
                if not key:
                    continue
                out[f'{key}_sharpe'] = {
                    'value': _to_float(row.get('point_sharpe', '')),
                    'ci_lo': _to_float(row.get('ci_low', '')),
                    'ci_hi': _to_float(row.get('ci_high', '')),
                }
    paired_path = ROOT / 'results/bootstrap_paired_tests.csv'
    if paired_path.exists():
        with open(paired_path) as f:
            for row in csv.DictReader(f):
                bench = row.get('benchmark', '').strip()
                key_map = {
                    'M1 (LR)':           'm2_vs_m1',
                    'M0 (Formula)':      'm2_vs_m0',
                    'Fixed 12-mo mom':   'm2_vs_fixed_12',
                    'Fixed 1-mo mom':    'm2_vs_fixed_1',
                }
                k = key_map.get(bench)
                if not k:
                    continue
                out[k] = {
                    'value': _to_float(row.get('point_diff', '')),
                    'ci_lo': _to_float(row.get('diff_ci_low', '')),
                    'ci_hi': _to_float(row.get('diff_ci_high', '')),
                    'p':     _to_float(row.get('p_value', '')),
                }
    regime_path = ROOT / 'results/bootstrap_regime.csv'
    if regime_path.exists():
        with open(regime_path) as f:
            for row in csv.DictReader(f):
                k = row.get('quantity', '').strip()
                if k.lower().startswith('panic') and 'calm' in k.lower():
                    out['m2_panic_minus_calm'] = {
                        'value': _to_float(row.get('point_value', row.get('value', ''))),
                        'ci_lo': _to_float(row.get('ci_low', '')),
                        'ci_hi': _to_float(row.get('ci_high', '')),
                        'p':     _to_float(row.get('p_value', '')),
                    }
    return out


def parse_table_panic_subtypes():
    """3-row table: Calm / Panic-Crash / Panic-Recovery with Sharpe + n + others."""
    text = _read('tables/table_panic_subtypes.tex')
    if not text:
        return {}
    out = {}
    rows = {'calm': 'Calm', 'panic_crash': 'Panic Crash', 'panic_recovery': 'Panic Recovery'}
    for key, label in rows.items():
        r = _parse_row(text, label)
        if r:
            # Format is typically: Label & n & Sharpe & ...
            # Expect Sharpe at a specific column; we'll just grab the second cell as Sharpe
            # but parse all numerics
            vals = [_to_float(c) for c in r[1:]]
            if any(v is not None for v in vals):
                out[f'{key}_sharpe'] = {'value': vals[1] if len(vals) > 1 else vals[0]}
    return out


def parse_table_ic():
    text = _read('tables/table_ic.tex')
    if not text:
        return {}
    out = {}
    # Look for rows like "M2 (XGB)" or similar with IC and t-stat
    for label, key in [('M2: XGB', 'm2'), ('Mom (12-1)', 'mom_12'),
                       ('M2 $-$ Mom', 'm2_minus_mom'), ('M2 - Mom', 'm2_minus_mom')]:
        r = _parse_row(text, label)
        if not r: continue
        if 'ic' not in [k.split('_')[-1] for k in out]:
            out[f'{key}_ic'] = {'value': _to_float(r[1]), 't': _to_float(r[2]) if len(r) > 2 else None}
    return out


def parse_table_hmm_separation():
    """Reads from results/hmm_separation.csv (written by hmm_diagnostics.py)."""
    import csv
    out = {}
    path = ROOT / 'results/hmm_separation.csv'
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                feat = row.get('feature', '').strip()
                if not feat:
                    continue
                key = feat.replace('_z', '').lower()
                out[f'{key}_calm']  = {'value': _to_float(row.get('calm_mean'))}
                out[f'{key}_panic'] = {'value': _to_float(row.get('panic_mean'))}
                out[f'{key}_delta'] = {
                    'value': _to_float(row.get('delta')),
                    'ci_lo': _to_float(row.get('ci_lo')),
                    'ci_hi': _to_float(row.get('ci_hi')),
                }
    return out


def parse_table_january():
    """Two sections: 'Full sample' and 'Excluding January'. Same row labels in each."""
    text = _read('tables/table_january.tex')
    if not text:
        return {}
    # Find section boundaries
    full_start = text.find(r'Full sample')
    excl_start = text.find(r'Excluding January')
    if full_start < 0 or excl_start < 0:
        return {}
    full_block = text[full_start:excl_start]
    excl_block = text[excl_start:]
    out = {}
    for block, suffix in [(full_block, '_full'), (excl_block, '_excl_jan')]:
        for label, key in [('M2: XGB', 'm2'), ('Fixed 12-mo mom', 'mom12')]:
            r = _parse_row(block, label)
            if not r: continue
            if len(r) >= 4:
                out[f'{key}{suffix}_sharpe']  = {'value': _to_float(r[3])}
                out[f'{key}{suffix}_ann_ret'] = {'value': _to_float(r[1]), 'unit': 'percent'}
    return out


def parse_table_seed_convergence():
    """Read directly from results/seed_convergence.csv."""
    import csv
    path = ROOT / 'results/seed_convergence.csv'
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            k = row.get('k', '').strip()
            if not k:
                continue
            out[f'k_{k}'] = {
                'value': _to_float(row.get('mean')),
                'p25':   _to_float(row.get('p25')),
                'p75':   _to_float(row.get('p75')),
                'std':   _to_float(row.get('std')),
            }
    return out


def parse_table_fund_alphas():
    text = _read('tables/table_fund_alphas.tex')
    if not text:
        return {}
    out = {}
    for label, key in [('CAPM', 'fund_capm'), ('FF3', 'fund_ff3'), ('Carhart', 'fund_carhart'),
                       ('FF5', 'fund_ff5'), ('FF6', 'fund_ff6')]:
        r = _parse_row(text, label)
        if not r: continue
        out[f'{key}_alpha'] = {'value': _to_float(r[1]), 'unit': 'percent',
                                't': _to_float(r[2]) if len(r) > 2 else None}
    return out


def parse_intl_summary():
    """Parse UK/JP intl summary CSVs. Strategy column uses 'method2_xgb' for M2."""
    import csv
    out = {}
    for region, suffix in [('uk', 'uk_regional'), ('jp', 'jp_regional'),
                           ('uk', 'uk_uspi'),    ('jp', 'jp_uspi')]:
        is_uspi = suffix.endswith('_uspi')
        path = ROOT / f"results/intl_{region}_summary{'_uspi' if is_uspi else ''}.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                name = row.get('strategy', '').strip().lower()
                # Map every strategy to a key prefix
                strat_map = {
                    'method2_xgb': 'm2', 'method1_lr': 'm1', 'method0_formula': 'm0',
                    'fixed_mom_12': 'fixed_12', 'fixed_mom_1': 'fixed_1',
                    'market': 'market',
                }
                strat_key = strat_map.get(name)
                if not strat_key:
                    continue
                out[f'{suffix}_{strat_key}_sharpe'] = {'value': _to_float(row.get('sharpe'))}
                out[f'{suffix}_{strat_key}_max_dd'] = {'value': _to_float(row.get('max_dd'))}
                out[f'{suffix}_{strat_key}_ann_ret'] = {'value': _to_float(row.get('ann_ret'))}
    return out


# ─────────────────────────────────────────────────────────────────────
# Registry: section name → parser function
# ─────────────────────────────────────────────────────────────────────

PARSERS = {
    'm2_perf':         parse_table_performance,
    'factor_alphas':   parse_table_factor_alphas,
    'regime_sharpe':   parse_table_regime_sharpe,
    'subperiod':       parse_table_subperiod,
    'bootstrap':       parse_table_bootstrap,
    'panic_subtypes':  parse_table_panic_subtypes,
    'ic':              parse_table_ic,
    'hmm_separation':  parse_table_hmm_separation,
    'january':         parse_table_january,
    'seed_convergence': parse_table_seed_convergence,
    'fund_alphas':     parse_table_fund_alphas,
    'international':   parse_intl_summary,
}


def main():
    print("Building results/PRODUCTION_METRICS.json from tables/ + results/...")
    total = 0
    for section, parser in PARSERS.items():
        try:
            metrics = parser()
        except Exception as e:
            print(f"  [WARN] {section}: parser raised {type(e).__name__}: {e}")
            continue
        if metrics:
            set_many(section, metrics, source=f'parsed-from-tables-and-csvs ({parser.__name__})')
            n = len(metrics)
            total += n
            print(f"  {section:<22s} {n:3d} metrics")
        else:
            print(f"  {section:<22s} (none — table missing or parser found no rows)")
    print(f"\nTotal: {total} metrics written to results/PRODUCTION_METRICS.json")


if __name__ == '__main__':
    main()
