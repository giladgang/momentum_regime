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
    cis_path = ROOT / 'results/thesis/bootstrap_sharpe_cis.csv'
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
    paired_path = ROOT / 'results/thesis/bootstrap_paired_tests.csv'
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
    regime_path = ROOT / 'results/thesis/bootstrap_regime.csv'
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
    """3-row table; columns: Ann.Ret | Ann.Vol | Sharpe | Months."""
    text = _read('tables/table_panic_subtypes.tex')
    if not text:
        return {}
    out = {}
    rows = {'calm': 'Calm', 'panic_crash': 'Panic: Crash', 'panic_recovery': 'Panic: Recovery'}
    for key, label in rows.items():
        r = _parse_row(text, label)
        if not r:
            continue
        # r = [label, ann_ret, ann_vol, sharpe, months]
        ann_ret = _to_float(r[1]) if len(r) > 1 else None
        ann_vol = _to_float(r[2]) if len(r) > 2 else None
        sharpe  = _to_float(r[3]) if len(r) > 3 else None
        months  = _to_float(r[4]) if len(r) > 4 else None
        out[f'{key}_sharpe']  = {'value': sharpe}
        out[f'{key}_ann_ret'] = {'value': ann_ret, 'unit': 'percent'}
        out[f'{key}_ann_vol'] = {'value': ann_vol, 'unit': 'percent'}
        if months is not None:
            out[f'{key}_months'] = {'value': int(months)}
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
    path = ROOT / 'results/thesis/hmm_separation.csv'
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
    path = ROOT / 'results/thesis/seed_convergence.csv'
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
        path = ROOT / f"results/thesis/intl_{region}_summary{'_uspi' if is_uspi else ''}.csv"
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


def _snapshot_config():
    """Read config.py and return a dict of the config knobs that influence
    thesis numbers. Stored in PRODUCTION_METRICS.json._metadata.config_snapshot
    so the next rerun can detect which knobs changed."""
    try:
        sys.path.insert(0, str(ROOT))
        import config  # noqa
    except Exception as e:
        print(f"  [WARN] could not import config.py: {e}")
        return {}
    snap = {}
    KNOBS = [
        'HMM_FEATURES', 'K_STATES', 'HMM_ITERATIONS', 'HMM_BURNIN',
        'MAX_DEPTH', 'LEARNING_RATE', 'N_ESTIMATORS', 'XGB_SEEDS',
        'FUND_FEATURES', 'MOM_FEATURES', 'TRADING_FEE',
        'TRAIN_END', 'PANEL_PATH',
    ]
    for k in KNOBS:
        if hasattr(config, k):
            v = getattr(config, k)
            # Normalise list-like to sorted tuple form for stable comparison
            if isinstance(v, (list, set, tuple)):
                snap[k] = sorted(v) if all(isinstance(x, (str, int, float)) for x in v) else list(v)
            else:
                snap[k] = v
    return snap


def _load_previous():
    """Load the previous PRODUCTION_METRICS.json snapshot (the metrics + config),
    returning ({}, {}) if not present."""
    import json
    path = ROOT / 'results/PRODUCTION_METRICS.json'
    if not path.exists():
        return {}, {}
    with open(path) as f:
        prev = json.load(f)
    prev_config = prev.get('_metadata', {}).get('config_snapshot', {})
    prev_metrics = {k: v for k, v in prev.items() if not k.startswith('_')}
    return prev_metrics, prev_config


def _flatten(metrics_by_section):
    """{'sec': {'key': {'value': ..., ...}}} → {'sec.key': {'value': ...}}."""
    out = {}
    for sec, items in metrics_by_section.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if isinstance(entry, dict):
                out[f'{sec}.{key}'] = entry
    return out


def _diff_metrics(prev, curr, atol=1e-6):
    """Return dict {flat_key: (old_val, new_val, delta)} for metrics whose
    'value' field changed since last run (or are new)."""
    p = _flatten(prev)
    c = _flatten(curr)
    diffs = {}
    for k in sorted(set(p) | set(c)):
        old = p.get(k, {}).get('value')
        new = c.get(k, {}).get('value')
        if old is None and new is None:
            continue
        if old is None:
            diffs[k] = (None, new, None)
        elif new is None:
            diffs[k] = (old, None, None)
        else:
            try:
                if abs(float(new) - float(old)) > atol:
                    diffs[k] = (old, new, float(new) - float(old))
            except (TypeError, ValueError):
                if old != new:
                    diffs[k] = (old, new, None)
    return diffs


def _diff_configs(prev, curr):
    """Return dict {knob: (old, new)} for config knobs whose value changed."""
    out = {}
    for k in sorted(set(prev) | set(curr)):
        if prev.get(k) != curr.get(k):
            out[k] = (prev.get(k), curr.get(k))
    return out


def _write_diff_report(metric_diffs, config_diffs, total_metrics):
    """Write results/METRICS_DIFF.md — a focused report of what changed
    this run, cross-referenced against the config dependency map."""
    from scripts._config_deps import CONFIG_DEPS, configs_touching_metric
    import datetime as _dt

    lines = [
        '# Metrics diff report',
        '',
        f'Generated: {_dt.datetime.now().isoformat(timespec="seconds")}  ',
        f'Total metrics tracked: {total_metrics}',
        '',
    ]

    # ── Config knobs that changed ─────────────────────────────────────────
    if config_diffs:
        lines += [
            '## ⚙️ Config knobs that changed since last run',
            '',
            '| Knob | Old → New | Expected metric ripple |',
            '|---|---|---|',
        ]
        for knob, (old, new) in config_diffs.items():
            dep = CONFIG_DEPS.get(knob, {})
            ripple = dep.get('downstream') or ', '.join(dep.get('metrics', [])) or '—'
            old_s = repr(old) if old is not None else '*(new)*'
            new_s = repr(new)
            lines.append(f'| `{knob}` | `{old_s}` → `{new_s}` | {ripple} |')
        lines.append('')
    else:
        lines += ['## ⚙️ Config knobs', '', 'No config knobs changed since last run.', '']

    # ── Metrics that changed ──────────────────────────────────────────────
    if not metric_diffs:
        lines += ['## 📊 Metrics', '', 'No metric values changed since last run.', '']
    else:
        lines += [
            f'## 📊 Metrics that changed ({len(metric_diffs)})',
            '',
            '| Metric (section.key) | Old | New | Δ | Likely config |',
            '|---|---|---|---|---|',
        ]
        for k, (old, new, delta) in metric_diffs.items():
            sec, _, key = k.partition('.')
            cfgs = configs_touching_metric(sec, key)
            cfg_str = ', '.join(cfgs) if cfgs else '—'
            old_s = f'{old:.4f}' if isinstance(old, (int, float)) else (str(old) if old is not None else '*(new)*')
            new_s = f'{new:.4f}' if isinstance(new, (int, float)) else (str(new) if new is not None else '*(removed)*')
            delta_s = f'{delta:+.4f}' if isinstance(delta, (int, float)) else '—'
            lines.append(f'| `{k}` | {old_s} | {new_s} | {delta_s} | {cfg_str} |')
        lines.append('')

    # ── Cross-check: expected vs actual ──────────────────────────────────
    if config_diffs:
        lines += [
            '## 🔍 Expected-vs-actual cross-check',
            '',
            "Did the actual metric drift match what we'd expect from the changed configs?",
            '',
        ]
        for knob, (old, new) in config_diffs.items():
            dep = CONFIG_DEPS.get(knob, {})
            patterns = dep.get('metrics', [])
            expected_keys = set()
            for pat in patterns:
                sec, _, key = pat.partition('.')
                for diff_key in metric_diffs:
                    diff_sec, _, diff_k = diff_key.partition('.')
                    if (sec == '*' or sec == diff_sec) and (key == '*' or key == diff_k):
                        expected_keys.add(diff_key)
            actual_drifted = set(metric_diffs.keys())
            unexpected = actual_drifted - expected_keys
            missing = set()  # we can't compute "expected but didn't drift" without enumerating canonical
            lines.append(f'**`{knob}` changed:**')
            if expected_keys:
                lines.append(f"- ✅ Expected drifts that occurred ({len(expected_keys)}): " +
                             ', '.join(f'`{k}`' for k in sorted(expected_keys)[:8]) +
                             (f' (+{len(expected_keys)-8} more)' if len(expected_keys) > 8 else ''))
            else:
                lines.append('- (no metrics drifted in the expected ripple set — could mean the change had no numeric effect, or the dependency map is incomplete)')
            lines.append('')

    if metric_diffs and config_diffs:
        # Look for unexpected drifts (any metric that changed but no listed config touches it)
        unexpected = []
        for k, (_, _, _) in metric_diffs.items():
            sec, _, key = k.partition('.')
            cfgs = configs_touching_metric(sec, key)
            changed_cfgs = set(cfgs) & set(config_diffs.keys())
            if not changed_cfgs:
                unexpected.append(k)
        if unexpected:
            lines += [
                '### ⚠ Unexpected drifts',
                '',
                "These metrics changed but **none of the changed configs lists them as dependencies**. ",
                "This could mean:",
                "- The dependency map (`scripts/_config_deps.py`) is incomplete (please update)",
                "- A config you forgot to track was changed",
                "- Stochastic seed drift (small) — usually fine if Δ is tiny",
                '',
            ]
            for k in unexpected[:15]:
                old, new, delta = metric_diffs[k]
                delta_s = f'Δ={delta:+.4f}' if isinstance(delta, (int, float)) else ''
                lines.append(f'- `{k}` ({delta_s})')
            if len(unexpected) > 15:
                lines.append(f'- ... and {len(unexpected) - 15} more')
            lines.append('')

    out_path = ROOT / 'results/reports/METRICS_DIFF.md'
    out_path.write_text('\n'.join(lines) + '\n')
    return out_path


def main():
    print("Building results/PRODUCTION_METRICS.json from tables/ + results/...")
    # Snapshot previous state for diffing
    prev_metrics, prev_config = _load_previous()

    # Run all parsers
    new_metrics_by_section = {}
    for section, parser in PARSERS.items():
        try:
            metrics = parser()
        except Exception as e:
            print(f"  [WARN] {section}: parser raised {type(e).__name__}: {e}")
            continue
        if metrics:
            set_many(section, metrics, source=f'parsed-from-tables-and-csvs ({parser.__name__})')
            new_metrics_by_section[section] = metrics
            n = len(metrics)
            print(f"  {section:<22s} {n:3d} metrics")
        else:
            print(f"  {section:<22s} (none — table missing or parser found no rows)")

    total = sum(len(v) for v in new_metrics_by_section.values())

    # Snapshot config.py knobs into _metadata.config_snapshot
    curr_config = _snapshot_config()
    import json
    metrics_path = ROOT / 'results/PRODUCTION_METRICS.json'
    with open(metrics_path) as f:
        data = json.load(f)
    data.setdefault('_metadata', {})['config_snapshot'] = curr_config
    with open(metrics_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

    # Compute diffs vs previous snapshot
    metric_diffs = _diff_metrics(prev_metrics, new_metrics_by_section)
    config_diffs = _diff_configs(prev_config, curr_config)

    # Write the focused diff report
    diff_path = _write_diff_report(metric_diffs, config_diffs, total)

    print(f'\nTotal: {total} metrics in PRODUCTION_METRICS.json')
    print(f'Diff vs last run: {len(metric_diffs)} metrics changed, {len(config_diffs)} config knobs changed')
    print(f'Report: {diff_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
