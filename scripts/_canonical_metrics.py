"""
Canonical metrics store — single source of truth for every published thesis number.

Every analysis script that produces a thesis-cited number MUST also write that
number to `results/PRODUCTION_METRICS.json` via the helpers below. The thesis
LaTeX (tables and prose) is then validated against this store, and macros for
inline-prose use are auto-generated from it (`latex/canonical_macros.tex`).

Schema:
    {
        "_metadata": {
            "schema_version": 1,
            "last_full_rerun": "2026-04-27T...",
            "data_source": "post-Shumway, 1990-2010 train"
        },
        "<section>": {
            "<metric_key>": {
                "value": <number>,
                "source_script": "scripts/foo.py",
                "updated": "2026-04-27T...",
                # optional fields:
                "ci_lo": ..., "ci_hi": ..., "stars": "***", "t": ..., "p": ...
            }
        }
    }

Usage from a generator script:
    from scripts._canonical_metrics import set_metric
    set_metric('m2_perf', 'sharpe', 1.114, source=__file__,
               ci_lo=0.66, ci_hi=1.54)
"""

import datetime
import json
import os
import sys

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'PRODUCTION_METRICS.json'
)


def _load():
    if os.path.exists(PATH):
        with open(PATH) as f:
            return json.load(f)
    return {
        '_metadata': {
            'schema_version': 1,
            'data_source': 'post-Shumway, 1990-2010 train, 2011-2025 test',
        }
    }


def _save(data):
    data['_metadata']['last_updated'] = datetime.datetime.now().isoformat(timespec='seconds')
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def set_metric(section, key, value, source=None, **extra):
    """Idempotent write of a single metric.

    Args:
        section: top-level grouping (e.g. 'm2_perf', 'factor_alphas', 'bootstrap').
        key: metric name within the section (e.g. 'sharpe', 'capm_alpha').
        value: the numeric value.
        source: script that computed it (typically pass __file__).
        **extra: optional fields like ci_lo, ci_hi, t, p, stars.
    """
    data = _load()
    section_dict = data.setdefault(section, {})
    entry = {
        'value': value,
        'updated': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    if source:
        entry['source_script'] = os.path.relpath(source, os.path.dirname(os.path.dirname(PATH)))
    entry.update({k: v for k, v in extra.items() if v is not None})
    section_dict[key] = entry
    _save(data)
    return entry


def set_many(section, metrics_dict, source=None):
    """Batch-update many metrics in one section. Each value can be a number
    or a dict with 'value' + extras. Single transaction (one save)."""
    data = _load()
    section_dict = data.setdefault(section, {})
    now = datetime.datetime.now().isoformat(timespec='seconds')
    for key, val in metrics_dict.items():
        if isinstance(val, dict):
            entry = dict(val)
            entry.setdefault('updated', now)
        else:
            entry = {'value': val, 'updated': now}
        if source:
            entry['source_script'] = os.path.relpath(source, os.path.dirname(os.path.dirname(PATH)))
        section_dict[key] = entry
    _save(data)


def get_metric(section, key, default=None):
    data = _load()
    return data.get(section, {}).get(key, default)


def get_value(section, key, default=None):
    """Convenience: get just the .value field, or default."""
    entry = get_metric(section, key)
    if entry is None:
        return default
    return entry.get('value', default)


if __name__ == '__main__':
    # Quick CLI: dump current metrics
    data = _load()
    print(json.dumps(data, indent=2, sort_keys=True))
