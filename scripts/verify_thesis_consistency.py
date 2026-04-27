"""
Verify that numbers appearing in thesis prose (latex/*.tex) match the canonical
values in results/PRODUCTION_METRICS.json.

Strategy: scan latex prose for inline numbers near keywords that map to known
metrics (e.g. "Sharpe ratio of 1.11", "alpha of 24.7\\%"), and compare against
the JSON. Flag any mismatch.

This is intentionally a SOFT verifier: it doesn't catch every drift, but it
does flag the headline numbers wherever they appear in prose, which is the
high-leverage class of error.

Run after rebuilding metrics:

    python scripts/build_metrics.py
    python scripts/verify_thesis_consistency.py
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
METRICS_PATH = ROOT / 'results' / 'PRODUCTION_METRICS.json'


# Inline-number checks: (metric_section, metric_key, [latex_search_patterns])
# Each pattern uses (?P<v>...) to capture the in-prose value to compare.
CHECKS = [
    # M2 Sharpe
    ('m2_perf', 'm2_sharpe', [
        r'M2[^.]{0,150}?Sharpe(?: ratio)?(?: of)?[^.\d]{0,20}(?P<v>\d+\.\d+)',
        r'Sharpe(?: ratio)?[^.\d]{0,20}(?P<v>\d+\.\d+)[^.]{0,80}?(?:M2|XGBoost)',
    ]),
    # NW t-stat
    ('m2_perf', 'm2_nw_t', [
        r'Newey[--]?West[^.]{0,80}?(?:t|t-stat)(?:istic)?(?:[^.\d]{0,15})(?P<v>\d+\.\d+)',
    ]),
    # Factor alphas (annualised %)
    ('factor_alphas', 'capm_alpha', [
        r'CAPM[^.]{0,80}?alpha(?:[^.\d]{0,15})(?P<v>\d+\.\d+)\\?%?',
    ]),
    ('factor_alphas', 'ff6_alpha', [
        r'FF6[^.]{0,80}?(?:alpha|\$\\alpha\$)(?:[^.\d]{0,15})(?P<v>\d+\.\d+)\\?%?',
        r'six-factor alpha[^.]{0,40}?(?P<v>\d+\.\d+)\\?%?',
    ]),
    ('factor_alphas', 'carhart_alpha', [
        r'Carhart[^.]{0,80}?(?:alpha|\$\\alpha\$)(?:[^.\d]{0,15})(?P<v>\d+\.\d+)\\?%?',
    ]),
    # Bootstrap
    ('bootstrap', 'm2_sharpe', [
        r"\\?\[\s*(?P<v>\d+\.\d+)[,\s]+\d+\.\d+\s*\\?\]",  # generic [x, y] interval — too noisy alone, but ok with section context above
    ]),
]

# Tolerance: how much can prose differ from JSON before flagging?
TOL_FRAC = 0.02   # 2% relative
TOL_ABS  = 0.02   # 0.02 absolute (for small values like Sharpe ~1)


def _close(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) <= max(TOL_ABS, TOL_FRAC * abs(max(abs(a), abs(b), 1)))


def main():
    if not METRICS_PATH.exists():
        print(f'ERROR: {METRICS_PATH} not found. Run scripts/build_metrics.py first.')
        sys.exit(1)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    flagged = []
    ok_count = 0

    for tex_path in sorted((ROOT / 'latex').rglob('*.tex')):
        rel = tex_path.relative_to(ROOT)
        text = tex_path.read_text()
        for section, key, patterns in CHECKS:
            entry = metrics.get(section, {}).get(key)
            if not entry:
                continue
            canonical = entry.get('value')
            for pat in patterns:
                for m in re.finditer(pat, text, flags=re.DOTALL):
                    try:
                        prose_value = float(m.group('v'))
                    except (KeyError, ValueError):
                        continue
                    # If the canonical is a percent number, the prose may be expressing
                    # it as a percent (e.g. 24.1) — both should match.
                    if entry.get('unit') == 'percent':
                        # Two-way: try percent-as-value (24.1 → 0.241) AND raw (24.1 → 24.1)
                        targets = [canonical, canonical * 100 if abs(canonical) < 1 else canonical / 100]
                    else:
                        targets = [canonical]
                    if not any(_close(prose_value, t) for t in targets):
                        line_no = text[:m.start()].count('\n') + 1
                        ctx = text[max(0, m.start() - 40):m.end() + 30].replace('\n', ' ')
                        flagged.append({
                            'file': str(rel), 'line': line_no,
                            'section': section, 'key': key,
                            'prose_value': prose_value, 'canonical': canonical,
                            'context': ctx,
                        })
                    else:
                        ok_count += 1

    print(f'\nThesis-consistency check: {ok_count} matches, {len(flagged)} flagged')
    print('=' * 78)
    if flagged:
        for f in flagged:
            print(f"\n  ⚠ {f['file']}:{f['line']}  [{f['section']}.{f['key']}]")
            print(f"    prose says: {f['prose_value']}  canonical: {f['canonical']}")
            print(f"    context: ...{f['context']}...")
        print('\nThis is a SOFT check — these flags are informational, not blocking.')
        print('The pipeline did not fail. Walk results/PROSE_DRIFT_REPORT.md for the full picture.')
    else:
        print('  ✓ All checked numbers match canonical metrics.')
    # Always exit 0 — drift is informational, not a pipeline failure.
    return 0


if __name__ == '__main__':
    sys.exit(main())
