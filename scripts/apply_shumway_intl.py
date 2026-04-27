"""
apply_shumway_intl.py
=====================
Compustat-Global analogue of `scripts/apply_shumway_delisting.py` for the
UK and Japan stock panels. Imputes a delisting return where a security
disappears from the panel mid-sample with a missing final-month `ret` and
a clear price-drop signal in the final available month.

Two modes
---------
**Strict mode** (preferred): when the panel includes the `secstat` and
`dldte` columns from `comp.g_security` (added by `import_intl_stocks.py`
in the re-pull required for this step), apply rule-based imputation
analogous to CRSP Shumway:

  Strict-1: `secstat = 'I'` AND row date is within 90 days of `dldte` AND
  `ret` is missing → `ret_adj = -0.30` (Shumway constant for distressed
  delistings without observed return).

  Strict-2: `secstat = 'I'` AND row date is within 90 days of `dldte` AND
  `ret` is present → compound `ret_adj = (1 + ret) * (1 - 0.30) - 1`.

  Strict-3: `secstat = 'A'` (active) OR row is not within the delisting
  window → passthrough `ret_adj = ret`.

We use Shumway's flat -30% because Compustat Global doesn't decompose
delisting reasons into performance vs. non-performance for most regions;
it is the conservative default consistent with Shumway (1997, 2001) and
matches the imputation our CRSP US script applies in its rule-4 case.

**Heuristic mode** (fallback): when `secstat`/`dldte` are absent (legacy
panels built before the field-augmented re-pull), fall back to a
price-drop heuristic that does NOT require those fields:

  Definition of "disappeared" — the security's last appearance is at
  least 90 days before the panel-end date.

  Rule 1 (compound observed `ret` if both `ret` and a prior-month
  `prc_close` are available, AND the security disappeared, AND the
  computed prc_close-based delisting return is materially negative)
  → `ret_adj = (1 + ret) * (1 + computed_dl_ret) - 1`
  We compound rather than overwrite to preserve the within-month
  observed return.

  Rule 2 (`ret` is missing for the final row but a price-drop signal is
  present: `prc_close / prev_prc_close < 0.7` and security disappeared)
  → `ret_adj = prc_close / prev_prc_close - 1` (use the drop directly).

  Rule 3 (`ret` is missing AND the security disappeared AND no clear
  price drop, OR the security disappeared with a normal final return)
  → `ret_adj = ret` (passthrough; could be a merger, going-private, or a
  data gap that we should not re-impute against).

  Rule 4 (security still active, OR `ret` is present and not extreme)
  → `ret_adj = ret` (passthrough).

The heuristic is intentionally LESS aggressive than CRSP Shumway. Use
strict mode whenever the panel supports it.

The script does NOT overwrite the existing `ret` column. It adds a new
`ret_adj` column. Downstream scripts (`scripts/cross_sectional_intl.py`)
must read `ret_adj` if present (with `ret` as fallback for backward
compatibility).

Backup + atomic write
---------------------
On the first run, the source file is copied to
`data/{region}_stock_panel.parquet.bak_before_shumway_intl`. Subsequent
runs reload from the backup so the rules are idempotent.

Usage
-----
    python scripts/apply_shumway_intl.py --region UK
    python scripts/apply_shumway_intl.py --region JP --force
    python scripts/apply_shumway_intl.py --region UK --dry-run

Outputs
-------
    data/{region}_stock_panel.parquet               (with new ret_adj column)
    data/{region}_stock_panel.parquet.bak_before_shumway_intl  (first run only)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

REQUIRED_COLS = {'secid', 'date', 'prc_close', 'ret'}
STRICT_COLS = {'secstat', 'dldte'}

# Disappearance grace window: securities whose last observation is older
# than this before the panel end are treated as disappeared.
DISAPPEAR_GRACE_DAYS = 90

# Strict mode: a row is "in the delisting window" if its date is within
# this many days of the security's `dldte`. The Shumway imputation fires
# only on the row(s) that fall in this window.
DELISTING_WINDOW_DAYS = 90

# Shumway's canonical imputed return for performance delistings without
# an observed return (CRSP-equivalent rule 4 / strict-1 here).
SHUMWAY_DLRET = -0.30

# Compustat-Global `dlrsn` (aliased from `dlrsni`) codes that we treat as
# performance delistings, analogous to CRSP `dlstcd ∈ [500, 599]`. These
# are bankruptcy (02) and liquidation (03). Other inactive codes (01
# acquisition/merger, 09 going-private, 20 data-source change, etc.)
# represent non-distressed delistings and get passthrough — same as CRSP
# Shumway leaves dlstcd ∉ [500, 599] alone.
PERFORMANCE_DLRSN = {'02', '03'}

# Price-drop threshold (heuristic Rule 2): only trigger imputation if the
# final month's price is < this fraction of the prior month's price.
PRICE_DROP_THRESHOLD = 0.70

# Heuristic Rule 1 trigger: a "materially negative" computed delisting
# return is anything below this (compound only when delisting return < -10%).
COMPOUND_TRIGGER = -0.10


def panel_path(region):
    return REPO / 'data' / f'{region.lower()}_stock_panel.parquet'


def backup_path(region):
    return REPO / 'data' / f'{region.lower()}_stock_panel.parquet.bak_before_shumway_intl'


def _ensure_backup(src, bak):
    """Create backup iff it doesn't already exist. Returns True if created."""
    src, bak = Path(src), Path(bak)
    if bak.exists():
        return False
    shutil.copy2(src, bak)
    return True


def _load_source(src, bak):
    """Load from backup if present (idempotent rerun); otherwise from src."""
    if Path(bak).exists():
        return pd.read_parquet(bak), bak
    return pd.read_parquet(src), src


def _atomic_write(df, dst, tmp):
    """Write to tmp, then rename to dst (atomic on POSIX)."""
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dst)


def _apply_strict(out):
    """Strict mode: rule-based Shumway analogue using secstat/dldte (and
    dlrsn if present).

    Fires on rows where:
      - secstat == 'I' (security is inactive), AND
      - the row's date is within DELISTING_WINDOW_DAYS of the security's
        dldte (i.e., this is the actual delisting month, not earlier
        history), AND
      - if `dlrsn` is present, it is in PERFORMANCE_DLRSN ({02, 03}).
        Without `dlrsn` we fall back to applying to ALL inactive
        in-window rows (legacy/conservative path).

    Returns rule counts dict; mutates `out['ret_adj']` in place."""
    counts = {
        'strict_1_impute_minus30_no_ret': 0,
        'strict_2_compound_ret_with_minus30': 0,
        'strict_3_passthrough_nonperformance': 0,
        'strict_4_passthrough_other': 0,
    }
    secstat = out['secstat'].astype(str).str.upper().str.strip()
    is_inactive = secstat == 'I'

    dldte = pd.to_datetime(out['dldte'], errors='coerce')
    days_to_dldte = (dldte - out['date']).dt.days.abs()
    in_window = is_inactive & dldte.notna() & (
        days_to_dldte <= DELISTING_WINDOW_DAYS
    )

    if 'dlrsn' in out.columns:
        dlrsn = out['dlrsn'].astype('string').str.strip()
        is_performance = dlrsn.isin(PERFORMANCE_DLRSN)
        impute_mask = in_window & is_performance
        # Track non-performance inactives separately for diagnostics
        counts['strict_3_passthrough_nonperformance'] = int(
            (in_window & ~is_performance).sum()
        )
    else:
        # Legacy panel without dlrsn — apply to all inactive-in-window
        # rows (less precise but conservative; the false-positive rate
        # is bounded by how many non-performance delistings exist).
        impute_mask = in_window
        counts['strict_3_passthrough_nonperformance'] = 0

    has_ret = out['ret'].notna()

    # Strict-1: imputable + ret missing → impute -0.30
    s1 = impute_mask & ~has_ret
    if s1.any():
        out.loc[s1, 'ret_adj'] = SHUMWAY_DLRET
        counts['strict_1_impute_minus30_no_ret'] = int(s1.sum())

    # Strict-2: imputable + ret present → compound (1+ret)(1-0.30)-1
    s2 = impute_mask & has_ret
    if s2.any():
        out.loc[s2, 'ret_adj'] = ((1 + out.loc[s2, 'ret'])
                                   * (1 + SHUMWAY_DLRET) - 1)
        counts['strict_2_compound_ret_with_minus30'] = int(s2.sum())

    # Strict-4: every other row in the panel (active securities, plus
    # inactive rows outside the delisting window — the security's
    # pre-distress history)
    counts['strict_4_passthrough_other'] = int(
        len(out) - counts['strict_1_impute_minus30_no_ret']
        - counts['strict_2_compound_ret_with_minus30']
        - counts['strict_3_passthrough_nonperformance']
    )
    return counts


def apply_shumway_intl(df):
    """Apply Shumway-analogue delisting treatment. Strict mode if the
    panel includes secstat/dldte columns; heuristic price-drop fallback
    otherwise. Returns (out_df, counts).

    out_df has the same rows as df plus a new `ret_adj` column.
    counts is a dict of rule firing counts (with a 'mode' key indicating
    which rule set fired)."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f'panel missing required columns: {missing}')

    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values(['secid', 'date']).reset_index(drop=True)

    # Initialise ret_adj as a copy of ret (cast to float so nullable
    # dtypes don't surprise downstream consumers)
    out['ret_adj'] = out['ret'].astype(float)

    if STRICT_COLS.issubset(set(df.columns)):
        strict_counts = _apply_strict(out)
        return out, {'mode': 'strict', **strict_counts}

    # Heuristic mode (legacy panels)
    panel_end = out['date'].max()
    cutoff = panel_end - pd.Timedelta(days=DISAPPEAR_GRACE_DAYS)

    # Per-security last-observation date
    last_dates = out.groupby('secid')['date'].transform('max')
    is_disappeared = last_dates < cutoff
    is_last_row = out['date'] == last_dates
    is_terminal_disappeared_row = is_disappeared & is_last_row

    # Compute prior-month prc_close per security via shift
    prev_close = out.groupby('secid', sort=False)['prc_close'].shift(1)

    counts = {
        'mode': 'heuristic',
        'rule_1_compound_ret_with_drop': 0,
        'rule_2_drop_only': 0,
        'rule_3_disappeared_no_signal': 0,
        'rule_4_unchanged': int(len(out)),
    }

    if not is_terminal_disappeared_row.any():
        return out, counts

    # Vectorised computation on the terminal-disappeared subset
    sub_idx = out.index[is_terminal_disappeared_row]
    prev_close_sub = prev_close.loc[sub_idx]
    last_close_sub = out.loc[sub_idx, 'prc_close']
    ret_sub = out.loc[sub_idx, 'ret']

    # Computed delisting return from price drop (NaN if prev not available
    # or zero/negative)
    safe_prev = prev_close_sub.where(prev_close_sub > 0, np.nan)
    computed_dl = (last_close_sub / safe_prev) - 1.0

    has_ret = ret_sub.notna()
    has_drop = computed_dl.notna() & (computed_dl < COMPOUND_TRIGGER)
    has_strong_drop = computed_dl.notna() & ((last_close_sub / safe_prev)
                                              < PRICE_DROP_THRESHOLD)

    # Rule 1: ret present + meaningful drop available → compound
    rule1_mask = has_ret & has_drop
    rule1_idx = sub_idx[rule1_mask]
    if len(rule1_idx) > 0:
        out.loc[rule1_idx, 'ret_adj'] = (
            (1 + ret_sub[rule1_mask]) * (1 + computed_dl[rule1_mask]) - 1
        )
        counts['rule_1_compound_ret_with_drop'] = int(len(rule1_idx))

    # Rule 2: ret missing + strong drop → use the drop directly
    rule2_mask = (~has_ret) & has_strong_drop
    rule2_idx = sub_idx[rule2_mask]
    if len(rule2_idx) > 0:
        out.loc[rule2_idx, 'ret_adj'] = computed_dl[rule2_mask]
        counts['rule_2_drop_only'] = int(len(rule2_idx))

    # Rule 3: disappeared without a usable signal → leave (counts only)
    rule3_mask = (~rule1_mask) & (~rule2_mask)
    counts['rule_3_disappeared_no_signal'] = int(rule3_mask.sum())

    # Rule 4: every other row in the panel
    n_touched = (counts['rule_1_compound_ret_with_drop']
                 + counts['rule_2_drop_only'])
    counts['rule_4_unchanged'] = int(len(out) - n_touched)

    return out, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', choices=['UK', 'JP'], required=True)
    parser.add_argument('--force', action='store_true',
                        help='Skip the confirmation prompt')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute counts but do not write any file')
    args = parser.parse_args()

    src = panel_path(args.region)
    bak = backup_path(args.region)
    if not src.exists():
        sys.exit(f'  [ERROR] {src} not found')

    print('=' * 70)
    print(f'  Compustat Global Shumway-analogue delisting treatment ({args.region})')
    print('=' * 70)
    df, src_used = _load_source(src, bak)
    print(f'  Loaded: {src_used} ({len(df):,} rows, '
          f'{df["secid"].nunique():,} securities)')

    out, counts = apply_shumway_intl(df)

    print('\n  Rule firings:')
    for k, v in counts.items():
        if isinstance(v, str):
            print(f'    {k:<35s}: {v:>10}')
        else:
            print(f'    {k:<35s}: {v:>10,}')

    n_changed = int((out['ret_adj'] != out['ret']).sum()
                    - (out['ret_adj'].isna() & out['ret'].isna()).sum())
    print(f'\n  ret_adj != ret rows: {n_changed:,} '
          f'({n_changed / len(out) * 100:.4f}% of panel)')

    if args.dry_run:
        print('\n  [DRY RUN] no file written.')
        return

    if not args.force:
        prompt = f'  Apply to {src} (creates backup if missing) [y/N]? '
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            ans = 'n'
        if ans != 'y':
            print('  aborted.')
            return

    if _ensure_backup(src, bak):
        print(f'  Backup created: {bak}')
    else:
        print(f'  Backup already exists: {bak}')

    tmp = Path(str(src) + '.tmp')
    _atomic_write(out, src, tmp)
    print(f'  Panel written:  {src}')
    print(f'\n  Downstream impact: scripts/cross_sectional_intl.py and any')
    print(f'  other consumer must use the new ret_adj column. ret is')
    print(f'  preserved unchanged for diagnostic comparison.')


if __name__ == '__main__':
    main()
