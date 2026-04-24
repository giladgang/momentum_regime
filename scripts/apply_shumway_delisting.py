"""
apply_shumway_delisting.py
==========================
Apply the Shumway (2001) delisting-return treatment to the CRSP monthly
panel at ``data/crsp_msf_raw.parquet``.

Why this script exists
----------------------
The thesis (`latex/data_section.tex`) states:

  "Following Shumway (2001), delisting returns are incorporated to
   prevent survivorship bias: actual delisting returns are used when
   available, and -30% is imputed for performance-related delistings
   with missing returns."

Prior to this script, the claim was not implemented: ``ret_adj`` in
the raw panel was essentially equal to ``ret`` for every row, ignoring
the 1,296 non-null ``dlret`` values and imputing only 1 performance
delisting. This script makes the claim true.

The five rules
--------------
For every stock-month row, ``ret_adj`` is set according to whichever
of the five mutually exclusive cases applies (first match wins):

  1. ``ret`` present AND ``dlret`` present
     -> ``ret_adj = (1 + ret) * (1 + dlret) - 1``
     (compound month return with delisting return)

  2. ``ret`` missing AND ``dlret`` present
     -> ``ret_adj = dlret``
     (delisting return is the only return observed)

  3. ``ret`` present AND ``dlret`` missing AND ``dlstcd`` in [500, 599]
     -> ``ret_adj = (1 + ret) * (1 + SHUMWAY_IMPUTE) - 1``
     (performance-related delisting with missing dlret -> impute the
      dlret at -30% per Shumway and compound with the month return)

  4. ``ret`` missing AND ``dlret`` missing AND ``dlstcd`` in [500, 599]
     -> ``ret_adj = SHUMWAY_IMPUTE``
     (both missing, performance delisting -> full -30%)

  5. otherwise
     -> ``ret_adj = ret`` (unchanged)

``dlstcd`` codes 500-599 are CRSP's "performance-related" delisting
codes, which is what Shumway's -30% imputation targets (bankruptcy,
insufficient capital, etc.).

Safety / idempotency
--------------------
A one-time backup is created at
``data/crsp_msf_raw.parquet.bak_before_shumway`` the first time this
script runs. On every subsequent run, the backup is treated as the
authoritative pre-Shumway state; the script restores from it and
re-applies the rules. Therefore:

  * Running twice yields the same output as running once.
  * If you need to revert, copy the backup over the main file.

The write is atomic: the new parquet is written to a ``.tmp`` path and
renamed over the original only after success.

IMPORTANT: downstream artefacts
-------------------------------
After running this script, the cached artefacts in
``artefacts/cs_artefacts_data.pkl`` and the panel in
``data/panel_with_regimes.parquet`` become stale. The full pipeline
(``python run_pipeline.py``) must be re-run to regenerate them. Your
headline Sharpe is expected to shift by a small amount (estimated
<0.05 given only ~1,300 affected rows out of 2M).

Usage
-----
    python scripts/apply_shumway_delisting.py --dry-run   # report only
    python scripts/apply_shumway_delisting.py             # apply
    python scripts/apply_shumway_delisting.py --force     # skip confirm
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

PANEL_PATH = 'data/crsp_msf_raw.parquet'
BACKUP_PATH = 'data/crsp_msf_raw.parquet.bak_before_shumway'
TMP_PATH = 'data/crsp_msf_raw.parquet.tmp'

# Shumway (2001) imputation for performance-related delistings with
# missing dlret. CRSP delisting codes 500-599 are "performance-related"
# (bankruptcy, insufficient capital, violations, liquidations).
SHUMWAY_IMPUTE = -0.30
PERF_DLSTCD_MIN = 500
PERF_DLSTCD_MAX = 599

# Required columns in the input panel.
REQUIRED_COLS = {'ret', 'dlret', 'dlstcd', 'ret_adj'}


# ═══════════════════════════════════════════════════════════════════════════════
# Core logic: apply the five rules
# ═══════════════════════════════════════════════════════════════════════════════

def apply_shumway(df):
    """Compute Shumway-adjusted ``ret_adj`` for every row in ``df``.

    Returns a new DataFrame (does not modify ``df``) plus a dict of
    counts describing how many rows each rule matched.

    The five rules are applied in order with mutually exclusive masks:
    no row can be touched by more than one rule.
    """
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input panel missing required columns: {sorted(missing)}. "
            f"Expected {sorted(REQUIRED_COLS)}, got {sorted(df.columns)}."
        )

    df = df.copy()
    ret = df['ret']
    dlret = df['dlret']
    dlstcd = df['dlstcd']

    # Force clean numpy-bool arrays. CRSP panels often use pandas nullable
    # dtypes (Float64 / Int64) where `.between()` on a NA returns NA; that
    # propagates through logical ops and corrupts row counts (NA is not
    # treated as False).
    ret_has = ret.notna().to_numpy(dtype=bool)
    dlret_has = dlret.notna().to_numpy(dtype=bool)
    is_perf = (
        dlstcd.between(PERF_DLSTCD_MIN, PERF_DLSTCD_MAX, inclusive='both')
              .fillna(False)
              .to_numpy(dtype=bool)
    )

    # Masks — each row matches at most one. All are plain numpy bools,
    # so ~, &, | behave as expected with no NA leakage.
    rule1 = ret_has & dlret_has                           # compound ret * dlret
    rule2 = (~ret_has) & dlret_has                        # dlret only
    rule3 = ret_has & (~dlret_has) & is_perf              # ret compounded with -30%
    rule4 = (~ret_has) & (~dlret_has) & is_perf           # -30% alone
    rule5 = ~(rule1 | rule2 | rule3 | rule4)              # unchanged

    # Safety: rules must partition the rows.
    total = rule1.sum() + rule2.sum() + rule3.sum() + rule4.sum() + rule5.sum()
    assert total == len(df), (
        f"Rule masks don't partition rows: "
        f"{rule1.sum()} + {rule2.sum()} + {rule3.sum()} + {rule4.sum()} + "
        f"{rule5.sum()} != {len(df)}"
    )

    # Apply. Start from `ret` and overwrite per rule.
    new_ret_adj = ret.copy().astype(float)

    new_ret_adj.loc[rule1] = (1.0 + ret.loc[rule1]) * (1.0 + dlret.loc[rule1]) - 1.0
    new_ret_adj.loc[rule2] = dlret.loc[rule2]
    new_ret_adj.loc[rule3] = (1.0 + ret.loc[rule3]) * (1.0 + SHUMWAY_IMPUTE) - 1.0
    new_ret_adj.loc[rule4] = SHUMWAY_IMPUTE
    # rule5 already has `ret` (including NaN for missing months)

    df['ret_adj'] = new_ret_adj

    counts = {
        'rule_1_compound_ret_dlret': int(rule1.sum()),
        'rule_2_dlret_only': int(rule2.sum()),
        'rule_3_compound_ret_shumway': int(rule3.sum()),
        'rule_4_shumway_only': int(rule4.sum()),
        'rule_5_unchanged': int(rule5.sum()),
    }
    return df, counts


# ═══════════════════════════════════════════════════════════════════════════════
# Summary reporting
# ═══════════════════════════════════════════════════════════════════════════════

def describe_changes(old_df, new_df, counts):
    """Human-readable summary of what the script did."""
    total = len(new_df)
    touched = counts['rule_1_compound_ret_dlret'] + counts['rule_2_dlret_only'] \
              + counts['rule_3_compound_ret_shumway'] + counts['rule_4_shumway_only']

    # Actual numerical changes (ret_adj differs)
    # Need to be careful about NaN comparison.
    old = old_df['ret_adj']
    new = new_df['ret_adj']
    both_na = old.isna() & new.isna()
    diff = ((old != new) | (old.isna() != new.isna())) & ~both_na
    n_changed = int(diff.sum())

    lines = [
        '=' * 70,
        '  Shumway (2001) delisting treatment — summary',
        '=' * 70,
        f'  Total rows:              {total:>10,}',
        f'  Touched by any rule:     {touched:>10,}  '
        f'({touched / total * 100:.3f}% of rows)',
        f'  ret_adj values changed:  {n_changed:>10,}  '
        f'(vs incoming file)',
        '',
        '  Per rule:',
        f'    1. ret + dlret -> compound:              {counts["rule_1_compound_ret_dlret"]:>6,}',
        f'    2. ret NaN, dlret present -> dlret:      {counts["rule_2_dlret_only"]:>6,}',
        f'    3. ret + perf delisting (no dlret)',
        f'       -> compound with -30%:                 {counts["rule_3_compound_ret_shumway"]:>6,}',
        f'    4. both NaN, perf delisting -> -30%:     {counts["rule_4_shumway_only"]:>6,}',
        f'    5. unchanged:                            {counts["rule_5_unchanged"]:>6,}',
        '=' * 70,
    ]
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# I/O: idempotent backup + atomic write
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_backup(panel_path, backup_path):
    """Create ``backup_path`` from ``panel_path`` iff the backup does
    not already exist. Return True iff a new backup was created."""
    if os.path.exists(backup_path):
        return False
    if not os.path.exists(panel_path):
        raise FileNotFoundError(panel_path)
    shutil.copy2(panel_path, backup_path)
    return True


def _load_source(panel_path, backup_path):
    """Load the pre-Shumway panel. If a backup exists, that's the
    authoritative source (idempotency); otherwise read the panel."""
    src = backup_path if os.path.exists(backup_path) else panel_path
    return pd.read_parquet(src), src


def _atomic_write(df, panel_path, tmp_path):
    """Write ``df`` to a tmp parquet next to the panel and rename over
    the original. Parquet writes are slow; a half-written file during
    a crash could corrupt the repo's primary data input. Rename is
    atomic on POSIX (same filesystem)."""
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, panel_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and report changes; do NOT write.')
    parser.add_argument('--force', action='store_true',
                        help='Skip the interactive confirmation prompt.')
    parser.add_argument('--panel', default=PANEL_PATH,
                        help=f'Path to CRSP monthly panel (default: {PANEL_PATH})')
    parser.add_argument('--backup', default=BACKUP_PATH,
                        help=f'Path to backup file (default: {BACKUP_PATH})')
    args = parser.parse_args()

    if not os.path.exists(args.panel):
        print(f'  ERROR: panel not found at {args.panel}', file=sys.stderr)
        return 1

    # Load the authoritative pre-Shumway state.
    src_df, src_path = _load_source(args.panel, args.backup)
    print(f'  Source: {src_path} ({len(src_df):,} rows)')

    # Apply rules.
    new_df, counts = apply_shumway(src_df)
    print(describe_changes(src_df, new_df, counts))

    if args.dry_run:
        print('\n  --dry-run: no files were modified.')
        return 0

    if not args.force:
        print('\n  This will OVERWRITE:', args.panel)
        print('  A one-time backup will be created (if not already) at:', args.backup)
        reply = input('  Proceed? [y/N]: ').strip().lower()
        if reply not in ('y', 'yes'):
            print('  Aborted.')
            return 1

    # Backup (idempotent) then atomic write.
    backup_created = _ensure_backup(args.panel, args.backup)
    _atomic_write(new_df, args.panel, TMP_PATH)

    if backup_created:
        print(f'  Backup created: {args.backup}')
    else:
        print(f'  Backup preserved: {args.backup}')
    print(f'  Panel written:  {args.panel}')
    print()
    print('  NEXT: artefacts/cs_artefacts_data.pkl and data/panel_with_regimes.parquet')
    print('        are now stale. Re-run the full pipeline:')
    print('            python run_pipeline.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
