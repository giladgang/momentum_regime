"""
validate_production.py
======================
Checks that production artefacts (pickle, tables, key numbers) are
internally consistent.  Designed to run:
  - As the final step of run_pipeline.py
  - After any experiment script restores production files
  - Manually at any time

Exit code 0 = all checks pass, 1 = at least one failure.
"""

import pickle, os, sys, re
import numpy as np
import pandas as pd

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load production pickle
# ══════════════════════════════════════════════════════════════════════════════

print("Validating production artefacts ...\n")

pkl_path = 'artefacts/cs_artefacts_data.pkl'
check("Pickle exists", os.path.exists(pkl_path))
if not os.path.exists(pkl_path):
    print("\nCannot continue without pickle. FAILED.")
    sys.exit(1)

with open(pkl_path, 'rb') as f:
    artefacts = pickle.load(f)

test = artefacts['test']
check("Test set loaded", len(test) > 0, f"len={len(test)}")
check("score_xgb in test", 'score_xgb' in test.columns)
check("score_lr in test", 'score_lr' in test.columns)
check("167 test months", test['date'].nunique() == 167,
      f"got {test['date'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Recompute key Sharpe ratios from the pickle and verify
# ══════════════════════════════════════════════════════════════════════════════

EXPECTED_SHARPES = {
    'score_xgb':     1.11,
    'score_lr':     -0.03,
    'score_mom12':   0.06,
    'score_mom1':    0.26,
    'score_formula': 0.11,
}

TOLERANCE = 0.02   # allow small float drift


def compute_ls_sharpe(test_df, score_col, fee=0.001):
    """Recompute long-short Sharpe from scratch."""
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in test_df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lme = longs['me'].sum()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return float('nan')
    r = pd.DataFrame(monthly).set_index('date')['ret']
    return round(r.mean() / r.std() * np.sqrt(12), 2)


print("\nSharpe ratio verification (from pickle scores):")
for score_col, expected in EXPECTED_SHARPES.items():
    actual = compute_ls_sharpe(test, score_col)
    check(f"{score_col} Sharpe={expected}",
          abs(actual - expected) <= TOLERANCE,
          f"got {actual}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Check that table_performance.tex matches the pickle
# ══════════════════════════════════════════════════════════════════════════════

print("\nTable consistency checks:")
perf_table = 'tables/table_performance.tex'
if os.path.exists(perf_table):
    with open(perf_table) as f:
        tex = f.read()

    # Extract M2 XGB Sharpe from table (look for "M2: XGB" row)
    # Pattern: M2 line with Sharpe value after Ann.Vol column
    m2_match = re.search(r'M2: XGB[^&]*&[^&]*&[^&]*&\s*([-\d.]+)', tex)
    if m2_match:
        table_sharpe = float(m2_match.group(1))
        check("table_performance M2 Sharpe matches pickle",
              abs(table_sharpe - EXPECTED_SHARPES['score_xgb']) <= TOLERANCE,
              f"table={table_sharpe}, expected={EXPECTED_SHARPES['score_xgb']}")
    else:
        check("table_performance M2 Sharpe parseable", False, "could not parse")

    # Check M1 LR Sharpe
    m1_match = re.search(r'M1: LR\b[^&]*&[^&]*&[^&]*&\s*([-\d.]+)', tex)
    if m1_match:
        table_sharpe = float(m1_match.group(1))
        check("table_performance M1 Sharpe matches pickle",
              abs(table_sharpe - EXPECTED_SHARPES['score_lr']) <= TOLERANCE,
              f"table={table_sharpe}, expected={EXPECTED_SHARPES['score_lr']}")

    # Check no duplicate rows -- "M2: XGB (mom+$\pi$)" is the main row;
    # "M2: XGB (mom+raw)" is a legitimate ablation row, not a duplicate.
    m2_main_count = len(re.findall(r'M2: XGB \(mom\+\$\\pi\$\)', tex))
    check("table_performance has exactly one M2 main row",
          m2_main_count == 1,
          f"found {m2_main_count} 'M2: XGB (mom+pi)' rows")
else:
    check("table_performance.tex exists", False)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Check panel_with_regimes matches
# ══════════════════════════════════════════════════════════════════════════════

print("\nPanel consistency checks:")
panel_path = 'data/panel_with_regimes.parquet'
if os.path.exists(panel_path):
    panel = pd.read_parquet(panel_path)
    check("panel has pi_filter", 'pi_filter' in panel.columns)
    n_months = panel['date'].nunique() if 'date' in panel.columns else 0
    check("panel covers full sample", n_months >= 400,
          f"got {n_months} months")
else:
    check("panel_with_regimes.parquet exists", False)

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 50}")
print(f"  {PASS} passed, {FAIL} failed")
print(f"{'=' * 50}")

sys.exit(1 if FAIL > 0 else 0)
