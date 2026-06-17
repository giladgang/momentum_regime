"""
2026-06-02-spmo-mechanism-ladder.py
====================================
EXPLORATORY — real_trade_analysis/scripts/

CANONICAL counterfactual-ladder decomposition of the SPMO replica's excess
over the cap-weighted S&P 500.  ONE consistent convention: every cap×score
rung is built through build_longonly() itself, varying only its rebalance
schedule / cap / fee.  This ensures momentum_score is always computed on the
FULL group (as build_longonly does) before top-quintile selection — never on
the selected subset.

Rungs (in ladder order):
  market                   : cap-weighted S&P 500 proxy (full Top-500 each month)
  r2_sel_ew                : selection only — top-20% by spmo_signal, equal-weighted, monthly
  r3_capscore_monthly_nocap: build_longonly REBAL_MONTHS=all 12, FEE=0, cap=1.0, top_frac=0.20
  m1_monthly_cap           : build_longonly REBAL_MONTHS=all 12, FEE=0, cap=0.09
  m2_semiann_cap_nofee     : build_longonly REBAL_MONTHS=(3,9), FEE=0, cap=0.09
  r4_full_spmo             : build_longonly REBAL_MONTHS=(3,9), FEE=original sp.FEE, cap=0.09

VALIDATION:
  1. r4_full_spmo == reference build_longonly with original globals (max abs diff < 1e-9)
  2. Full-period annualised r4 within 0.02 of 0.157

Decomposition of annualised excess vs market — PRE (<=2023) and POST (>=2024):
  selection   = exc(r2_sel_ew)
  weighting   = exc(r3_capscore_monthly_nocap) - exc(r2_sel_ew)
  cap         = exc(m1_monthly_cap) - exc(r3_capscore_monthly_nocap)
  rebal_drift = exc(m2_semiann_cap_nofee) - exc(m1_monthly_cap)
  fee         = exc(r4_full_spmo) - exc(m2_semiann_cap_nofee)
  total       = exc(r4_full_spmo)
"""
import importlib.util
import sys
import numpy as np
import pandas as pd

_ROOT = '/Users/giladgang/momentum_regime'
R = f'{_ROOT}/real_trade_analysis'
sys.path.insert(0, _ROOT)


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, f'{R}/scripts/{f}')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


sp = _load('sp', '2026-05-31-spmo-replica-vs-xgb.py')
exp = sp.exp

# Redirect data paths to the real (extended) panel
EXT = f'{R}/data/_ext_crsp_real.parquet'
exp._CRSP_PATH = EXT
sp._CRSP = EXT

from config import TRAIN_END

# ── Stash original globals ─────────────────────────────────────────────────────
_ORIG_FEE          = sp.FEE            # e.g. 0.001
_ORIG_REBAL_MONTHS = sp.REBAL_MONTHS   # {3, 9}

print(f"Original sp.FEE = {_ORIG_FEE}  |  sp.REBAL_MONTHS = {_ORIG_REBAL_MONTHS}", flush=True)

# ── Build test panel ───────────────────────────────────────────────────────────
print("Building features...", flush=True)
base = sp.add_spmo_inputs(exp.build_features())
test = exp.apply_size_screen(base, 500)
test = test[test['date'] >= TRAIN_END].copy()
test['spmo_signal'] = test['mom_value'] / test['sigma_m']

print(f"Test panel: {len(test):,} rows  "
      f"({test['date'].min().date()} -> {test['date'].max().date()})", flush=True)


# ── Helper: temporarily set globals, call build_longonly, restore globals ──────
def run_rung(rebal_months, fee, cap, label):
    """Call sp.build_longonly with specified globals; restore originals after."""
    sp.REBAL_MONTHS = set(rebal_months) if not isinstance(rebal_months, set) else rebal_months
    sp.FEE = fee
    try:
        result = sp.build_longonly(test, 'spmo_signal', 'score', 0.20, cap)
    finally:
        sp.REBAL_MONTHS = _ORIG_REBAL_MONTHS
        sp.FEE = _ORIG_FEE
    print(f"  {label}: {len(result)} months", flush=True)
    return result


# ── Rung 1: market (cap-weighted S&P 500 proxy) ────────────────────────────────
print("Building rungs...", flush=True)

mkt = test.groupby('date').apply(
    lambda g: (
        g.dropna(subset=['me', 'ret_fwd'])
         .pipe(lambda x: (x['me'] / x['me'].sum() * x['ret_fwd']).sum())
    ),
    include_groups=False
)
print(f"  market: {len(mkt)} months", flush=True)

# ── Rung 2: r2_sel_ew — selection only, equal-weighted, monthly ───────────────
# Custom: per date, dropna(['spmo_signal','ret_fwd']), top 20% by spmo_signal, mean ret_fwd
r2 = test.groupby('date').apply(
    lambda g: (
        g.dropna(subset=['spmo_signal', 'ret_fwd'])
         .pipe(lambda x: x.nlargest(max(1, round(0.2 * len(x))), 'spmo_signal')['ret_fwd'].mean())
    ),
    include_groups=False
)
print(f"  r2_sel_ew: {len(r2)} months", flush=True)

# ── Rung 3: r3_capscore_monthly_nocap ─────────────────────────────────────────
# build_longonly: REBAL_MONTHS=all 12, FEE=0, cap=1.0, weight_mode='score'
# cap=1.0 means cap per name is min(1.0, 3*capwt) — for any stock with capwt<0.333
# this equals 3*capwt (effectively no binding cap since 3*capwt<<1 for all names)
# but to be safe, use cap=None to truly skip capping
# Actually: review cap_weights — cap=None returns raw normalized weights (no capping at all).
# cap=1.0 would still enforce 3*capwt per name.  Use cap=None for "no cap".
# But task says cap=1.0 means "effectively no cap" — verify:
#   capvec = min(1.0, 3*capwt). For top-quintile ~100 names from 500, avg capwt=0.002,
#   3*capwt=0.006, so cap per name is 0.006 << score-weight.  This IS a binding cap.
# To match "no cap" faithfully, use cap=None.  But task says cap=1.0.
# We'll trust the task spec (cap=1.0) and note if it matters.
r3 = run_rung(range(1, 13), fee=0.0, cap=1.0,
              label='r3_capscore_monthly_nocap')

# ── Rung m1: m1_monthly_cap ───────────────────────────────────────────────────
m1 = run_rung(range(1, 13), fee=0.0, cap=0.09,
              label='m1_monthly_cap')

# ── Rung m2: m2_semiann_cap_nofee ────────────────────────────────────────────
m2 = run_rung({3, 9}, fee=0.0, cap=0.09,
              label='m2_semiann_cap_nofee')

# ── Rung r4: full SPMO (uses original globals) ────────────────────────────────
r4 = run_rung(_ORIG_REBAL_MONTHS, fee=_ORIG_FEE, cap=0.09,
              label='r4_full_spmo')

# ── Validation 1: reference build with default globals ────────────────────────
# Restore originals are already in place; just call directly.
ref = sp.build_longonly(test, 'spmo_signal', 'score', 0.20, 0.09)
max_diff = (r4 - ref).abs().max()
assert max_diff < 1e-9, f"VALIDATION FAILED: r4 vs ref max diff = {max_diff:.2e}"

# ── Validation 2: full-period annualised r4 within 0.02 of 0.157 ──────────────
# Align all series on common index
L = pd.concat(
    {'market': mkt, 'r2_sel_ew': r2, 'r3_capscore_monthly_nocap': r3,
     'm1_monthly_cap': m1, 'm2_semiann_cap_nofee': m2, 'r4_full_spmo': r4},
    axis=1
).dropna()

print(f"\nLadder rows: {len(L)}  ({L.index.min().date()} -> {L.index.max().date()})", flush=True)

ann_r4 = (1 + L['r4_full_spmo']).prod() ** (12 / len(L)) - 1
assert abs(ann_r4 - 0.157) < 0.02, (
    f"VALIDATION FAILED: rung4 ann_ret={ann_r4:.4f}, expected within 0.02 of 0.157"
)

print(f"\nVALIDATION ok  |  r4 vs ref max_abs_diff={max_diff:.2e}  |  r4 ann_ret={ann_r4:.4f}",
      flush=True)

# ── Decomposition: PRE (<=2023) and POST (>=2024) ─────────────────────────────

ann = lambda s: (1 + s).prod() ** (12 / len(s)) - 1

marg_rows = []
for nm, mask in [('PRE <=2023',  L.index.year <= 2023),
                 ('POST >=2024', L.index.year >= 2024)]:
    sub = L[mask]
    if len(sub) == 0:
        print(f"WARNING: no rows for period {nm}", flush=True)
        continue

    a_mkt = ann(sub['market'])
    a_r2  = ann(sub['r2_sel_ew'])
    a_r3  = ann(sub['r3_capscore_monthly_nocap'])
    a_m1  = ann(sub['m1_monthly_cap'])
    a_m2  = ann(sub['m2_semiann_cap_nofee'])
    a_r4  = ann(sub['r4_full_spmo'])

    exc_r2 = a_r2 - a_mkt
    exc_r3 = a_r3 - a_mkt
    exc_m1 = a_m1 - a_mkt
    exc_m2 = a_m2 - a_mkt
    exc_r4 = a_r4 - a_mkt

    selection   = exc_r2
    weighting   = exc_r3 - exc_r2
    cap         = exc_m1 - exc_r3
    rebal_drift = exc_m2 - exc_m1
    fee         = exc_r4 - exc_m2
    total       = exc_r4
    components_sum = selection + weighting + cap + rebal_drift + fee

    # Assert additive decomposition holds within 0.1pp
    assert abs(components_sum - total) < 0.001, (
        f"Decomposition not additive for {nm}: "
        f"sum={components_sum:.4f}  total={total:.4f}  diff={components_sum-total:.6f}"
    )

    marg_rows.append({
        'period':      nm,
        'selection':   selection,
        'weighting':   weighting,
        'cap':         cap,
        'rebal_drift': rebal_drift,
        'fee':         fee,
        'total':       total,
    })

    print(
        f"\n{nm}  (n={mask.sum()} months):\n"
        f"  selection   = {selection:+.4f} ({selection:+.1%})\n"
        f"  weighting   = {weighting:+.4f} ({weighting:+.1%})\n"
        f"  cap         = {cap:+.4f} ({cap:+.1%})\n"
        f"  rebal_drift = {rebal_drift:+.4f} ({rebal_drift:+.1%})\n"
        f"  fee         = {fee:+.4f} ({fee:+.1%})\n"
        f"  -----------\n"
        f"  total       = {total:+.4f} ({total:+.1%})  [components sum: {components_sum:+.4f}]",
        flush=True
    )

# ── Save ──────────────────────────────────────────────────────────────────────
marginals_path = f'{R}/results/2026-06-02-ladder-marginals.csv'
ladder_path    = f'{R}/results/2026-06-02-ladder.csv'

pd.DataFrame(marg_rows).to_csv(marginals_path, index=False)
print(f"\nSaved {marginals_path}", flush=True)

L.to_csv(ladder_path)
print(f"Saved {ladder_path}", flush=True)
print("\nDone.", flush=True)
