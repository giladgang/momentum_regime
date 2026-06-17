"""
2026-06-02-mechanics-decomp.py
==============================
EXPLORATORY — real_trade_analysis/scripts/

Decomposes the SPMO "mechanics" rung from the counterfactual ladder
(+11.4% POST excess) into three components:
  cap effect           = adding 9% cap to monthly cap×score   (m1 − r3)
  rebalance/drift      = monthly → semi-annual rebalance       (m2 − m1)
  fee effect           = transaction costs                     (r4 − m2)

Four portfolios:
  r3 = monthly, cap×score weights, NO cap,  NO fee
       [uses STATELESS groupby approach — identical to ladder's r3_capscore]
  m1 = monthly, cap×score weights, WITH 9% cap, NO fee
  m2 = semi-annual (Mar/Sep), WITH 9% cap, NO fee
  r4 = semi-annual (Mar/Sep), WITH 9% cap, WITH fee  (= full SPMO replica)

NOTE on r3 construction:
  The ladder's sel_capscore selects top-n by raw spmo_signal, then applies
  momentum_score to the SELECTED subgroup (100 stocks). build_longonly in the
  sp module applies momentum_score to the FULL 500-stock group first, then
  selects top-n — the z-score population differs, changing weights.
  To ensure r3 matches the ladder exactly, we use the stateless groupby
  (same as ladder) for r3. m1/m2/r4 use a local stateful builder (build_v2)
  that replicates select-then-score with added state for drift/caps/fees.

Also reports average monthly turnover for r3 vs r4, and NVIDIA (permno 86580)
avg/max weight in POST period for each.
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


sp  = _load('sp',  '2026-05-31-spmo-replica-vs-xgb.py')
exp = sp.exp

from config import TRAIN_END

EXT = f'{R}/data/_ext_crsp_real.parquet'
exp._CRSP_PATH = EXT
sp._CRSP       = EXT

ORIG_FEE          = sp.FEE          # 0.001
ORIG_REBAL_MONTHS = sp.REBAL_MONTHS  # {3, 9}
print(f"Original FEE={ORIG_FEE}  REBAL_MONTHS={ORIG_REBAL_MONTHS}", flush=True)

# ── Build test panel (same as ladder) ─────────────────────────────────────────
print("Building features...", flush=True)
base = sp.add_spmo_inputs(exp.build_features())
test = exp.apply_size_screen(base, 500)
test = test[test['date'] >= TRAIN_END].copy()
test['spmo_signal'] = test['mom_value'] / test['sigma_m']

print(f"Test panel: {len(test):,} rows  "
      f"({test['date'].min().date()} -> {test['date'].max().date()})", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ann(s):
    s = s.dropna()
    return (1 + s).prod() ** (12 / len(s)) - 1


NVIDIA_PERMNO = 86580


# ── r3: stateless groupby (identical to ladder's sel_capscore) ────────────────
# Selects top-n by spmo_signal, THEN applies momentum_score to SELECTED subgroup.
# No drift, no state, no cap, no fee.

def _r3_month(g):
    g = g.dropna(subset=['spmo_signal', 'me', 'ret_fwd'])
    n = max(1, round(0.2 * len(g)))
    s = g.nlargest(n, 'spmo_signal').copy()
    ms = sp.momentum_score(s['spmo_signal'])
    ms = pd.Series(ms.values if hasattr(ms, 'values') else ms, index=s.index)
    raw = s['me'] * ms
    w = raw / raw.sum()
    return (w * s['ret_fwd']).sum()


# ── build_v2: stateful builder that replicates ladder scoring convention ──────
# Selects top-n by raw signal, THEN applies momentum_score to the selected
# subgroup (matching ladder's population). Adds drift between non-rebal months,
# cap, and fee.  rebal_months, fee, and cap are explicit args (no monkeypatching).

def build_v2(test, top_frac, cap, rebal_months, fee, track_permno=None):
    """
    Long-only, hold-and-drift between rebalances.
    select-then-score convention (matches ladder r3 scoring population).
    Returns (ret_series, avg_monthly_turnover, permno_avg_wt, permno_max_wt).
    """
    monthly   = []
    turnovers = []
    perm_wts  = []
    prev_w    = {}

    for date, grp in test.groupby('date'):
        is_rebal = (date.month in rebal_months) or (not prev_w)
        if is_rebal:
            g = grp.dropna(subset=['me', 'spmo_signal', 'ret_fwd']).copy()
            if len(g) < 20:
                continue
            n_sel = max(1, round(top_frac * len(g)))
            # select top-n by raw signal first
            sel   = g.nlargest(n_sel, 'spmo_signal').copy()
            # then apply momentum_score to the SELECTED subgroup
            ms    = sp.momentum_score(sel['spmo_signal'])
            ms    = pd.Series(ms.values if hasattr(ms, 'values') else ms,
                              index=sel.index)
            raw   = sel['me'] * ms
            w_s   = sp.cap_weights(raw, sel['me'], cap, sp.WCAP_MULT)
            w     = {p: float(v) for p, v in zip(sel['permno'], w_s)}
            retmap = sel.set_index('permno')['ret_fwd']
            turn  = 0.5 * sum(abs(w.get(p, 0) - prev_w.get(p, 0))
                              for p in set(w) | set(prev_w))
            r     = sum(w[p] * retmap[p] for p in w) - fee * turn
            turnovers.append(turn)
            if track_permno is not None:
                perm_wts.append({'date': date, 'wt': w.get(track_permno, 0.0)})
            prev_w = sp._drift(w, retmap)
            monthly.append({'date': date, 'ret': r})
        else:
            held = grp[grp['permno'].isin(prev_w)].dropna(subset=['me', 'ret_fwd'])
            if held.empty:
                continue
            retmap = held.set_index('permno')['ret_fwd']
            wp     = {p: prev_w[p] for p in prev_w if p in retmap.index}
            tot    = sum(wp.values())
            if tot <= 0:
                continue
            wp = {p: v / tot for p, v in wp.items()}
            r  = sum(wp[p] * retmap[p] for p in wp)
            if track_permno is not None:
                perm_wts.append({'date': date, 'wt': wp.get(track_permno, 0.0)})
            prev_w = sp._drift(wp, retmap)
            monthly.append({'date': date, 'ret': r})

    ret_s    = pd.DataFrame(monthly).set_index('date')['ret']
    # avg_turn: average per-rebalance-event turnover (0.5*sum|Δw|)
    # annualized = avg_turn * (number of rebal events / number of months)
    n_months = len(monthly)
    n_rebal  = len(turnovers)
    avg_turn_per_event = np.mean(turnovers) if turnovers else np.nan
    # annualized turnover = per-event * events/yr = per-event * (n_rebal/n_months*12)
    avg_turn_ann = avg_turn_per_event * (n_rebal / n_months * 12) if n_months > 0 else np.nan
    avg_turn = (avg_turn_per_event, avg_turn_ann)   # return both
    if track_permno is not None and perm_wts:
        pw_df  = pd.DataFrame(perm_wts).set_index('date')
        avg_pw = pw_df['wt'].mean()
        max_pw = pw_df['wt'].max()
    else:
        avg_pw = max_pw = np.nan
    return ret_s, avg_turn, avg_pw, max_pw


# ── Market (cap-weighted, same as ladder) ─────────────────────────────────────
def market_ret(g):
    g = g.dropna(subset=['me', 'ret_fwd'])
    return (g['me'] / g['me'].sum() * g['ret_fwd']).sum()

print("Building market...", flush=True)
mkt = test.groupby('date').apply(market_ret, include_groups=False)

# ── r3: stateless (exact ladder match) ───────────────────────────────────────
print("Building r3 (stateless groupby, no cap, no fee)...", flush=True)
r3 = test.groupby('date').apply(_r3_month, include_groups=False)

# Compute r3 turnover and NVIDIA weights separately via stateful monthly build
# (same weights as stateless, but tracks transitions)
print("Building r3 tracked (for turnover / NVDA weight)...", flush=True)
r3_tracked, (turn_r3_evt, turn_r3_ann), nvda_avg_r3, nvda_max_r3 = build_v2(
    test, 0.20, None, tuple(range(1, 13)), 0.0, track_permno=NVIDIA_PERMNO)

# Verify r3 and r3_tracked match (should be identical up to float precision)
common = r3.index.intersection(r3_tracked.index)
max_diff = (r3.reindex(common) - r3_tracked.reindex(common)).abs().max()
print(f"  r3 vs r3_tracked max abs diff: {max_diff:.2e}", flush=True)

# ── m1: monthly, 9% cap, no fee ───────────────────────────────────────────────
print("Building m1 (monthly, 9% cap, no fee)...", flush=True)
m1, _turn_m1, _, _ = build_v2(test, 0.20, 0.09, tuple(range(1, 13)), 0.0)

# ── m2: semi-annual, 9% cap, no fee ──────────────────────────────────────────
print("Building m2 (semi-annual, 9% cap, no fee)...", flush=True)
m2, _turn_m2, _, _ = build_v2(test, 0.20, 0.09, (3, 9), 0.0)

# ── r4: semi-annual, 9% cap, WITH fee  (= full SPMO replica) ─────────────────
print("Building r4 (semi-annual, 9% cap, fee)...", flush=True)
r4, (turn_r4_evt, turn_r4_ann), nvda_avg_r4, nvda_max_r4 = build_v2(
    test, 0.20, 0.09, (3, 9), ORIG_FEE, track_permno=NVIDIA_PERMNO)

# Validate r4 matches the original sp.build_longonly call from the ladder
print("Validating r4 vs original sp.build_longonly...", flush=True)
sp.REBAL_MONTHS = ORIG_REBAL_MONTHS
sp.FEE          = ORIG_FEE
r4_orig = sp.build_longonly(test, 'spmo_signal', 'score', 0.20, 0.09)
common_r4 = r4.index.intersection(r4_orig.index)
max_diff_r4 = (r4.reindex(common_r4) - r4_orig.reindex(common_r4)).abs().max()
print(f"  r4 vs sp.build_longonly max abs diff: {max_diff_r4:.2e}", flush=True)
if max_diff_r4 > 0.01:
    print("  WARNING: r4 deviates from original by more than 1pp per month — check scoring convention")

# ── Align on common date index ─────────────────────────────────────────────────
L = pd.concat({'market': mkt, 'r3': r3, 'm1': m1, 'm2': m2, 'r4': r4}, axis=1).dropna()
print(f"Aligned rows: {len(L)}  ({L.index.min().date()} -> {L.index.max().date()})",
      flush=True)

pre  = L[L.index.year <= 2023]
post = L[L.index.year >= 2024]

# ── Cross-check r3 and r4 against the saved ladder CSV ────────────────────────
try:
    ladder_csv = f'{R}/results/2026-06-02-ladder.csv'
    ladder = pd.read_csv(ladder_csv, index_col=0, parse_dates=True)
    common_l = L.index.intersection(ladder.index)
    r3_diff = (L.loc[common_l, 'r3'] - ladder.loc[common_l, 'r3_capscore']).abs().max()
    r4_diff = (L.loc[common_l, 'r4'] - ladder.loc[common_l, 'r4_full_spmo']).abs().max()
    print(f"  Cross-check vs ladder CSV: r3_diff={r3_diff:.2e}  r4_diff={r4_diff:.2e}", flush=True)
except Exception as e:
    print(f"  Could not cross-check vs ladder CSV: {e}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  MECHANICS DECOMPOSITION  —  annualized returns and excess vs cap-weighted market")
print("=" * 90)
print(f"\n  {'Portfolio':<40} {'PRE ret':>9} {'PRE exc':>9} {'POST ret':>9} {'POST exc':>9}")
print("  " + "-" * 82)
for label, col in [
    ('r3  monthly, cap×score, no cap, no fee  [ladder]', 'r3'),
    ('m1  monthly, cap×score, 9% cap, no fee',           'm1'),
    ('m2  semi-ann, 9% cap, no fee',                     'm2'),
    ('r4  semi-ann, 9% cap, WITH fee (SPMO replica)',    'r4'),
]:
    pre_ret  = ann(pre[col])
    post_ret = ann(post[col])
    pre_exc  = pre_ret  - ann(pre['market'])
    post_exc = post_ret - ann(post['market'])
    print(f"  {label:<40} {pre_ret:>8.1%}  {pre_exc:>+8.1%}  {post_ret:>8.1%}  {post_exc:>+8.1%}")

mkt_pre  = ann(pre['market'])
mkt_post = ann(post['market'])
print(f"\n  {'market (cap-wt S&P500 proxy)':<40} {mkt_pre:>8.1%}  {'--':>9}  {mkt_post:>8.1%}  {'--':>9}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  3-WAY MECHANICS DECOMPOSITION  (marginal annualized excess contributions)")
print("=" * 90)

for period, sub in [('PRE  (<=2023)', pre), ('POST (>=2024)', post)]:
    r3_exc = ann(sub['r3']) - ann(sub['market'])
    m1_exc = ann(sub['m1']) - ann(sub['market'])
    m2_exc = ann(sub['m2']) - ann(sub['market'])
    r4_exc = ann(sub['r4']) - ann(sub['market'])

    cap_effect   = m1_exc - r3_exc
    rebal_effect = m2_exc - m1_exc
    fee_effect   = r4_exc - m2_exc
    total_mech   = r4_exc - r3_exc

    print(f"\n  {period}")
    print(f"    cap effect (adding 9% cap to monthly):   {cap_effect:>+7.1%}")
    print(f"    rebal/drift (monthly -> semi-annual):    {rebal_effect:>+7.1%}")
    print(f"    fee effect  (10 bps on turnover):        {fee_effect:>+7.1%}")
    print(f"    sum of 3 components:                     {cap_effect+rebal_effect+fee_effect:>+7.1%}")
    print(f"    ────────────────────────────────────────────────")
    print(f"    TOTAL mechanics gap (r4 - r3):           {total_mech:>+7.1%}")
    ladder_ref = 0.1144 if 'POST' in period else -0.0040
    delta = total_mech - ladder_ref
    flag = " <-- within 0.5pp OK" if abs(delta) < 0.005 else f" <-- DEVIATION {delta:+.1%}"
    print(f"    Ladder's mechanics marginal ({ladder_ref:+.1%}):        {delta:+.2%}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  TURNOVER AND NVIDIA WEIGHT")
print("=" * 90)
print(f"\n  Turnover (full period):")
print(f"    {'Portfolio':<32}  {'per-event':>10}  {'annualized':>10}")
print(f"    {'-'*56}")
print(f"    {'r3  (monthly, no cap, no fee)':<32}  {turn_r3_evt:>9.1%}  {turn_r3_ann:>9.1%}")
print(f"    {'r4  (semi-ann, 9% cap, fee)':<32}  {turn_r4_evt:>9.1%}  {turn_r4_ann:>9.1%}")
print(f"    (per-event = 0.5*sum|Dw| at each rebal; annualized = per-event * events/yr)")

print(f"\n  NVIDIA (permno {NVIDIA_PERMNO}) weights — FULL period (2011-2025):")
print(f"    r3 (monthly):    avg={nvda_avg_r3:.2%}   max={nvda_max_r3:.2%}")
print(f"    r4 (semi-ann):   avg={nvda_avg_r4:.2%}   max={nvda_max_r4:.2%}")

# POST-only NVIDIA weights using the aligned common dates
post_dates_set = set(post.index)
post_test = test[test['date'].isin(post_dates_set)].copy()

print("\n  Re-running POST-only slice for NVIDIA weights...", flush=True)
_, _t3, nvda_avg_r3_post, nvda_max_r3_post = build_v2(
    post_test, 0.20, None, tuple(range(1, 13)), 0.0, track_permno=NVIDIA_PERMNO)
_, _t4, nvda_avg_r4_post, nvda_max_r4_post = build_v2(
    post_test, 0.20, 0.09, (3, 9), ORIG_FEE, track_permno=NVIDIA_PERMNO)

print(f"\n  NVIDIA (permno {NVIDIA_PERMNO}) POST-ONLY (>=2024) weights:")
print(f"    r3 (monthly, no cap):  avg={nvda_avg_r3_post:.2%}   max={nvda_max_r3_post:.2%}")
print(f"    r4 (semi-ann, 9% cap): avg={nvda_avg_r4_post:.2%}   max={nvda_max_r4_post:.2%}")
print(f"  (r4 has 9% cap; cap is applied at rebalance, drift can push weight above cap between rebalances)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)

post_r3_exc = ann(post['r3']) - ann(post['market'])
post_m1_exc = ann(post['m1']) - ann(post['market'])
post_m2_exc = ann(post['m2']) - ann(post['market'])
post_r4_exc = ann(post['r4']) - ann(post['market'])
post_cap_eff   = post_m1_exc - post_r3_exc
post_rebal_eff = post_m2_exc - post_m1_exc
post_fee_eff   = post_r4_exc - post_m2_exc

print("  VERDICT:")
print(f"  POST mechanics gap (r4 - r3 excess) = {post_r4_exc - post_r3_exc:+.1%}")
print(f"    cap={post_cap_eff:+.1%}  rebal/drift={post_rebal_eff:+.1%}  fee={post_fee_eff:+.1%}")
if abs(post_rebal_eff) > 0.05 and abs(post_rebal_eff) > abs(post_cap_eff):
    print(f"  CONCLUSION: The mechanics gap is dominated by the rebal/drift effect "
          f"({post_rebal_eff:+.1%}).")
    print(f"  This is a real 'low-turnover lets winners run' effect: semi-annual rebalancing")
    print(f"  avoids selling momentum winners at each month-end, compounding their gains.")
    print(f"  The 9% cap is a minor drag ({post_cap_eff:+.1%}) and fees are negligible ({post_fee_eff:+.1%}).")
    print(f"  NOT an artifact of r3's no-cap/no-fee construction: m1 (monthly WITH cap, no fee)")
    print(f"  still yields only {post_m1_exc:+.1%} excess, vs r4's {post_r4_exc:+.1%} excess.")
elif abs(post_cap_eff) >= abs(post_rebal_eff):
    print(f"  CONCLUSION: The mechanics gap is dominated by the CAP effect ({post_cap_eff:+.1%}),")
    print(f"  not the rebal/drift effect ({post_rebal_eff:+.1%}). Interpret with caution.")
else:
    print(f"  CONCLUSION: no dominant single component — inspect all three.")

print("=" * 90)
print("\nDone.", flush=True)
