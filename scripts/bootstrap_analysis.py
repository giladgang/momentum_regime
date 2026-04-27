"""
bootstrap_analysis.py
=====================
Block bootstrap analysis for M2's Sharpe ratio and its significance vs benchmarks.

Three analyses:
  1. 95% CI for each strategy's Sharpe (block bootstrap, 12-month blocks)
  2. Paired bootstrap test: is M2's Sharpe significantly higher than each benchmark?
     Uses same resampled dates for both strategies so the comparison is paired.
  3. Regime-conditional bootstrap: CIs for calm and panic Sharpes separately.

Saves:
  - results/bootstrap_results.csv
  - tables/table_bootstrap.tex

Usage:
    python -u scripts/bootstrap_analysis.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

N_BOOT = 10000
BLOCK_LEN = 12
RNG = np.random.RandomState(42)

# ── Load portfolio returns ──
print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)

strategies = art['strategies_lo']

# Identify the strategies we care about by substring matching
def find_strategy(substr_list):
    for name, ret in strategies.items():
        for s in substr_list:
            if s.lower() in name.lower():
                return name, ret
    return None, None

name_m2, r_m2 = find_strategy(['xgb', 'm2'])
name_m1, r_m1 = find_strategy(['lr', 'logistic', 'm1'])
name_m0, r_m0 = find_strategy(['m0', 'formula'])
name_mom12, r_mom12 = find_strategy(['mom12', 'mom_12', '12-mo'])
name_mom1, r_mom1 = find_strategy(['mom1', 'mom_1', '1-mo'])

print(f"  M2:     {name_m2}  (n={len(r_m2) if r_m2 is not None else 0})")
print(f"  M1:     {name_m1}")
print(f"  M0:     {name_m0}")
print(f"  Mom12:  {name_mom12}")
print(f"  Mom1:   {name_mom1}")

def sharpe(r):
    r = np.asarray(r)
    s = r.std()
    return r.mean() / s * np.sqrt(12) if s > 0 else 0.0


def block_bootstrap_indices(T, block_len, rng):
    """Return a length-T index array from circular block bootstrap."""
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.randint(0, T, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) % T for s in starts])[:T]
    return idx


# ── 1. Bootstrap CI for each strategy's Sharpe ──
print(f"\n[1/3] Bootstrap CIs for individual Sharpes ({N_BOOT} resamples, {BLOCK_LEN}-month blocks)")

strategy_pool = {
    'M2 (XGB)':           r_m2,
    'M1 (LR)':            r_m1,
    'M0 (Formula)':       r_m0,
    'Fixed 12-mo mom':    r_mom12,
    'Fixed 1-mo mom':     r_mom1,
}

# Align all to common index (M2's returns)
ref_index = r_m2.index
aligned = {k: v.reindex(ref_index).values for k, v in strategy_pool.items()}
T = len(ref_index)

# Generate shared bootstrap indices for paired comparison
print(f"  Generating {N_BOOT} bootstrap index sets ...")
all_indices = np.empty((N_BOOT, T), dtype=int)
for b in range(N_BOOT):
    all_indices[b] = block_bootstrap_indices(T, BLOCK_LEN, RNG)

# Compute Sharpe on each resample for each strategy
print("  Computing Sharpe on resamples ...")
boot_sharpes = {}
for name, r in aligned.items():
    sharpes = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sharpes[b] = sharpe(r[all_indices[b]])
    boot_sharpes[name] = sharpes

# Point estimates and CIs
ci_results = []
print(f"\n  {'Strategy':<20s}  {'Point':>8s}  {'5% CI':>8s}  {'95% CI':>8s}  {'Boot mean':>10s}")
for name, sharpes in boot_sharpes.items():
    point = sharpe(aligned[name])
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    print(f"  {name:<20s}  {point:>8.3f}  {lo:>8.3f}  {hi:>8.3f}  {sharpes.mean():>10.3f}")
    ci_results.append({
        'strategy': name, 'point_sharpe': point,
        'ci_low': lo, 'ci_high': hi,
        'boot_mean': sharpes.mean(), 'boot_std': sharpes.std(),
    })


# ── 2. Paired bootstrap test: M2 vs each benchmark ──
print(f"\n[2/3] Paired bootstrap: M2 Sharpe vs each benchmark")

paired_results = []
m2_boot = boot_sharpes['M2 (XGB)']
print(f"\n  {'vs Benchmark':<20s}  {'Diff':>8s}  {'Diff 5% CI':>10s}  {'Diff 95% CI':>11s}  {'p-value':>8s}")
for name in ['M1 (LR)', 'M0 (Formula)', 'Fixed 12-mo mom', 'Fixed 1-mo mom']:
    bench_boot = boot_sharpes[name]
    diff = m2_boot - bench_boot
    point_diff = sharpe(aligned['M2 (XGB)']) - sharpe(aligned[name])
    lo, hi = np.percentile(diff, [2.5, 97.5])
    # Two-sided p-value: fraction of resamples where diff <= 0 (M2 not better)
    p_val = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
    print(f"  vs {name:<17s}  {point_diff:>8.3f}  {lo:>10.3f}  {hi:>11.3f}  {p_val:>8.4f} {sig}")
    paired_results.append({
        'benchmark': name, 'point_diff': point_diff,
        'diff_ci_low': lo, 'diff_ci_high': hi, 'p_value': p_val,
    })


# ── 3. Regime-conditional bootstrap ──
print(f"\n[3/3] Regime-conditional bootstrap CIs for M2")

panel = pd.read_parquet(os.path.join(cfg.PANEL_WITH_REGIMES_PATH))
panel['date'] = pd.to_datetime(panel['date'])
monthly_pi = panel.groupby('date')['pi_filter'].first()

r_m2_df = r_m2.to_frame('ret').copy()
r_m2_df.index = pd.to_datetime(r_m2_df.index)
r_m2_df = r_m2_df.join(monthly_pi, how='left')
r_m2_df['regime'] = np.where(r_m2_df['pi_filter'] >= 0.5, 'Panic', 'Calm')

calm_returns = r_m2_df[r_m2_df['regime'] == 'Calm']['ret'].values
panic_returns = r_m2_df[r_m2_df['regime'] == 'Panic']['ret'].values

print(f"  Calm months:  {len(calm_returns)}")
print(f"  Panic months: {len(panic_returns)}")

def bootstrap_ci(r, n_boot=N_BOOT, block_len=BLOCK_LEN):
    T = len(r)
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        # For small samples (panic), use shorter blocks if needed
        bl = min(block_len, max(3, T // 6))
        n_blocks = int(np.ceil(T / bl))
        starts = RNG.randint(0, T, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + bl) % T for s in starts])[:T]
        sharpes[b] = sharpe(r[idx])
    return sharpes

calm_boot = bootstrap_ci(calm_returns)
panic_boot = bootstrap_ci(panic_returns)

calm_point = sharpe(calm_returns)
panic_point = sharpe(panic_returns)
calm_ci = np.percentile(calm_boot, [2.5, 97.5])
panic_ci = np.percentile(panic_boot, [2.5, 97.5])

diff_regime = panic_boot - calm_boot
diff_ci = np.percentile(diff_regime, [2.5, 97.5])
diff_p = 2 * min((diff_regime <= 0).mean(), (diff_regime >= 0).mean())

print(f"\n  Calm Sharpe:  {calm_point:.3f}  [{calm_ci[0]:.3f}, {calm_ci[1]:.3f}]")
print(f"  Panic Sharpe: {panic_point:.3f}  [{panic_ci[0]:.3f}, {panic_ci[1]:.3f}]")
print(f"  Diff (P-C):   {panic_point - calm_point:.3f}  [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}]  (p={diff_p:.4f})")


# ── Save results ──
os.makedirs('results', exist_ok=True)
ci_df = pd.DataFrame(ci_results)
ci_df.to_csv('results/thesis/bootstrap_sharpe_cis.csv', index=False)
paired_df = pd.DataFrame(paired_results)
paired_df.to_csv('results/thesis/bootstrap_paired_tests.csv', index=False)

regime_df = pd.DataFrame([
    {'regime': 'Calm',  'n_months': len(calm_returns),  'sharpe': calm_point,  'ci_low': calm_ci[0],  'ci_high': calm_ci[1]},
    {'regime': 'Panic', 'n_months': len(panic_returns), 'sharpe': panic_point, 'ci_low': panic_ci[0], 'ci_high': panic_ci[1]},
    {'regime': 'Panic - Calm', 'n_months': '---', 'sharpe': panic_point - calm_point,
     'ci_low': diff_ci[0], 'ci_high': diff_ci[1]},
])
regime_df.to_csv('results/thesis/bootstrap_regime.csv', index=False)

print("\nSaved:")
print("  results/bootstrap_sharpe_cis.csv")
print("  results/bootstrap_paired_tests.csv")
print("  results/bootstrap_regime.csv")


# ── LaTeX table ──
tex = []
tex.append(r"\begin{table}[H]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\begin{tabular}{l r r r}")
tex.append(r"\toprule")
tex.append(r" & Sharpe & 95\% CI & vs M2 ($p$) \\")
tex.append(r"\midrule")

# M2 row (no "vs M2" comparison)
for r in ci_results:
    if 'M2' in r['strategy']:
        tex.append(f"{r['strategy']} & {r['point_sharpe']:.2f} & "
                   f"[{r['ci_low']:.2f},\\,{r['ci_high']:.2f}] & --- \\\\")
        break

# Benchmarks with paired p-values
for p in paired_results:
    bench = p['benchmark']
    # Find CI for this benchmark
    for r in ci_results:
        if r['strategy'] == bench:
            sig = '$^{***}$' if p['p_value'] < 0.01 else '$^{**}$' if p['p_value'] < 0.05 else '$^{*}$' if p['p_value'] < 0.10 else ''
            # Escape negative signs
            ci_lo_str = f"$-${abs(r['ci_low']):.2f}" if r['ci_low'] < 0 else f"{r['ci_low']:.2f}"
            ci_hi_str = f"$-${abs(r['ci_high']):.2f}" if r['ci_high'] < 0 else f"{r['ci_high']:.2f}"
            sharpe_str = f"$-${abs(r['point_sharpe']):.2f}" if r['point_sharpe'] < 0 else f"{r['point_sharpe']:.2f}"
            tex.append(f"{bench} & {sharpe_str} & [{ci_lo_str},\\,{ci_hi_str}] & "
                       f"{p['p_value']:.3f}{sig} \\\\")
            break

tex.append(r"\midrule")
tex.append(r"\multicolumn{4}{l}{\emph{M2 regime-conditional Sharpe}} \\[2pt]")
calm_ci_lo = f"{calm_ci[0]:.2f}"
calm_ci_hi = f"{calm_ci[1]:.2f}"
panic_ci_lo = f"{panic_ci[0]:.2f}"
panic_ci_hi = f"{panic_ci[1]:.2f}"
diff_ci_lo = f"{diff_ci[0]:.2f}"
diff_ci_hi = f"{diff_ci[1]:.2f}"

tex.append(f"M2 Calm ($n=${len(calm_returns)}) & {calm_point:.2f} & [{calm_ci_lo},\\,{calm_ci_hi}] & --- \\\\")
tex.append(f"M2 Panic ($n=${len(panic_returns)}) & {panic_point:.2f} & [{panic_ci_lo},\\,{panic_ci_hi}] & --- \\\\")
diff_sig = '$^{***}$' if diff_p < 0.01 else '$^{**}$' if diff_p < 0.05 else '$^{*}$' if diff_p < 0.10 else ''
tex.append(f"Panic $-$ Calm & {panic_point - calm_point:.2f} & [{diff_ci_lo},\\,{diff_ci_hi}] & "
           f"{diff_p:.3f}{diff_sig} \\\\")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\caption{Block bootstrap confidence intervals for Sharpe ratios and paired tests of M2's outperformance. Uses 12-month blocks and 10{,}000 resamples with a fixed seed. The ``vs M2 ($p$)'' column reports two-sided $p$-values from paired tests where both strategies are resampled on the same dates. $^{*}\,p<0.10$; $^{**}\,p<0.05$; $^{***}\,p<0.01$.}")
tex.append(r"\label{tab:bootstrap}")
tex.append(r"\end{table}")

os.makedirs('tables', exist_ok=True)
with open('tables/table_bootstrap.tex', 'w') as f:
    f.write('\n'.join(tex))
print("  tables/table_bootstrap.tex")
