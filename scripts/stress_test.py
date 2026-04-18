"""
stress_test.py
==============
Empirical vulnerability analysis and stress test for the L/S momentum strategy.

Part A: Diagnose vulnerabilities from actual returns
    - When did the strategy lose money?
    - What were the market conditions during losses?
    - Is the strategy exposed to prolonged bear markets?

Part B: Formal stress tests based on diagnosed vulnerabilities
    - Tail risk metrics (VaR, CVaR)
    - Drawdown conditioning (early, sustained, recovery)
    - Regime transition analysis
    - Historical scenario performance
    - Reverse stress test
    - Return distribution (skewness, kurtosis)

Usage:
    python scripts/stress_test.py
"""

import numpy as np
import pandas as pd
import pickle, warnings, os, sys
import statsmodels.api as sm
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE, TRAIN_END, MOM_FEATURES

print("=" * 70)
print("  STRESS TEST: VULNERABILITY ANALYSIS")
print("=" * 70)

# ── Load data ──
with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
r_mkt = art['r_mkt']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

# Get strategy returns (from most recent pipeline run)
# We need the L/S returns -- load or recompute
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

# Use production scores from artefacts to compute L/S returns
print("  Using production XGB scores from artefacts ...")
from config import TRADING_FEE as FEE
REDUCED = MOM_FEATURES + ['pi_filter']

def long_short_port(df_test, score_col, fee=FEE):
    monthly, plw, psw = [], {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        L = grp[grp[score_col] >= hi]; S = grp[grp[score_col] <= lo]
        if L['me'].sum() == 0 or S['me'].sum() == 0: continue
        lme = L['me'].sum(); nlw = (L.set_index('permno')['me'] / lme).to_dict()
        rl = (L['ret_fwd'] * L['me']).sum() / lme
        sme = S['me'].sum(); nsw = (S.set_index('permno')['me'] / sme).to_dict()
        rs = (S['ret_fwd'] * S['me']).sum() / sme
        tl = sum(abs(nlw.get(p, 0) - plw.get(p, 0)) for p in set(nlw) | set(plw)) / 2
        ts = sum(abs(nsw.get(p, 0) - psw.get(p, 0)) for p in set(nsw) | set(psw)) / 2
        monthly.append({'date': date, 'ret': rl - rs - fee * (tl + ts)})
        plw, psw = nlw, nsw
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

test['score'] = test['score_xgb']
r_strat = long_short_port(test, 'score')

# Align returns with regime data
pi_aligned = pi_monthly.reindex(r_strat.index)['pi_filter'].fillna(0.5)
mkt_aligned = r_mkt.reindex(r_strat.index).fillna(0)

# ═══════════════════════════════════════════════════════════════════════════════
#  PART A: DIAGNOSE VULNERABILITIES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART A: EMPIRICAL VULNERABILITY DIAGNOSIS")
print("=" * 70)

# A1: When did the strategy lose money?
print("\n[ A1 ] Loss analysis ...")
negative_months = r_strat[r_strat < 0]
positive_months = r_strat[r_strat >= 0]
print(f"  Negative months: {len(negative_months)} / {len(r_strat)} ({len(negative_months)/len(r_strat):.1%})")
print(f"  Positive months: {len(positive_months)} / {len(r_strat)} ({len(positive_months)/len(r_strat):.1%})")
print(f"  Average loss (negative months): {negative_months.mean():.2%}")
print(f"  Average gain (positive months): {positive_months.mean():.2%}")

# Worst 10 months
print("\n  10 worst months:")
worst = r_strat.nsmallest(10)
for date, ret in worst.items():
    pi = pi_aligned.get(date, np.nan)
    mkt = mkt_aligned.get(date, np.nan)
    regime = 'PANIC' if pi >= 0.5 else 'CALM'
    mkt_dir = 'UP' if mkt > 0 else 'DOWN'
    print(f"    {date.strftime('%Y-%m')}: {ret:>+7.2%}  pi={pi:.2f} [{regime}]  mkt={mkt:>+6.2%} [{mkt_dir}]")

# A2: Losses by regime
print("\n[ A2 ] Returns by regime ...")
calm_mask = pi_aligned < 0.5
panic_mask = pi_aligned >= 0.5

r_calm = r_strat[calm_mask]
r_panic = r_strat[panic_mask]

print(f"\n  {'':15s} {'Calm':>10s} {'Panic':>10s}")
print(f"  {'-'*37}")
print(f"  {'N months':<15s} {len(r_calm):>10d} {len(r_panic):>10d}")
print(f"  {'Mean return':<15s} {r_calm.mean():>10.2%} {r_panic.mean():>10.2%}")
print(f"  {'Std':<15s} {r_calm.std():>10.2%} {r_panic.std():>10.2%}")
print(f"  {'% negative':<15s} {(r_calm<0).mean():>10.1%} {(r_panic<0).mean():>10.1%}")
print(f"  {'Worst month':<15s} {r_calm.min():>10.2%} {r_panic.min():>10.2%}")
print(f"  {'Best month':<15s} {r_calm.max():>10.2%} {r_panic.max():>10.2%}")

# A3: Losses by market direction
print("\n[ A3 ] Returns by market direction ...")
mkt_up = mkt_aligned > 0
mkt_down = mkt_aligned <= 0

print(f"\n  {'':15s} {'Mkt UP':>10s} {'Mkt DOWN':>10s}")
print(f"  {'-'*37}")
print(f"  {'N months':<15s} {mkt_up.sum():>10d} {mkt_down.sum():>10d}")
print(f"  {'Mean return':<15s} {r_strat[mkt_up].mean():>10.2%} {r_strat[mkt_down].mean():>10.2%}")
print(f"  {'% negative':<15s} {(r_strat[mkt_up]<0).mean():>10.1%} {(r_strat[mkt_down]<0).mean():>10.1%}")

# A4: Regime x Market direction (the vulnerability matrix)
print("\n[ A4 ] Vulnerability matrix (regime x market direction) ...")
print(f"\n  {'':20s} {'Mkt UP':>10s} {'Mkt DOWN':>10s}")
print(f"  {'-'*42}")

for regime_name, regime_mask in [('CALM', calm_mask), ('PANIC', panic_mask)]:
    for mkt_name, mkt_mask in [('Mkt UP', mkt_up), ('Mkt DOWN', mkt_down)]:
        combined = regime_mask & mkt_mask
        if combined.sum() > 0:
            r_sub = r_strat[combined]
            label = f"{regime_name} + {mkt_name}"
            print(f"  {label:<20s} n={combined.sum():>3d}  mean={r_sub.mean():>+6.2%}  "
                  f"neg={( r_sub<0).mean():>5.1%}  worst={r_sub.min():>+6.2%}")

# A5: Regime transitions
print("\n[ A5 ] Regime transition analysis ...")
regime_binary = (pi_aligned >= 0.5).astype(int)
transitions = regime_binary.diff().abs()
transition_months = transitions[transitions == 1].index

print(f"  Total regime transitions: {len(transition_months)}")

# Returns around transitions
if len(transition_months) > 2:
    pre_ret = []
    post_ret = []
    dates_list = r_strat.index.tolist()
    for td in transition_months:
        if td in dates_list:
            idx = dates_list.index(td)
            if idx > 0:
                pre_ret.append(r_strat.iloc[idx - 1])
            post_ret.append(r_strat.iloc[idx])

    print(f"  Mean return month BEFORE transition: {np.mean(pre_ret):>+6.2%}")
    print(f"  Mean return month OF transition:     {np.mean(post_ret):>+6.2%}")

# A6: Consecutive losses
print("\n[ A6 ] Consecutive loss analysis ...")
is_loss = (r_strat < 0).astype(int)
streaks = []
current = 0
for v in is_loss:
    if v == 1:
        current += 1
    else:
        if current > 0:
            streaks.append(current)
        current = 0
if current > 0:
    streaks.append(current)

print(f"  Longest losing streak: {max(streaks) if streaks else 0} months")
print(f"  Average losing streak: {np.mean(streaks):.1f} months")
print(f"  Number of losing streaks: {len(streaks)}")

# A7: Drawdown duration analysis
print("\n[ A7 ] Drawdown duration analysis ...")
cum = (1 + r_strat).cumprod()
peak = cum.cummax()
drawdown = (cum - peak) / peak

# Current drawdown periods
in_dd = drawdown < 0
dd_months = in_dd.sum()
print(f"  Months in drawdown: {dd_months} / {len(r_strat)} ({dd_months/len(r_strat):.1%})")
print(f"  Max drawdown: {drawdown.min():.1%}")

# Time to recovery after max drawdown
max_dd_date = drawdown.idxmin()
recovery_dates = cum[cum.index > max_dd_date]
recovered = recovery_dates[recovery_dates >= peak[max_dd_date]]
if len(recovered) > 0:
    recovery_time = (recovered.index[0] - max_dd_date).days // 30
    print(f"  Max DD date: {max_dd_date.strftime('%Y-%m')}")
    print(f"  Recovery time: {recovery_time} months")
else:
    print(f"  Max DD date: {max_dd_date.strftime('%Y-%m')}")
    print(f"  Recovery: NOT YET RECOVERED")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART B: FORMAL STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART B: FORMAL STRESS TESTS")
print("=" * 70)

# B1: Tail risk metrics
print("\n[ B1 ] Tail risk metrics ...")
r_arr = r_strat.values

var_95 = np.percentile(r_arr, 5)
var_99 = np.percentile(r_arr, 1)
cvar_95 = r_arr[r_arr <= var_95].mean()
cvar_99 = r_arr[r_arr <= var_99].mean()

print(f"  VaR (95%):  {var_95:>+7.2%}  (5% chance of losing more than this in a month)")
print(f"  VaR (99%):  {var_99:>+7.2%}  (1% chance of losing more than this in a month)")
print(f"  CVaR (95%): {cvar_95:>+7.2%}  (average loss in worst 5% of months)")
print(f"  CVaR (99%): {cvar_99:>+7.2%}  (average loss in worst 1% of months)")

# B2: Return distribution
print("\n[ B2 ] Return distribution ...")
skew = scipy_stats.skew(r_arr)
kurt = scipy_stats.kurtosis(r_arr)
jb_stat, jb_p = scipy_stats.jarque_bera(r_arr)

print(f"  Mean:     {r_arr.mean():>+7.2%}")
print(f"  Std:      {r_arr.std():>7.2%}")
print(f"  Skewness: {skew:>+7.3f}  ({'negative skew = crash prone' if skew < 0 else 'positive skew = good'})")
print(f"  Kurtosis: {kurt:>+7.3f}  ({'fat tails' if kurt > 0 else 'thin tails'})")
print(f"  Jarque-Bera: {jb_stat:.1f} (p={jb_p:.4f})  ({'non-normal' if jb_p < 0.05 else 'normal'})")

# B3: Drawdown conditioning
print("\n[ B3 ] Returns by drawdown phase ...")

# Compute market drawdown
mkt_cum = (1 + mkt_aligned).cumprod()
mkt_peak = mkt_cum.cummax()
mkt_dd = (mkt_cum - mkt_peak) / mkt_peak

# Count consecutive months in drawdown
dd_duration = pd.Series(0, index=mkt_dd.index)
count = 0
for i, (date, dd) in enumerate(mkt_dd.items()):
    if dd < -0.02:  # market is in drawdown (>2% below peak)
        count += 1
    else:
        count = 0
    dd_duration.iloc[i] = count

print(f"\n  {'Phase':<25s} {'N':>4s} {'Mean Ret':>9s} {'% Neg':>7s} {'Worst':>8s}")
print(f"  {'-'*55}")

for label, mask in [
    ('No drawdown',          dd_duration == 0),
    ('Early DD (mo 1-3)',    (dd_duration >= 1) & (dd_duration <= 3)),
    ('Mid DD (mo 4-6)',      (dd_duration >= 4) & (dd_duration <= 6)),
    ('Late DD (mo 7+)',      dd_duration >= 7),
]:
    r_sub = r_strat[mask]
    if len(r_sub) > 0:
        print(f"  {label:<25s} {len(r_sub):>4d} {r_sub.mean():>+8.2%} {(r_sub<0).mean():>6.1%} {r_sub.min():>+7.2%}")

# B4: Historical scenarios
print("\n[ B4 ] Historical scenario performance ...")

scenarios = [
    ('2018 Q4 sell-off',    '2018-10-01', '2018-12-31'),
    ('COVID crash',         '2020-02-01', '2020-04-30'),
    ('COVID recovery',      '2020-04-01', '2020-08-31'),
    ('2022 rate hikes',     '2022-01-01', '2022-10-31'),
    ('2022 recovery',       '2022-10-01', '2023-03-31'),
]

print(f"\n  {'Scenario':<25s} {'Cum Ret':>8s} {'Sharpe':>7s} {'Worst Mo':>9s} {'Mkt Ret':>8s}")
print(f"  {'-'*60}")

for name, start, end in scenarios:
    r_sub = r_strat[(r_strat.index >= start) & (r_strat.index <= end)]
    m_sub = mkt_aligned[(mkt_aligned.index >= start) & (mkt_aligned.index <= end)]
    if len(r_sub) > 1:
        cum_ret = (1 + r_sub).prod() - 1
        sh = r_sub.mean() / r_sub.std() * np.sqrt(12) if r_sub.std() > 0 else 0
        mkt_cum = (1 + m_sub).prod() - 1
        print(f"  {name:<25s} {cum_ret:>+7.1%} {sh:>7.2f} {r_sub.min():>+8.2%} {mkt_cum:>+7.1%}")

# B5: Reverse stress test
print("\n[ B5 ] Reverse stress test ...")
print("  Question: what conditions cause the strategy to lose > 5% in a month?")

big_losses = r_strat[r_strat < -0.05]
if len(big_losses) > 0:
    print(f"\n  Months with loss > 5%: {len(big_losses)}")
    for date, ret in big_losses.items():
        pi = pi_aligned.get(date, np.nan)
        mkt = mkt_aligned.get(date, np.nan)
        dd = mkt_dd.get(date, np.nan)
        dur = dd_duration.get(date, 0)
        print(f"    {date.strftime('%Y-%m')}: ret={ret:>+6.2%}  pi={pi:.2f}  "
              f"mkt={mkt:>+6.2%}  mkt_dd={dd:>+6.1%}  dd_month={dur}")
else:
    print("  No months with loss > 5%")
    print("  Checking > 3%:")
    big_losses = r_strat[r_strat < -0.03]
    if len(big_losses) > 0:
        print(f"  Months with loss > 3%: {len(big_losses)}")
        for date, ret in big_losses.items():
            pi = pi_aligned.get(date, np.nan)
            mkt = mkt_aligned.get(date, np.nan)
            print(f"    {date.strftime('%Y-%m')}: ret={ret:>+6.2%}  pi={pi:.2f}  mkt={mkt:>+6.2%}")

# B6: Correlation with market in different regimes
print("\n[ B6 ] Market correlation by regime ...")
for regime_name, mask in [('All', pd.Series(True, index=r_strat.index)),
                           ('Calm', calm_mask), ('Panic', panic_mask)]:
    r_s = r_strat[mask]
    r_m = mkt_aligned[mask]
    if len(r_s) > 5:
        corr = r_s.corr(r_m)
        beta = np.cov(r_s, r_m)[0, 1] / np.var(r_m) if np.var(r_m) > 0 else 0
        print(f"  {regime_name:<8s}: corr={corr:>+6.3f}  beta={beta:>+6.3f}  n={len(r_s)}")


# B7: Prolonged bear market simulation
print("\n[ B7 ] Prolonged bear market simulation ...")
print("  Bootstrap from actual panic-month returns to simulate extended bears")

# Get strategy returns during panic months
panic_returns = r_strat[panic_mask].values

if len(panic_returns) > 5:
    np.random.seed(42)
    N_SIM = 1000
    scenarios = [6, 12, 18, 24]

    print(f"\n  Based on {len(panic_returns)} actual panic-month returns:")
    print(f"  Panic month mean: {panic_returns.mean():>+6.2%}  std: {panic_returns.std():>5.2%}")
    print(f"  Panic month worst: {panic_returns.min():>+6.2%}  best: {panic_returns.max():>+6.2%}")

    print(f"\n  {'Duration':<15s} {'Median Loss':>12s} {'95th Loss':>10s} {'99th Loss':>10s} {'Worst Case':>11s} {'% Positive':>11s}")
    print(f"  {'-'*72}")

    for n_months in scenarios:
        cum_returns = np.zeros(N_SIM)
        for sim in range(N_SIM):
            # Draw n_months random panic returns with replacement
            sampled = np.random.choice(panic_returns, size=n_months, replace=True)
            cum_returns[sim] = (1 + sampled).prod() - 1

        median_loss = np.percentile(cum_returns, 50)
        p5_loss = np.percentile(cum_returns, 5)
        p1_loss = np.percentile(cum_returns, 1)
        worst = cum_returns.min()
        pct_positive = (cum_returns > 0).mean()

        print(f"  {n_months:>2d} months       {median_loss:>+11.1%} {p5_loss:>+9.1%} {p1_loss:>+9.1%} {worst:>+10.1%} {pct_positive:>10.1%}")

    # Also simulate: what if panic returns were 50% worse (tail scenario)
    print(f"\n  EXTREME SCENARIO: panic returns 50% worse than observed")
    print(f"  {'Duration':<15s} {'Median Loss':>12s} {'95th Loss':>10s} {'Worst Case':>11s}")
    print(f"  {'-'*50}")

    extreme_returns = panic_returns * 1.5  # 50% worse
    for n_months in scenarios:
        cum_returns = np.zeros(N_SIM)
        for sim in range(N_SIM):
            sampled = np.random.choice(extreme_returns, size=n_months, replace=True)
            cum_returns[sim] = (1 + sampled).prod() - 1

        median_loss = np.percentile(cum_returns, 50)
        p5_loss = np.percentile(cum_returns, 5)
        worst = cum_returns.min()

        print(f"  {n_months:>2d} months       {median_loss:>+11.1%} {p5_loss:>+9.1%} {worst:>+10.1%}")
else:
    print("  Not enough panic months for simulation")

# B8: Recovery analysis
print("\n[ B8 ] Recovery after large losses ...")
print("  What happens in the 1, 3, 6 months after the strategy's worst months?")

worst_months = r_strat.nsmallest(10).index
dates_list = r_strat.index.tolist()

print(f"\n  {'Bad Month':<12s} {'Loss':>7s} {'Next 1mo':>9s} {'Next 3mo':>9s} {'Next 6mo':>9s}")
print(f"  {'-'*50}")

for date in worst_months:
    idx = dates_list.index(date)
    loss = r_strat.iloc[idx]

    next1 = r_strat.iloc[idx+1] if idx+1 < len(r_strat) else np.nan
    next3 = (1 + r_strat.iloc[idx+1:idx+4]).prod() - 1 if idx+3 < len(r_strat) else np.nan
    next6 = (1 + r_strat.iloc[idx+1:idx+7]).prod() - 1 if idx+6 < len(r_strat) else np.nan

    n1 = f"{next1:>+8.2%}" if not np.isnan(next1) else "     N/A"
    n3 = f"{next3:>+8.2%}" if not np.isnan(next3) else "     N/A"
    n6 = f"{next6:>+8.2%}" if not np.isnan(next6) else "     N/A"

    print(f"  {date.strftime('%Y-%m'):<12s} {loss:>+6.2%} {n1} {n3} {n6}")

avg_next1 = []
avg_next3 = []
for date in worst_months:
    idx = dates_list.index(date)
    if idx+1 < len(r_strat):
        avg_next1.append(r_strat.iloc[idx+1])
    if idx+3 < len(r_strat):
        avg_next3.append((1 + r_strat.iloc[idx+1:idx+4]).prod() - 1)

if avg_next1:
    print(f"\n  Average recovery after worst 10 months:")
    print(f"    Next 1 month: {np.mean(avg_next1):>+6.2%}")
    if avg_next3:
        print(f"    Next 3 months: {np.mean(avg_next3):>+6.2%}")
    reversal = np.mean(avg_next1) > 0
    print(f"    Pattern: {'MEAN REVERTING (losses tend to recover)' if reversal else 'PERSISTENT (losses compound)'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRESS TEST SUMMARY")
print("=" * 70)

print(f"""
  TAIL RISK:
    VaR (95%):  {var_95:>+7.2%}  |  CVaR (95%): {cvar_95:>+7.2%}
    VaR (99%):  {var_99:>+7.2%}  |  CVaR (99%): {cvar_99:>+7.2%}

  DISTRIBUTION:
    Skewness: {skew:>+.3f}  |  Kurtosis: {kurt:>+.3f}

  VULNERABILITY:
    Worst regime: {'PANIC' if r_panic.mean() < r_calm.mean() else 'CALM'} (mean={min(r_panic.mean(), r_calm.mean()):>+.2%})
    Worst market condition: {'MKT DOWN' if r_strat[mkt_down].mean() < r_strat[mkt_up].mean() else 'MKT UP'}
    Longest losing streak: {max(streaks) if streaks else 0} months
    Max drawdown: {drawdown.min():.1%}

  DRAWDOWN RESILIENCE:
    Strategy loses most in: {'sustained (7+ mo) drawdowns' if dd_duration.max() >= 7 else 'early drawdowns'}
""")

print("=" * 70)
print("  STRESS TEST COMPLETE")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORT: STRESS SCENARIO TABLE (LaTeX + CSV)
# ═══════════════════════════════════════════════════════════════════════════════

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Compute the recession injection table matching the thesis:
# Each row injects a recession of N months where M2 loses its average
# losing panic-month return per month, at every possible insertion point.
# MDD = worst-case maximum drawdown across all insertion points.

# Average losing panic-month return
panic_losing = r_strat[(panic_mask) & (r_strat < 0)]
avg_panic_loss = panic_losing.mean()
print(f"\n  Avg losing panic-month return: {avg_panic_loss:.2%}")

# Baseline MDD (no recession injected)
cum_base = (1 + r_strat).cumprod()
mdd_base = ((cum_base - cum_base.cummax()) / cum_base.cummax()).min()

recession_durations = [3, 6, 12, 18, 24]
stress_rows = []
stress_rows.append({
    'duration': 'Baseline (no recession)',
    'mkt_loss': np.nan,
    'm2_loss': np.nan,
    'mdd_worst': mdd_base
})

r_arr_strat = r_strat.values
n_months_total = len(r_arr_strat)

for n_rec in recession_durations:
    # Market loss: compounded loss at avg_panic_loss rate
    mkt_loss = (1 + avg_panic_loss) ** n_rec - 1
    m2_loss = mkt_loss  # same rate assumption

    # Worst-case MDD: try every insertion point
    worst_mdd = mdd_base
    for start in range(n_months_total - n_rec + 1):
        # Create modified return series: replace months start..start+n_rec with avg_panic_loss
        r_mod = r_arr_strat.copy()
        r_mod[start:start + n_rec] = avg_panic_loss
        cum_mod = np.cumprod(1 + r_mod)
        peak_mod = np.maximum.accumulate(cum_mod)
        dd_mod = (cum_mod - peak_mod) / peak_mod
        mdd_mod = dd_mod.min()
        if mdd_mod < worst_mdd:
            worst_mdd = mdd_mod

    stress_rows.append({
        'duration': f'{n_rec} months',
        'mkt_loss': mkt_loss,
        'm2_loss': m2_loss,
        'mdd_worst': worst_mdd
    })

# Save CSV
stress_df = pd.DataFrame(stress_rows)
stress_df.to_csv(os.path.join(TABLES_DIR, 'table_stress_scenarios.csv'), index=False, float_format='%.4f')
print(f"Saved: {os.path.join(TABLES_DIR, 'table_stress_scenarios.csv')}")

# Build LaTeX table
def fmt_pct_round(v):
    """Format as rounded percentage with $-$ for negative."""
    pct = v * 100
    s = f"{abs(pct):.0f}\\%"
    return f"$-${s}" if v < 0 else s

tex_lines = []
tex_lines.append(r'\begin{table}[H]')
tex_lines.append(r'\centering')
tex_lines.append(r'\small')
tex_lines.append(r'\begin{tabular}{l r r r}')
tex_lines.append(r'\toprule')
tex_lines.append(r'Recession Duration & Market Loss & M2 Loss & MDD (worst case) \\')
tex_lines.append(r'\midrule')

for row in stress_rows:
    dur = row['duration']
    if np.isnan(row['mkt_loss']):
        mkt = '---'
        m2 = '---'
    else:
        mkt = fmt_pct_round(row['mkt_loss'])
        m2 = fmt_pct_round(row['m2_loss'])
    mdd = fmt_pct_round(row['mdd_worst'])
    tex_lines.append(f"{dur} & {mkt} & {m2} & {mdd} \\\\")

tex_lines.append(r'\bottomrule')
tex_lines.append(r'\end{tabular}')
tex_lines.append(r"\caption{Prolonged bear market simulation. Each row injects a recession during which M2 loses " +
                 f"{avg_panic_loss:.2%}".replace('-', '') +
                 r" per month (its average losing panic-month return). MDD is the worst-case maximum drawdown across all possible insertion points.}")
tex_lines.append(r'\label{tab:stress_scenarios}')
tex_lines.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'table_stress_scenarios.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex_lines) + '\n')
print(f"Saved: {tex_path}")
