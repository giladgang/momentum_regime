"""
ghm_benchmark.py
=================
Implements the Goulding, Harvey & Mazzoleni (2023) "Momentum Turning Points"
method, adapted to the cross-sectional setting, and benchmarked against the
regime-dependent strategies from cross_sectional_model.py.

GHM (2023) key idea
--------------------
Use the agreement/disagreement of SLOW (12-month) and FAST (1-month)
trailing *market* returns to classify each month into one of four states:
  Bull       : SLOW >= 0, FAST >= 0   (uptrend confirmed)
  Correction : SLOW >= 0, FAST < 0    (possible turning point down)
  Bear       : SLOW < 0,  FAST < 0    (downtrend confirmed)
  Rebound    : SLOW < 0,  FAST >= 0   (possible turning point up)

Cross-sectional adaptation
--------------------------
GHM apply time-series momentum (long/short the market). We adapt their
framework to cross-sectional stock selection:

  rank_slow = within-month percentile rank of mom_12
  rank_fast = within-month percentile rank of mom_1
  score(a)  = (1 - a) * rank_slow  +  a * rank_fast

Static strategies: a = 0 (SLOW), 0.25, 0.50 (MED), 0.75, 1.0 (FAST)

Dynamic strategy (DYN): vary a by market cycle
  After Bull/Bear: a = 0.50 (both signals agree, blend equally)
  After Correction: a = a_Co  (estimated from training data)
  After Rebound: a = a_Re     (estimated from training data)

DYN estimation: grid search over (a_Co, a_Re) on training-period portfolio
performance, selecting the pair that maximises the training Sharpe ratio.

Evaluation
----------
Same test set, portfolio construction, and metrics as cross_sectional_model.py:
  - Test: 2011-01 to 2024-11
  - Top-decile long-only, value-weighted, NYSE breakpoints
  - 10 bps one-way transaction cost
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ── Section 1: Load data ─────────────────────────────────────────────────────

print("[ 1/5 ] Loading data ...")

stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

# Eligibility: ordinary common shares on major exchanges, price > $1
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.reset_index(drop=True)

# Load regime data for pi_filter (needed for comparison) and market returns
regimes = pd.read_parquet('data/panel_with_regimes.parquet')
regimes['date'] = pd.to_datetime(regimes['date'])
regimes = regimes[['date', 'pi_filter', 'ret_next', 'vwretd']].drop_duplicates('date').sort_values('date')

stocks = stocks.merge(regimes[['date', 'pi_filter']], on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

# ── Section 2: Momentum signals (same as cross_sectional_model.py) ───────────

print("[ 2/5 ] Computing momentum signals ...")

MOM_LBS = list(range(1, 13))
stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)

for lb in MOM_LBS:
    roll_sum = (
        stocks.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb)
        .sum()
        .reset_index(level='permno', drop=True)
        .sort_index()
    )
    stocks[f'mom_{lb}'] = np.expm1(roll_sum)

stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

# Size: log market equity lagged 1 month
stocks['log_me'] = np.log(
    stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)

# Forward return
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

# Drop rows missing core features
MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
CORE = MOM_FEATURES + ['pi_filter', 'log_me']
df = stocks.dropna(subset=['ret_fwd'] + CORE).copy().reset_index(drop=True)

train_mask = df['date'] < '2011-01-01'
test_mask  = df['date'] >= '2011-01-01'
train = df[train_mask].copy()
test  = df[test_mask].copy()

print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
print(f"  Test dates: {test['date'].min().date()} → {test['date'].max().date()}")

# ── Section 3: Market cycle classification (GHM method) ──────────────────────

print("[ 3/5 ] Classifying market cycles (Bull/Correction/Bear/Rebound) ...")

# Build trailing market returns from value-weighted market return series
mkt_ret = regimes[['date', 'vwretd']].dropna().sort_values('date').reset_index(drop=True)
mkt_ret = mkt_ret.rename(columns={'vwretd': 'r_mkt'})

# Trailing 1-month market return: just the current month's return
mkt_ret['mkt_fast'] = mkt_ret['r_mkt']

# Trailing 12-month market return: arithmetic average of past 12 months
# (GHM use arithmetic average monthly return over 12 months)
mkt_ret['mkt_slow'] = mkt_ret['r_mkt'].rolling(12, min_periods=12).mean()

# Classify market cycles
def classify_cycle(row):
    if pd.isna(row['mkt_slow']):
        return np.nan
    if row['mkt_slow'] >= 0 and row['mkt_fast'] >= 0:
        return 'Bull'
    elif row['mkt_slow'] >= 0 and row['mkt_fast'] < 0:
        return 'Correction'
    elif row['mkt_slow'] < 0 and row['mkt_fast'] < 0:
        return 'Bear'
    else:  # mkt_slow < 0 and mkt_fast >= 0
        return 'Rebound'

mkt_ret['cycle'] = mkt_ret.apply(classify_cycle, axis=1)

# Print cycle frequencies
cycle_freq = mkt_ret['cycle'].value_counts(normalize=True).sort_index()
print("  Market cycle frequencies (full sample):")
for c, f in cycle_freq.items():
    print(f"    {c:12s}: {f:.1%}")

# Merge cycles into stock data
cycle_map = mkt_ret[['date', 'cycle', 'mkt_slow', 'mkt_fast']].dropna()
train = train.merge(cycle_map[['date', 'cycle']], on='date', how='left')
test  = test.merge(cycle_map[['date', 'cycle']], on='date', how='left')

# Print test-period cycle frequencies
test_cycles = test[['date', 'cycle']].drop_duplicates('date')['cycle'].value_counts(normalize=True).sort_index()
print("  Market cycle frequencies (test period 2011+):")
for c, f in test_cycles.items():
    print(f"    {c:12s}: {f:.1%}")

# ── Section 4: Portfolio construction (same as cross_sectional_model.py) ─────

TRADING_FEE = 0.001  # 10 bps one-way

def long_only_port(df_test, score_col, fee=TRADING_FEE):
    """Top-decile long-only portfolio, value-weighted by me, monthly rebalancing."""
    monthly      = []
    prev_weights = {}

    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        hi    = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0:
            continue
        total_me    = longs['me'].sum()
        new_weights = (longs.set_index('permno')['me'] / total_me).to_dict()
        all_permnos = set(new_weights) | set(prev_weights)
        turnover    = sum(abs(new_weights.get(p, 0) - prev_weights.get(p, 0))
                         for p in all_permnos) / 2
        r_gross     = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_weights

    return pd.DataFrame(monthly).set_index('date')['ret']


def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


# ── Section 5: GHM cross-sectional strategies ────────────────────────────────

print("[ 4/5 ] Running GHM strategies ...")

def compute_blended_score(data, a):
    """
    Compute GHM-style blended cross-sectional score:
      score = (1 - a) * pctrank(mom_12) + a * pctrank(mom_1)

    Uses within-month percentile ranks so mom_12 and mom_1 are comparable.
    """
    rank_slow = data.groupby('date')['mom_12'].rank(pct=True)
    rank_fast = data.groupby('date')['mom_1'].rank(pct=True)
    return (1 - a) * rank_slow + a * rank_fast


def compute_dyn_score(data, a_bu, a_co, a_be, a_re):
    """
    Dynamic (state-dependent) blended score.
    Speed parameter a varies by market cycle.
    """
    rank_slow = data.groupby('date')['mom_12'].rank(pct=True)
    rank_fast = data.groupby('date')['mom_1'].rank(pct=True)

    a = pd.Series(np.nan, index=data.index)
    a[data['cycle'] == 'Bull']       = a_bu
    a[data['cycle'] == 'Correction'] = a_co
    a[data['cycle'] == 'Bear']       = a_be
    a[data['cycle'] == 'Rebound']    = a_re
    a = a.fillna(0.5)  # fallback

    return (1 - a) * rank_slow + a * rank_fast


# --- 5a: Static speed strategies ---
print("  Static speed strategies ...")

static_speeds = {
    'GHM SLOW (a=0)':    0.0,
    'GHM a=0.25':        0.25,
    'GHM MED (a=0.5)':   0.50,
    'GHM a=0.75':        0.75,
    'GHM FAST (a=1)':    1.0,
}

ghm_returns = {}

for name, a in static_speeds.items():
    test[f'score_{name}'] = compute_blended_score(test, a).values
    ghm_returns[name] = long_only_port(test, f'score_{name}')
    ar, av, sh, mdd = metrics(ghm_returns[name])
    print(f"    {name:<22s}  Sharpe={sh:.2f}  Ann.Ret={ar:.1%}")


# --- 5b: DYN strategy — estimate (a_Co, a_Re) from training data ---
print("\n  Estimating DYN speeds from training data ...")

# Grid search over (a_Co, a_Re) on training data
# a_Bu and a_Be don't matter much (both signals agree), fix at 0.5
best_sharpe = -999
best_pair   = (0.5, 0.5)
grid = np.arange(0.0, 1.05, 0.1)

for a_co in grid:
    for a_re in grid:
        train[f'_dyn_score'] = compute_dyn_score(train, 0.5, a_co, 0.5, a_re).values
        r = long_only_port(train, '_dyn_score')
        if len(r) < 12:
            continue
        sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
        if sh > best_sharpe:
            best_sharpe = sh
            best_pair   = (a_co, a_re)

a_co_hat, a_re_hat = best_pair
print(f"    Best training speeds: a_Co={a_co_hat:.2f}, a_Re={a_re_hat:.2f}  "
      f"(training Sharpe={best_sharpe:.2f})")

# Apply DYN to test data
test['score_dyn'] = compute_dyn_score(test, 0.5, a_co_hat, 0.5, a_re_hat).values
ghm_returns['GHM DYN'] = long_only_port(test, 'score_dyn')

# Also clean up temp column
if '_dyn_score' in train.columns:
    train.drop(columns=['_dyn_score'], inplace=True)


# ── Section 6: Load existing strategy results for comparison ─────────────────

print("\n[ 5/5 ] Comparing with thesis methods ...")

with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

existing = artefacts['strategies_lo']
r_mkt = artefacts['r_mkt']

# Combine all strategies
all_strategies = {
    # Existing thesis methods
    'Market (buy & hold)':   r_mkt,
    'Fixed 12-mo mom':       existing['Fixed 12-mo mom'],
    'Fixed 1-mo mom':        existing['Fixed 1-mo mom'],
    'Thesis M0: Formula':    existing['Method 0: Formula'],
    'Thesis M1: LR':         existing['Method 1: LR'],
    'Thesis M2: XGB':        existing['Method 2: XGB'],
    # GHM strategies
    'GHM SLOW (a=0)':        ghm_returns['GHM SLOW (a=0)'],
    'GHM MED (a=0.5)':       ghm_returns['GHM MED (a=0.5)'],
    'GHM FAST (a=1)':        ghm_returns['GHM FAST (a=1)'],
    'GHM DYN':               ghm_returns['GHM DYN'],
}

# ── Performance table ────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  COMPARISON: Thesis methods vs GHM (2023) — TEST: 2011–2025")
print(f"  Monthly rebalancing, long-only top decile, net of 10 bps fees")
print(f"{'='*80}")
print(f"\n{'Strategy':<26} {'Ann.Ret':>9} {'Ann.Vol':>9} {'Sharpe':>8} {'Max DD':>9}")
print("-" * 66)

# Print thesis methods first
for name in ['Market (buy & hold)', 'Fixed 12-mo mom', 'Fixed 1-mo mom',
             'Thesis M0: Formula', 'Thesis M1: LR', 'Thesis M2: XGB']:
    ar, av, sh, mdd = metrics(all_strategies[name])
    print(f"{name:<26} {ar:>8.1%} {av:>8.1%} {sh:>8.2f} {mdd:>8.1%}")

print("-" * 66)

# Then GHM methods
for name in ['GHM SLOW (a=0)', 'GHM MED (a=0.5)', 'GHM FAST (a=1)', 'GHM DYN']:
    ar, av, sh, mdd = metrics(all_strategies[name])
    print(f"{name:<26} {ar:>8.1%} {av:>8.1%} {sh:>8.2f} {mdd:>8.1%}")

# Also print full GHM speed sweep
print(f"\n--- GHM static speed sweep ---")
print(f"{'Speed a':<22} {'Ann.Ret':>9} {'Ann.Vol':>9} {'Sharpe':>8} {'Max DD':>9}")
print("-" * 56)
for name, a in static_speeds.items():
    ar, av, sh, mdd = metrics(ghm_returns[name])
    print(f"{name:<22} {ar:>8.1%} {av:>8.1%} {sh:>8.2f} {mdd:>8.1%}")
ar, av, sh, mdd = metrics(ghm_returns['GHM DYN'])
print(f"{'GHM DYN':<22} {ar:>8.1%} {av:>8.1%} {sh:>8.2f} {mdd:>8.1%}")

# ── Regime-conditional Sharpe comparison ─────────────────────────────────────

test_dates = test[['date', 'pi_filter']].drop_duplicates('date').set_index('date')

def regime_sharpe(r):
    r = r.dropna()
    pi = test_dates.reindex(r.index)['pi_filter']
    calm  = r[pi < 0.5]
    panic = r[pi >= 0.5]
    def sr(x): return x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    return sr(r), sr(calm), sr(panic)

print(f"\n--- Sharpe by regime (calm/panic from HMM) ---")
print(f"{'Strategy':<26} {'Full':>7} {'Calm':>7} {'Panic':>7}")
print("-" * 50)
for name, r in all_strategies.items():
    full, calm, panic = regime_sharpe(r)
    panic_str = f"{panic:>7.2f}" if not np.isnan(panic) else "   N/A"
    print(f"{name:<26} {full:>7.2f} {calm:>7.2f} {panic_str}")

# ── Plot: Cumulative performance comparison ──────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]
plot_strats = {
    'Market (buy & hold)':   ('black',      '--', 1.0),
    'Fixed 12-mo mom':       ('grey',       '--', 0.8),
    'Thesis M0: Formula':    ('steelblue',  '-',  1.2),
    'Thesis M2: XGB':        ('darkorange', '-',  1.6),
    'GHM SLOW (a=0)':        ('seagreen',   '-.', 1.0),
    'GHM MED (a=0.5)':       ('purple',     '-',  1.4),
    'GHM DYN':               ('crimson',    '-',  1.4),
}

for name, (c, ls, lw) in plot_strats.items():
    r = all_strategies[name].dropna().sort_index()
    ax.plot(r.index, (1 + r).cumprod(), color=c, linestyle=ls, linewidth=lw, label=name)

ax.axhline(1, color='black', linewidth=0.4, linestyle=':')
ax.set_ylabel('Cumulative wealth ($1)', fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Thesis methods vs GHM (2023) — long-only top decile, 2011-2025', fontsize=11)

# Bottom panel: market cycles
ax2 = axes[1]
cycle_dates = test[['date', 'cycle']].drop_duplicates('date').sort_values('date')
cycle_colors = {'Bull': '#2ca02c', 'Correction': '#ff7f0e', 'Bear': '#d62728', 'Rebound': '#1f77b4'}
for _, row in cycle_dates.iterrows():
    if pd.notna(row['cycle']):
        ax2.axvspan(row['date'] - pd.Timedelta(days=15),
                    row['date'] + pd.Timedelta(days=15),
                    alpha=0.6, color=cycle_colors.get(row['cycle'], 'grey'),
                    linewidth=0)
# Legend for cycles
for cyc, col in cycle_colors.items():
    ax2.fill_between([], [], color=col, alpha=0.6, label=cyc)
ax2.set_ylabel('Market cycle', fontsize=9)
ax2.set_yticks([])
ax2.legend(fontsize=8, loc='upper left', ncol=4)
ax2.set_xlabel('Date')

plt.tight_layout()
fig.savefig('ghm_comparison.png', dpi=150, bbox_inches='tight')
fig.savefig('ghm_vs_thesis.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nPlot saved: ghm_comparison.png + ghm_vs_thesis.png")

# ── Plot: Sharpe ratio bar chart comparison ──────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

bar_strats = [
    'Fixed 12-mo mom', 'Fixed 1-mo mom',
    'GHM SLOW (a=0)', 'GHM MED (a=0.5)', 'GHM FAST (a=1)', 'GHM DYN',
    'Thesis M0: Formula', 'Thesis M1: LR', 'Thesis M2: XGB',
]

sharpes = [metrics(all_strategies[n])[2] for n in bar_strats]

bar_colors = []
for n in bar_strats:
    if n.startswith('GHM'):
        bar_colors.append('#7b2d8e')    # purple for GHM
    elif n.startswith('Thesis'):
        bar_colors.append('#e67e22')    # orange for thesis
    else:
        bar_colors.append('steelblue')  # blue for baselines

bars = ax.bar(range(len(bar_strats)), sharpes, color=bar_colors, alpha=0.85, edgecolor='white')

# Add value labels
for bar, sh in zip(bars, sharpes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{sh:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(bar_strats)))
ax.set_xticklabels(bar_strats, rotation=35, ha='right', fontsize=8)
ax.set_ylabel('Sharpe ratio (annualized)', fontsize=10)
ax.set_title('Sharpe ratio comparison: GHM (2023) vs Thesis methods\n'
             'Long-only top decile, 2011-2025, net of fees', fontsize=11)
ax.axhline(0, color='black', linewidth=0.5)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', alpha=0.85, label='Baselines'),
    Patch(facecolor='#7b2d8e',   alpha=0.85, label='GHM (2023)'),
    Patch(facecolor='#e67e22',   alpha=0.85, label='Thesis methods'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

plt.tight_layout()
fig.savefig('ghm_sharpe_comparison.png', dpi=150)
plt.close(fig)
print("Plot saved: ghm_sharpe_comparison.png")

print("\nDone.")
