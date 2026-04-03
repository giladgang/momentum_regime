"""
test_orthogonal_features.py
===========================
Compute candidate HMM features and measure their orthogonality to ALL
features already used in the thesis:
  - 4 HMM features (DD, VOL, CS, LVIX)
  - pi_filter (regime signal)
  - 12 momentum lookbacks (monthly cross-sectional medians)
  - 7 fundamentals (monthly cross-sectional medians)

Candidates (all real-time observable, monthly):
  TERM   — Term spread (10Y minus 2Y Treasury): yield curve slope; inverts before recessions
  TED    — TED spread (3-mo LIBOR minus T-bill): interbank lending stress
  HY_OAS — High-yield option-adjusted spread (ICE BofA): credit stress for junk bonds
  DISP   — Return dispersion: cross-sectional std of individual stock returns
  ADR    — Advance-decline ratio: fraction of stocks with positive monthly returns
  SKEW   — Cross-sectional return skewness: asymmetry in the stock return distribution
  REL_N  — Relative market participation: log ratio of # stocks trading to 12-mo avg

Orthogonality is measured three ways:
  1. Pairwise correlations with every existing feature
  2. R² from regressing each candidate on ALL existing features (novelty = 1 - R²)
  3. Regime discrimination: Welch t-test of candidate means in calm vs panic months
  → Composite score = novelty × normalised regime discrimination
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load existing data ───────────────────────────────────────────────────────

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
panel['year_month'] = panel['date'].dt.to_period('M')

stock_data = pd.read_parquet('data/crsp_msf_raw.parquet')
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['year_month'] = stock_data['date'].dt.to_period('M')

print("=" * 80)
print("  ORTHOGONALITY ANALYSIS: candidate features vs ALL existing features")
print("=" * 80)

# ── Compute monthly aggregates of stock-level features ───────────────────────

print("\nAggregating stock-level features to monthly cross-sectional medians ...")

# Momentum lookbacks
for k in range(1, 13):
    stock_data[f'mom_{k}'] = (
        stock_data.groupby('permno')['ret_adj']
        .transform(lambda r: (1 + r).rolling(k).apply(np.prod, raw=True) - 1)
    )

mom_cols = [f'mom_{k}' for k in range(1, 13)]
fund_cols = ['bm', 'roe', 'gross_profit_a', 'asset_growth', 'leverage',
             'earnings_growth', 'me']

# Cross-sectional median per month
agg_mom = stock_data.groupby('year_month')[mom_cols].median().reset_index()
agg_fund = stock_data.groupby('year_month')[fund_cols].median().reset_index()

# Log market equity median
agg_fund['log_me'] = np.log(agg_fund['me'].clip(lower=0.001))
agg_fund = agg_fund.drop(columns=['me'])

# Merge onto panel
panel = panel.merge(agg_mom, on='year_month', how='left')
panel = panel.merge(agg_fund, on='year_month', how='left')

# ── Define ALL existing features ────────────────────────────────────────────

HMM_FEATURES = {
    'DD_z':   'Market drawdown from 12-mo peak (z-scored)',
    'VOL_z':  'Log realised volatility from daily returns (z-scored)',
    'CS_z':   'Credit spread, BAA minus AAA yield (z-scored)',
    'LVIX_z': 'Log VIX, implied 30-day volatility (z-scored)',
}

REGIME_FEATURES = {
    'pi_filter': 'HMM filtered panic probability',
}

MOM_FEATURES = {f'mom_{k}': f'{k}-month trailing momentum (cross-sect median)'
                for k in range(1, 13)}

FUND_FEATURES = {
    'bm':              'Book-to-market ratio (cross-sect median)',
    'roe':             'Return on equity (cross-sect median)',
    'gross_profit_a':  'Gross profitability / assets (cross-sect median)',
    'asset_growth':    'Year-over-year asset growth (cross-sect median)',
    'leverage':        'Total debt / total assets (cross-sect median)',
    'earnings_growth': 'Signed-log YoY earnings growth (cross-sect median)',
    'log_me':          'Log market equity (cross-sect median)',
}

ALL_EXISTING = {}
ALL_EXISTING.update(HMM_FEATURES)
ALL_EXISTING.update(REGIME_FEATURES)
ALL_EXISTING.update(MOM_FEATURES)
ALL_EXISTING.update(FUND_FEATURES)

EXISTING_COLS = list(ALL_EXISTING.keys())
print(f"  Total existing features: {len(EXISTING_COLS)}")

# ── Compute candidate features ──────────────────────────────────────────────

print("\nFetching candidate features from FRED ...")
import pandas_datareader.data as web

CANDIDATES = {}

# 1. Term spread: 10Y minus 2Y Treasury yield
try:
    t10 = web.DataReader('DGS10', 'fred', start='1970-01-01', end='2025-12-31')
    t2 = web.DataReader('DGS2', 'fred', start='1970-01-01', end='2025-12-31')
    term = (t10['DGS10'] - t2['DGS2']).resample('ME').mean()
    term.name = 'TERM'
    term = term.to_frame()
    term.index = term.index.to_period('M')
    term_df = term.reset_index().rename(columns={term.index.name or term.columns[0]: 'year_month'})
    if term_df.columns[0] != 'year_month':
        term_df = term_df.rename(columns={term_df.columns[0]: 'year_month'})
    panel = panel.merge(term_df[['year_month', 'TERM']], on='year_month', how='left')
    CANDIDATES['TERM'] = ('Term spread (10Y minus 2Y Treasury yield). Measures yield curve '
                          'slope; inverts before recessions as markets price in rate cuts. '
                          'Captures monetary policy stance and recession expectations from '
                          'the fixed-income market.')
    print("  TERM: OK")
except Exception as e:
    print(f"  TERM: FAILED ({e})")

# 2. TED spread
try:
    ted_data = web.DataReader('TEDRATE', 'fred', start='1970-01-01', end='2025-12-31')
    ted = ted_data.resample('ME').mean()
    ted.columns = ['TED']
    ted.index = ted.index.to_period('M')
    ted_df = ted.reset_index()
    ted_df.columns = ['year_month', 'TED']
    panel = panel.merge(ted_df, on='year_month', how='left')
    CANDIDATES['TED'] = ('TED spread (3-month LIBOR minus 3-month T-bill yield). Measures '
                         'stress in interbank lending markets; spikes when banks distrust '
                         'each other\'s solvency. Peaked at 4.6% during the 2008 crisis. '
                         'Note: LIBOR was discontinued in June 2023; SOFR-based alternatives '
                         'exist but the series is truncated.')
    print(f"  TED: OK ({ted['TED'].notna().sum()} months)")
except Exception as e:
    print(f"  TED: FAILED ({e})")

# 3. High-yield OAS
try:
    hy = web.DataReader('BAMLH0A0HYM2', 'fred', start='1970-01-01', end='2025-12-31')
    hy = hy.resample('ME').mean()
    hy.columns = ['HY_OAS']
    hy.index = hy.index.to_period('M')
    hy_df = hy.reset_index()
    hy_df.columns = ['year_month', 'HY_OAS']
    panel = panel.merge(hy_df, on='year_month', how='left')
    CANDIDATES['HY_OAS'] = ('ICE BofA US High-Yield Option-Adjusted Spread. Measures the '
                            'extra yield investors demand for holding junk bonds over '
                            'Treasuries, adjusted for embedded options. Broader than the '
                            'BAA-AAA credit spread (CS) because it includes the riskiest '
                            'corporate issuers. Available from 1997.')
    print(f"  HY_OAS: OK ({hy['HY_OAS'].notna().sum()} months)")
except Exception as e:
    print(f"  HY_OAS: FAILED ({e})")

print("\nComputing features from CRSP data ...")

# 4. Return dispersion
disp = (
    stock_data.groupby('year_month')['ret_adj']
    .std()
    .reset_index()
    .rename(columns={'ret_adj': 'DISP'})
)
disp['DISP'] = np.log(disp['DISP'])
panel = panel.merge(disp[['year_month', 'DISP']], on='year_month', how='left')
CANDIDATES['DISP'] = ('Cross-sectional return dispersion (log). The standard deviation of '
                      'individual stock returns in each month, log-transformed. High '
                      'dispersion means stocks are moving in different directions, creating '
                      'larger payoffs for any signal that correctly ranks them. Spikes during '
                      'crises when idiosyncratic risk dominates.')

# 5. Advance-decline ratio
adr = (
    stock_data.groupby('year_month')
    .apply(lambda g: (g['ret_adj'] > 0).mean(), include_groups=False)
    .reset_index()
    .rename(columns={0: 'ADR'})
)
panel = panel.merge(adr[['year_month', 'ADR']], on='year_month', how='left')
CANDIDATES['ADR'] = ('Advance-decline ratio. The fraction of stocks with positive monthly '
                     'returns. Measures market breadth: in healthy rallies most stocks '
                     'participate (ADR > 0.5); in narrow or deteriorating markets, fewer '
                     'stocks advance. Low ADR signals fragile conditions even if the index '
                     'is flat or rising.')

# 6. Cross-sectional skewness
skew = (
    stock_data.groupby('year_month')['ret_adj']
    .skew()
    .reset_index()
    .rename(columns={'ret_adj': 'SKEW'})
)
panel = panel.merge(skew[['year_month', 'SKEW']], on='year_month', how='left')
CANDIDATES['SKEW'] = ('Cross-sectional return skewness. Measures asymmetry in the '
                      'distribution of individual stock returns within each month. '
                      'Positive skew means a few stocks had outsized gains (lottery-like); '
                      'negative skew means a few had outsized losses (crash-like). May '
                      'capture tail risk dynamics not reflected in volatility measures.')

# 7. Relative # stocks trading
n_stocks = (
    stock_data.groupby('year_month')['permno']
    .nunique()
    .reset_index()
    .rename(columns={'permno': 'N_STOCKS'})
)
n_stocks = n_stocks.sort_values('year_month')
n_stocks['N_STOCKS_MA12'] = n_stocks['N_STOCKS'].rolling(12).mean()
n_stocks['REL_N'] = np.log(n_stocks['N_STOCKS'] / n_stocks['N_STOCKS_MA12'])
panel = panel.merge(n_stocks[['year_month', 'REL_N']], on='year_month', how='left')
CANDIDATES['REL_N'] = ('Relative market participation. Log ratio of the number of stocks '
                       'trading this month to the trailing 12-month average. Captures '
                       'changes in market breadth and listing activity relative to recent '
                       'norms. Drops during stress as delistings accelerate and IPOs freeze; '
                       'rises during recovery as new listings resume.')

CAND_COLS = list(CANDIDATES.keys())
print(f"\n  Candidate features: {len(CAND_COLS)}")

# ── Restrict to 1990+ and standardize ───────────────────────────────────────

df = panel[panel['date'] >= '1990-01-01'].copy()
train_mask = df['date'] < '2011-01-01'

# Standardize candidates
for col in CAND_COLS:
    vals = df[col].astype(float)
    mu = vals[train_mask].mean()
    sd = vals[train_mask].std()
    if sd > 0 and not np.isnan(sd):
        df[f'{col}_z'] = (vals - mu) / sd
    else:
        df[f'{col}_z'] = np.nan

CAND_Z = [f'{c}_z' for c in CAND_COLS]

print(f"  Analysis sample: {df['date'].min().date()} to {df['date'].max().date()} "
      f"({len(df)} months)")

# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: Pairwise correlations with ALL existing features
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  1. PAIRWISE CORRELATIONS: each candidate vs every existing feature")
print("=" * 80)

# Group existing features for display
groups = [
    ("HMM",         list(HMM_FEATURES.keys())),
    ("Regime",      list(REGIME_FEATURES.keys())),
    ("Momentum",    [f'mom_{k}' for k in [1, 6, 12]]),  # show 1, 6, 12 as representatives
    ("Fundamentals", list(FUND_FEATURES.keys())),
]

for cand in CAND_COLS:
    cz = f'{cand}_z' if f'{cand}_z' in df.columns else cand
    print(f"\n  {cand}: {CANDIDATES[cand][:80]}...")
    for group_name, group_cols in groups:
        corrs = []
        for ec in group_cols:
            valid = df[[cz, ec]].dropna()
            if len(valid) > 20:
                r = valid[cz].corr(valid[ec])
                corrs.append((ec, r))
        if corrs:
            corr_str = "  ".join(f"{name}={r:+.2f}" for name, r in corrs)
            print(f"    {group_name:>13}: {corr_str}")

# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: R² against ALL existing features
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  2. REDUNDANCY: R² of each candidate regressed on ALL existing features")
print("     (novelty = 1 - R²; higher = more new information)")
print("=" * 80)

from sklearn.linear_model import LinearRegression

r2_results = {}
for cand in CAND_COLS:
    cz = f'{cand}_z' if f'{cand}_z' in df.columns else cand
    subset = df[EXISTING_COLS + [cz]].dropna()
    if len(subset) < 30:
        print(f"  {cand:<10}  insufficient data ({len(subset)} obs)")
        continue
    X = subset[EXISTING_COLS].values
    y = subset[cz].values
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    r2_results[cand] = r2
    novelty = 1 - r2
    bar = "█" * int(novelty * 30) + "░" * (30 - int(novelty * 30))
    print(f"  {cand:<10}  R² = {r2:.3f}  novelty = {novelty:.3f}  {bar}  "
          f"({novelty*100:.0f}% new)")

# Also show R² against HMM features only (for comparison)
print("\n  (For comparison: R² against HMM features only)")
hmm_cols = list(HMM_FEATURES.keys())
for cand in CAND_COLS:
    cz = f'{cand}_z' if f'{cand}_z' in df.columns else cand
    subset = df[hmm_cols + [cz]].dropna()
    if len(subset) < 30:
        continue
    X = subset[hmm_cols].values
    y = subset[cz].values
    reg = LinearRegression().fit(X, y)
    r2_hmm = reg.score(X, y)
    r2_all = r2_results.get(cand, np.nan)
    extra = r2_all - r2_hmm if not np.isnan(r2_all) else 0
    print(f"  {cand:<10}  R²(HMM only) = {r2_hmm:.3f}  "
          f"R²(all) = {r2_all:.3f}  "
          f"extra explained by mom+fund = {extra:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: Regime discrimination
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  3. REGIME DISCRIMINATION: calm vs panic means (Welch t-test)")
print("=" * 80)

calm = df[df['pi_smooth'] <= 0.5]
panic = df[df['pi_smooth'] > 0.5]
print(f"\n  Months: {len(calm)} calm, {len(panic)} panic\n")

header = f"  {'Feature':<12} {'Description':<55} {'μ_calm':>7} {'μ_panic':>7} {'|t|':>6} {'p':>7} {'Sig':>3}"
print(header)
print("  " + "-" * len(header))

regime_t = {}

# Existing features
for col, desc in {**HMM_FEATURES, **REGIME_FEATURES}.items():
    c_vals, p_vals = calm[col].dropna(), panic[col].dropna()
    if len(c_vals) < 10 or len(p_vals) < 10:
        continue
    t, p = stats.ttest_ind(c_vals, p_vals, equal_var=False)
    regime_t[col] = abs(t)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    print(f"  {col:<12} {desc[:55]:<55} {c_vals.mean():>+7.3f} {p_vals.mean():>+7.3f} "
          f"{abs(t):>6.1f} {p:>7.4f} {sig:>3}")

print("  " + "-" * len(header))

# Candidate features
for col in CAND_COLS:
    c_vals, p_vals = calm[col].dropna(), panic[col].dropna()
    if len(c_vals) < 10 or len(p_vals) < 10:
        continue
    t, p = stats.ttest_ind(c_vals, p_vals, equal_var=False)
    regime_t[col] = abs(t)
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    desc_short = CANDIDATES[col][:55]
    print(f"  {col:<12} {desc_short:<55} {c_vals.mean():>+7.3f} {p_vals.mean():>+7.3f} "
          f"{abs(t):>6.1f} {p:>7.4f} {sig:>3}")

# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4: Composite ranking
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  4. COMPOSITE RANKING")
print("     Score = novelty(vs ALL features) × normalised regime discrimination")
print("=" * 80)

# Max |corr| with any existing feature
max_corr_results = {}
for cand in CAND_COLS:
    cz = f'{cand}_z' if f'{cand}_z' in df.columns else cand
    max_abs = 0
    for ec in EXISTING_COLS:
        valid = df[[cz, ec]].dropna()
        if len(valid) > 20:
            r = abs(valid[cz].corr(valid[ec]))
            if r > max_abs:
                max_abs = r
    max_corr_results[cand] = max_abs

max_existing_disc = max(regime_t.get(e, 0) for e in list(HMM_FEATURES.keys()) + ['pi_filter'])

scores = {}
for c in CAND_COLS:
    if c not in r2_results or c not in regime_t:
        continue
    novelty = 1 - r2_results[c]
    disc = regime_t[c]
    disc_norm = disc / max_existing_disc if max_existing_disc > 0 else 0
    score = novelty * disc_norm
    scores[c] = {
        'novelty': novelty,
        'r2': r2_results[c],
        'disc': disc,
        'disc_norm': disc_norm,
        'score': score,
        'max_corr': max_corr_results.get(c, 1.0),
    }

ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)

print(f"\n  {'Rank':<5} {'Feature':<10} {'R²(all)':>8} {'Novelty':>8} "
      f"{'|t|':>6} {'max|r|':>7} {'Score':>7}")
print("  " + "-" * 65)
for i, (c, s) in enumerate(ranked, 1):
    print(f"  {i:<5} {c:<10} {s['r2']:>8.3f} {s['novelty']:>8.3f} "
          f"{s['disc']:>6.1f} {s['max_corr']:>7.3f} {s['score']:>7.3f}")

print(f"\n  Detailed descriptions:")
print("  " + "-" * 75)
for i, (c, s) in enumerate(ranked, 1):
    desc = CANDIDATES[c]
    print(f"\n  {i}. {c} (score = {s['score']:.3f})")
    print(f"     R²(all existing) = {s['r2']:.3f} → {s['novelty']*100:.0f}% genuinely new info")
    print(f"     Regime t-stat = {s['disc']:.1f}, max |corr| with existing = {s['max_corr']:.2f}")
    # Word-wrap description at ~75 chars
    words = desc.split()
    line = "     "
    for w in words:
        if len(line) + len(w) + 1 > 80:
            print(line)
            line = "     " + w
        else:
            line += " " + w if line.strip() else w
    if line.strip():
        print(line)

# ═══════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════

print("\nGenerating plots ...")

# ── Plot 1: Correlation heatmap (candidates vs existing, compact) ────────

# Select representative existing features to keep heatmap readable
repr_existing = list(HMM_FEATURES.keys()) + ['pi_filter'] + ['mom_1', 'mom_6', 'mom_12'] + \
                ['bm', 'roe', 'log_me', 'leverage']
repr_labels = {
    'DD_z': 'DD', 'VOL_z': 'VOL', 'CS_z': 'CS', 'LVIX_z': 'LVIX',
    'pi_filter': 'π_filter',
    'mom_1': 'mom₁', 'mom_6': 'mom₆', 'mom_12': 'mom₁₂',
    'bm': 'B/M', 'roe': 'ROE', 'log_me': 'log(ME)', 'leverage': 'Leverage',
}
cand_labels = {f'{c}_z': c for c in CAND_COLS}
cand_z_available = [f'{c}_z' for c in CAND_COLS if f'{c}_z' in df.columns]

all_plot_cols = repr_existing + cand_z_available
valid_for_plot = df[all_plot_cols].dropna()
corr_plot = valid_for_plot.corr()

# Rename for display
all_labels = {**repr_labels, **cand_labels}
corr_display = corr_plot.rename(index=all_labels, columns=all_labels)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_display.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
n = len(corr_display)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(corr_display.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(corr_display.index, fontsize=8)

for i in range(n):
    for j in range(n):
        val = corr_display.iloc[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)

# Separator between existing and candidate features
sep = len(repr_existing)
ax.axhline(sep - 0.5, color='black', linewidth=2.5)
ax.axvline(sep - 0.5, color='black', linewidth=2.5)

# Group labels
ax.text(sep/2, -1.5, 'Existing features', ha='center', fontsize=9, fontweight='bold')
ax.text(sep + len(cand_z_available)/2, -1.5, 'Candidates', ha='center',
        fontsize=9, fontweight='bold', color='darkgreen')

plt.colorbar(im, ax=ax, shrink=0.75, label='Pearson correlation')
ax.set_title('Correlation: existing features vs candidate features\n'
             '(black line separates existing from candidates)', fontsize=11)
plt.tight_layout()
fig.savefig('orthogonality_heatmap.png', dpi=150)
plt.close(fig)
print("  Saved: orthogonality_heatmap.png")

# ── Plot 2: Ranking summary ─────────────────────────────────────────────

if len(ranked) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [c for c, _ in ranked]
    novelties = [s['novelty'] for _, s in ranked]
    discs = [s['disc'] for _, s in ranked]
    composites = [s['score'] for _, s in ranked]

    # Panel 1: Novelty
    colors = ['steelblue' if n > 0.5 else 'silver' for n in novelties]
    axes[0].barh(names, novelties, color=colors, alpha=0.85, edgecolor='white')
    axes[0].set_xlabel('Novelty (1 - R²)', fontsize=10)
    axes[0].set_title('Orthogonality to ALL existing\n(higher = more new info)', fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].axvline(0.5, color='black', linewidth=0.5, linestyle=':')
    axes[0].invert_yaxis()

    # Panel 2: Regime discrimination
    colors = ['crimson' if d > 3 else 'silver' for d in discs]
    axes[1].barh(names, discs, color=colors, alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('|t-stat| (calm vs panic)', fontsize=10)
    axes[1].set_title('Regime discrimination\n(higher = better separator)', fontsize=10)
    axes[1].axvline(1.96, color='black', linewidth=0.5, linestyle=':', label='p=0.05')
    axes[1].legend(fontsize=7)
    axes[1].invert_yaxis()

    # Panel 3: Composite
    colors = ['forestgreen' if s > 0.1 else 'silver' for s in composites]
    axes[2].barh(names, composites, color=colors, alpha=0.85, edgecolor='white')
    axes[2].set_xlabel('Composite score', fontsize=10)
    axes[2].set_title('Overall value added\n(novelty × discrimination)', fontsize=10)
    axes[2].invert_yaxis()

    plt.suptitle('Candidate feature evaluation for HMM\n'
                 '(orthogonality measured against all 24 existing features)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig('orthogonality_ranking.png', dpi=150)
    plt.close(fig)
    print("  Saved: orthogonality_ranking.png")

print("\n" + "=" * 80)
print("  CORRELATION ANALYSIS DONE — now running HMM experiments")
print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: HMM EXPERIMENTS
#  Fit a 2-state Bayesian HMM with different feature combinations and evaluate
#  regime detection quality. Uses the same Gibbs sampler as hmm_model.py but
#  self-contained — does NOT modify any saved files.
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.stats import multivariate_normal, invwishart
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

# ── HMM helper functions (copied from hmm_model.py, parameterised) ──────────

def log_emission(Z, mu, Sigma, K):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)
    ])

def ffbs(Z, mu, Sigma, P, K, init_prior=None):
    n = len(Z)
    log_emit = log_emission(Z, mu, Sigma, K)
    log_alpha = np.zeros((n, K))
    if init_prior is None:
        init_prior = np.full(K, 1.0 / K)
    log_alpha[0] = np.log(init_prior + 1e-300) + log_emit[0]
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    alpha = np.exp(log_alpha)
    s = np.zeros(n, dtype=int)
    s[n-1] = np.random.choice(K, p=alpha[n-1])
    for t in range(n - 2, -1, -1):
        probs = alpha[t] * P[:, s[t+1]]
        probs /= probs.sum()
        s[t] = np.random.choice(K, p=probs)
    return s

def forward_filter(Z, mu, Sigma, P, K):
    n = len(Z)
    log_emit = log_emission(Z, mu, Sigma, K)
    log_alpha = np.zeros((n, K))
    log_alpha[0] = np.log(0.5) + log_emit[0]
    for t in range(1, n):
        for k in range(K):
            log_alpha[t, k] = log_emit[t, k] + logsumexp(
                log_alpha[t-1] + np.log(P[:, k] + 1e-300))
    log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha)


def fit_hmm(Z_train, Z_test, n_iter=1000, n_burnin=300, seed=2201):
    """
    Fit a 2-state Bayesian HMM via Gibbs sampling on Z_train.
    Returns posterior mean parameters and filtered probabilities on full data.
    """
    np.random.seed(seed)
    K = 2
    T, D = Z_train.shape

    # Priors
    m_0 = np.zeros(D)
    kappa_0 = 0.01
    nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    # Initialise from VOL (first feature that looks like volatility)
    # Use first feature's median as crude split
    states = (Z_train[:, 0] > np.median(Z_train[:, 0])).astype(int)

    mu = np.zeros((K, D))
    Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)

    P_mat = np.array([[0.95, 0.05], [0.10, 0.90]])

    n_keep = n_iter - n_burnin
    mu_draws = np.zeros((n_keep, K, D))
    Sigma_draws = np.zeros((n_keep, K, D, D))
    P_draws = np.zeros((n_keep, K, K))
    state_draws = np.zeros((n_keep, T), dtype=int)

    for m in range(n_iter):
        # FFBS
        states = ffbs(Z_train, mu, Sigma, P_mat, K)

        # NIW per regime
        for k in range(K):
            Z_k = Z_train[states == k]
            n_k = len(Z_k)
            if n_k < D + 2:
                continue
            x_bar = Z_k.mean(axis=0)
            S_k = (Z_k - x_bar).T @ (Z_k - x_bar)
            kappa_n = kappa_0 + n_k
            m_n = (kappa_0 * m_0 + n_k * x_bar) / kappa_n
            nu_n = nu_0 + n_k
            Psi_n = Psi_0 + S_k + (kappa_0 * n_k / kappa_n) * np.outer(x_bar - m_0, x_bar - m_0)
            Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
            mu[k] = np.random.multivariate_normal(m_n, Sigma[k] / kappa_n)

        # Dirichlet for transition matrix
        for i in range(K):
            counts = np.array([np.sum((states[:-1] == i) & (states[1:] == j))
                               for j in range(K)], dtype=float)
            P_mat[i] = np.random.dirichlet(alpha_dir[i] + counts)

        if m >= n_burnin:
            idx = m - n_burnin
            mu_draws[idx] = mu
            Sigma_draws[idx] = Sigma
            P_draws[idx] = P_mat
            state_draws[idx] = states

    # Posterior means
    mu_post = mu_draws.mean(axis=0)
    Sigma_post = Sigma_draws.mean(axis=0)
    P_post = P_draws.mean(axis=0)

    # Identify panic regime (higher mean of first feature after DD_z,
    # or use the feature with highest absolute separation)
    # Use separation across all features
    sep = mu_post[1] - mu_post[0]
    # Panic = regime with higher volatility-like features (positive separation)
    # If most separations are positive for state 1, state 1 is panic
    # Simple heuristic: use the mean of absolute feature values
    panic_state = 1 if np.mean(np.abs(mu_post[1])) > np.mean(np.abs(mu_post[0])) else 0

    # For robustness: check if the first feature (typically DD or VOL) helps identify
    # Actually, let's use a more robust approach: correlate with known crisis dates
    # Simpler: whichever state has higher mean for features 1+ (VOL-like) is panic
    # Just pick the state with larger overall spread from zero
    if np.sum(mu_post[1]**2) < np.sum(mu_post[0]**2):
        panic_state = 0
    else:
        panic_state = 1

    # Smoothed probabilities (train)
    pi_smooth_train = (state_draws == panic_state).mean(axis=0)

    # Filtered probabilities (full sample)
    Z_full = np.vstack([Z_train, Z_test])
    filtered = forward_filter(Z_full, mu_post, Sigma_post, P_post, K)
    pi_filter = filtered[:, panic_state]

    # Smoothed test (500 FFBS draws with fixed params)
    T_test = len(Z_test)
    last_filter = pi_filter[T - 1]
    test_init = np.zeros(K)
    test_init[panic_state] = last_filter
    test_init[1 - panic_state] = 1.0 - last_filter
    smooth_test_draws = np.zeros((200, T_test), dtype=int)
    for i in range(200):
        smooth_test_draws[i] = ffbs(Z_test, mu_post, Sigma_post, P_post, K,
                                    init_prior=test_init)
    pi_smooth_test = (smooth_test_draws == panic_state).mean(axis=0)

    pi_smooth = np.concatenate([pi_smooth_train, pi_smooth_test])
    pi_filter_train = pi_filter[:T]
    pi_filter_test = pi_filter[T:]

    return {
        'mu_post': mu_post,
        'Sigma_post': Sigma_post,
        'P_post': P_post,
        'panic_state': panic_state,
        'pi_filter': pi_filter,
        'pi_filter_train': pi_filter_train,
        'pi_filter_test': pi_filter_test,
        'pi_smooth': pi_smooth,
        'pi_smooth_train': pi_smooth_train,
        'pi_smooth_test': pi_smooth_test,
        'mu_draws': mu_draws,
    }


def evaluate_hmm(result, dates_train, dates_test, ret_train, ret_test, label=""):
    """Evaluate HMM regime quality using multiple metrics."""
    pi_f_train = result['pi_filter_train']
    pi_f_test = result['pi_filter_test']
    pi_s_train = result['pi_smooth_train']
    pi_s_test = result['pi_smooth_test']

    metrics = {}

    # 1. Regime separation: mean |Δμ| across features
    mu = result['mu_post']
    panic = result['panic_state']
    calm = 1 - panic
    sep = np.abs(mu[panic] - mu[calm])
    metrics['mean_sep'] = sep.mean()
    metrics['min_sep'] = sep.min()

    # 2. Crisis detection: mean pi_filter during known crisis months
    crisis_periods = [
        ('2000-03-01', '2002-10-01'),  # Dot-com
        ('2007-10-01', '2009-06-01'),  # GFC
        ('2020-02-01', '2021-12-01'),  # COVID
    ]
    all_dates = np.concatenate([dates_train, dates_test])
    pi_filter_full = result['pi_filter']

    crisis_scores = []
    calm_scores = []
    is_crisis = np.zeros(len(all_dates), dtype=bool)
    for start, end in crisis_periods:
        mask = (all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))
        is_crisis |= mask
        if mask.sum() > 0:
            crisis_scores.append(pi_filter_full[mask].mean())

    calm_mask = ~is_crisis
    if calm_mask.sum() > 0:
        calm_scores.append(pi_filter_full[calm_mask].mean())

    metrics['crisis_pi'] = np.mean(crisis_scores) if crisis_scores else np.nan
    metrics['calm_pi'] = np.mean(calm_scores) if calm_scores else np.nan
    metrics['crisis_contrast'] = metrics['crisis_pi'] - metrics['calm_pi']

    # 3. Binary classification: crisis months as ground truth
    y_true = is_crisis.astype(float)
    try:
        metrics['auc_filter'] = roc_auc_score(y_true, pi_filter_full)
    except:
        metrics['auc_filter'] = np.nan
    try:
        metrics['brier_filter'] = brier_score_loss(y_true, pi_filter_full)
    except:
        metrics['brier_filter'] = np.nan

    # 4. Out-of-sample crisis detection (test only)
    test_crisis_periods = [('2020-02-01', '2021-12-01')]
    for start, end in test_crisis_periods:
        mask = (dates_test >= pd.Timestamp(start)) & (dates_test <= pd.Timestamp(end))
        if mask.sum() > 0:
            metrics['covid_pi'] = pi_f_test[mask].mean()
            metrics['covid_detected'] = (pi_f_test[mask] > 0.5).mean()

    # 5. Return prediction: correlation between pi_filter and next-month |return|
    #    (panic should predict large absolute returns)
    ret_full = np.concatenate([ret_train, ret_test])
    valid = ~np.isnan(ret_full) & ~np.isnan(pi_filter_full)
    if valid.sum() > 20:
        metrics['corr_abs_ret'] = np.corrcoef(pi_filter_full[valid],
                                               np.abs(ret_full[valid]))[0, 1]

    # 6. Transition matrix quality: persistence
    P = result['P_post']
    metrics['persist_calm'] = P[calm, calm]
    metrics['persist_panic'] = P[panic, panic]

    # 7. Regime balance: fraction of months in panic (should be ~25-40%)
    metrics['frac_panic_train'] = (pi_f_train > 0.5).mean()
    metrics['frac_panic_test'] = (pi_f_test > 0.5).mean()

    return metrics


# ── Build feature matrices for each combination ────────────────────────────

print("\n" + "=" * 80)
print("  5. HMM EXPERIMENTS: fitting 2-state HMM with different feature sets")
print("=" * 80)

# Reload clean panel for feature construction
panel_raw = pd.read_parquet('data/panel.parquet')
panel_raw['date'] = pd.to_datetime(panel_raw['date'])
panel_raw['year_month'] = panel_raw['date'].dt.to_period('M')

# Standardize raw features on train data (pre-2011)
train_mask_raw = panel_raw['date'] < '2011-01-01'

# Already have DD_z, VOL_z, CS_z, LVIX_z in panel_raw
# Need to add candidate features and standardize them

# Merge candidates onto raw panel
for feat_df, col in [(disp, 'DISP'), (adr, 'ADR'), (skew, 'SKEW'),
                      (n_stocks, 'REL_N')]:
    panel_raw = panel_raw.merge(feat_df[['year_month', col]], on='year_month', how='left')

if 'TERM' in CANDIDATES:
    panel_raw = panel_raw.merge(term_df[['year_month', 'TERM']], on='year_month', how='left')
if 'TED' in CANDIDATES:
    panel_raw = panel_raw.merge(ted_df[['year_month', 'TED']], on='year_month', how='left')
if 'HY_OAS' in CANDIDATES:
    panel_raw = panel_raw.merge(hy_df[['year_month', 'HY_OAS']], on='year_month', how='left')

# Standardize candidates
for col in CAND_COLS:
    if col in panel_raw.columns:
        vals = panel_raw[col].astype(float)
        mu_tr = vals[train_mask_raw].mean()
        sd_tr = vals[train_mask_raw].std()
        if sd_tr > 0 and not np.isnan(sd_tr):
            panel_raw[f'{col}_z'] = (vals - mu_tr) / sd_tr

# Restrict to 1990+ (LVIX availability)
panel_hmm = panel_raw.dropna(subset=['DD_z', 'VOL_z', 'CS_z', 'LVIX_z']).copy()
panel_hmm = panel_hmm.reset_index(drop=True)

dates_all = panel_hmm['date'].values
ret_all = panel_hmm['vwretd'].values

split_idx = (panel_hmm['date'] < '2011-01-01').sum()
dates_train_hmm = dates_all[:split_idx]
dates_test_hmm = dates_all[split_idx:]
ret_train_hmm = ret_all[:split_idx]
ret_test_hmm = ret_all[split_idx:]

# ── Define feature combinations to test ──────────────────────────────────

BASE = ['DD_z', 'VOL_z', 'CS_z', 'LVIX_z']

COMBOS = {
    'Baseline (4)':           BASE,
    '+ DISP (5)':             BASE + ['DISP_z'],
    '+ TERM (5)':             BASE + ['TERM_z'],
    '+ REL_N (5)':            BASE + ['REL_N_z'],
    '+ TED (5)':              BASE + ['TED_z'],
    '+ SKEW (5)':             BASE + ['SKEW_z'],
    '+ HY_OAS (5)':           BASE + ['HY_OAS_z'],
    '+ ADR (5)':              BASE + ['ADR_z'],
    '+ DISP + TERM (6)':      BASE + ['DISP_z', 'TERM_z'],
    '+ DISP + REL_N (6)':     BASE + ['DISP_z', 'REL_N_z'],
    '+ DISP + TERM + REL_N (7)': BASE + ['DISP_z', 'TERM_z', 'REL_N_z'],
    '+ DISP+TERM+REL_N+SKEW (8)': BASE + ['DISP_z', 'TERM_z', 'REL_N_z', 'SKEW_z'],
    '+ DISP+TERM+REL_N+TED (8)': BASE + ['DISP_z', 'TERM_z', 'REL_N_z', 'TED_z'],
}

# Filter out combos with unavailable features
valid_combos = {}
for name, feats in COMBOS.items():
    available = all(f in panel_hmm.columns for f in feats)
    has_data = True
    if available:
        sub = panel_hmm[feats].dropna()
        has_data = len(sub) > 200  # need enough months
    if available and has_data:
        valid_combos[name] = feats
    else:
        missing = [f for f in feats if f not in panel_hmm.columns]
        print(f"  SKIP {name}: missing {missing}")

print(f"\n  Testing {len(valid_combos)} feature combinations")
print(f"  Gibbs sampler: 1000 iterations, 300 burn-in per run\n")

# ── Run HMM for each combination ────────────────────────────────────────

all_results = {}
for combo_name, feat_list in valid_combos.items():
    print(f"  Running: {combo_name} ({len(feat_list)} features) ...", end=" ", flush=True)

    sub = panel_hmm[['date'] + feat_list].dropna().reset_index(drop=True)
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']

    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)

    d_tr = sub_train['date'].values
    d_te = sub_test['date'].values
    # Match returns by index
    r_tr = panel_hmm.set_index('date').loc[pd.DatetimeIndex(d_tr), 'vwretd'].values
    r_te = panel_hmm.set_index('date').loc[pd.DatetimeIndex(d_te), 'vwretd'].values

    try:
        result = fit_hmm(Z_tr, Z_te, n_iter=1000, n_burnin=300, seed=2201)
        metrics = evaluate_hmm(result, d_tr, d_te, r_tr, r_te, label=combo_name)
        metrics['n_features'] = len(feat_list)
        metrics['n_params'] = 2 * len(feat_list) + 2 * len(feat_list) * (len(feat_list) + 1) // 2 + 2
        metrics['obs_per_param'] = len(Z_tr) / metrics['n_params']
        all_results[combo_name] = metrics
        print(f"AUC={metrics['auc_filter']:.3f}  "
              f"crisis_contrast={metrics['crisis_contrast']:.3f}  "
              f"COVID_det={metrics.get('covid_detected', 0):.0%}")
    except Exception as e:
        print(f"FAILED: {e}")

# ── Summary table ────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  6. HMM EXPERIMENT RESULTS")
print("=" * 80)

# Sort by AUC
sorted_results = sorted(all_results.items(),
                         key=lambda x: x[1].get('auc_filter', 0), reverse=True)

print(f"\n  {'Combo':<35} {'#F':>3} {'#P':>4} {'O/P':>5} "
      f"{'AUC':>6} {'Brier':>6} {'Cris↑':>6} {'Calm↓':>6} "
      f"{'Δ':>6} {'COV%':>5} {'Sep':>5} {'%Pan':>5}")
print("  " + "-" * 115)

for name, m in sorted_results:
    print(f"  {name:<35} {m['n_features']:>3} {m['n_params']:>4} "
          f"{m['obs_per_param']:>5.1f} "
          f"{m.get('auc_filter', 0):>6.3f} "
          f"{m.get('brier_filter', 1):>6.3f} "
          f"{m.get('crisis_pi', 0):>6.3f} "
          f"{m.get('calm_pi', 1):>6.3f} "
          f"{m.get('crisis_contrast', 0):>+6.3f} "
          f"{m.get('covid_detected', 0)*100:>5.0f} "
          f"{m.get('mean_sep', 0):>5.2f} "
          f"{m.get('frac_panic_test', 0)*100:>5.0f}")

print(f"""
  Legend:
    #F     = number of features
    #P     = number of HMM parameters (means + covariances + transition)
    O/P    = observations per parameter (higher = more reliable estimates)
    AUC    = area under ROC (pi_filter vs known crisis months, full sample)
    Brier  = Brier score (lower = better calibrated probabilities)
    Cris↑  = mean pi_filter during crisis months (want HIGH, close to 1)
    Calm↓  = mean pi_filter during calm months (want LOW, close to 0)
    Δ      = crisis - calm contrast (higher = better separation)
    COV%   = % of COVID months (2020-2021) correctly detected (pi > 0.5)
    Sep    = mean absolute regime-mean separation across features
    %Pan   = % of test months classified as panic (want ~25-35%)
""")

# ── Plot: comparison ─────────────────────────────────────────────────────

if len(sorted_results) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = [n for n, _ in sorted_results]
    aucs = [m.get('auc_filter', 0) for _, m in sorted_results]
    contrasts = [m.get('crisis_contrast', 0) for _, m in sorted_results]
    covid_det = [m.get('covid_detected', 0) * 100 for _, m in sorted_results]

    # Highlight baseline
    colors_auc = ['gold' if 'Baseline' in n else 'steelblue' for n in names]
    colors_con = ['gold' if 'Baseline' in n else 'crimson' for n in names]
    colors_cov = ['gold' if 'Baseline' in n else 'forestgreen' for n in names]

    axes[0].barh(names, aucs, color=colors_auc, alpha=0.85, edgecolor='white')
    axes[0].set_xlabel('AUC (crisis detection)', fontsize=10)
    axes[0].set_title('Full-sample AUC\n(pi_filter vs known crises)', fontsize=10)
    axes[0].invert_yaxis()

    axes[1].barh(names, contrasts, color=colors_con, alpha=0.85, edgecolor='white')
    axes[1].set_xlabel('Crisis - Calm contrast', fontsize=10)
    axes[1].set_title('Regime contrast\n(higher = cleaner separation)', fontsize=10)
    axes[1].invert_yaxis()

    axes[2].barh(names, covid_det, color=colors_cov, alpha=0.85, edgecolor='white')
    axes[2].set_xlabel('% COVID months detected', fontsize=10)
    axes[2].set_title('Out-of-sample COVID detection\n(% months with pi > 0.5)', fontsize=10)
    axes[2].set_xlim(0, 105)
    axes[2].invert_yaxis()

    plt.suptitle('HMM feature combination comparison\n'
                 '(gold = current baseline with 4 features)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig('hmm_feature_comparison.png', dpi=150)
    plt.close(fig)
    print("  Saved: hmm_feature_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3: DOWNSTREAM PORTFOLIO TEST
#  For each HMM variant, use its pi_filter as input to the cross-sectional
#  models (M0 formula, M1 LR, M2 XGB) and compare portfolio Sharpe ratios.
#  This answers: does a better HMM actually produce better trading results?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  7. DOWNSTREAM PORTFOLIO TEST")
print("     Using each HMM's pi_filter in the cross-sectional pipeline")
print("=" * 80)

from sklearn.linear_model import LogisticRegression as LR_sklearn
from sklearn.preprocessing import StandardScaler

# Load stock data
print("\n  Loading stock data ...")
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw.sort_values(['permno', 'date']).reset_index(drop=True)
stocks_raw = stocks_raw[stocks_raw['shrcd'].isin([10, 11])]
stocks_raw = stocks_raw[stocks_raw['exchcd'].isin([1, 2, 3])]
stocks_raw = stocks_raw[stocks_raw['prc'].abs() > 1.0]
stocks_raw = stocks_raw.reset_index(drop=True)

# Momentum signals
print("  Computing momentum signals ...")
stocks_raw['_log_ret'] = np.log1p(stocks_raw['ret_adj'].clip(lower=-0.999))
stocks_raw['_log_ret_s1'] = stocks_raw.groupby('permno')['_log_ret'].shift(1)
for lb in range(1, 13):
    roll_sum = (
        stocks_raw.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb)
        .sum()
        .reset_index(level='permno', drop=True)
        .sort_index()
    )
    stocks_raw[f'mom_{lb}'] = np.expm1(roll_sum)
stocks_raw.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

# Log market equity
stocks_raw['log_me'] = np.log(
    stocks_raw.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)

# Forward return
stocks_raw['ret_fwd'] = stocks_raw.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

MOM_FEATURES_CS = [f'mom_{lb}' for lb in range(1, 13)]
ALL_FEATURES_CS = MOM_FEATURES_CS + [
    'pi_filter', 'bm', 'roe', 'earnings_growth',
    'leverage', 'asset_growth', 'gross_profit_a', 'log_me'
]
CORE_FEATURES_CS = MOM_FEATURES_CS + ['pi_filter', 'log_me']

TRADING_FEE = 0.001  # 10 bps


def long_only_port_test(df_test, score_col, fee=TRADING_FEE):
    """Top-decile long-only portfolio, value-weighted, monthly rebalance."""
    monthly = []
    prev_weights = {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0:
            continue
        total_me = longs['me'].sum()
        new_weights = (longs.set_index('permno')['me'] / total_me).to_dict()
        all_p = set(new_weights) | set(prev_weights)
        turnover = sum(abs(new_weights.get(p, 0) - prev_weights.get(p, 0))
                       for p in all_p) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_weights
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']


def compute_metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 12:
        return 0, 0, 0, 0
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd


def run_portfolio_test(pi_filter_series, combo_name):
    """
    Given a pi_filter time series (date -> value), run M0/M1/M2 and return metrics.
    """
    # Merge pi_filter onto stock data
    if isinstance(pi_filter_series, pd.DataFrame):
        pi_df = pi_filter_series[['date', 'pi_filter']].copy()
    else:
        pi_df = pi_filter_series.reset_index()
        pi_df.columns = ['date', 'pi_filter']
    stocks_with_pi = stocks_raw.merge(pi_df, on='date', how='left')
    stocks_with_pi['pi_filter'] = stocks_with_pi['pi_filter'].ffill()

    # Drop rows missing core features
    df_cs = stocks_with_pi.dropna(subset=['ret_fwd'] + CORE_FEATURES_CS).copy()
    df_cs = df_cs.reset_index(drop=True)

    train_cs = df_cs[df_cs['date'] < '2011-01-01'].copy()
    test_cs = df_cs[df_cs['date'] >= '2011-01-01'].copy()

    if len(train_cs) < 1000 or len(test_cs) < 1000:
        return None

    results = {}

    # ── M0: Deterministic formula ──
    pi_by_month = test_cs.groupby('date')['pi_filter'].first()
    lb_by_month = np.clip(np.round(12 - 11 * pi_by_month).astype(int), 1, 12)
    date_to_lb = lb_by_month.to_dict()
    test_cs['score_m0'] = np.nan
    for date, lb in date_to_lb.items():
        mask = test_cs['date'] == date
        test_cs.loc[mask, 'score_m0'] = test_cs.loc[mask, f'mom_{lb}']
    r_m0 = long_only_port_test(test_cs, 'score_m0')
    if len(r_m0) > 12:
        results['M0_Formula'] = compute_metrics(r_m0)

    # ── M1: Logistic Regression (full features) ──
    # Impute missing fundamentals with train median for LR
    train_cs_imp = train_cs.copy()
    test_cs_imp = test_cs.copy()
    fund_cols_cs = ['bm', 'roe', 'earnings_growth', 'leverage',
                    'asset_growth', 'gross_profit_a']
    for col in fund_cols_cs:
        med = train_cs_imp[col].median()
        train_cs_imp[col] = train_cs_imp[col].fillna(med)
        test_cs_imp[col] = test_cs_imp[col].fillna(med)

    X_tr = train_cs_imp[ALL_FEATURES_CS].values.astype(float)
    X_te = test_cs_imp[ALL_FEATURES_CS].values.astype(float)

    # Standardize
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Target: above cross-sectional median
    train_cs_imp['above_med'] = train_cs_imp.groupby('date')['ret_fwd'].transform(
        lambda x: (x > x.median()).astype(int))
    y_tr = train_cs_imp['above_med'].values

    lr_model = LR_sklearn(max_iter=1000, solver='lbfgs', random_state=42)
    lr_model.fit(X_tr_s, y_tr)
    test_cs_imp['score_lr'] = lr_model.predict_proba(X_te_s)[:, 1]
    r_lr = long_only_port_test(test_cs_imp, 'score_lr')
    if len(r_lr) > 12:
        results['M1_LR'] = compute_metrics(r_lr)

    # ── M2: XGBoost (full features, native NaN handling) ──
    try:
        from xgboost import XGBRegressor
        X_tr_xgb = train_cs[ALL_FEATURES_CS].values.astype(float)
        X_te_xgb = test_cs[ALL_FEATURES_CS].values.astype(float)
        y_tr_xgb = train_cs['ret_fwd'].values.astype(float)

        xgb_model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_model.fit(X_tr_xgb, y_tr_xgb)
        test_cs['score_xgb'] = xgb_model.predict(X_te_xgb)
        r_xgb = long_only_port_test(test_cs, 'score_xgb')
        if len(r_xgb) > 12:
            results['M2_XGB'] = compute_metrics(r_xgb)
    except Exception as e:
        print(f"    XGB failed: {e}")

    return results


# ── Run portfolio test for each HMM variant ──────────────────────────────

# First, get baseline pi_filter from saved data
baseline_regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'pi_filter']]
baseline_regimes['date'] = pd.to_datetime(baseline_regimes['date'])

portfolio_results = {}

# Baseline (original 4 features)
print("\n  Running portfolio test: Baseline (saved pi_filter) ...")
r_baseline = run_portfolio_test(baseline_regimes, 'Baseline (4)')
if r_baseline:
    portfolio_results['Baseline (4)'] = r_baseline
    for method, (ar, av, sr, mdd) in r_baseline.items():
        print(f"    {method}: Sharpe={sr:.3f}  Return={ar:.1%}  Vol={av:.1%}")

# Each HMM experiment
for combo_name, combo_metrics in all_results.items():
    if combo_name == 'Baseline (4)':
        continue  # already done with saved data

    feat_list = valid_combos[combo_name]
    print(f"\n  Running portfolio test: {combo_name} ...", end=" ", flush=True)

    # Re-fit HMM to get pi_filter (reuse the fit_hmm function)
    sub = panel_hmm[['date'] + feat_list].dropna().reset_index(drop=True)
    sub_train = sub[sub['date'] < '2011-01-01']
    sub_test = sub[sub['date'] >= '2011-01-01']
    Z_tr = sub_train[feat_list].values.astype(float)
    Z_te = sub_test[feat_list].values.astype(float)

    result = fit_hmm(Z_tr, Z_te, n_iter=1000, n_burnin=300, seed=2201)

    # Build pi_filter series
    dates_combo = sub['date'].values
    pi_series = pd.DataFrame({
        'date': dates_combo,
        'pi_filter': result['pi_filter']
    })

    port_res = run_portfolio_test(pi_series, combo_name)
    if port_res:
        portfolio_results[combo_name] = port_res
        for method, (ar, av, sr, mdd) in port_res.items():
            print(f"{method}: SR={sr:.3f}", end="  ")
        print()
    else:
        print("FAILED")

# ── Portfolio results summary ────────────────────────────────────────────

print("\n" + "=" * 80)
print("  8. PORTFOLIO RESULTS SUMMARY")
print("     Sharpe ratios by HMM feature set and cross-sectional method")
print("=" * 80)

# Collect all methods
all_methods = set()
for combo, methods in portfolio_results.items():
    all_methods.update(methods.keys())
all_methods = sorted(all_methods)

# Header
header = f"  {'HMM Features':<38}"
for m in all_methods:
    header += f" {m+' SR':>10} {m+' Vol':>10}"
print(f"\n{header}")
print("  " + "-" * (38 + len(all_methods) * 22))

# Sort by M1_LR Sharpe (the main metric)
sorted_port = sorted(portfolio_results.items(),
                      key=lambda x: x[1].get('M1_LR', (0,0,0,0))[2], reverse=True)

for combo, methods in sorted_port:
    line = f"  {combo:<38}"
    for m in all_methods:
        if m in methods:
            ar, av, sr, mdd = methods[m]
            line += f" {sr:>10.3f} {av:>9.1%}"
        else:
            line += f" {'--':>10} {'--':>10}"
    print(line)

# Detailed view: full metrics for top combinations
print(f"\n  Detailed results for top 5 combinations (by M1 LR Sharpe):\n")
for combo, methods in sorted_port[:5]:
    print(f"  {combo}")
    for m in all_methods:
        if m in methods:
            ar, av, sr, mdd = methods[m]
            print(f"    {m:<12}  Return={ar:>6.1%}  Vol={av:>6.1%}  "
                  f"Sharpe={sr:>6.3f}  MaxDD={mdd:>7.1%}")
    print()

# ── Final comparison plot ─────────────────────────────────────────────────

if len(sorted_port) > 1:
    fig, axes = plt.subplots(1, len(all_methods), figsize=(7 * len(all_methods), 7))
    if len(all_methods) == 1:
        axes = [axes]

    for ax_idx, method in enumerate(all_methods):
        names_plot = []
        sharpes_plot = []
        for combo, methods in sorted_port:
            if method in methods:
                names_plot.append(combo)
                sharpes_plot.append(methods[method][2])

        colors = ['gold' if 'Baseline' in n else 'steelblue' for n in names_plot]
        axes[ax_idx].barh(names_plot, sharpes_plot, color=colors, alpha=0.85,
                          edgecolor='white')
        axes[ax_idx].set_xlabel('Sharpe Ratio', fontsize=10)
        axes[ax_idx].set_title(f'{method}\n(gold = baseline)', fontsize=11)
        axes[ax_idx].invert_yaxis()

        # Add value labels
        for i, v in enumerate(sharpes_plot):
            axes[ax_idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

    plt.suptitle('Portfolio Sharpe ratios by HMM feature set\n'
                 'Same cross-sectional models, different regime signals',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig('hmm_portfolio_comparison.png', dpi=150)
    plt.close(fig)
    print("  Saved: hmm_portfolio_comparison.png")

print("\n" + "=" * 80)
print("  ALL DONE")
print("=" * 80)
