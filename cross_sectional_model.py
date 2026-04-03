"""
cross_sectional_model.py
========================
Regime-dependent cross-sectional momentum with four methods.

Each month, score all stocks and form a long-short portfolio:
  Long  : top decile by score (value-weighted, NYSE breakpoints)
  Short : bottom decile by score (value-weighted, NYSE breakpoints)

Features per stock:
  mom_1 … mom_12               — trailing momentum at 12 monthly lookbacks
  pi_filter                    — HMM regime state (panic probability)
  bm, roe, earnings_growth     — value / profitability
  leverage, asset_growth       — risk / investment
  gross_profit_a               — Novy-Marx profitability
  log_me                       — size (log market equity)

Three methods
-------------
  Method 0 : Deterministic formula — lb = round(12 - 11*pi), score = mom_{lb}
  Method 1 : Logistic Regression   — P(above median return) from all features
  Method 2 : XGBoost Regressor     — predicted forward return from all features

Train: pre-2011   |   Test: 2011–2025 (out-of-sample)

SHAP analysis (Method 2) shows pi_filter's contribution vs other features.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap

# ── Section 1: Load & merge data ──────────────────────────────────────────────

print("[ 1/6 ] Loading data ...")
stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)

# Eligibility: ordinary common shares on major exchanges, price > $1
stocks = stocks[stocks['shrcd'].isin([10, 11])]       # ordinary common shares only
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]     # NYSE / AMEX / NASDAQ
stocks = stocks[stocks['prc'].abs() > 1.0]            # exclude penny stocks
stocks = stocks.reset_index(drop=True)

regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'pi_filter', 'ret_next']]
regimes['date'] = pd.to_datetime(regimes['date'])

stocks = stocks.merge(regimes[['date', 'pi_filter']], on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

print(f"  Stocks: {len(stocks):,} rows  |  {stocks['permno'].nunique():,} permnos  "
      f"|  {stocks['date'].min().date()} → {stocks['date'].max().date()}")

# ── Section 2: Per-stock momentum signals ─────────────────────────────────────

print("[ 2/6 ] Computing momentum signals ...")

MOM_LBS = list(range(1, 13))  # 1 through 12 — full monthly resolution

# Fast log-sum approach: log(prod(1+r)) = sum(log(1+r))
# Use groupby().rolling() which is fully C-level (no Python per group/window)
stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)  # shift avoids look-ahead

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

# Size: log market equity lagged 1 month (avoid look-ahead)
stocks['log_me'] = np.log(
    stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)

# ── Section 3: Target variable ────────────────────────────────────────────────

stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

MOM_FEATURES  = [f'mom_{lb}' for lb in MOM_LBS]   # mom_1 … mom_12
FEATURES = MOM_FEATURES + [
    'pi_filter',
    'bm', 'roe', 'earnings_growth',
    'leverage', 'asset_growth', 'gross_profit_a',
    'log_me',
]

# Drop rows missing target or core momentum/regime features only.
# XGBoost handles NaN fundamentals natively — do NOT drop on those.
CORE_FEATURES = MOM_FEATURES + ['pi_filter', 'log_me']
df = stocks.dropna(subset=['ret_fwd'] + CORE_FEATURES).copy()
df = df.reset_index(drop=True)

print(f"  Usable rows after dropna: {len(df):,}  "
      f"|  {df['date'].min().date()} → {df['date'].max().date()}")

# ── Section 4: Train / test split ─────────────────────────────────────────────

train_mask = df['date'] < '2011-01-01'
test_mask  = df['date'] >= '2011-01-01'
train = df[train_mask].copy()
test  = df[test_mask].copy()

X_train = train[FEATURES].values.astype(float)
y_train = train['ret_fwd'].values.astype(float)
X_test  = test[FEATURES].values.astype(float)

# Cross-sectional median label for LR
train['above_med'] = train.groupby('date')['ret_fwd'].transform(
    lambda x: (x > x.median()).astype(int)
)

print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")

# ── Section 5: Portfolio construction helper ──────────────────────────────────

# One-way transaction cost per unit of portfolio turnover.
# 10 bps is standard for a value-weighted, large-cap-biased strategy
# (NYSE breakpoints tilt toward big liquid stocks).
# Round-trip cost = 2 × 10 bps = 20 bps per full position change.
TRADING_FEE = 0.001  # 10 bps one-way

QUARTERLY_MONTHS = {3, 6, 9, 12}   # end-of-quarter rebalancing dates

def long_only_port(df_test, score_col, fee=TRADING_FEE, rebal_months=None):
    """
    Top-decile long-only portfolio, value-weighted by me.

    rebal_months : set of month numbers on which to rebalance (default: every month).
                   Use QUARTERLY_MONTHS for quarterly rebalancing.
                   In non-rebalancing months the existing holdings are held at
                   current market-cap weights — no trading cost is incurred.

    Fee is applied only on rebalancing months:
      cost = fee * one_way_turnover
      one_way_turnover = 0.5 * sum(|new_weight - prev_weight|)
    """
    if rebal_months is None:
        rebal_months = set(range(1, 13))

    monthly      = []
    prev_weights = {}   # permno -> weight after last rebalance

    for date, grp in df_test.groupby('date'):
        if date.month in rebal_months:
            # ── Rebalancing month: score stocks, trade to new portfolio ──
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
        else:
            # ── Hold month: keep previous holdings, no fee ──
            if not prev_weights:
                continue
            held = grp[grp['permno'].isin(prev_weights)]
            if held.empty or held['me'].sum() == 0:
                continue
            total_me    = held['me'].sum()
            r_gross     = (held['ret_fwd'] * held['me']).sum() / total_me
            # drift weights to current market caps for next hold/rebal month
            prev_weights = (held.set_index('permno')['me'] / total_me).to_dict()
            monthly.append({'date': date, 'ret': r_gross})

    return pd.DataFrame(monthly).set_index('date')['ret']

def long_short_port(df_test, score_col):
    """
    Each month: NYSE P10/P90 breakpoints on score_col.
    Long >= P90, Short <= P10, value-weighted by me.
    Returns pd.Series of monthly long-short returns indexed by date.
    """
    monthly = []
    for date, grp in df_test.groupby('date'):
        nyse    = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi  = nyse.quantile(0.10), nyse.quantile(0.90)
        longs   = grp[grp[score_col] >= hi]
        shorts  = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        r_long  = (longs['ret_fwd']  * longs['me']).sum()  / longs['me'].sum()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        monthly.append({'date': date, 'ret': r_long - r_short})
    return pd.DataFrame(monthly).set_index('date')['ret']

def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd

# ── Section 6: Benchmarks ─────────────────────────────────────────────────────

print("[ 3/6 ] Computing benchmarks ...")
test['score_mom12'] = test['mom_12']
test['score_mom1']  = test['mom_1']
r_mom12_lo    = long_only_port(test, 'score_mom12')
r_mom1_lo     = long_only_port(test, 'score_mom1')
r_mom12_lo_q  = long_only_port(test, 'score_mom12', rebal_months=QUARTERLY_MONTHS)
r_mom1_lo_q   = long_only_port(test, 'score_mom1',  rebal_months=QUARTERLY_MONTHS)

# Market buy & hold
mkt = regimes[(regimes['date'] >= '2011-01-01') & regimes['ret_next'].notna()].copy()
r_mkt = mkt.set_index('date')['ret_next']

# ── Section 7: Method 0 — Deterministic formula ───────────────────────────────

print("[ 4/6 ] Running three methods ...")
print("  Method 0: Deterministic formula ...")

# Vectorized: compute lb per month (167 unique months), not per row (557k rows)
pi_by_month  = test.groupby('date')['pi_filter'].first()
lb_by_month  = np.clip(np.round(12 - 11 * pi_by_month).astype(int), 1, 12)
date_to_lb   = lb_by_month.to_dict()

test['score_formula'] = np.nan
for date, lb in date_to_lb.items():
    mask = test['date'] == date
    test.loc[mask, 'score_formula'] = test.loc[mask, f'mom_{lb}']
r_formula_lo   = long_only_port(test, 'score_formula')
r_formula_lo_q = long_only_port(test, 'score_formula', rebal_months=QUARTERLY_MONTHS)
print(f"    {len(r_formula_lo)} monthly observations")

# ── Section 8: Method 1 — Logistic Regression ────────────────────────────────

print("  Method 1: Logistic Regression ...")
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# LR cannot handle NaN — median-impute fundamentals on train, apply same to test
imputer  = SimpleImputer(strategy='median')
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(imputer.fit_transform(X_train))
X_te_s   = scaler.transform(imputer.transform(X_test))

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_tr_s, train['above_med'].values)
test = test.copy()
test['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
r_lr_lo   = long_only_port(test, 'score_lr')
r_lr_lo_q = long_only_port(test, 'score_lr', rebal_months=QUARTERLY_MONTHS)

# LR weights plot
fig_lr, ax_lr = plt.subplots(figsize=(8, 6))
coef = pd.Series(lr.coef_[0], index=FEATURES).sort_values()
colors_lr = ['crimson' if c < 0 else 'steelblue' for c in coef]
coef.plot(kind='barh', ax=ax_lr, color=colors_lr, alpha=0.8)
ax_lr.axvline(0, color='black', linewidth=0.8)
ax_lr.set_xlabel('Learned weight (standardized)', fontsize=9)
ax_lr.set_title('Logistic Regression weights\n(positive = predicts above-median return)', fontsize=10)
plt.tight_layout()
fig_lr.savefig('cs_lr_weights.png', dpi=150)
plt.close(fig_lr)
print("  LR weights saved: cs_lr_weights.png")

# ── Section 9: Method 2 — XGBoost Regressor ──────────────────────────────────

print("  Method 2: XGBoost ...")
xgb = XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   tree_method='hist', random_state=42, verbosity=0)
xgb.fit(X_train, y_train)


test['score_xgb'] = xgb.predict(X_test)
r_xgb_lo   = long_only_port(test, 'score_xgb')
r_xgb_lo_q = long_only_port(test, 'score_xgb', rebal_months=QUARTERLY_MONTHS)

# ── Average tree: most common feature & threshold at each (depth, position) ──
import re
from collections import defaultdict, Counter

def parse_all_trees(booster, feature_names):
    """Return list of dicts: {node_id: {leaf,feature,threshold,yes,no,depth}}."""
    dump  = booster.get_dump(dump_format='text')
    trees = []
    for raw in dump:
        nodes = {}
        for line in raw.strip().split('\n'):
            depth   = len(line) - len(line.lstrip('\t'))
            line    = line.strip()
            node_id = int(re.match(r'(\d+):', line).group(1))
            if 'leaf' in line:
                val = float(re.search(r'leaf=([-\d.e+]+)', line).group(1))
                nodes[node_id] = dict(leaf=True, value=val, depth=depth)
            else:
                m     = re.search(r'\[f(\d+)<([-\d.e+]+)\].*yes=(\d+),no=(\d+)', line)
                fidx  = int(m.group(1))
                nodes[node_id] = dict(leaf=False, depth=depth,
                                      feature=feature_names[fidx] if fidx < len(feature_names) else f'f{fidx}',
                                      threshold=float(m.group(2)),
                                      yes=int(m.group(3)), no=int(m.group(4)))
        trees.append(nodes)
    return trees

def assign_positions(tree, node_id=0, pos_str='root'):
    """Map each node to a (depth, position-string) key, e.g. (2, 'L-R')."""
    mapping = {node_id: pos_str}
    n = tree[node_id]
    if not n['leaf']:
        mapping.update(assign_positions(tree, n['yes'], pos_str + '-L'))
        mapping.update(assign_positions(tree, n['no'],  pos_str + '-R'))
    return mapping

# Aggregate: for each (depth, position) collect all features used
pos_features   = defaultdict(list)   # (depth,pos) -> [feature, ...]
pos_thresholds = defaultdict(list)   # (depth,pos) -> [threshold, ...]
pos_leaf_vals  = defaultdict(list)   # (depth,pos) -> [leaf_val, ...]

all_trees = parse_all_trees(xgb.get_booster(), FEATURES)
for tree in all_trees:
    pos_map = assign_positions(tree)
    for nid, pos_str in pos_map.items():
        depth = tree[nid]['depth']
        key   = (depth, pos_str)
        if tree[nid]['leaf']:
            pos_leaf_vals[key].append(tree[nid]['value'])
        else:
            pos_features[key].append(tree[nid]['feature'])
            pos_thresholds[key].append(tree[nid]['threshold'])

# Build "average tree": most common feature + median threshold at each position
def build_avg_tree(max_depth=4):
    """Return dict {pos_str: node_info} for the consensus tree."""
    avg = {}
    queue = [('root', 0)]
    while queue:
        pos_str, depth = queue.pop(0)
        key = (depth, pos_str)
        if depth == max_depth or key not in pos_features:
            # leaf
            vals = pos_leaf_vals.get(key, [0.0])
            avg[pos_str] = dict(leaf=True, value=float(np.mean(vals)),
                                count=len(vals))
        else:
            feat  = Counter(pos_features[key]).most_common(1)[0][0]
            freq  = Counter(pos_features[key]).most_common(1)[0][1]
            pct   = freq / len(pos_features[key])
            thresh = float(np.median([t for f, t in
                                      zip(pos_features[key], pos_thresholds[key])
                                      if f == feat]))
            avg[pos_str] = dict(leaf=False, feature=feat, threshold=thresh,
                                pct=pct, count=len(pos_features[key]))
            queue.append((pos_str + '-L', depth + 1))
            queue.append((pos_str + '-R', depth + 1))
    return avg

avg_tree = build_avg_tree(max_depth=4)

# Layout and draw
def layout_avg(pos_str='root', x=0.5, y=1.0, dx=0.22):
    positions = {pos_str: (x, y)}
    node = avg_tree.get(pos_str)
    if node and not node['leaf']:
        positions.update(layout_avg(pos_str + '-L', x - dx, y - 0.20, dx * 0.5))
        positions.update(layout_avg(pos_str + '-R', x + dx, y - 0.20, dx * 0.5))
    return positions

positions = layout_avg()

fig_avg, ax_avg = plt.subplots(figsize=(22, 9))
for pos_str, (x, y) in positions.items():
    node = avg_tree[pos_str]
    if node['leaf']:
        label = f"leaf\navg={node['value']:+.5f}"
        bbox  = dict(boxstyle='round,pad=0.3', fc='#d4edda', ec='#28a745', lw=1.2)
    else:
        label = f"{node['feature']}\n< {node['threshold']:.4f}\n({node['pct']:.0%} of trees)"
        bbox  = dict(boxstyle='round,pad=0.3', fc='#cce5ff', ec='#004085', lw=1.2)
    ax_avg.text(x, y, label, ha='center', va='center', fontsize=7,
                bbox=bbox, zorder=3)
    if not node['leaf']:
        for child_key, side in [(pos_str + '-L', 'Y'), (pos_str + '-R', 'N')]:
            if child_key in positions:
                cx, cy = positions[child_key]
                ax_avg.annotate('', xy=(cx, cy + 0.025), xytext=(x, y - 0.028),
                                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))
                color = '#28a745' if side == 'Y' else '#dc3545'
                ax_avg.text((x + cx) / 2, (y + cy) / 2, side, fontsize=7,
                            color=color, ha='center', va='center', fontweight='bold')

ax_avg.set_xlim(-0.05, 1.05)
ax_avg.set_ylim(0.50, 1.10)
ax_avg.axis('off')
ax_avg.set_title('Average XGBoost tree (most common feature & median threshold\n'
                 'at each split position, aggregated over all 500 trees)', fontsize=11, pad=10)
plt.tight_layout()
fig_avg.savefig('cs_avg_tree.png', dpi=150, bbox_inches='tight')
plt.close(fig_avg)
print("  Average tree saved: cs_avg_tree.png")

# ── Section 10: Performance table ────────────────────────────────────────────

strategies_lo = {
    'Market (buy & hold)':   r_mkt,
    'Fixed 12-mo mom':       r_mom12_lo,
    'Fixed 1-mo mom':        r_mom1_lo,
    'Method 0: Formula':     r_formula_lo,
    'Method 1: LR':          r_lr_lo,
    'Method 2: XGB':         r_xgb_lo,
}
strategies_lo_q = {
    'Market (buy & hold)':   r_mkt,
    'Fixed 12-mo mom':       r_mom12_lo_q,
    'Fixed 1-mo mom':        r_mom1_lo_q,
    'Method 0: Formula':     r_formula_lo_q,
    'Method 1: LR':          r_lr_lo_q,
    'Method 2: XGB':         r_xgb_lo_q,
}

# Use monthly long-only as the main strategies dict for plots
strategies = strategies_lo

# Regime-conditional Sharpe helper
test_dates = test[['date', 'pi_filter']].drop_duplicates('date').set_index('date')
def regime_sharpe(r):
    r = r.dropna()
    pi = test_dates.reindex(r.index)['pi_filter']
    calm  = r[pi < 0.5]
    panic = r[pi >= 0.5]
    def sr(x): return x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    return sr(r), sr(calm), sr(panic)

def print_perf_table(label, strats):
    print(f"\n--- {label} ---")
    print(f"{'Strategy':<26} {'Ann.Ret':>9} {'Ann.Vol':>9} {'Sharpe':>8} {'Max DD':>9}")
    print("-" * 66)
    for name, r in strats.items():
        ar, av, sh, mdd = metrics(r)
        print(f"{name:<26} {ar:>8.1%} {av:>8.1%} {sh:>8.2f} {mdd:>8.1%}")

def print_regime_table(label, strats):
    print(f"\n--- Sharpe by regime — {label} ---")
    print(f"{'Strategy':<26} {'Full':>7} {'Calm':>7} {'Panic':>7}")
    print("-" * 50)
    for name, r in strats.items():
        full, calm, panic = regime_sharpe(r)
        print(f"{name:<26} {full:>7.2f} {calm:>7.2f} {panic:>7.2f}")

print_perf_table("Monthly rebalancing — net of fees (TEST: 2011–2025)", strategies_lo)
print_regime_table("monthly rebalancing", strategies_lo)

print_perf_table("Quarterly rebalancing — net of fees (TEST: 2011–2025)", strategies_lo_q)
print_regime_table("quarterly rebalancing", strategies_lo_q)

# ── Monthly vs Quarterly comparison: Sharpe difference ───────────────────────
print(f"\n--- Monthly vs Quarterly: Sharpe difference (monthly − quarterly) ---")
print(f"{'Strategy':<26} {'Monthly':>9} {'Quarterly':>10} {'Diff':>8}")
print("-" * 58)
for name in strategies_lo:
    sh_m = metrics(strategies_lo[name])[2]
    sh_q = metrics(strategies_lo_q[name])[2]
    print(f"{name:<26} {sh_m:>9.2f} {sh_q:>10.2f} {sh_m - sh_q:>+8.2f}")

# ── Section 12: SHAP feature importance (Method 2) ───────────────────────────

print("\n[ 5/6 ] Computing SHAP values ...")

explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)   # (N_test, n_features)

mean_shap = np.abs(shap_values).mean(axis=0)
feat_imp_raw = pd.Series(mean_shap, index=FEATURES)

# Aggregate all momentum signals into a single "Momentum" group
def aggregate_momentum(imp):
    """Sum momentum feature importances into a single 'Momentum' entry."""
    mom_total = imp[MOM_FEATURES].sum()
    other     = imp.drop(MOM_FEATURES)
    return pd.concat([other, pd.Series({'Momentum': mom_total})]).sort_values(ascending=True)

feat_imp = aggregate_momentum(feat_imp_raw)

# Pretty display names for plots
DISPLAY_NAMES = {
    'pi_filter': r'$\pi^{filter}$',
    'log_me': 'Log market equity',
    'bm': 'Book-to-market',
    'roe': 'Return on equity',
    'asset_growth': 'Asset growth',
    'gross_profit_a': 'Gross profitability',
    'leverage': 'Leverage',
    'earnings_growth': 'Earnings growth',
    'Momentum': 'Momentum (aggregate)',
}

def rename_index(s):
    return s.rename(index=lambda x: DISPLAY_NAMES.get(x, x))

feat_imp = rename_index(feat_imp)

# Regime-split SHAP: mean |SHAP| for pi_filter in calm vs panic months
pi_test_per_row = test['pi_filter'].values
pi_idx = FEATURES.index('pi_filter')
shap_pi_calm  = np.abs(shap_values[pi_test_per_row < 0.5,  pi_idx]).mean()
shap_pi_panic = np.abs(shap_values[pi_test_per_row >= 0.5, pi_idx]).mean()

print(f"  pi_filter mean |SHAP|: calm={shap_pi_calm:.5f}  panic={shap_pi_panic:.5f}")

# Calm-only and panic-only XGBoost models for regime-split feature importance
calm_mask  = train['pi_filter'] < 0.5
panic_mask = train['pi_filter'] >= 0.5

xgb_calm = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method='hist', random_state=42, verbosity=0)
xgb_calm.fit(X_train[calm_mask.values], y_train[calm_mask.values])

xgb_panic = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         tree_method='hist', random_state=42, verbosity=0)
xgb_panic.fit(X_train[panic_mask.values], y_train[panic_mask.values])

shap_calm_vals  = shap.TreeExplainer(xgb_calm).shap_values(X_test)
shap_panic_vals = shap.TreeExplainer(xgb_panic).shap_values(X_test)

imp_calm_raw  = pd.Series(np.abs(shap_calm_vals).mean(axis=0),  index=FEATURES)
imp_panic_raw = pd.Series(np.abs(shap_panic_vals).mean(axis=0), index=FEATURES)
imp_calm  = rename_index(aggregate_momentum(imp_calm_raw))
imp_panic = rename_index(aggregate_momentum(imp_panic_raw))

# ── Section 13: Partial dependence of pi_filter ───────────────────────────────

median_row = pd.DataFrame([np.nanmedian(X_test, axis=0)], columns=FEATURES)
pi_grid    = np.linspace(0, 1, 100)
pdp_scores = []
for pv in pi_grid:
    row = median_row.copy()
    row['pi_filter'] = pv
    pdp_scores.append(xgb.predict(row.values)[0])

# ── Section 14: Plots ─────────────────────────────────────────────────────────

print("[ 6/6 ] Saving plots ...")

# Plot 1: Cumulative performance
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
ax = axes[0]
colors = {
    'Market (buy & hold)':   ('black',      '--', 1.2),
    'Fixed 12-mo mom':       ('steelblue',  '--', 1.0),
    'Fixed 1-mo mom':        ('seagreen',   '--', 1.0),
    'Method 0: Formula':     ('grey',       '-',  1.2),
    'Method 1: LR':          ('crimson',    '-',  1.4),
    'Method 2: XGB':         ('darkorange', '-',  1.6),
}
for name, r in strategies.items():
    c, ls, lw = colors[name]
    r = r.dropna().sort_index()
    ax.plot(r.index, (1 + r).cumprod(), color=c, linestyle=ls, linewidth=lw, label=name)
ax.axhline(1, color='black', linewidth=0.4, linestyle=':')
ax.set_ylabel('Cumulative wealth ($1)', fontsize=9)
ax.legend(fontsize=8)
ax.set_title('Cross-sectional momentum: long-only top decile — 2011–2025', fontsize=10)

ax2 = axes[1]
pi_monthly = test[['date', 'pi_filter']].drop_duplicates('date').sort_values('date')
ax2.fill_between(pi_monthly['date'], pi_monthly['pi_filter'],
                 alpha=0.5, color='crimson', label='pi_filter')
ax2.axhline(0.5, color='black', linewidth=0.5, linestyle='--')
ax2.set_ylim(0, 1)
ax2.set_ylabel('pi_filter (panic prob)', fontsize=9)
ax2.set_xlabel('Date')
ax2.legend(fontsize=8)
plt.tight_layout()
fig.savefig('cs_performance.png', dpi=150)
plt.close(fig)

# Plot 2: SHAP feature importance (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Panel 1: Overall mean |SHAP|
ax = axes[0]
feat_imp.plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
pi_display = r'$\pi^{filter}$'
ax.axvline(feat_imp[pi_display], color='crimson', linewidth=1.5,
           linestyle='--', label=f'{pi_display} = {feat_imp[pi_display]:.4f}')
ax.set_title('Overall feature importance\n(mean |SHAP|, XGBoost)', fontsize=10)
ax.set_xlabel('Mean |SHAP value|', fontsize=9)
ax.legend(fontsize=8)

# Panel 2: pi_filter SHAP in calm vs panic
ax = axes[1]
ax.bar(['Calm\n($\\pi < 0.5$)', 'Panic\n($\\pi \\geq 0.5$)'],
       [shap_pi_calm, shap_pi_panic],
       color=['steelblue', 'crimson'], alpha=0.8, width=0.4)
ax.set_ylabel(f'Mean |SHAP| of {pi_display}', fontsize=9)
ax.set_title(f'How much does {pi_display}\nmatter by regime?', fontsize=10)
for i, v in enumerate([shap_pi_calm, shap_pi_panic]):
    ax.text(i, v + 0.00002, f'{v:.5f}', ha='center', fontsize=9)

# Panel 3: All grouped features calm vs panic model
ax = axes[2]
all_feats = sorted(set(imp_calm.index) | set(imp_panic.index),
                   key=lambda f: imp_calm.get(f, 0) + imp_panic.get(f, 0))
x = np.arange(len(all_feats))
w = 0.35
calm_vals  = [imp_calm.get(f, 0)  for f in all_feats]
panic_vals = [imp_panic.get(f, 0) for f in all_feats]
ax.barh(x - w/2, calm_vals,  w, label='Calm model',  color='steelblue', alpha=0.8)
ax.barh(x + w/2, panic_vals, w, label='Panic model', color='crimson',   alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(all_feats, fontsize=8)
ax.set_xlabel('Mean |SHAP|', fontsize=9)
ax.set_title('Feature importance:\ncalm vs panic regime models', fontsize=10)
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig('cs_feature_importance.png', dpi=150)
plt.close(fig)

# Plot 3: Partial dependence of pi_filter
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pi_grid, pdp_scores, color='crimson', linewidth=2)
ax.axvline(0.5, color='black', linewidth=0.8, linestyle='--', label='pi=0.5 threshold')
ax.fill_betweenx([min(pdp_scores), max(pdp_scores)], 0,   0.5, alpha=0.07,
                 color='steelblue', label='Calm regime')
ax.fill_betweenx([min(pdp_scores), max(pdp_scores)], 0.5, 1.0, alpha=0.07,
                 color='crimson',   label='Panic regime')
ax.set_xlabel('pi_filter (panic probability)', fontsize=10)
ax.set_ylabel('Predicted forward return (score)', fontsize=10)
ax.set_title('Partial dependence of pi_filter\n(all other features at median)', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig('cs_pdp_pi_filter.png', dpi=150)
plt.close(fig)

print("Plots saved: cs_performance.png  cs_feature_importance.png  cs_pdp_pi_filter.png")

# ── Save artefacts for fast re-plotting ──────────────────────────────────────
import pickle, joblib

joblib.dump(xgb, 'cs_artefacts_xgb.pkl')
joblib.dump(lr,  'cs_artefacts_lr.pkl')

with open('cs_artefacts_data.pkl', 'wb') as f:
    pickle.dump({
        'test':        test,
        'train':       train,
        'X_test':      X_test,
        'X_train':     X_train,
        'X_te_s':      X_te_s,
        'X_tr_s':      X_tr_s,
        'y_train':     y_train,
        'FEATURES':    FEATURES,
        'strategies_lo': strategies_lo,
        'r_mkt':       r_mkt,
        'shap_values': shap_values,
        'imp_calm':    imp_calm,
        'imp_panic':   imp_panic,
        'imp_calm_raw':  imp_calm_raw,
        'imp_panic_raw': imp_panic_raw,
        'pi_grid':     pi_grid,
        'pdp_scores':  pdp_scores,
        'all_trees':   all_trees,
        'avg_tree':    avg_tree,
    }, f)

print("Artefacts saved: cs_artefacts_*.pkl  (load with cs_plots.py)")
