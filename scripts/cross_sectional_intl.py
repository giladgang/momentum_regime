"""
cross_sectional_intl.py
=======================
Regional cross-sectional momentum + regime model for UK or Japan, mirroring
the US production `scripts/cross_sectional_model.py` but:
  - Reads regional stock panel and regional regime panel
  - Uses momentum-only feature set (no fundamentals, by design — see
    INTL_VALIDATION_PLAN.md for the rationale)
  - Uses unconditional decile breakpoints across the regional universe
    each month (no NYSE-equivalent in UK/JP main-listing common equity)
  - Optionally swaps the regional `pi_filter` for the US-trained one
    (`--use-us-pi`) to test the global financial cycle hypothesis (Test A)

Self-contained: does NOT import cross_sectional_model.py to avoid running
its US fit at import time.

Usage
-----
    python scripts/cross_sectional_intl.py --region UK
    python scripts/cross_sectional_intl.py --region JP
    python scripts/cross_sectional_intl.py --region UK --use-us-pi  # Test A
    python scripts/cross_sectional_intl.py --region UK --xgb-seeds 5 --smoke

Outputs
-------
    results/intl_{region}_returns[_uspi].csv   monthly returns per method
    results/intl_{region}_summary[_uspi].csv   Sharpe / ann_ret / vol / MDD per method
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRAIN_END, TRADING_FEE as CFG_TRADING_FEE,
                    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE)
from src.utils import metrics

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--region', choices=['UK', 'JP'], required=True)
parser.add_argument('--use-us-pi', action='store_true',
                    help='Test A: use US-trained pi_filter instead of regional')
parser.add_argument('--xgb-seeds', type=int, default=10,
                    help='Number of XGBoost seeds to ensemble (production = 50)')
parser.add_argument('--smoke', action='store_true',
                    help='Smoke test: 3 XGB seeds only')
args = parser.parse_args()

if args.smoke:
    args.xgb_seeds = 3

REGION = args.region
suffix = '_uspi' if args.use_us_pi else ''

STOCK_PANEL  = f'data/{REGION.lower()}_stock_panel.parquet'
REGIONAL_REG = f'data/{REGION.lower()}_panel_with_regimes.parquet'
US_REG       = 'data/panel_with_regimes.parquet'
OUT_RETURNS  = f'results/intl_{REGION.lower()}_returns{suffix}.csv'
OUT_SUMMARY  = f'results/intl_{REGION.lower()}_summary{suffix}.csv'

print(f"=== Regional cross-sectional model: {REGION} "
      f"({'US pi_filter (Test A)' if args.use_us_pi else 'regional pi_filter'}) ===")
print(f"Stock panel:  {STOCK_PANEL}")
print(f"Regime panel: {US_REG if args.use_us_pi else REGIONAL_REG}")
print(f"XGB seeds:    {args.xgb_seeds}")

# ── Section 1: Load and merge ────────────────────────────────────────────────

stocks = pd.read_parquet(STOCK_PANEL)
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['secid', 'date']).reset_index(drop=True)

# Prefer Shumway-corrected returns if `apply_shumway_intl.py` has been run
# (it writes a new `ret_adj` column without overwriting `ret`). Recompute
# `ret_fwd` from the corrected returns so the cross-sectional model trains
# and predicts on the corrected series. If `ret_adj` is absent we fall back
# to the original `ret` (legacy / un-treated panel).
if 'ret_adj' in stocks.columns:
    n_changed = int((stocks['ret_adj'] != stocks['ret']).sum()
                    - (stocks['ret_adj'].isna() & stocks['ret'].isna()).sum())
    print(f"\n[ 1/5 ] ret_adj column present ({n_changed:,} rows differ "
          f"from ret); recomputing ret_fwd from ret_adj.")
    # Recompute ret_fwd from corrected returns. We do NOT overwrite the
    # original `ret` column — apply_shumway_intl.py preserves it for
    # diagnostic comparison, and nothing downstream reads `ret` directly
    # (the model trains/predicts on FEATURES + ret_fwd only).
    stocks['ret_fwd'] = stocks.groupby('secid')['ret_adj'].shift(-1)
print(f"\n[ 1/5 ] Loaded stock panel: {len(stocks):,} rows  |  "
      f"{stocks['secid'].nunique():,} securities  |  "
      f"{stocks['date'].min().date()} -> {stocks['date'].max().date()}")

regime_path = US_REG if args.use_us_pi else REGIONAL_REG
regimes = pd.read_parquet(regime_path)
regimes['date'] = pd.to_datetime(regimes['date'])
regimes = regimes[['date', 'pi_filter']].drop_duplicates(subset='date').sort_values('date')
print(f"  Regime panel: {len(regimes):,} months  |  "
      f"{regimes['date'].min().date()} -> {regimes['date'].max().date()}")

stocks = stocks.merge(regimes, on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

# ── Section 2: Features and target ───────────────────────────────────────────

MOM_FEATURES = [f'mom_{k}' for k in range(1, 13)]
FEATURES     = MOM_FEATURES + ['pi_filter']

# Drop rows missing target or core features
df = stocks.dropna(subset=['ret_fwd'] + FEATURES).copy()
df = df.reset_index(drop=True)
print(f"  Usable rows after dropna: {len(df):,}  |  "
      f"{df['date'].min().date()} -> {df['date'].max().date()}")

# Train / test split
train_end = pd.Timestamp(TRAIN_END)
train = df[df['date'] < train_end].copy()
test  = df[df['date'] >= train_end].copy()
print(f"  Train: {len(train):,} rows ({train['date'].min().date()} -> {train['date'].max().date()})")
print(f"  Test:  {len(test):,} rows ({test['date'].min().date()}  -> {test['date'].max().date()})")

X_train = train[FEATURES].values.astype(float)
y_train = train['ret_fwd'].values.astype(float)
X_test  = test [FEATURES].values.astype(float)

train['above_med'] = train.groupby('date')['ret_fwd'].transform(
    lambda x: (x > x.median()).astype(int)
)

# ── Section 3: Portfolio constructor (long/short, value-weighted, unconditional deciles) ──

TRADING_FEE = CFG_TRADING_FEE


def long_short_unconditional(df_test, score_col, fee=TRADING_FEE):
    """
    Each month: form unconditional top-decile (long) and bottom-decile (short)
    portfolios across all eligible stocks. Value-weighted by `me`. Transaction
    cost applied to both legs based on month-over-month one-way turnover.
    Returns pd.Series of monthly L-S returns indexed by date.
    """
    monthly  = []
    prev_lw  = {}
    prev_sw  = {}
    for date, grp in df_test.groupby('date'):
        scores = grp[score_col].dropna()
        if len(scores) < 50:
            continue
        lo = scores.quantile(0.10)
        hi = scores.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        lme = longs ['me'].sum()
        sme = shorts['me'].sum()
        if lme == 0 or sme == 0:
            continue
        new_lw = (longs.set_index ('secid')['me'] / lme).to_dict()
        new_sw = (shorts.set_index('secid')['me'] / sme).to_dict()
        r_long  = (longs ['ret_fwd'] * longs ['me']).sum() / lme
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']


# ── Section 4: Methods ───────────────────────────────────────────────────────

print(f"\n[ 2/5 ] Computing benchmarks ...")

# Benchmarks: fixed momentum lookbacks
test['score_mom12'] = test['mom_12']
test['score_mom1']  = test['mom_1']
r_mom12 = long_short_unconditional(test, 'score_mom12')
r_mom1  = long_short_unconditional(test, 'score_mom1')

# Market: VW market return from regime panel (regional or US, depending on flag)
mkt_panel = pd.read_parquet(REGIONAL_REG)  # always use regional market return as benchmark
mkt_panel['date'] = pd.to_datetime(mkt_panel['date'])
mkt = mkt_panel[mkt_panel['date'] >= train_end].set_index('date')['mkt_ret']

print(f"\n[ 3/5 ] Method 0: Deterministic formula ...")
# lb = round(12 - 11*pi); pi=0 -> 12-mo mom, pi=1 -> 1-mo mom
pi_by_month = test.groupby('date')['pi_filter'].first()
lb_by_month = np.clip(np.round(12 - 11 * pi_by_month).astype(int), 1, 12)
test['score_formula'] = np.nan
for date, lb in lb_by_month.items():
    mask = test['date'] == date
    test.loc[mask, 'score_formula'] = test.loc[mask, f'mom_{lb}']
r_formula = long_short_unconditional(test, 'score_formula')

print(f"\n[ 4/5 ] Method 1: Logistic Regression ...")
imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()
X_tr_s = scaler.fit_transform(imputer.fit_transform(X_train))
X_te_s = scaler.transform(    imputer.transform(    X_test))

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_tr_s, train['above_med'].values)
test = test.copy()
test['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
r_lr = long_short_unconditional(test, 'score_lr')

# Ridge baseline (linear, continuous-target)
ridge = Ridge(alpha=1.0)
ridge.fit(X_tr_s, y_train)
test['score_ridge'] = ridge.predict(X_te_s)
r_ridge = long_short_unconditional(test, 'score_ridge')
ridge_pi_coef = dict(zip(FEATURES, ridge.coef_)).get('pi_filter', float('nan'))
print(f"  Ridge coef on pi_filter: {ridge_pi_coef:+.4f}")

print(f"\n[ 5/5 ] Method 2: XGBoost ensemble ({args.xgb_seeds} seeds) ...")
xgb_predictions = np.zeros(len(X_test))
for xgb_seed in range(1, args.xgb_seeds + 1):
    xgb_i = XGBRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE, tree_method='hist',
        random_state=xgb_seed, verbosity=0, n_jobs=2,
    )
    xgb_i.fit(X_train, y_train)
    xgb_predictions += xgb_i.predict(X_test)
xgb_predictions /= args.xgb_seeds
test['score_xgb'] = xgb_predictions
r_xgb = long_short_unconditional(test, 'score_xgb')

# ── Section 5: Report and save ───────────────────────────────────────────────

strats = {
    'market':         mkt,
    'fixed_mom_12':   r_mom12,
    'fixed_mom_1':    r_mom1,
    'method0_formula': r_formula,
    'method1_lr':      r_lr,
    'method1b_ridge':  r_ridge,
    'method2_xgb':     r_xgb,
}

summary_rows = []
for name, r in strats.items():
    if len(r) == 0:
        summary_rows.append({'strategy': name, 'n_months': 0,
                              'ann_ret': np.nan, 'ann_vol': np.nan,
                              'sharpe': np.nan, 'max_dd': np.nan})
        continue
    ann_ret, ann_vol, sharpe, mdd = metrics(r)
    summary_rows.append({
        'strategy': name, 'n_months': len(r),
        'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'max_dd': mdd,
    })

summary = pd.DataFrame(summary_rows)
print("\n" + "=" * 60)
print(f"  {REGION} cross-sectional results "
      f"({'US pi_filter' if args.use_us_pi else 'regional pi_filter'})")
print("=" * 60)
print(summary.round(4).to_string(index=False))

# Save returns (one column per strategy, indexed by date)
returns_df = pd.DataFrame(strats).sort_index()
os.makedirs('results', exist_ok=True)
returns_df.to_csv(OUT_RETURNS, index_label='date')
summary.to_csv(OUT_SUMMARY, index=False)
print(f"\n  Saved: {OUT_RETURNS}")
print(f"  Saved: {OUT_SUMMARY}")
