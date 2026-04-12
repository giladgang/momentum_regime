"""
economic_mechanism.py
=====================
Analyzes WHY momentum behaves differently in calm vs panic regimes.

Analyses:
1. Portfolio characteristics by regime (size, beta, B/M, profitability, leverage)
2. Forward return decomposition (stock recovery vs market recovery)
3. Holding period analysis (1, 3, 6 month returns of panic selections)
4. Sector rotation (which sectors in calm vs panic)

Usage:
    python scripts/economic_mechanism.py
"""

import numpy as np
import pandas as pd
import pickle, warnings, os, sys
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRADING_FEE, TRAIN_END, MOM_FEATURES, N_ESTIMATORS,
                    MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE, XGB_SEEDS)

print("=" * 70)
print("  ECONOMIC MECHANISM ANALYSIS")
print("  Why does momentum behave differently in calm vs panic?")
print("=" * 70)

# ── Load data ──
with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

REDUCED = MOM_FEATURES + ['pi_filter']
FEE = TRADING_FEE

# Get regime classification for each month
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

# Train XGB ensemble
from xgboost import XGBRegressor

X_tr = train[REDUCED].values.astype(float)
X_te = test[REDUCED].values.astype(float)
y_tr = train['ret_fwd'].values.astype(float)

print("\n  Training XGB ensemble ...")
preds = np.zeros(len(X_te))
for xs in XGB_SEEDS[:5]:
    xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                       learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                       colsample_bytree=COLSAMPLE, tree_method='hist',
                       random_state=xs, verbosity=0)
    xgb.fit(X_tr, y_tr)
    preds += xgb.predict(X_te)
preds /= 5
test['score'] = preds

# Classify each test month
test_pi = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test_pi['pi_month'] = test_pi['pi_month'].fillna(0.5)
test_pi['regime'] = np.where(test_pi['pi_month'] >= 0.5, 'Panic', 'Calm')

# Identify long and short legs each month
test_pi['leg'] = 'middle'
for date, grp in test_pi.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    long_mask = test_pi.loc[grp.index, 'score'] >= hi
    short_mask = test_pi.loc[grp.index, 'score'] <= lo
    test_pi.loc[grp.index[long_mask], 'leg'] = 'long'
    test_pi.loc[grp.index[short_mask], 'leg'] = 'short'

# Filter to only long/short legs
portfolio = test_pi[test_pi['leg'].isin(['long', 'short'])].copy()

print(f"  Portfolio observations: {len(portfolio):,}")
print(f"  Long leg: {(portfolio['leg']=='long').sum():,}")
print(f"  Short leg: {(portfolio['leg']=='short').sum():,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PORTFOLIO CHARACTERISTICS BY REGIME
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  1. PORTFOLIO CHARACTERISTICS BY REGIME")
print("=" * 70)

# Characteristics to analyze
char_cols = {
    'mom_1': '1-month momentum',
    'mom_12': '12-month momentum',
    'me': 'Market equity ($M)',
    'bm': 'Book-to-market',
    'roe': 'Return on equity',
    'leverage': 'Leverage',
    'gross_profit_a': 'Gross profitability',
    'ret_fwd': 'Forward return',
}

available_chars = [c for c in char_cols if c in portfolio.columns]

for regime in ['Calm', 'Panic']:
    print(f"\n  --- {regime} months ---")
    print(f"  {'Characteristic':<25s} {'Long leg':>12s} {'Short leg':>12s} {'Diff':>10s}")
    print(f"  {'-'*62}")

    for col in available_chars:
        long_vals = portfolio[(portfolio['regime'] == regime) & (portfolio['leg'] == 'long')][col]
        short_vals = portfolio[(portfolio['regime'] == regime) & (portfolio['leg'] == 'short')][col]

        if len(long_vals) > 0 and len(short_vals) > 0:
            long_mean = long_vals.mean()
            short_mean = short_vals.mean()
            diff = long_mean - short_mean
            label = char_cols[col]

            if col == 'me':
                print(f"  {label:<25s} {long_mean:>12,.0f} {short_mean:>12,.0f} {diff:>+10,.0f}")
            elif col in ['ret_fwd', 'mom_1', 'mom_12', 'roe', 'leverage', 'gross_profit_a']:
                print(f"  {label:<25s} {long_mean:>12.3f} {short_mean:>12.3f} {diff:>+10.3f}")
            else:
                print(f"  {label:<25s} {long_mean:>12.2f} {short_mean:>12.2f} {diff:>+10.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FORWARD RETURN DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  2. FORWARD RETURN DECOMPOSITION")
print("  How much of the profit comes from stock-specific vs market recovery?")
print("=" * 70)

r_mkt = art['r_mkt']

for regime in ['Calm', 'Panic']:
    regime_months = portfolio[portfolio['regime'] == regime]['date'].unique()

    long_rets = []
    short_rets = []
    mkt_rets = []

    for date in regime_months:
        long_stocks = portfolio[(portfolio['date'] == date) & (portfolio['leg'] == 'long')]
        short_stocks = portfolio[(portfolio['date'] == date) & (portfolio['leg'] == 'short')]

        if len(long_stocks) > 0:
            r_long = (long_stocks['ret_fwd'] * long_stocks['me']).sum() / long_stocks['me'].sum()
            long_rets.append(r_long)

        if len(short_stocks) > 0:
            r_short = (short_stocks['ret_fwd'] * short_stocks['me']).sum() / short_stocks['me'].sum()
            short_rets.append(r_short)

        if date in r_mkt.index:
            mkt_rets.append(r_mkt[date])

    if long_rets and short_rets and mkt_rets:
        avg_long = np.mean(long_rets)
        avg_short = np.mean(short_rets)
        avg_mkt = np.mean(mkt_rets)
        avg_ls = avg_long - avg_short

        # Decompose: L/S return = (long - mkt) - (short - mkt) = long_alpha - short_alpha
        long_alpha = avg_long - avg_mkt
        short_alpha = avg_short - avg_mkt

        print(f"\n  --- {regime} ({len(regime_months)} months) ---")
        print(f"    Market return:       {avg_mkt:>+7.2%}")
        print(f"    Long leg return:     {avg_long:>+7.2%}  (alpha: {long_alpha:>+6.2%})")
        print(f"    Short leg return:    {avg_short:>+7.2%}  (alpha: {short_alpha:>+6.2%})")
        print(f"    L/S return:          {avg_ls:>+7.2%}")
        print(f"    From stock selection: {long_alpha - short_alpha:>+7.2%}  ({abs(long_alpha-short_alpha)/abs(avg_ls)*100 if avg_ls != 0 else 0:.0f}% of L/S)")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HOLDING PERIOD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  3. HOLDING PERIOD ANALYSIS")
print("  If you held the panic portfolio longer, would reversal bets still work?")
print("=" * 70)

# Compute forward returns at different horizons
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])

# Get forward returns at 1, 3, 6 month horizons
for horizon, label in [(1, '1-month'), (3, '3-month'), (6, '6-month')]:
    col_name = f'ret_fwd_{horizon}'
    if col_name not in test_pi.columns:
        # Compute cumulative forward return over horizon months
        stocks_temp = stocks_raw[['permno', 'date', 'ret_adj']].copy()
        stocks_temp = stocks_temp.sort_values(['permno', 'date'])

        cum_fwd = stocks_temp.groupby('permno')['ret_adj'].transform(
            lambda x: x.shift(-1).rolling(horizon, min_periods=horizon).apply(
                lambda y: (1 + y).prod() - 1, raw=True)
        )
        stocks_temp[col_name] = cum_fwd

        test_pi = test_pi.merge(
            stocks_temp[['permno', 'date', col_name]],
            on=['permno', 'date'], how='left', suffixes=('', '_new')
        )
        if f'{col_name}_new' in test_pi.columns:
            test_pi[col_name] = test_pi[f'{col_name}_new']
            test_pi.drop(columns=[f'{col_name}_new'], inplace=True)

# Report results for panic months
panic_portfolio = test_pi[(test_pi['regime'] == 'Panic') & (test_pi['leg'].isin(['long', 'short']))]

print(f"\n  Panic months: forward returns by holding period")
print(f"  {'Horizon':<12s} {'Long leg':>10s} {'Short leg':>10s} {'L/S':>10s}")
print(f"  {'-'*45}")

for horizon, label in [(1, '1-month'), (3, '3-month'), (6, '6-month')]:
    col_name = f'ret_fwd_{horizon}'
    if col_name in panic_portfolio.columns:
        long_ret = panic_portfolio[panic_portfolio['leg'] == 'long'][col_name].mean()
        short_ret = panic_portfolio[panic_portfolio['leg'] == 'short'][col_name].mean()

        if not np.isnan(long_ret) and not np.isnan(short_ret):
            print(f"  {label:<12s} {long_ret:>+9.2%} {short_ret:>+9.2%} {long_ret-short_ret:>+9.2%}")
        else:
            print(f"  {label:<12s}       N/A       N/A       N/A")
    else:
        print(f"  {label:<12s}  (not computed)")

# Same for calm
calm_portfolio = test_pi[(test_pi['regime'] == 'Calm') & (test_pi['leg'].isin(['long', 'short']))]

print(f"\n  Calm months: forward returns by holding period")
print(f"  {'Horizon':<12s} {'Long leg':>10s} {'Short leg':>10s} {'L/S':>10s}")
print(f"  {'-'*45}")

for horizon, label in [(1, '1-month'), (3, '3-month'), (6, '6-month')]:
    col_name = f'ret_fwd_{horizon}'
    if col_name in calm_portfolio.columns:
        long_ret = calm_portfolio[calm_portfolio['leg'] == 'long'][col_name].mean()
        short_ret = calm_portfolio[calm_portfolio['leg'] == 'short'][col_name].mean()

        if not np.isnan(long_ret) and not np.isnan(short_ret):
            print(f"  {label:<12s} {long_ret:>+9.2%} {short_ret:>+9.2%} {long_ret-short_ret:>+9.2%}")
        else:
            print(f"  {label:<12s}       N/A       N/A       N/A")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SECTOR ROTATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  4. SECTOR ROTATION")
print("  Does the strategy rotate into different sectors by regime?")
print("=" * 70)

# Use SIC codes to approximate sectors if available
if 'siccd' in portfolio.columns or 'sic' in portfolio.columns:
    sic_col = 'siccd' if 'siccd' in portfolio.columns else 'sic'

    def sic_to_sector(sic):
        if pd.isna(sic) or sic == 0:
            return 'Unknown'
        sic = int(sic)
        if sic < 1000: return 'Agriculture'
        elif sic < 1500: return 'Mining'
        elif sic < 1800: return 'Construction'
        elif sic < 4000: return 'Manufacturing'
        elif sic < 5000: return 'Transport/Utilities'
        elif sic < 5200: return 'Wholesale'
        elif sic < 6000: return 'Retail'
        elif sic < 6800: return 'Finance'
        elif sic < 9000: return 'Services'
        else: return 'Other'

    portfolio['sector'] = portfolio[sic_col].apply(sic_to_sector)

    for regime in ['Calm', 'Panic']:
        print(f"\n  --- {regime}: Top sectors by leg ---")
        for leg in ['long', 'short']:
            sub = portfolio[(portfolio['regime'] == regime) & (portfolio['leg'] == leg)]
            if len(sub) > 0:
                sector_pct = sub['sector'].value_counts(normalize=True).head(5)
                print(f"    {leg.upper()} leg:")
                for sector, pct in sector_pct.items():
                    print(f"      {sector:<25s} {pct:>6.1%}")
else:
    # Use exchange as a rough proxy
    print("\n  SIC codes not available. Using exchange distribution as proxy.")
    for regime in ['Calm', 'Panic']:
        print(f"\n  --- {regime} ---")
        for leg in ['long', 'short']:
            sub = portfolio[(portfolio['regime'] == regime) & (portfolio['leg'] == leg)]
            if len(sub) > 0:
                exch_pct = sub['exchcd'].value_counts(normalize=True)
                exch_map = {1: 'NYSE', 2: 'AMEX', 3: 'NASDAQ'}
                print(f"    {leg.upper()} leg:")
                for exch, pct in exch_pct.items():
                    print(f"      {exch_map.get(exch, f'Exch {exch}'):<15s} {pct:>6.1%}")

    # Also show average size (proxy for sector tilt)
    print(f"\n  Average market cap by regime and leg:")
    print(f"  {'':15s} {'Calm Long':>12s} {'Calm Short':>12s} {'Panic Long':>12s} {'Panic Short':>12s}")
    print(f"  {'-'*55}")
    for stat_name, func in [('Mean ($M)', 'mean'), ('Median ($M)', 'median')]:
        vals = []
        for regime in ['Calm', 'Panic']:
            for leg in ['long', 'short']:
                sub = portfolio[(portfolio['regime'] == regime) & (portfolio['leg'] == leg)]['me']
                if len(sub) > 0:
                    vals.append(getattr(sub, func)())
                else:
                    vals.append(0)
        print(f"  {stat_name:<15s} {vals[0]:>12,.0f} {vals[1]:>12,.0f} {vals[2]:>12,.0f} {vals[3]:>12,.0f}")


print("\n" + "=" * 70)
print("  ECONOMIC MECHANISM ANALYSIS COMPLETE")
print("=" * 70)
