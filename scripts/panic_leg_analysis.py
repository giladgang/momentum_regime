"""
panic_leg_analysis.py
=====================
Deep dive into panic-regime long vs short leg:
1. Stock overlap between legs across months
2. Where the return spread actually comes from
3. Joint momentum profiles (not marginal z-scores)
4. SHAP direction analysis (signed SHAP, not absolute)
"""

import numpy as np
import pandas as pd
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load data ──
print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)

test = art['test'].copy()
shap_values = art['shap_values']
FEATURES = art['FEATURES']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

test = test.merge(pi_monthly.reset_index()[['date', 'pi_filter']].rename(
    columns={'pi_filter': 'pi_month'}), on='date', how='left')
test['regime'] = np.where(test['pi_month'] >= 0.5, 'Panic', 'Calm')

# ── Assign legs ──
print("Assigning portfolio legs ...")
test['leg'] = 'middle'
for date, grp in test.groupby('date'):
    nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
    if len(nyse) < 10:
        continue
    lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
    test.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    test.loc[grp[grp['score_xgb'] <= lo].index, 'leg'] = 'short'

panic = test[test['regime'] == 'Panic'].copy()
horizons = list(range(1, 13))
mom_cols = [f'mom_{h}' for h in horizons]

# ══════════════════════════════════════════════════════════════════════════════
# 1. STOCK OVERLAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  1. STOCK OVERLAP BETWEEN LONG AND SHORT LEGS (PANIC)")
print("="*60)

overlap_pcts = []
for date, grp in panic.groupby('date'):
    long_permnos = set(grp[grp['leg'] == 'long']['permno'])
    short_permnos = set(grp[grp['leg'] == 'short']['permno'])
    if len(long_permnos) == 0 or len(short_permnos) == 0:
        continue
    overlap = long_permnos & short_permnos
    overlap_pcts.append(len(overlap) / min(len(long_permnos), len(short_permnos)) * 100)

print(f"  Average overlap: {np.mean(overlap_pcts):.1f}%")
print(f"  Max overlap: {np.max(overlap_pcts):.1f}%")
print(f"  (0% means completely different stocks in long vs short)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ACTUAL FORWARD RETURNS BY LEG
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  2. FORWARD RETURNS BY LEG (PANIC)")
print("="*60)

for leg in ['long', 'short']:
    rets = panic[panic['leg'] == leg]['ret_fwd']
    print(f"  {leg}: mean={rets.mean():.4f}, median={rets.median():.4f}, n={len(rets)}")

# Monthly VW returns
monthly_spread = []
for date, grp in panic.groupby('date'):
    longs = grp[grp['leg'] == 'long']
    shorts = grp[grp['leg'] == 'short']
    if len(longs) == 0 or len(shorts) == 0:
        continue
    r_long = (longs['ret_fwd'] * longs['me']).sum() / longs['me'].sum()
    r_short = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
    monthly_spread.append({
        'date': date, 'r_long': r_long, 'r_short': r_short,
        'spread': r_long - r_short
    })

spread_df = pd.DataFrame(monthly_spread)
print(f"\n  Value-weighted monthly returns (panic):")
print(f"    Long:   {spread_df['r_long'].mean():+.4f} ({spread_df['r_long'].mean()*100:+.2f}%/mo)")
print(f"    Short:  {spread_df['r_short'].mean():+.4f} ({spread_df['r_short'].mean()*100:+.2f}%/mo)")
print(f"    Spread: {spread_df['spread'].mean():+.4f} ({spread_df['spread'].mean()*100:+.2f}%/mo)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. JOINT MOMENTUM PROFILES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  3. JOINT MOMENTUM PROFILES (PANIC) - RAW VALUES")
print("="*60)

for leg in ['long', 'short']:
    leg_data = panic[panic['leg'] == leg]
    print(f"\n  {leg.upper()} LEG - Mean raw momentum:")
    for h in horizons:
        col = f'mom_{h}'
        print(f"    mom_{h:>2d}: {leg_data[col].mean():+.4f} ({leg_data[col].mean()*100:+.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Z-SCORE SPREAD (LONG - SHORT) AT EACH HORIZON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  4. Z-SCORE SPREAD (LONG - SHORT) BY HORIZON (PANIC)")
print("="*60)

print(f"  {'Horizon':>8s} {'Long z':>8s} {'Short z':>8s} {'Spread':>8s}")
print("  " + "-"*36)
for h in horizons:
    col = f'mom_{h}'
    monthly_spreads = []
    for date, grp in panic.groupby('date'):
        mean = grp[col].mean()
        std = grp[col].std()
        if std == 0:
            continue
        long_z = ((grp.loc[grp['leg'] == 'long', col] - mean) / std).mean()
        short_z = ((grp.loc[grp['leg'] == 'short', col] - mean) / std).mean()
        monthly_spreads.append({'long_z': long_z, 'short_z': short_z})
    ms = pd.DataFrame(monthly_spreads)
    print(f"  mom_{h:>2d}   {ms['long_z'].mean():>+7.3f}  {ms['short_z'].mean():>+7.3f}  {(ms['long_z'] - ms['short_z']).mean():>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SIGNED SHAP (not absolute) - DIRECTION OF INFLUENCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  5. SIGNED SHAP BY HORIZON AND LEG (PANIC)")
print("="*60)

mom_indices = [FEATURES.index(f'mom_{h}') for h in horizons]
regime_mask = test['regime'] == 'Panic'
leg_arr = test['leg'].values

print(f"  {'Horizon':>8s} {'Long SHAP':>10s} {'Short SHAP':>11s} {'Diff':>8s}")
print("  " + "-"*42)
for i, h in enumerate(horizons):
    idx = mom_indices[i]
    long_mask = regime_mask & (test['leg'] == 'long')
    short_mask = regime_mask & (test['leg'] == 'short')
    long_shap = shap_values[long_mask.values, idx].mean()
    short_shap = shap_values[short_mask.values, idx].mean()
    print(f"  mom_{h:>2d}   {long_shap:>+9.5f}  {short_shap:>+10.5f}  {long_shap - short_shap:>+7.5f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  6. XGB SCORE DISTRIBUTION BY LEG (PANIC)")
print("="*60)

for leg in ['long', 'short', 'middle']:
    scores = panic[panic['leg'] == leg]['score_xgb']
    print(f"  {leg:>6s}: mean={scores.mean():+.5f}, std={scores.std():.5f}, "
          f"min={scores.min():+.5f}, max={scores.max():+.5f}, n={len(scores)}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. CORRELATION: MONTH-1 MOM vs FORWARD RETURN IN PANIC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  7. MONTHLY IC (SPEARMAN) BY HORIZON IN PANIC")
print("="*60)

from scipy.stats import spearmanr

print(f"  {'Horizon':>8s} {'Mean IC':>8s} {'t-stat':>8s}")
print("  " + "-"*28)
for h in horizons:
    col = f'mom_{h}'
    monthly_ics = []
    for date, grp in panic.groupby('date'):
        valid = grp[[col, 'ret_fwd']].dropna()
        if len(valid) < 30:
            continue
        rho, _ = spearmanr(valid[col], valid['ret_fwd'])
        monthly_ics.append(rho)
    ics = np.array(monthly_ics)
    t = ics.mean() / (ics.std() / np.sqrt(len(ics))) if ics.std() > 0 else 0
    print(f"  mom_{h:>2d}   {ics.mean():>+7.4f}  {t:>+7.2f}")

print("\nDone.")
