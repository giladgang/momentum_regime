"""
generate_plots.py
=================
Generates all plots for the thesis using current pipeline results.
Reads from cs_artefacts_data.pkl and panel_with_regimes.parquet.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, warnings, os, sys
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PORTFOLIO_TYPE, TRADING_FEE, TRAIN_END, CS_FEATURES,
                    MOM_FEATURES, XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH,
                    LEARNING_RATE, SUBSAMPLE, COLSAMPLE, HMM_FEATURES,
                    USE_FUNDAMENTALS)

print("=" * 70)
print("  GENERATING ALL PLOTS")
print("=" * 70)

# ── Load data ──
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
train = art['train'].copy()
r_mkt = art['r_mkt']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

REDUCED = MOM_FEATURES + ['pi_filter']
FEATURES = CS_FEATURES
FEE = TRADING_FEE

def long_short_port(df_test, score_col, fee=FEE):
    monthly, prev_lw, prev_sw = [], {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]; shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0)) for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0)) for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def long_only_port(df_data, score_col, fee=FEE):
    monthly, prev_weights = [], {}
    for date, grp in df_data.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0: continue
        total_me = longs['me'].sum()
        new_w = (longs.set_index('permno')['me'] / total_me).to_dict()
        turnover = sum(abs(new_w.get(p, 0) - prev_weights.get(p, 0)) for p in set(new_w) | set(prev_weights)) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_w
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

build_port = long_short_port if PORTFOLIO_TYPE == 'long_short' else long_only_port

# ── Use production scores from artefacts (no retraining) ──
print("\n[ 1 ] Using production XGB scores from artefacts ...")
test['score_m2_red'] = test['score_xgb']

if USE_FUNDAMENTALS:
    test['score_m2_full'] = test['score_xgb']

# Prepare features for M1
X_tr_red = train[REDUCED].values.astype(float)
X_te_red = test[REDUCED].values.astype(float)
y_tr = train['ret_fwd'].values.astype(float)

# M1
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(imp.fit_transform(X_tr_red))
X_te_s = scaler.transform(imp.transform(X_te_red))
train_c = train.copy()
train_c['above_med'] = train_c.groupby('date')['ret_fwd'].transform(lambda x: (x >= x.median()).astype(int))
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_tr_s, train_c['above_med'].values)
test['score_m1'] = lr.predict_proba(X_te_s)[:, 1]

# Fixed mom
test['score_mom12'] = test.groupby('date')['mom_12'].rank(pct=True)

# Compute returns
r_mom12 = build_port(test, 'score_mom12')
r_m1 = build_port(test, 'score_m1')
r_m2_red = build_port(test, 'score_m2_red')
if USE_FUNDAMENTALS:
    r_m2_full = build_port(test, 'score_m2_full')

# ═══════════════════════════════════════════════════════════════════
# PLOT 1: Cumulative wealth paths
# ═══════════════════════════════════════════════════════════════════

print("[ 2 ] Cumulative wealth plot ...")

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
pi_monthly = panel[['date', 'pi_filter']].dropna().drop_duplicates('date').set_index('date')

strategies = [
    ('Market', r_mkt, '-', 'grey', 1.5),
    ('Fixed 12-mo L/S', r_mom12, '--', 'brown', 1.0),
    ('M1: LR', r_m1, '--', 'blue', 1.0),
    ('M2: XGB (mom+pi)', r_m2_red, '-', 'red', 2.0),
]
if USE_FUNDAMENTALS:
    strategies.append(('M2: XGB (full)', r_m2_full, '-', 'darkred', 1.5))

for name, r, ls, c, lw in strategies:
    cum = (1 + r).cumprod()
    sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    ax.plot(cum.index, cum, label=f'{name} (SR={sh:.2f})', linestyle=ls, color=c, linewidth=lw)

# Shade panic
for date in r_m2_red.index:
    pi_val = pi_monthly.reindex([date])
    if len(pi_val) > 0 and pi_val['pi_filter'].iloc[0] >= 0.5:
        ax.axvspan(date, date + pd.offsets.MonthEnd(1), alpha=0.1, color='red')

port_label = 'Long-Short' if PORTFOLIO_TYPE == 'long_short' else 'Long-Only'
ax.set_ylabel('Cumulative Wealth (from $1)')
ax.set_title(f'{port_label} Portfolio Performance ({int(TRADING_FEE*10000)} bps costs)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
ax.axhline(1, color='black', linewidth=0.5, linestyle=':')
plt.tight_layout()
fig.savefig('plots/cs_performance_regime_shaded.png', dpi=150)
plt.close(fig)
print("  Saved: cs_performance_regime_shaded.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 2: PDP for pi_filter
# ═══════════════════════════════════════════════════════════════════

print("[ 3 ] PDP plot ...")

import joblib
xgb = joblib.load('artefacts/cs_artefacts_xgb.pkl')

X_te = test[REDUCED].values.astype(float)
pi_grid = np.linspace(0, 1, 100)
pi_idx = REDUCED.index('pi_filter')

pdp_scores = []
for pi_val in pi_grid:
    X_mod = X_te.copy()
    X_mod[:, pi_idx] = pi_val
    preds = xgb.predict(X_mod)
    pdp_scores.append(preds.mean())

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(pi_grid, pdp_scores, color='darkblue', linewidth=2)
ax.set_xlabel('pi_filter', fontsize=13)
ax.set_ylabel('Mean Predicted Return', fontsize=13)
ax.set_title('Partial Dependence of XGBoost Predicted Return on pi_filter')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
fig.savefig('plots/cs_pdp_pi_filter.png', dpi=150)
plt.close(fig)
print("  Saved: cs_pdp_pi_filter.png")

# ═══════════════════════════════════════════════════════════════════
# PLOT 3: SHAP importance by regime
# ═══════════════════════════════════════════════════════════════════

print("[ 4 ] SHAP importance plot ...")

explainer = shap.TreeExplainer(xgb)
shap_vals = explainer.shap_values(X_te)
abs_shap = np.abs(shap_vals)

test_pi = test.groupby('date')['pi_filter'].first()
obs_pi = test['date'].map(test_pi.to_dict()).fillna(0.5)
calm_mask = obs_pi.values < 0.5

mean_shap_calm = abs_shap[calm_mask].mean(axis=0)
mean_shap_panic = abs_shap[~calm_mask].mean(axis=0)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
x = np.arange(len(REDUCED))
width = 0.35
ax.bar(x - width / 2, mean_shap_calm, width, label='Calm', color='steelblue', alpha=0.8)
ax.bar(x + width / 2, mean_shap_panic, width, label='Panic', color='indianred', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(REDUCED, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean |SHAP|')
ax.set_title('Feature Importance by Regime')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig('plots/cs_feature_importance.png', dpi=150)
plt.close(fig)
print("  Saved: cs_feature_importance.png")

print("\n" + "=" * 70)
print("  ALL PLOTS GENERATED")
print("=" * 70)
