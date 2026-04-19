"""
Risk-aversion analysis: train XGBoost on mean-variance utility targets
    y_s = r_{t+1}^s  -  (gamma/2) * sigma_s^2
for a range of gamma values, then compare portfolio performance and
SHAP feature importances across risk-aversion levels.

Output:
  - risk_aversion_performance.png   (Sharpe, Return, Vol vs gamma)
  - risk_aversion_shap.png          (feature importance vs gamma)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import shap
import pickle, warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load artefacts (same sample as main results)
# ══════════════════════════════════════════════════════════════════════════════

print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

train = artefacts['train'].copy()
test  = artefacts['test'].copy()
X_train = artefacts['X_train']
X_test  = artefacts['X_test']
y_train = artefacts['y_train']
FEATURES = artefacts['FEATURES']

MOM_LBS = list(range(1, 13))
MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]

# Compute trailing realized variance on the artefact sample
# (needed for risk-adjusted targets, but NaN rows are kept -- we fill with median)
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw.sort_values(['permno', 'date'])
stocks_raw['trail_var'] = (
    stocks_raw.groupby('permno')['ret_adj']
    .transform(lambda x: x.shift(1).rolling(12, min_periods=6).var())
)
trail_var_map = stocks_raw.set_index(['permno', 'date'])['trail_var']

train['trail_var'] = train.set_index(['permno', 'date']).index.map(
    lambda idx: trail_var_map.get(idx, np.nan))
test['trail_var'] = test.set_index(['permno', 'date']).index.map(
    lambda idx: trail_var_map.get(idx, np.nan))

# Fill missing trail_var with median (don't drop rows -- keep same sample)
train_var_median = train['trail_var'].median()
train['trail_var'] = train['trail_var'].fillna(train_var_median)
test['trail_var'] = test['trail_var'].fillna(train_var_median)

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")
print(f"  Trail var coverage: train={train['trail_var'].notna().mean():.1%}  test={test['trail_var'].notna().mean():.1%}")

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Portfolio helper
# ══════════════════════════════════════════════════════════════════════════════

TRADING_FEE = 0.001

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
        lw = (longs.set_index('permno')['me'] / longs['me'].sum()).to_dict()
        sw = (shorts.set_index('permno')['me'] / shorts['me'].sum()).to_dict()
        tl = sum(abs(lw.get(p,0)-prev_lw.get(p,0)) for p in set(lw)|set(prev_lw)) / 2
        ts = sum(abs(sw.get(p,0)-prev_sw.get(p,0)) for p in set(sw)|set(prev_sw)) / 2
        r_l = (longs['ret_fwd'] * longs['me']).sum() / longs['me'].sum()
        r_s = (shorts['ret_fwd'] * shorts['me']).sum() / shorts['me'].sum()
        monthly.append({'date': date, 'ret': r_l - r_s - fee*(tl+ts)})
        prev_lw, prev_sw = lw, sw
    return pd.DataFrame(monthly).set_index('date')['ret']

def metrics(r):
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    return ann_ret, ann_vol, sharpe, mdd

# ══════════════════════════════════════════════════════════════════════════════
# 3. Sweep over gamma values
# ══════════════════════════════════════════════════════════════════════════════

GAMMAS = list(range(0, 11))

results = []
shap_by_gamma = {}

import time

for i_gamma, gamma in enumerate(GAMMAS):
    if i_gamma > 0:
        time.sleep(10)  # cool down between runs
    print(f"\n{'='*60}")
    print(f"  gamma = {gamma}")
    print(f"{'='*60}")

    score_col = f'score_g{gamma}'

    if gamma == 0:
        # Use production artefact scores for baseline (same as main results)
        test[score_col] = test['score_xgb']
        import joblib
        model = joblib.load('artefacts/cs_artefacts_xgb.pkl')
        print("  Using production artefact scores for baseline")
    else:
        # Compute risk-adjusted target and retrain
        y_train_g = train['ret_fwd'].values - (gamma / 2) * train['trail_var'].values

        XGB_SEEDS = list(range(1, 51))  # 50 seeds (same as production baseline)
        preds = np.zeros(len(X_test))
        last_model = None
        for xs in XGB_SEEDS:
            model = XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method='hist', random_state=xs, verbosity=0
            )
            model.fit(X_train, y_train_g)
            preds += model.predict(X_test)
            last_model = model
        preds /= len(XGB_SEEDS)
        test[score_col] = preds
        model = last_model

    # Portfolio performance
    r = long_short_port(test, score_col)
    ann_ret, ann_vol, sharpe, mdd = metrics(r)
    results.append({
        'gamma': gamma, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'mdd': mdd, 'final_wealth': (1 + r).prod()
    })
    print(f"  Ann Ret: {ann_ret:.1%}  |  Vol: {ann_vol:.1%}  |  "
          f"Sharpe: {sharpe:.2f}  |  MDD: {mdd:.1%}")

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES)

    # Aggregate momentum into single entry
    mom_total = mean_abs_shap[MOM_FEATURES].sum()
    other = mean_abs_shap.drop(MOM_FEATURES)
    agg = pd.concat([pd.Series({'Momentum (agg)': mom_total}), other])
    agg = agg.sort_values(ascending=False)
    shap_by_gamma[gamma] = agg

    print(f"  Top 5 SHAP: {agg.head(5).to_dict()}")

res_df = pd.DataFrame(results)
res_df.to_csv('results/risk_aversion_results.csv', index=False, float_format='%.4f')
print("\n\nSummary:")
print(res_df.to_string(index=False))
print("\nSaved: risk_aversion_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Combined plot
# ══════════════════════════════════════════════════════════════════════════════

# Build normalised SHAP DataFrame (proportions)
all_features = shap_by_gamma[0].index.tolist()
shap_df = pd.DataFrame({g: shap_by_gamma[g].reindex(all_features) for g in GAMMAS})
shap_norm = shap_df.div(shap_df.sum(axis=0), axis=1) * 100  # percentages

# Top features by mean normalised importance
mean_imp = shap_norm.mean(axis=1).sort_values(ascending=False)
top_feats = mean_imp.head(8).index.tolist()

# Rename for display
DISPLAY_NAMES = {
    'Momentum (agg)': 'Momentum (all horizons)',
    'pi_filter': r'$\pi_t^{\mathrm{filter}}$',
    'log_me': 'Log market equity',
    'bm': 'Book-to-market',
    'roe': 'Return on equity',
    'asset_growth': 'Asset growth',
    'earnings_growth': 'Earnings growth',
    'gross_profit_a': 'Gross profitability',
    'leverage': 'Leverage',
}

FEAT_COLORS = {
    'Momentum (agg)': '#1B4F8A',
    'pi_filter': '#E65100',
    'log_me': '#2E6B4F',
    'bm': '#5B3A8C',
    'roe': '#9B2226',
    'asset_growth': '#0277BD',
    'earnings_growth': '#558B2F',
    'gross_profit_a': '#6D4C41',
    'leverage': '#455A64',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

# ── Top row: Performance panels ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(res_df['gamma'], res_df['sharpe'], 's-', color='#1B4F8A',
         linewidth=2.2, markersize=7, markerfacecolor='white', markeredgewidth=2)
for g, s in zip(res_df['gamma'], res_df['sharpe']):
    ax1.annotate(f'{s:.2f}', (g, s), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=8.5, color='#1B4F8A')
ax1.set_xlabel(r'Risk aversion $\gamma$')
ax1.set_ylabel('Sharpe Ratio')
ax1.set_title('Sharpe Ratio', fontweight='bold')
ax1.grid(True, alpha=0.2)
ax1.set_xlim(-0.5, 10.5)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(res_df['gamma'], res_df['ann_ret'] * 100, 's-', color='#2E6B4F',
         linewidth=2.2, markersize=7, markerfacecolor='white', markeredgewidth=2)
for g, r in zip(res_df['gamma'], res_df['ann_ret']):
    ax2.annotate(f'{r:.1%}', (g, r*100), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=8.5, color='#2E6B4F')
ax2.set_xlabel(r'Risk aversion $\gamma$')
ax2.set_ylabel('Annualized Return (%)')
ax2.set_title('Annualized Return', fontweight='bold')
ax2.grid(True, alpha=0.2)
ax2.set_xlim(-0.5, 10.5)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(res_df['gamma'], res_df['ann_vol'] * 100, 's-', color='#9B2226',
         linewidth=2.2, markersize=7, markerfacecolor='white', markeredgewidth=2)
for g, v in zip(res_df['gamma'], res_df['ann_vol']):
    ax3.annotate(f'{v:.1%}', (g, v*100), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=8.5, color='#9B2226')
ax3.set_xlabel(r'Risk aversion $\gamma$')
ax3.set_ylabel('Annualized Volatility (%)')
ax3.set_title('Annualized Volatility', fontweight='bold')
ax3.grid(True, alpha=0.2)
ax3.set_xlim(-0.5, 10.5)

# ── Bottom row: Stacked area (normalised SHAP) ──
ax4 = fig.add_subplot(gs[1, :])

bottoms = np.zeros(len(GAMMAS))
for feat in reversed(top_feats):
    vals = np.array([shap_norm.loc[feat, g] for g in GAMMAS])
    label = DISPLAY_NAMES.get(feat, feat)
    color = FEAT_COLORS.get(feat, '#888888')
    ax4.fill_between(GAMMAS, bottoms, bottoms + vals, alpha=0.85,
                     label=label, color=color, linewidth=0)
    ax4.plot(GAMMAS, bottoms + vals, color=color, linewidth=0.8, alpha=0.6)
    bottoms += vals

ax4.set_xlabel(r'Risk aversion $\gamma$')
ax4.set_ylabel('Share of total SHAP importance (%)')
ax4.set_title('Relative Feature Importance vs Risk Aversion', fontweight='bold')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 100)

handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles[::-1], labels[::-1], loc='center left',
           bbox_to_anchor=(1.01, 0.5), frameon=True, framealpha=0.9)

fig.suptitle(r'XGBoost with Mean-Variance Utility Target:  $y_s = r_{t+1}^s - \frac{\gamma}{2}\,\sigma_s^2$',
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('plots/risk_aversion_analysis.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("\nSaved risk_aversion_analysis.png")
