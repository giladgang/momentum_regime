"""
momentum_pi_analysis.py
=======================
How does the momentum signal interact with pi_filter?

Three views:
  1. Regime-conditional IC: which lookbacks predict returns in calm vs panic?
  2. Time-varying best lookback vs pi_filter: does the optimal horizon shift?
  3. SHAP dependence: how does XGBoost use momentum given pi_filter?
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pickle, joblib

# ── Load artefacts from cross_sectional_model.py ────────────────────────────

with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test      = artefacts['test']
X_test    = artefacts['X_test']
FEATURES  = artefacts['FEATURES']
shap_vals = artefacts['shap_values']

xgb = joblib.load('cs_artefacts_xgb.pkl')

MOM_LBS      = list(range(1, 13))
MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]

# ── 1. Regime-conditional Information Coefficient (IC) ───────────────────────
# For each month, compute Spearman rank correlation between each mom signal
# and forward returns. Then average separately for calm vs panic months.

print("Computing regime-conditional ICs ...")

ic_records = []
for date, grp in test.groupby('date'):
    pi = grp['pi_filter'].iloc[0]
    regime = 'Panic' if pi >= 0.5 else 'Calm'
    for lb in MOM_LBS:
        col = f'mom_{lb}'
        valid = grp[[col, 'ret_fwd']].dropna()
        if len(valid) < 30:
            continue
        rho, _ = spearmanr(valid[col], valid['ret_fwd'])
        ic_records.append({'date': date, 'lb': lb, 'ic': rho,
                           'regime': regime, 'pi_filter': pi})

ic_df = pd.DataFrame(ic_records)

# Average IC by lookback and regime
avg_ic = ic_df.groupby(['lb', 'regime'])['ic'].mean().unstack('regime')
avg_ic = avg_ic[['Calm', 'Panic']]  # consistent order

print("\nAverage IC (Spearman) by lookback and regime:")
print(avg_ic.round(4).to_string())

# ── 2. Time-varying best lookback ────────────────────────────────────────────
# Each month: which lookback had the highest cross-sectional IC?

print("\nComputing time-varying best lookback ...")

monthly_best = (ic_df.loc[ic_df.groupby('date')['ic'].idxmax()]
                [['date', 'lb', 'pi_filter', 'regime']].sort_values('date'))

# Also compute a "momentum IC" for each month (average across all lookbacks)
monthly_avg_ic = ic_df.groupby('date').agg(
    avg_ic=('ic', 'mean'),
    pi_filter=('pi_filter', 'first'),
    regime=('regime', 'first')
).reset_index()

# ── 3. SHAP dependence: momentum SHAP vs pi_filter ──────────────────────────
# For each stock-month, sum SHAP from all 12 momentum features,
# then plot against pi_filter to see how XGBoost shifts its reliance.

print("Computing SHAP dependence ...")

mom_indices = [FEATURES.index(f) for f in MOM_FEATURES]
shap_mom_total = shap_vals[:, mom_indices].sum(axis=1)  # total momentum SHAP per row
pi_per_row = test['pi_filter'].values

# ── Plots ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Average IC by lookback and regime (bar chart)
ax = axes[0, 0]
x = np.arange(len(MOM_LBS))
w = 0.35
ax.bar(x - w/2, avg_ic['Calm'],  w, label='Calm', color='steelblue', alpha=0.8)
ax.bar(x + w/2, avg_ic['Panic'], w, label='Panic', color='crimson', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(lb) for lb in MOM_LBS])
ax.set_xlabel('Momentum lookback (months)', fontsize=9)
ax.set_ylabel('Average Spearman IC', fontsize=9)
ax.axhline(0, color='black', linewidth=0.5)
ax.legend(fontsize=9)
ax.set_title('Which momentum lookbacks predict returns?\n'
             '(cross-sectional IC, calm vs panic)', fontsize=10)

# Panel B: Time-varying best lookback vs pi_filter
ax = axes[0, 1]
dates = monthly_best['date'].values
lbs = monthly_best['lb'].values
pi = monthly_best['pi_filter'].values
colors_scatter = np.where(monthly_best['regime'] == 'Panic', 'crimson', 'steelblue')
ax.scatter(dates, lbs, c=colors_scatter, s=20, alpha=0.7, zorder=2)

# Overlay pi_filter on secondary axis
ax2 = ax.twinx()
ax2.fill_between(dates, pi, alpha=0.15, color='crimson')
ax2.set_ylabel('pi_filter', fontsize=9, color='crimson')
ax2.set_ylim(0, 1)

ax.set_ylabel('Best lookback (months)', fontsize=9)
ax.set_xlabel('Date')
ax.set_ylim(0.5, 12.5)
ax.set_yticks(MOM_LBS)
ax.set_title('Best lookback each month\n'
             '(dots = highest IC lookback, shading = pi_filter)', fontsize=10)

# Panel C: Total momentum SHAP vs pi_filter (hexbin)
ax = axes[1, 0]
hb = ax.hexbin(pi_per_row, shap_mom_total, gridsize=40, cmap='RdBu_r',
               mincnt=1, reduce_C_function=np.mean)
ax.set_xlabel('pi_filter', fontsize=9)
ax.set_ylabel('Total momentum SHAP\n(sum of all 12 lookbacks)', fontsize=9)
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
ax.axvline(0.5, color='black', linewidth=0.5, linestyle='--')
plt.colorbar(hb, ax=ax, label='Mean SHAP')
ax.set_title('How does XGBoost use momentum\nacross regimes?', fontsize=10)

# Panel D: Monthly average momentum IC vs pi_filter (scatter)
ax = axes[1, 1]
calm_m  = monthly_avg_ic[monthly_avg_ic['regime'] == 'Calm']
panic_m = monthly_avg_ic[monthly_avg_ic['regime'] == 'Panic']
ax.scatter(calm_m['pi_filter'], calm_m['avg_ic'],
           color='steelblue', alpha=0.6, s=30, label='Calm month')
ax.scatter(panic_m['pi_filter'], panic_m['avg_ic'],
           color='crimson', alpha=0.6, s=30, label='Panic month')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0.5, color='black', linewidth=0.5, linestyle='--')

# Trend line
from numpy.polynomial.polynomial import polyfit
all_pi = monthly_avg_ic['pi_filter'].values
all_ic = monthly_avg_ic['avg_ic'].values
mask = np.isfinite(all_pi) & np.isfinite(all_ic)
b, m = polyfit(all_pi[mask], all_ic[mask], 1)
x_line = np.linspace(0, 1, 100)
ax.plot(x_line, b + m * x_line, color='black', linewidth=1.5,
        linestyle='--', label=f'Trend (slope={m:.3f})')

ax.set_xlabel('pi_filter', fontsize=9)
ax.set_ylabel('Average momentum IC\n(mean across all lookbacks)', fontsize=9)
ax.legend(fontsize=8)
ax.set_title('Does momentum predictability\nchange with regime?', fontsize=10)

plt.tight_layout()
fig.savefig('momentum_pi_interaction.png', dpi=150)
plt.close(fig)
print("\nSaved: momentum_pi_interaction.png")

# ── Summary stats ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  MOMENTUM × REGIME INTERACTION SUMMARY")
print("=" * 50)

print(f"\n  Avg momentum IC in calm:  {calm_m['avg_ic'].mean():.4f}")
print(f"  Avg momentum IC in panic: {panic_m['avg_ic'].mean():.4f}")

best_calm  = avg_ic['Calm'].idxmax()
best_panic = avg_ic['Panic'].idxmax()
print(f"\n  Best lookback in calm:  mom_{best_calm} (IC={avg_ic.loc[best_calm, 'Calm']:.4f})")
print(f"  Best lookback in panic: mom_{best_panic} (IC={avg_ic.loc[best_panic, 'Panic']:.4f})")

print(f"\n  Trend slope (IC vs pi_filter): {m:.4f}")
if m < -0.01:
    print("  → Momentum is LESS predictive in panic (negative slope)")
elif m > 0.01:
    print("  → Momentum is MORE predictive in panic (positive slope)")
else:
    print("  → Momentum predictability is roughly constant across regimes")
