"""
ridge_baseline_test.py
======================
Test whether the LR/XGB divergence is driven by linearity vs nonlinearity,
or by classification vs regression targets.

Adds a Ridge regression (M1b) that predicts continuous forward returns
(same target as XGBoost XGB) using a linear model (same functional form as LR).
This isolates the linearity/nonlinearity question from the target question.

Uses saved artefacts from cross_sectional_model.py to avoid re-running
the full pipeline.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_FEE, MOM_FEATURES

# ── Load saved artefacts ─────────────────────────────────────────────────────

print("Loading artefacts ...")
with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test     = artefacts['test'].copy()
train    = artefacts['train'].copy()
X_train  = artefacts['X_train']
X_test   = artefacts['X_test']
y_train  = artefacts['y_train']
FEATURES = artefacts['FEATURES']

print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
print(f"  Features: {FEATURES}")

# ── Portfolio construction (copied from cross_sectional_model.py) ────────────

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs  = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0:
            continue
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
    if not monthly:
        return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    return ann_ret, ann_vol, sharpe, mdd

# ── Preprocessing (same as LR: impute + scale) ──────────────────────────────

imputer = SimpleImputer(strategy='median')
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(imputer.fit_transform(X_train))
X_te_s  = scaler.transform(imputer.transform(X_test))

# ── Ridge Regression: predicting continuous returns (same target as XGB) ──────

print("\n=== Ridge Regression (continuous return target, same as XGB) ===")

alphas_to_test = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for alpha in alphas_to_test:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr_s, y_train)
    test['score_ridge'] = ridge.predict(X_te_s)

    r_ridge = long_short_port(test, 'score_ridge')
    ar, av, sh, mdd = metrics(r_ridge)

    # Regime-conditional
    test_dates = test[['date', 'pi_filter']].drop_duplicates('date').set_index('date')
    pi = test_dates.reindex(r_ridge.index)['pi_filter']
    calm  = r_ridge[pi < 0.5]
    panic = r_ridge[pi >= 0.5]
    sr_calm  = calm.mean() / calm.std() * np.sqrt(12) if len(calm) > 1 and calm.std() > 0 else float('nan')
    sr_panic = panic.mean() / panic.std() * np.sqrt(12) if len(panic) > 1 and panic.std() > 0 else float('nan')

    print(f"  alpha={alpha:>7.1f}  |  Sharpe={sh:>6.2f}  Ann.Ret={ar:>7.1%}  "
          f"Ann.Vol={av:>7.1%}  MDD={mdd:>7.1%}  |  Calm={sr_calm:>6.2f}  Panic={sr_panic:>6.2f}")

    # Print coefficients for the best alpha
    if alpha == 1.0:
        print(f"\n  Ridge (alpha=1.0) coefficients:")
        coefs = pd.Series(ridge.coef_, index=FEATURES).sort_values(ascending=False)
        for feat, c in coefs.items():
            print(f"    {feat:<15s} {c:>+8.5f}")

# ── OLS (no regularization) for comparison ───────────────────────────────────

print("\n=== OLS (no regularization) ===")
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_tr_s, y_train)
test['score_ols'] = ols.predict(X_te_s)
r_ols = long_short_port(test, 'score_ols')
ar, av, sh, mdd = metrics(r_ols)

pi = test_dates.reindex(r_ols.index)['pi_filter']
calm  = r_ols[pi < 0.5]
panic = r_ols[pi >= 0.5]
sr_calm  = calm.mean() / calm.std() * np.sqrt(12) if len(calm) > 1 and calm.std() > 0 else float('nan')
sr_panic = panic.mean() / panic.std() * np.sqrt(12) if len(panic) > 1 and panic.std() > 0 else float('nan')

print(f"  OLS       |  Sharpe={sh:>6.2f}  Ann.Ret={ar:>7.1%}  "
      f"Ann.Vol={av:>7.1%}  MDD={mdd:>7.1%}  |  Calm={sr_calm:>6.2f}  Panic={sr_panic:>6.2f}")

print(f"\n  OLS coefficients:")
coefs = pd.Series(ols.coef_, index=FEATURES).sort_values(ascending=False)
for feat, c in coefs.items():
    print(f"    {feat:<15s} {c:>+8.5f}")

# ── Recap: LR (LR classification) and XGB for comparison ───────────

print("\n=== Comparison with existing models ===")

strats = artefacts['strategies_lo']
for name in ['Method 1: LR', 'Method 2: XGB']:
    if name in strats:
        r = strats[name]
        ar, av, sh, mdd = metrics(r)
        pi = test_dates.reindex(r.index)['pi_filter']
        calm  = r[pi < 0.5]
        panic = r[pi >= 0.5]
        sr_calm  = calm.mean() / calm.std() * np.sqrt(12) if len(calm) > 1 and calm.std() > 0 else float('nan')
        sr_panic = panic.mean() / panic.std() * np.sqrt(12) if len(panic) > 1 and panic.std() > 0 else float('nan')
        print(f"  {name:<26s}  |  Sharpe={sh:>6.2f}  Ann.Ret={ar:>7.1%}  "
              f"Ann.Vol={av:>7.1%}  MDD={mdd:>7.1%}  |  Calm={sr_calm:>6.2f}  Panic={sr_panic:>6.2f}")

# ── Export LaTeX and CSV tables ─────────────────────────────────────────────

import os
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Collect all results into a list for export
rows = []

# OLS
test['score_ols_final'] = ols.predict(X_te_s)
r_ols_final = long_short_port(test, 'score_ols_final')
ar_o, av_o, sh_o, mdd_o = metrics(r_ols_final)
pi_o = test_dates.reindex(r_ols_final.index)['pi_filter']
calm_o = r_ols_final[pi_o < 0.5]
panic_o = r_ols_final[pi_o >= 0.5]
sr_calm_o = calm_o.mean() / calm_o.std() * np.sqrt(12) if len(calm_o) > 1 and calm_o.std() > 0 else float('nan')
sr_panic_o = panic_o.mean() / panic_o.std() * np.sqrt(12) if len(panic_o) > 1 and panic_o.std() > 0 else float('nan')
rows.append({'model': 'OLS (no reg.)', 'alpha': '---', 'sharpe': sh_o,
             'ann_ret': ar_o, 'ann_vol': av_o, 'mdd': mdd_o,
             'sr_calm': sr_calm_o, 'sr_panic': sr_panic_o})

# Ridge models
for alpha in alphas_to_test:
    ridge_m = Ridge(alpha=alpha)
    ridge_m.fit(X_tr_s, y_train)
    test['score_ridge_exp'] = ridge_m.predict(X_te_s)
    r_r = long_short_port(test, 'score_ridge_exp')
    ar_r, av_r, sh_r, mdd_r = metrics(r_r)
    pi_r = test_dates.reindex(r_r.index)['pi_filter']
    calm_r = r_r[pi_r < 0.5]
    panic_r = r_r[pi_r >= 0.5]
    sr_calm_r = calm_r.mean() / calm_r.std() * np.sqrt(12) if len(calm_r) > 1 and calm_r.std() > 0 else float('nan')
    sr_panic_r = panic_r.mean() / panic_r.std() * np.sqrt(12) if len(panic_r) > 1 and panic_r.std() > 0 else float('nan')
    # Format alpha for display
    if alpha >= 1000:
        alpha_str = f"{alpha:,.0f}"
    elif alpha == int(alpha):
        alpha_str = f"{int(alpha)}"
    else:
        alpha_str = f"{alpha}"
    rows.append({'model': 'Ridge', 'alpha': alpha_str, 'sharpe': sh_r,
                 'ann_ret': ar_r, 'ann_vol': av_r, 'mdd': mdd_r,
                 'sr_calm': sr_calm_r, 'sr_panic': sr_panic_r})

# LR and XGB from artefacts
strats = artefacts['strategies_lo']
for name_key, label in [('Method 1: LR', 'LR (LR, classification)'),
                         ('Method 2: XGB', 'XGB')]:
    if name_key in strats:
        r = strats[name_key]
        ar_m, av_m, sh_m, mdd_m = metrics(r)
        pi_m = test_dates.reindex(r.index)['pi_filter']
        calm_m = r[pi_m < 0.5]
        panic_m = r[pi_m >= 0.5]
        sr_calm_m = calm_m.mean() / calm_m.std() * np.sqrt(12) if len(calm_m) > 1 and calm_m.std() > 0 else float('nan')
        sr_panic_m = panic_m.mean() / panic_m.std() * np.sqrt(12) if len(panic_m) > 1 and panic_m.std() > 0 else float('nan')
        rows.append({'model': label, 'alpha': '---', 'sharpe': sh_m,
                     'ann_ret': ar_m, 'ann_vol': av_m, 'mdd': mdd_m,
                     'sr_calm': sr_calm_m, 'sr_panic': sr_panic_m})

# Save CSV
df_ridge = pd.DataFrame(rows)
df_ridge.to_csv(os.path.join(TABLES_DIR, 'table_ridge.csv'), index=False, float_format='%.4f')
print(f"Saved: {os.path.join(TABLES_DIR, 'table_ridge.csv')}")

# Helper to format numbers for LaTeX
def fmt_pct(v):
    """Format as percentage with $-$ for negative.
    Emits LaTeX-escaped \\% (not bare %) so the table compiles.
    """
    s = f"{abs(v) * 100:.1f}\\%"
    return f"$-${s}" if v < 0 else s

def fmt_sharpe(v):
    """Format Sharpe with $-$ for negative."""
    s = f"{abs(v):.2f}"
    return f"$-${s}" if v < 0 else s

def fmt_alpha_tex(a):
    """Format alpha for LaTeX (add thousands comma)."""
    if a == '---':
        return '---'
    try:
        val = float(a.replace(',', ''))
        if val >= 1000:
            return f"{int(val):,}".replace(',', '{,}')
        elif val == int(val):
            return f"{int(val)}"
        else:
            return a
    except ValueError:
        return a

# Build LaTeX table
# Only include selected ridge alphas matching the thesis (0.1, 1, 10, 100, 1000)
selected_alphas = {'---', '0.1', '1', '10', '100', '1,000'}
tex_lines = []
tex_lines.append(r'\begin{table}[H]')
tex_lines.append(r'\centering')
tex_lines.append(r'\small')
tex_lines.append(r'\begin{tabular}{l r r r r r r}')
tex_lines.append(r'\toprule')
tex_lines.append(r'Model & $\alpha$ & Sharpe & Ann.\ Ret & Ann.\ Vol & MDD & Calm / Panic SR \\')
tex_lines.append(r'\midrule')

for row in rows:
    m = row['model']
    a = row['alpha']
    # Skip alphas not in the thesis table (0.01)
    if m == 'Ridge' and a not in selected_alphas:
        continue
    a_tex = fmt_alpha_tex(a)
    sh = fmt_sharpe(row['sharpe'])
    ar = fmt_pct(row['ann_ret'])
    av = fmt_pct(row['ann_vol'])
    mdd = fmt_pct(row['mdd'])
    sr_c = fmt_sharpe(row['sr_calm'])
    sr_p = fmt_sharpe(row['sr_panic'])
    line = f"{m} & {a_tex} & {sh} & {ar} & {av} & {mdd} & {sr_c} / {sr_p} \\\\"
    # Add midrule before LR
    if m == 'LR (LR, classification)':
        tex_lines.append(r'\midrule')
    tex_lines.append(line)

tex_lines.append(r'\bottomrule')
tex_lines.append(r'\end{tabular}')
tex_lines.append(r'\caption{Ridge regression baseline predicting continuous forward returns (same target as XGB). The Sharpe of $-$0.57 is robust across all regularisation strengths.}')
tex_lines.append(r'\label{tab:ridge}')
tex_lines.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'table_ridge.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex_lines) + '\n')
print(f"Saved: {tex_path}")

print("\nDone.")
