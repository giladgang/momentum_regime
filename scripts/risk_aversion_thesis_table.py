"""
risk_aversion_thesis_table.py
==============================
Reproduces the exact risk aversion table in the thesis (Section 5.5).

Three target families:
  1. Sharpe-like:    y = r / sigma^a          for a in [0.5, 1.0]
  2. Mean-variance:  y = r - (gamma/2)*sigma^2 for gamma in [0.1, 0.2, 0.5]
  3. Log return:     y = log(1 + r)

Baseline (y = r) uses production artefact scores.
All variants use 50 XGB seeds on artefact sample.
Saves: risk_aversion_thesis_results.csv

Usage:
    python -u scripts/risk_aversion_thesis_table.py
"""

import numpy as np
import pandas as pd
import pickle, sys, os, time
from xgboost import XGBRegressor
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load artefacts ──
print("Loading artefacts ...")
with open('cs_artefacts_data.pkl', 'rb') as f:
    art = pickle.load(f)

train = art['train'].copy()
test  = art['test'].copy()
X_train = art['X_train']
X_test  = art['X_test']
y_train = art['y_train']
FEATURES = art['FEATURES']

MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]
TRADING_FEE = 0.001
XGB_SEEDS = list(range(1, 51))

# ── Compute trailing variance on artefact sample ──
print("Computing trailing variance ...")
stocks_raw = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks_raw['date'] = pd.to_datetime(stocks_raw['date'])
stocks_raw = stocks_raw.sort_values(['permno', 'date'])
stocks_raw['trail_var'] = (
    stocks_raw.groupby('permno')['ret_adj']
    .transform(lambda x: x.shift(1).rolling(12, min_periods=6).var())
)

# Merge trail_var into train/test via permno+date
trail_lookup = stocks_raw[['permno', 'date', 'trail_var']].drop_duplicates(['permno', 'date'])
train = train.merge(trail_lookup, on=['permno', 'date'], how='left')
test = test.merge(trail_lookup, on=['permno', 'date'], how='left')

# Fill missing with training median
train_var_median = train['trail_var'].median()
train['trail_var'] = train['trail_var'].fillna(train_var_median)
test['trail_var'] = test['trail_var'].fillna(train_var_median)

# Also need trailing sigma for Sharpe-like targets
train['trail_sigma'] = np.sqrt(np.array(train['trail_var'].values, dtype=float).clip(min=1e-8))
test['trail_sigma'] = np.sqrt(np.array(test['trail_var'].values, dtype=float).clip(min=1e-8))

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")
print(f"  Trail var NaN after fill: train={train['trail_var'].isna().sum()}, test={test['trail_var'].isna().sum()}")

# ── Portfolio helper ──
def long_short_port(df_test, score_col, fee=TRADING_FEE):
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_test.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10: continue
        lo, hi = nyse.quantile(0.10), nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        shorts = grp[grp[score_col] <= lo]
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
        lme = longs['me'].sum()
        nlw = (longs.set_index('permno')['me'] / lme).to_dict()
        rl = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        nsw = (shorts.set_index('permno')['me'] / sme).to_dict()
        rs = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(nlw.get(p,0)-prev_lw.get(p,0)) for p in set(nlw)|set(prev_lw))/2
        ts = sum(abs(nsw.get(p,0)-prev_sw.get(p,0)) for p in set(nsw)|set(prev_sw))/2
        monthly.append({'date': date, 'ret': rl - rs - fee*(tl+ts)})
        prev_lw, prev_sw = nlw, nsw
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def metrics(r):
    r = r.dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    return ann_ret, ann_vol, sharpe

def compute_pi_share(model, X_test, features, mom_features):
    """Compute pi_filter's share of total feature importance."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    abs_shap = np.abs(sv).mean(axis=0)
    total = abs_shap.sum()
    pi_idx = features.index('pi_filter')
    return abs_shap[pi_idx] / total * 100

def train_and_evaluate(name, y_target, X_train, X_test, test_df, features, mom_features):
    """Train 50-seed XGB, build portfolio, compute metrics and pi share."""
    print(f"\n  Training 50-seed ensemble for: {name}")
    preds = np.zeros(len(X_test))
    last_model = None
    for xs in XGB_SEEDS:
        model = XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', random_state=xs, verbosity=0
        )
        model.fit(X_train, y_target)
        preds += model.predict(X_test)
        last_model = model
    preds /= len(XGB_SEEDS)

    score_col = f'score_{name}'
    test_df[score_col] = preds
    r = long_short_port(test_df, score_col)
    ann_ret, ann_vol, sharpe = metrics(r)
    pi_share = compute_pi_share(last_model, X_test, features, mom_features)

    print(f"    Sharpe={sharpe:.2f}  Ann.Ret={ann_ret:.1%}  pi_share={pi_share:.0f}%")
    return {'name': name, 'sharpe': sharpe, 'ann_ret': ann_ret, 'ann_vol': ann_vol, 'pi_share': pi_share}

# ── Define all targets ──
r_train = np.array(train['ret_fwd'].values, dtype=float)
sigma_train = np.array(train['trail_sigma'].values, dtype=float)
var_train = np.array(train['trail_var'].values, dtype=float)

targets = [
    # Baseline
    ('Baseline (r)', r_train),
    # Sharpe-like: r / sigma^a
    ('Sharpe a=0.5', r_train / (sigma_train ** 0.5)),
    ('Sharpe a=1.0', r_train / sigma_train),
    # Mean-variance: r - (gamma/2) * sigma^2
    ('MV gamma=0.1', r_train - 0.05 * var_train),
    ('MV gamma=0.2', r_train - 0.10 * var_train),
    ('MV gamma=0.5', r_train - 0.25 * var_train),
    # Log return
    ('Log return', np.log1p(np.clip(r_train, -0.999, None))),
]

# ── Run all ──
print("\n" + "=" * 60)
print("  RISK AVERSION ANALYSIS (50 seeds, artefact sample)")
print("=" * 60)

results = []
for i, (name, y_target) in enumerate(targets):
    if i > 0:
        time.sleep(5)  # brief cooldown

    if name == 'Baseline (r)':
        # Use production artefact scores
        print(f"\n  Baseline: using production artefact scores")
        test['score_baseline'] = test['score_xgb']
        r = long_short_port(test, 'score_baseline')
        ann_ret, ann_vol, sharpe = metrics(r)
        import joblib
        model = joblib.load('cs_artefacts_xgb.pkl')
        pi_share = compute_pi_share(model, X_test, FEATURES, MOM_FEATURES)
        results.append({'name': name, 'sharpe': sharpe, 'ann_ret': ann_ret,
                       'ann_vol': ann_vol, 'pi_share': pi_share})
        print(f"    Sharpe={sharpe:.2f}  Ann.Ret={ann_ret:.1%}  pi_share={pi_share:.0f}%")
    else:
        result = train_and_evaluate(name, y_target, X_train, X_test, test,
                                    FEATURES, MOM_FEATURES)
        results.append(result)

# ── Save results ──
res_df = pd.DataFrame(results)
res_df.to_csv('risk_aversion_thesis_results.csv', index=False, float_format='%.4f')

print("\n\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"{'Target':<20s} {'Sharpe':>8s} {'Pi share':>9s}")
print("-" * 40)
for _, row in res_df.iterrows():
    print(f"  {row['name']:<18s} {row['sharpe']:>7.2f} {row['pi_share']:>8.0f}%")

print(f"\nSaved: risk_aversion_thesis_results.csv")

# ── Export LaTeX and CSV tables ─────────────────────────────────────────────

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# Save CSV to tables dir as well
res_df.to_csv(os.path.join(TABLES_DIR, 'table_risk_aversion.csv'), index=False, float_format='%.4f')

# Build LaTeX table matching the thesis format
def fmt_sharpe(v):
    s = f"{abs(v):.2f}"
    return f"$-${s}" if v < 0 else s

tex_lines = []
tex_lines.append(r'\begin{table}[H]')
tex_lines.append(r'\centering')
tex_lines.append(r'\begin{tabular}{l l r r}')
tex_lines.append(r'\toprule')
tex_lines.append(r'Target family & Parameter & Sharpe & $\pi$ share \\')
tex_lines.append(r'\midrule')

# Map result names to table rows
# Row order and grouping must match thesis
for _, row in res_df.iterrows():
    name = row['name']
    sh = fmt_sharpe(row['sharpe'])
    pi_pct = f"{row['pi_share']:.0f}\\%"

    if name == 'Baseline (r)':
        tex_lines.append(f"Baseline ($r$) & --- & {sh} & {pi_pct} \\\\")
        tex_lines.append(r'\midrule')
        tex_lines.append(r"\multicolumn{4}{l}{\textit{Sharpe-like: $r\,/\,\sigma^a$}} \\[2pt]")
    elif name == 'Sharpe a=0.5':
        tex_lines.append(f" & $a = 0.5$ & {sh} & {pi_pct} \\\\")
    elif name == 'Sharpe a=1.0':
        tex_lines.append(f" & $a = 1.0$ & {sh} & {pi_pct} \\\\")
        tex_lines.append(r'\midrule')
        tex_lines.append(r"\multicolumn{4}{l}{\textit{Mean-variance: $r - \frac{\gamma}{2}\sigma^2$}} \\[2pt]")
    elif name == 'MV gamma=0.1':
        tex_lines.append(f" & $\\gamma = 0.1$ & {sh} & {pi_pct} \\\\")
    elif name == 'MV gamma=0.2':
        tex_lines.append(f" & $\\gamma = 0.2$ & {sh} & {pi_pct} \\\\")
    elif name == 'MV gamma=0.5':
        tex_lines.append(f" & $\\gamma = 0.5$ & {sh} & {pi_pct} \\\\")
        tex_lines.append(r'\midrule')
        tex_lines.append(r"\multicolumn{4}{l}{\textit{Log return: $\log(1+r)$}} \\[2pt]")
    elif name == 'Log return':
        tex_lines.append(f" & --- & {sh} & {pi_pct} \\\\")

tex_lines.append(r'\bottomrule')
tex_lines.append(r'\end{tabular}')
tex_lines.append(r"\caption{Risk-adjusted training targets and regime signal importance. ``$\pi$ share'' is the regime signal's fraction of total feature importance (remainder is momentum). All variants use 50 XGB seeds on the same sample as the main results.}")
tex_lines.append(r'\label{tab:risk_aversion}')
tex_lines.append(r'\end{table}')

tex_path = os.path.join(TABLES_DIR, 'table_risk_aversion.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex_lines) + '\n')
print(f"Saved: {tex_path}")

print("Done.")
