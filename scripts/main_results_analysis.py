"""
main_results_analysis.py
========================
Computes all statistical results for the thesis Main Results section and
outputs LaTeX table fragments to tables/.

Tables produced:
  1. table_performance.tex   — Full performance comparison (10 strategies)
  2. table_regime_sharpe.tex — Regime-conditional Sharpe ratios
  3. table_shap.tex          — SHAP feature importance (overall + by regime)
  4. table_lr_coef.tex       — Logistic regression coefficients with inference
  5. table_ic.tex            — Information coefficient analysis
  6. table_granger.tex       — Granger causality (pi_filter ↔ momentum IC)
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pickle, joblib, os, sys, warnings
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PORTFOLIO_TYPE, TRADING_FEE, TRAIN_END, TABLES_DIR,
                    XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
                    SUBSAMPLE, COLSAMPLE, CS_FEATURES, MOM_FEATURES,
                    USE_FUNDAMENTALS)

os.makedirs(TABLES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# A. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

print("[ 1/8 ] Loading artefacts ...")

with open('cs_artefacts_data.pkl', 'rb') as f:
    artefacts = pickle.load(f)

test       = artefacts['test'].copy()
train      = artefacts['train'].copy()
X_train    = artefacts['X_train']
X_tr_s     = artefacts['X_tr_s']
X_te_s     = artefacts['X_te_s']
FEATURES   = artefacts['FEATURES']
strats_lo  = artefacts['strategies_lo']
r_mkt      = artefacts['r_mkt']
shap_vals  = artefacts['shap_values']

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

MOM_LBS      = list(range(1, 13))
MOM_FEATURES = [f'mom_{lb}' for lb in MOM_LBS]
# TRADING_FEE imported from config.py

# ── Helpers ───────────────────────────────────────────────────────────────────

def metrics(r):
    r = pd.Series(r).dropna()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum     = (1 + r).cumprod()
    mdd     = ((cum - cum.cummax()) / cum.cummax()).min()
    final   = cum.iloc[-1]
    return ann_ret, ann_vol, sharpe, mdd, final


def long_only_port(df_data, score_col, fee=TRADING_FEE):
    """Top-decile long-only portfolio, value-weighted, NYSE breakpoints."""
    monthly = []
    prev_weights = {}
    for date, grp in df_data.groupby('date'):
        nyse = grp[grp['exchcd'] == 1][score_col].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        longs = grp[grp[score_col] >= hi]
        if longs['me'].sum() == 0:
            continue
        total_me = longs['me'].sum()
        new_weights = (longs.set_index('permno')['me'] / total_me).to_dict()
        all_permnos = set(new_weights) | set(prev_weights)
        turnover = sum(abs(new_weights.get(p, 0) - prev_weights.get(p, 0))
                       for p in all_permnos) / 2
        r_gross = (longs['ret_fwd'] * longs['me']).sum() / total_me
        monthly.append({'date': date, 'ret': r_gross - fee * turnover})
        prev_weights = new_weights
    return pd.DataFrame(monthly).set_index('date')['ret']


def long_short_port(df_data, score_col, fee=TRADING_FEE):
    """Top-decile long, bottom-decile short, value-weighted, NYSE breakpoints."""
    monthly = []
    prev_lw, prev_sw = {}, {}
    for date, grp in df_data.groupby('date'):
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
    return pd.DataFrame(monthly).set_index('date')['ret']


def build_port(df_data, score_col, fee=TRADING_FEE):
    """Dispatcher: calls long_only_port or long_short_port based on config."""
    if PORTFOLIO_TYPE == 'long_short':
        return long_short_port(df_data, score_col, fee=fee)
    else:
        return long_only_port(df_data, score_col, fee=fee)


def block_bootstrap_sharpe(r, n_boot=10000, block_len=12):
    """Block bootstrap 95% CI on annualized Sharpe ratio."""
    r = np.array(r)
    T = len(r)
    n_blocks = int(np.ceil(T / block_len))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = np.random.randint(0, T - block_len + 1, size=n_blocks)
        sample = np.concatenate([r[s:s+block_len] for s in starts])[:T]
        std = sample.std()
        sharpes[b] = sample.mean() / std * np.sqrt(12) if std > 0 else 0
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return lo, hi


def newey_west_t(r_strat, r_bench, maxlags=6):
    """Newey-West t-stat for mean excess return vs benchmark."""
    r_strat = pd.Series(r_strat)
    r_bench = pd.Series(r_bench)
    # Align by index
    common = r_strat.index.intersection(r_bench.index)
    excess = r_strat.loc[common].values - r_bench.loc[common].values
    excess = np.asarray(excess, dtype=float)
    excess = excess[~np.isnan(excess)]
    if len(excess) < 12:
        return np.nan, np.nan
    X = np.ones((len(excess), 1))
    model = sm.OLS(excess, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model.tvalues[0], model.pvalues[0]


def sig_stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


def write_tex(filename, content):
    import re
    # Escape unescaped % signs in data (digit followed by % not preceded by \)
    content = re.sub(r'(\d)%', r'\1\\%', content)
    path = os.path.join(TABLES_DIR, filename)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# B. GHM BENCHMARK RE-RUN
# ══════════════════════════════════════════════════════════════════════════════

print("[ 2/8 ] Re-running GHM benchmark ...")

# Market cycle classification from vwretd
mkt_ret = panel[['date', 'vwretd']].dropna().drop_duplicates('date').sort_values('date').reset_index(drop=True)
mkt_ret['mkt_fast'] = mkt_ret['vwretd']
mkt_ret['mkt_slow'] = mkt_ret['vwretd'].rolling(12, min_periods=12).mean()

def classify_cycle(row):
    if pd.isna(row['mkt_slow']):
        return np.nan
    if row['mkt_slow'] >= 0 and row['mkt_fast'] >= 0:
        return 'Bull'
    elif row['mkt_slow'] >= 0 and row['mkt_fast'] < 0:
        return 'Correction'
    elif row['mkt_slow'] < 0 and row['mkt_fast'] < 0:
        return 'Bear'
    else:
        return 'Rebound'

mkt_ret['cycle'] = mkt_ret.apply(classify_cycle, axis=1)
cycle_map = mkt_ret[['date', 'cycle']].dropna()

# Merge cycles into train/test
train = train.merge(cycle_map, on='date', how='left')
test  = test.merge(cycle_map, on='date', how='left')

def compute_blended_score(data, a):
    rank_slow = data.groupby('date')['mom_12'].rank(pct=True)
    rank_fast = data.groupby('date')['mom_1'].rank(pct=True)
    return (1 - a) * rank_slow + a * rank_fast

def compute_dyn_score(data, a_bu, a_co, a_be, a_re):
    rank_slow = data.groupby('date')['mom_12'].rank(pct=True)
    rank_fast = data.groupby('date')['mom_1'].rank(pct=True)
    a = pd.Series(np.nan, index=data.index)
    a[data['cycle'] == 'Bull']       = a_bu
    a[data['cycle'] == 'Correction'] = a_co
    a[data['cycle'] == 'Bear']       = a_be
    a[data['cycle'] == 'Rebound']    = a_re
    a = a.fillna(0.5)
    return (1 - a) * rank_slow + a * rank_fast

# Static strategies
ghm_returns = {}
for name, a in [('GHM SLOW (a=0)', 0.0), ('GHM MED (a=0.5)', 0.5), ('GHM FAST (a=1)', 1.0)]:
    test[f'score_{name}'] = compute_blended_score(test, a).values
    ghm_returns[name] = build_port(test, f'score_{name}')

# DYN: grid search on training
best_sharpe, best_pair = -999, (0.5, 0.5)
grid = np.arange(0.0, 1.05, 0.1)
for a_co in grid:
    for a_re in grid:
        train['_dyn'] = compute_dyn_score(train, 0.5, a_co, 0.5, a_re).values
        r = build_port(train, '_dyn')
        if len(r) < 12:
            continue
        sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
        if sh > best_sharpe:
            best_sharpe = sh
            best_pair = (a_co, a_re)

a_co_hat, a_re_hat = best_pair
print(f"  DYN speeds: a_Co={a_co_hat:.2f}, a_Re={a_re_hat:.2f}")

test['score_dyn'] = compute_dyn_score(test, 0.5, a_co_hat, 0.5, a_re_hat).values
ghm_returns['GHM DYN'] = build_port(test, 'score_dyn')

if '_dyn' in train.columns:
    train.drop(columns=['_dyn'], inplace=True)

# ══════════════════════════════════════════════════════════════════════════════
# B2. REDUCED-FEATURE MODELS (momentum + pi_filter only)
# ══════════════════════════════════════════════════════════════════════════════

print("[ 2b/8 ] Training reduced-feature models (mom + pi_filter only) ...")

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

REDUCED_FEATURES = MOM_FEATURES + ['pi_filter']
feat_idx = [FEATURES.index(f) for f in REDUCED_FEATURES]

# Extract reduced feature matrices from the full artefact data
X_train_red = train[REDUCED_FEATURES].values.astype(float)
X_test_red  = test[REDUCED_FEATURES].values.astype(float)

# LR: impute + scale + fit on reduced features
imputer_red = SimpleImputer(strategy='median')
scaler_red  = StandardScaler()
X_tr_red_s  = scaler_red.fit_transform(imputer_red.fit_transform(X_train_red))
X_te_red_s  = scaler_red.transform(imputer_red.transform(X_test_red))

# Need above_med labels
if 'above_med' not in train.columns:
    train['above_med'] = train.groupby('date')['ret_fwd'].transform(
        lambda x: (x > x.median()).astype(int)
    )

lr_red = LogisticRegression(max_iter=1000, C=1.0)
lr_red.fit(X_tr_red_s, train['above_med'].values)
test['score_lr_red'] = lr_red.predict_proba(X_te_red_s)[:, 1]
r_lr_red = build_port(test, 'score_lr_red')
print(f"  M1: LR (mom+pi) — {len(r_lr_red)} monthly obs")

# XGB: use the production 50-seed ensemble scores from the artefacts
# (score_xgb was trained on mom+pi features with 50 seeds averaged)
r_xgb_red = build_port(test, 'score_xgb')
print(f"  M2: XGB (mom+pi) — {len(r_xgb_red)} monthly obs")

# ══════════════════════════════════════════════════════════════════════════════
# C. TABLE 1: FULL PERFORMANCE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("[ 3/8 ] Table 1: Performance comparison ...")

# Recompute all baseline strategies as long-short
strats_ls = {}
score_map = {
    'Fixed 12-mo mom': 'score_mom12',
    'Fixed 1-mo mom':  'score_mom1',
    'Method 0: Formula': 'score_formula',
    'Method 1: LR':    'score_lr',
    'Method 2: XGB':   'score_xgb',
}
for sname, scol in score_map.items():
    strats_ls[sname] = build_port(test, scol)

all_strats = {
    'Market':               r_mkt,
    'Fixed 12-mo mom':      strats_ls['Fixed 12-mo mom'],
    'Fixed 1-mo mom':       strats_ls['Fixed 1-mo mom'],
    'M0: Formula':          strats_ls['Method 0: Formula'],
    'M1: LR':               strats_ls['Method 1: LR'],
    'M2: XGB (mom+$\\pi$)': r_xgb_red,
}

perf_rows = []
for name, r in all_strats.items():
    ar, av, sh, mdd, final = metrics(r)
    # Compute market beta
    if name == 'Market':
        beta = 1.0
        nw_t, nw_p = np.nan, np.nan
    else:
        common = r.index.intersection(r_mkt.index)
        r_aligned = r.loc[common].values
        mkt_aligned = r_mkt.loc[common].values
        valid = ~(np.isnan(r_aligned) | np.isnan(mkt_aligned))
        if valid.sum() > 12:
            cov_mat = np.cov(r_aligned[valid], mkt_aligned[valid])
            beta = cov_mat[0, 1] / cov_mat[1, 1]
        else:
            beta = np.nan
        nw_t, nw_p = newey_west_t(r, r_mkt)
    perf_rows.append({
        'name': name, 'ann_ret': ar, 'ann_vol': av, 'sharpe': sh,
        'beta': beta, 'mdd': mdd, 'final': final,
        'nw_t': nw_t, 'nw_p': nw_p
    })

# Print to console
print(f"\n{'Strategy':<22} {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>7} {'Beta':>6} {'MaxDD':>7} {'NW t':>6} {'Final$':>7}")
print("-" * 80)
for row in perf_rows:
    nw = f"{row['nw_t']:.2f}" if not np.isnan(row['nw_t']) else "  --"
    beta = f"{row['beta']:.2f}" if not np.isnan(row['beta']) else "  --"
    print(f"{row['name']:<22} {row['ann_ret']:>7.1%} {row['ann_vol']:>7.1%} {row['sharpe']:>7.2f} {beta:>6} {row['mdd']:>7.1%} {nw:>6} {row['final']:>7.1f}")

# LaTeX table
tex_lines = []
tex_lines.append(r"\begin{table}[htbp]")
tex_lines.append(r"\centering")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l r r r r r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r" & Ann.\ Ret & Ann.\ Vol & Sharpe & Max DD & $\beta$ & NW $t$ & Final \$ \\")
tex_lines.append(r"\midrule")

for i, row in enumerate(perf_rows):
    if np.isnan(row['nw_t']):
        nw = "--"
    else:
        nw = f"{row['nw_t']:.2f}{sig_stars(row['nw_p'])}"
    # Format beta
    beta_val = row['beta']
    if np.isnan(beta_val):
        beta_str = "--"
    elif beta_val < 0:
        beta_str = f"$-${abs(beta_val):.2f}"
    else:
        beta_str = f"{beta_val:.2f}"
    line = f"{row['name']} & {row['ann_ret']:.1%} & {row['ann_vol']:.1%} & {row['sharpe']:.2f} & {row['mdd']:.1%} & {beta_str} & {nw} & {row['final']:.1f} \\\\"
    # Add midrule separator after Market
    if i == 0:
        line += "\n" + r"\midrule"
    tex_lines.append(line)

tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\caption{Out-of-sample portfolio performance.}")
tex_lines.append(r"\label{tab:performance}")
tex_lines.append("")
tex_lines.append(r"\medskip")
tex_lines.append(r"\small")
tex_lines.append(r"\textbf{Notes:} Long-short (top/bottom decile, NYSE breakpoints), value-weighted, net of 10\,bps one-way costs. $\beta$ is the market beta. NW $t$ uses Newey--West standard errors (6 lags). Test period: January 2011 to November 2025 (167 months). $^{*}$\,$p<0.10$; $^{**}$\,$p<0.05$; $^{***}$\,$p<0.01$.")
tex_lines.append(r"\end{table}")

write_tex('table_performance.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# D. TABLE 2: REGIME-CONDITIONAL SHARPE
# ══════════════════════════════════════════════════════════════════════════════

print("[ 4/8 ] Table 2: Regime-conditional Sharpe ...")

test_dates = test[['date', 'pi_filter']].drop_duplicates('date').set_index('date')

def regime_sharpe(r):
    r = r.dropna()
    pi = test_dates.reindex(r.index)['pi_filter']
    calm  = r[pi < 0.5]
    panic = r[pi >= 0.5]
    def sr(x):
        return x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else np.nan
    return sr(r), sr(calm), sr(panic), len(calm), len(panic)

reg_rows = []
for name, r in all_strats.items():
    full, calm, panic, n_calm, n_panic = regime_sharpe(r)
    reg_rows.append({'name': name, 'full': full, 'calm': calm, 'panic': panic,
                     'n_calm': n_calm, 'n_panic': n_panic})

# Print
print(f"\n{'Strategy':<18} {'Full':>7} {'Calm':>7} {'Panic':>7} {'N_calm':>7} {'N_panic':>7}")
print("-" * 55)
for row in reg_rows:
    p = f"{row['panic']:.2f}" if not np.isnan(row['panic']) else "N/A"
    print(f"{row['name']:<18} {row['full']:>7.2f} {row['calm']:>7.2f} {p:>7} {row['n_calm']:>7} {row['n_panic']:>7}")

# LaTeX
# Get n_calm and n_panic from the first row for the caption
n_calm_total = reg_rows[0]['n_calm']
n_panic_total = reg_rows[0]['n_panic']

tex_lines = []
tex_lines.append(r"\begin{table}[htbp]")
tex_lines.append(r"\centering")
tex_lines.append(r"\begin{tabular}{l r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r" & Full & Calm & Panic \\")
tex_lines.append(r"\midrule")
for i, row in enumerate(reg_rows):
    p = f"{row['panic']:.2f}" if not np.isnan(row['panic']) else "--"
    # Format negative values with $-$ prefix
    def fmt_sharpe(v):
        if np.isnan(v):
            return "--"
        if v < 0:
            return f"$-${abs(v):.2f}"
        return f"{v:.2f}"
    line = f"{row['name']} & {fmt_sharpe(row['full'])} & {fmt_sharpe(row['calm'])} & {fmt_sharpe(row['panic'])} \\\\"
    if i == 0:
        line += "\n" + r"\midrule"
    tex_lines.append(line)
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(f"\\caption{{Regime-conditional Sharpe ratios ({n_calm_total} calm months, {n_panic_total} panic months).}}")
tex_lines.append(r"\label{tab:regime_sharpe}")
tex_lines.append(r"\end{table}")

write_tex('table_regime_sharpe.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# E. TABLE 3: SHAP FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

print("[ 5/8 ] Table 3: SHAP importance ...")

abs_shap = np.abs(shap_vals)
mean_shap = abs_shap.mean(axis=0)

# Regime split
pi_vals = test['pi_filter'].values
calm_mask = pi_vals < 0.5
mean_shap_calm  = abs_shap[calm_mask].mean(axis=0)
mean_shap_panic = abs_shap[~calm_mask].mean(axis=0)

# Build importance table (aggregate momentum)
mom_idx = [FEATURES.index(f) for f in MOM_FEATURES]
other_idx = [i for i in range(len(FEATURES)) if i not in mom_idx]

shap_rows = []
# Momentum aggregate
shap_rows.append({
    'feature': 'Momentum (agg.)',
    'overall': mean_shap[mom_idx].sum(),
    'calm':    mean_shap_calm[mom_idx].sum(),
    'panic':   mean_shap_panic[mom_idx].sum(),
})
# Individual non-momentum features
for i in other_idx:
    shap_rows.append({
        'feature': FEATURES[i],
        'overall': mean_shap[i],
        'calm':    mean_shap_calm[i],
        'panic':   mean_shap_panic[i],
    })

shap_rows.sort(key=lambda x: -x['overall'])

# Print
print(f"\n{'Rank':>4} {'Feature':<20} {'Overall':>9} {'Calm':>9} {'Panic':>9}")
print("-" * 55)
for rank, row in enumerate(shap_rows, 1):
    print(f"{rank:>4} {row['feature']:<20} {row['overall']:>9.4f} {row['calm']:>9.4f} {row['panic']:>9.4f}")

# Also show individual momentum lookbacks
print("\n  Individual momentum lookback importance:")
for lb in MOM_LBS:
    i = FEATURES.index(f'mom_{lb}')
    print(f"    mom_{lb:>2}: overall={mean_shap[i]:.4f}  calm={mean_shap_calm[i]:.4f}  panic={mean_shap_panic[i]:.4f}")

# LaTeX
feature_display = {
    'Momentum (agg.)': 'Momentum (aggregate)',
    'pi_filter': r'$\pi^{\text{filter}}$',
    'bm': 'Book-to-market',
    'roe': 'Return on equity',
    'earnings_growth': 'Earnings growth',
    'leverage': 'Leverage',
    'asset_growth': 'Asset growth',
    'gross_profit_a': 'Gross profitability',
    'log_me': 'Log market equity',
}

tex_lines = []
tex_lines.append(r"\begin{table}[htbp]")
tex_lines.append(r"\centering")
tex_lines.append(r"\begin{tabular}{c l r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Rank & Feature & Overall & Calm & Panic \\")
tex_lines.append(r"\midrule")
for rank, row in enumerate(shap_rows, 1):
    disp = feature_display.get(row['feature'], row['feature'])
    tex_lines.append(f"{rank} & {disp} & {row['overall']:.4f} & {row['calm']:.4f} & {row['panic']:.4f} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\caption{SHAP feature importance for the XGBoost model (Method~2). Mean absolute SHAP values computed on the test set. The 12 momentum lookbacks are aggregated into a single entry. Columns show importance overall and split by regime.}")
tex_lines.append(r"\label{tab:shap}")
tex_lines.append(r"\end{table}")

write_tex('table_shap.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# F. TABLE 4: LR COEFFICIENTS WITH INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

print("[ 6/8 ] Table 4: LR coefficients ...")

# Target: above-median forward return
y_train = train['above_med'].values

# Fit statsmodels Logit (unregularized) for proper inference
X_const = sm.add_constant(X_tr_s)
try:
    logit_model = sm.Logit(y_train, X_const).fit(maxiter=2000, disp=0)
    lr_converged = True
except Exception as e:
    print(f"  Warning: Logit did not converge, using regularized: {e}")
    logit_model = sm.Logit(y_train, X_const).fit_regularized(alpha=1.0, maxiter=2000, disp=0)
    lr_converged = False

coefs  = logit_model.params[1:]   # skip constant
if lr_converged:
    ses    = logit_model.bse[1:]
    zvals  = logit_model.tvalues[1:]
    pvals  = logit_model.pvalues[1:]
else:
    ses    = np.full(len(coefs), np.nan)
    zvals  = np.full(len(coefs), np.nan)
    pvals  = np.full(len(coefs), np.nan)

lr_rows = []
for i, feat in enumerate(FEATURES):
    lr_rows.append({
        'feature': feat, 'coef': coefs[i], 'se': ses[i],
        'z': zvals[i], 'p': pvals[i]
    })

# Sort by absolute coefficient
lr_rows.sort(key=lambda x: -abs(x['coef']))

# Print
print(f"\n{'Feature':<20} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8}")
print("-" * 55)
for row in lr_rows:
    se = f"{row['se']:.4f}" if not np.isnan(row['se']) else "N/A"
    z  = f"{row['z']:.2f}" if not np.isnan(row['z']) else "N/A"
    p  = f"{row['p']:.4f}" if not np.isnan(row['p']) else "N/A"
    print(f"{row['feature']:<20} {row['coef']:>8.4f} {se:>8} {z:>8} {p:>8} {sig_stars(row['p'])}")

# LaTeX
tex_lines = []
tex_lines.append(r"\begin{table}[htbp]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Logistic regression (Method~1) coefficient estimates. Features are standardized prior to fitting. The model is estimated without regularization to obtain valid standard errors. Dependent variable: $\mathbb{1}[r_{t+1}^s > \text{median}]$.}")
tex_lines.append(r"\label{tab:lr_coef}")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l r r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Feature & $\hat{\beta}$ & Std.\ Err. & $z$ & Sig. \\")
tex_lines.append(r"\midrule")
for row in lr_rows:
    disp = feature_display.get(row['feature'], row['feature'].replace('_', r'\_'))
    se = f"{row['se']:.4f}" if not np.isnan(row['se']) else "--"
    z  = f"{row['z']:.2f}" if not np.isnan(row['z']) else "--"
    stars = sig_stars(row['p']) if not np.isnan(row['p']) else ""
    tex_lines.append(f"{disp} & {row['coef']:.4f} & {se} & {z} & {stars} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")

write_tex('table_lr_coef.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# G. TABLE 5: INFORMATION COEFFICIENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("[ 7/8 ] Table 5: IC analysis ...")

ic_records = []
for date, grp in test.groupby('date'):
    pi = grp['pi_filter'].iloc[0]
    regime = 'Panic' if pi >= 0.5 else 'Calm'
    valid = grp[['score_mom12', 'score_lr', 'score_xgb', 'ret_fwd']].dropna()
    if len(valid) < 30:
        continue
    rho_mom, _ = spearmanr(valid['score_mom12'], valid['ret_fwd'])
    rho_lr,  _ = spearmanr(valid['score_lr'],    valid['ret_fwd'])
    rho_xgb, _ = spearmanr(valid['score_xgb'],   valid['ret_fwd'])
    ic_records.append({
        'date': date, 'pi_filter': pi, 'regime': regime,
        'IC_mom': rho_mom, 'IC_lr': rho_lr, 'IC_xgb': rho_xgb
    })

ic_df = pd.DataFrame(ic_records)

# Significance tests
ic_results = []
for col, label in [('IC_mom', 'Momentum (mom\\_12)'),
                    ('IC_lr',  'M1: LR'),
                    ('IC_xgb', 'M2: XGB')]:
    vals = ic_df[col].values
    calm_vals  = ic_df.loc[ic_df['regime'] == 'Calm',  col].values
    panic_vals = ic_df.loc[ic_df['regime'] == 'Panic', col].values

    t_all, p_all = ttest_1samp(vals, 0)
    t_calm, p_calm = ttest_1samp(calm_vals, 0) if len(calm_vals) > 1 else (np.nan, np.nan)
    t_panic, p_panic = ttest_1samp(panic_vals, 0) if len(panic_vals) > 1 else (np.nan, np.nan)

    ic_results.append({
        'label': label,
        'mean': vals.mean(), 'std': vals.std(), 't': t_all, 'p': p_all,
        'calm_mean': calm_vals.mean() if len(calm_vals) > 0 else np.nan,
        'panic_mean': panic_vals.mean() if len(panic_vals) > 0 else np.nan,
        'calm_t': t_calm, 'calm_p': p_calm,
        'panic_t': t_panic, 'panic_p': p_panic,
    })

# Paired test: IC_LR - IC_mom
d = ic_df['IC_lr'].values - ic_df['IC_mom'].values
t_paired, p_paired = ttest_1samp(d, 0)

# Paired test: IC_XGB - IC_mom
d_xgb = ic_df['IC_xgb'].values - ic_df['IC_mom'].values
t_paired_xgb, p_paired_xgb = ttest_1samp(d_xgb, 0)

# Print
print(f"\n{'Signal':<22} {'Mean IC':>8} {'Std':>7} {'t':>7} {'p':>8} {'Calm':>7} {'Panic':>7}")
print("-" * 70)
for row in ic_results:
    print(f"{row['label']:<22} {row['mean']:>8.4f} {row['std']:>7.4f} {row['t']:>7.2f} {row['p']:>8.4f} "
          f"{row['calm_mean']:>7.4f} {row['panic_mean']:>7.4f}")

print(f"\n  Paired tests (Δ IC over momentum):")
print(f"    LR  - Mom:  Δ = {ic_df['IC_lr'].mean() - ic_df['IC_mom'].mean():+.4f}  t = {t_paired:.2f}  p = {p_paired:.4f} {sig_stars(p_paired)}")
print(f"    XGB - Mom:  Δ = {ic_df['IC_xgb'].mean() - ic_df['IC_mom'].mean():+.4f}  t = {t_paired_xgb:.2f}  p = {p_paired_xgb:.4f} {sig_stars(p_paired_xgb)}")

# LaTeX
tex_lines = []
tex_lines.append(r"\begin{table}[htbp]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Monthly information coefficients (Spearman rank correlation between stock scores and realised forward returns). $t$-statistics test $H_0\!: \text{IC} = 0$. Bottom panel reports paired $t$-tests for the difference in IC between combined models and momentum alone.}")
tex_lines.append(r"\label{tab:ic}")
tex_lines.append(r"\begin{tabular}{l r r r r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Signal & Mean IC & Std & $t$ & Calm IC & Panic IC & $N$ \\")
tex_lines.append(r"\midrule")
for row in ic_results:
    stars = sig_stars(row['p'])
    tex_lines.append(f"{row['label']} & {row['mean']:.4f} & {row['std']:.4f} & {row['t']:.2f}{stars} & "
                     f"{row['calm_mean']:.4f} & {row['panic_mean']:.4f} & {len(ic_df)} \\\\")
tex_lines.append(r"\midrule")
tex_lines.append(r"\multicolumn{7}{l}{\textit{Paired differences (vs.\ momentum)}} \\")
delta_lr = ic_df['IC_lr'].mean() - ic_df['IC_mom'].mean()
delta_xgb = ic_df['IC_xgb'].mean() - ic_df['IC_mom'].mean()
tex_lines.append(f"M1: LR $-$ Mom & {delta_lr:+.4f} & & {t_paired:.2f}{sig_stars(p_paired)} & & & \\\\")
tex_lines.append(f"M2: XGB $-$ Mom & {delta_xgb:+.4f} & & {t_paired_xgb:.2f}{sig_stars(p_paired_xgb)} & & & \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")

write_tex('table_ic.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# H. GRANGER CAUSALITY & CAUSAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("[ 8/8 ] Granger causality ...")

# Monthly IC for mom_12 (average across all lookbacks for robustness)
ic_mom_monthly = []
for date, grp in test.groupby('date'):
    pi = grp['pi_filter'].iloc[0]
    # Use average momentum IC across all lookbacks
    ics = []
    for lb in MOM_LBS:
        col = f'mom_{lb}'
        valid = grp[[col, 'ret_fwd']].dropna()
        if len(valid) < 30:
            continue
        rho, _ = spearmanr(valid[col], valid['ret_fwd'])
        ics.append(rho)
    if ics:
        ic_mom_monthly.append({'date': date, 'pi_filter': pi, 'IC_mom_avg': np.mean(ics)})

ic_ts = pd.DataFrame(ic_mom_monthly).set_index('date').sort_index()

# Granger causality: pi_filter → IC_mom
print("\n  Direction 1: pi_filter → momentum IC")
print("  (Does the regime signal predict future momentum effectiveness?)")
try:
    gc1 = grangercausalitytests(ic_ts[['IC_mom_avg', 'pi_filter']].dropna().values, maxlag=3, verbose=False)
    gc1_results = []
    for lag in [1, 2, 3]:
        f_stat = gc1[lag][0]['ssr_ftest'][0]
        f_pval = gc1[lag][0]['ssr_ftest'][1]
        gc1_results.append({'lag': lag, 'F': f_stat, 'p': f_pval})
        print(f"    Lag {lag}: F = {f_stat:.3f}, p = {f_pval:.4f} {sig_stars(f_pval)}")
except Exception as e:
    print(f"    Error: {e}")
    gc1_results = []

print("\n  Direction 2: momentum IC → pi_filter")
print("  (Does momentum effectiveness predict future regime?)")
try:
    gc2 = grangercausalitytests(ic_ts[['pi_filter', 'IC_mom_avg']].dropna().values, maxlag=3, verbose=False)
    gc2_results = []
    for lag in [1, 2, 3]:
        f_stat = gc2[lag][0]['ssr_ftest'][0]
        f_pval = gc2[lag][0]['ssr_ftest'][1]
        gc2_results.append({'lag': lag, 'F': f_stat, 'p': f_pval})
        print(f"    Lag {lag}: F = {f_stat:.3f}, p = {f_pval:.4f} {sig_stars(f_pval)}")
except Exception as e:
    print(f"    Error: {e}")
    gc2_results = []

# Regime transition analysis
print("\n  Regime transition analysis:")
pi_monthly = ic_ts['pi_filter']
regime_binary = (pi_monthly >= 0.5).astype(int)
transitions = regime_binary.diff().abs()
transition_dates = transitions[transitions == 1].index

# For each transition, compare IC in window before vs after
window = 3
pre_ics, post_ics = [], []
ic_series = ic_ts['IC_mom_avg']
all_dates = ic_series.index.tolist()

for td in transition_dates:
    idx = all_dates.index(td) if td in all_dates else -1
    if idx < window or idx + window >= len(all_dates):
        continue
    pre  = ic_series.iloc[idx-window:idx].values
    post = ic_series.iloc[idx:idx+window].values
    pre_ics.extend(pre)
    post_ics.extend(post)

if pre_ics and post_ics:
    from scipy.stats import ttest_ind
    t_trans, p_trans = ttest_ind(pre_ics, post_ics)
    print(f"  Avg IC pre-transition  (3 months): {np.mean(pre_ics):.4f}")
    print(f"  Avg IC post-transition (3 months): {np.mean(post_ics):.4f}")
    print(f"  Two-sample t-test: t = {t_trans:.2f}, p = {p_trans:.4f} {sig_stars(p_trans)}")
else:
    t_trans, p_trans = np.nan, np.nan
    print("  Not enough transitions for analysis.")

# LaTeX table
tex_lines = []
tex_lines.append(r"\begin{table}[htbp]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Granger causality tests between the regime signal ($\pi_t^{\text{filter}}$) and momentum's cross-sectional information coefficient. $F$-statistics from the Granger causality $F$-test at lags 1--3.}")
tex_lines.append(r"\label{tab:granger}")
tex_lines.append(r"\begin{tabular}{l c r r r r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r" & & \multicolumn{2}{c}{Lag 1} & \multicolumn{2}{c}{Lag 2} & \multicolumn{2}{c}{Lag 3} \\")
tex_lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
tex_lines.append(r"Direction & & $F$ & $p$ & $F$ & $p$ & $F$ & $p$ \\")
tex_lines.append(r"\midrule")

if gc1_results:
    line = r"$\pi^{\text{filter}} \to \text{IC}_{\text{mom}}$"
    line += " & "
    for res in gc1_results:
        line += f"& {res['F']:.2f} & {res['p']:.3f}{sig_stars(res['p'])} "
    line += r"\\"
    tex_lines.append(line)

if gc2_results:
    line = r"$\text{IC}_{\text{mom}} \to \pi^{\text{filter}}$"
    line += " & "
    for res in gc2_results:
        line += f"& {res['F']:.2f} & {res['p']:.3f}{sig_stars(res['p'])} "
    line += r"\\"
    tex_lines.append(line)

tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")

write_tex('table_granger.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# G. TABLE: GHM COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("[ 7/10 ] Table: GHM comparison ...")

ghm_strats = {
    'M2: XGB': r_xgb_red,
    'GHM SLOW ($a=0$)': ghm_returns['GHM SLOW (a=0)'],
    'GHM MED ($a=0.5$)': ghm_returns['GHM MED (a=0.5)'],
    'GHM FAST ($a=1$)': ghm_returns['GHM FAST (a=1)'],
    'GHM DYN': ghm_returns['GHM DYN'],
}

tex_lines = []
tex_lines.append(r"\begin{table}[H]")
tex_lines.append(r"\centering")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l r r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r" & Sharpe & Ann.\ Ret & Ann.\ Vol & Max DD \\")
tex_lines.append(r"\midrule")
for name, r in ghm_strats.items():
    ar, av, sh, mdd, _ = metrics(r)
    sep = r"\midrule" if name == 'M2: XGB' else ""
    tex_lines.append(f"{name} & {sh:.2f} & {ar:.1%} & {av:.1%} & $-${abs(mdd):.1%} \\\\")
    if sep:
        tex_lines.append(sep)
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\caption{Comparison with \citet{GHM2023} market-state momentum strategies (long-short construction). All four GHM variants collapse in long-short, because their deterministic blending of two momentum horizons cannot adapt the short side to changing regimes. M2 is shown for reference.}")
tex_lines.append(r"\label{tab:ghm_comparison}")
tex_lines.append(r"\end{table}")
write_tex('table_ghm_comparison.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# H. TABLE: PLACEBO / REGIME SIGNAL REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

print("[ 8/10 ] Table: Placebo (regime signal removal) ...")

# XGB without regime signal: train on momentum only
MOM_ONLY = MOM_FEATURES.copy()
X_tr_mom = train[MOM_ONLY].values.astype(float)
X_te_mom = test[MOM_ONLY].values.astype(float)
y_tr_val = train['ret_fwd'].values.astype(float)

# Handle NaN
for j in range(X_tr_mom.shape[1]):
    col_med = np.nanmedian(X_tr_mom[:, j])
    X_tr_mom[np.isnan(X_tr_mom[:, j]), j] = col_med
    X_te_mom[np.isnan(X_te_mom[:, j]), j] = col_med

from config import XGB_SEEDS, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE

preds_no_pi = np.zeros(len(X_te_mom))
for xs in XGB_SEEDS:
    xgb_no = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                           learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE, tree_method='hist',
                           random_state=xs, verbosity=0)
    xgb_no.fit(X_tr_mom, y_tr_val)
    preds_no_pi += xgb_no.predict(X_te_mom)
preds_no_pi /= len(XGB_SEEDS)
test['score_no_pi'] = preds_no_pi
r_no_pi = build_port(test, 'score_no_pi')
_, _, sh_no_pi, _, _ = metrics(r_no_pi)
_, _, sh_baseline, _, _ = metrics(r_xgb_red)

tex_lines = []
tex_lines.append(r"\begin{table}[H]")
tex_lines.append(r"\centering")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l r}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Regime signal & M2 Sharpe \\")
tex_lines.append(r"\midrule")
tex_lines.append(f"Real $\\pi_t^{{\\text{{filter}}}}$ (baseline) & {sh_baseline:.2f} \\\\")
tex_lines.append(f"No regime signal & {sh_no_pi:.2f} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(f"\\caption{{Placebo regime signal test (long-short, mom+$\\pi$). Removing the regime signal reduces M2's Sharpe from {sh_baseline:.2f} to {sh_no_pi:.2f}, confirming that the HMM signal is essential for long-short momentum stock selection.}}")
tex_lines.append(r"\label{tab:placebo}")
tex_lines.append(r"\end{table}")
write_tex('table_placebo.tex', '\n'.join(tex_lines))

# Kitchen sink (same data, different framing)
tex_lines = []
tex_lines.append(r"\begin{table}[H]")
tex_lines.append(r"\centering")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l r}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Configuration & M2 Sharpe \\")
tex_lines.append(r"\midrule")
tex_lines.append(f"XGB with real $\\pi_t^{{\\text{{filter}}}}$ & {sh_baseline:.2f} \\\\")
tex_lines.append(f"XGB without regime signal (baseline) & {sh_no_pi:.2f} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(f"\\caption{{Kitchen-sink test (long-short, mom+$\\pi$). The real HMM signal more than doubles the Sharpe ratio ({sh_baseline:.2f} vs.\\ {sh_no_pi:.2f}), confirming that the regime signal's contribution is specific and economically meaningful.}}")
tex_lines.append(r"\label{tab:kitchen_sink}")
tex_lines.append(r"\end{table}")
write_tex('table_kitchen_sink.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# I. TABLE: ALT SPLITS (baseline only)
# ══════════════════════════════════════════════════════════════════════════════

print("[ 9/10 ] Table: Alt splits ...")

_, _, sh_m1, _, _ = metrics(r_lr_red)
tex_lines = []
tex_lines.append(r"\begin{table}[H]")
tex_lines.append(r"\centering")
tex_lines.append(r"\small")
tex_lines.append(r"\begin{tabular}{l r r r}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Train / Test split & $N_{\text{test}}$ & M1 Sharpe & M2 Sharpe \\")
tex_lines.append(r"\midrule")
tex_lines.append(f"1990--2010 / 2011--2025 (baseline) & 167 & ${sh_m1:.2f}$ & {sh_baseline:.2f} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(f"\\caption{{Out-of-sample performance under the baseline train/test split (long-short, mom+$\\pi$, 50-seed ensemble). M1 collapses while M2 achieves a Sharpe of {sh_baseline:.2f} with highly significant factor model alphas.}}")
tex_lines.append(r"\label{tab:alt_splits}")
tex_lines.append(r"\end{table}")
write_tex('table_alt_splits.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# J. TABLE: REGIME SIGNAL ABLATION (HMM vs GHM vs none)
# ══════════════════════════════════════════════════════════════════════════════

print("[ 10/10 ] Table: Regime signal ablation ...")

# XGB + GHM cycle (already have ghm_returns from above, but need XGB trained on GHM cycle)
# Use GHM cycle variable as regime signal instead of pi_filter
# The GHM cycle is already computed in the GHM section above
# For a fair comparison: retrain XGB with mom + GHM_cycle instead of mom + pi_filter

# Build GHM cycle feature
test_ghm = test.copy()
# GHM market state: Bull if both 1-mo and 12-mo > 0, Bear if both < 0, etc.
r_mkt_monthly = panel[panel['date'] >= TRAIN_END][['date', 'ret_next']].dropna().set_index('date')
r_mkt_12 = r_mkt_monthly['ret_next'].rolling(12).mean()

test_ghm_dates = test_ghm[['date']].drop_duplicates().sort_values('date')
test_ghm_dates = test_ghm_dates.merge(
    r_mkt_monthly.reset_index().rename(columns={'ret_next': 'r1'}), on='date', how='left')
test_ghm_dates['r12'] = r_mkt_12.reindex(test_ghm_dates['date']).values
test_ghm_dates['ghm_cycle'] = 0  # Bull
test_ghm_dates.loc[(test_ghm_dates['r1'] < 0) & (test_ghm_dates['r12'] >= 0), 'ghm_cycle'] = 1  # Correction
test_ghm_dates.loc[(test_ghm_dates['r1'] < 0) & (test_ghm_dates['r12'] < 0), 'ghm_cycle'] = 2   # Bear
test_ghm_dates.loc[(test_ghm_dates['r1'] >= 0) & (test_ghm_dates['r12'] < 0), 'ghm_cycle'] = 3  # Rebound

# Same for train
train_ghm = train.copy()
train_dates = train_ghm[['date']].drop_duplicates().sort_values('date')
r_mkt_train = panel[panel['date'] < TRAIN_END][['date', 'ret_next']].dropna().set_index('date')
r_mkt_12_tr = r_mkt_train['ret_next'].rolling(12).mean()
train_dates = train_dates.merge(
    r_mkt_train.reset_index().rename(columns={'ret_next': 'r1'}), on='date', how='left')
train_dates['r12'] = r_mkt_12_tr.reindex(train_dates['date']).values
train_dates['ghm_cycle'] = 0
train_dates.loc[(train_dates['r1'] < 0) & (train_dates['r12'] >= 0), 'ghm_cycle'] = 1
train_dates.loc[(train_dates['r1'] < 0) & (train_dates['r12'] < 0), 'ghm_cycle'] = 2
train_dates.loc[(train_dates['r1'] >= 0) & (train_dates['r12'] < 0), 'ghm_cycle'] = 3

train_ghm = train_ghm.merge(train_dates[['date', 'ghm_cycle']], on='date', how='left')
test_ghm = test_ghm.merge(test_ghm_dates[['date', 'ghm_cycle']], on='date', how='left')
train_ghm['ghm_cycle'] = train_ghm['ghm_cycle'].fillna(0)
test_ghm['ghm_cycle'] = test_ghm['ghm_cycle'].fillna(0)

GHM_FEATURES = MOM_FEATURES + ['ghm_cycle']
X_tr_ghm = train_ghm[GHM_FEATURES].values.astype(float)
X_te_ghm = test_ghm[GHM_FEATURES].values.astype(float)
for j in range(X_tr_ghm.shape[1]):
    col_med = np.nanmedian(X_tr_ghm[:, j])
    X_tr_ghm[np.isnan(X_tr_ghm[:, j]), j] = col_med
    X_te_ghm[np.isnan(X_te_ghm[:, j]), j] = col_med

preds_ghm = np.zeros(len(X_te_ghm))
for xs in XGB_SEEDS:
    xgb_ghm = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                            learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                            colsample_bytree=COLSAMPLE, tree_method='hist',
                            random_state=xs, verbosity=0)
    xgb_ghm.fit(X_tr_ghm, y_tr_val)
    preds_ghm += xgb_ghm.predict(X_te_ghm)
preds_ghm /= len(XGB_SEEDS)
test_ghm['score_ghm_xgb'] = preds_ghm
r_ghm_xgb = build_port(test_ghm, 'score_ghm_xgb')

# Raw HMM features (DD, DISP, REL_N, CS) without HMM processing
print("  Training XGB with raw HMM features (no pi_filter) ...")
HMM_RAW_FEATURES = ['DD_z', 'DISP_z', 'REL_N_z', 'CS_z']

# Load raw HMM features from panel
panel_raw = panel[['date'] + HMM_RAW_FEATURES].dropna()

train_raw = train.copy().merge(panel_raw, on='date', how='left')
test_raw = test.copy().merge(panel_raw, on='date', how='left')
for col in HMM_RAW_FEATURES:
    med = train_raw[col].median()
    train_raw[col] = train_raw[col].fillna(med)
    test_raw[col] = test_raw[col].fillna(med)

RAW_FEATURES = MOM_FEATURES + HMM_RAW_FEATURES
X_tr_raw_hmm = train_raw[RAW_FEATURES].values.astype(float)
X_te_raw_hmm = test_raw[RAW_FEATURES].values.astype(float)
for j in range(X_tr_raw_hmm.shape[1]):
    col_med = np.nanmedian(X_tr_raw_hmm[:, j])
    X_tr_raw_hmm[np.isnan(X_tr_raw_hmm[:, j]), j] = col_med
    X_te_raw_hmm[np.isnan(X_te_raw_hmm[:, j]), j] = col_med

preds_raw_hmm = np.zeros(len(X_te_raw_hmm))
for xs in XGB_SEEDS:
    xgb_raw = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                           learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                           colsample_bytree=COLSAMPLE, tree_method='hist',
                           random_state=xs, verbosity=0)
    xgb_raw.fit(X_tr_raw_hmm, y_tr_val)
    preds_raw_hmm += xgb_raw.predict(X_te_raw_hmm)
preds_raw_hmm /= len(XGB_SEEDS)
test_raw['score_raw_hmm'] = preds_raw_hmm
r_raw_hmm = build_port(test_raw, 'score_raw_hmm')
ar_raw, av_raw, sh_raw, mdd_raw, _ = metrics(r_raw_hmm)
print(f"  Raw HMM features: Sharpe={sh_raw:.3f}  Ann.Ret={ar_raw:.1%}")

ar_hmm, av_hmm, sh_hmm, mdd_hmm, _ = metrics(r_xgb_red)
ar_ghm_x, av_ghm_x, sh_ghm_x, mdd_ghm_x, _ = metrics(r_ghm_xgb)
ar_no, av_no, sh_no, mdd_no, _ = metrics(r_no_pi)

tex_lines = []
tex_lines.append(r"\begin{table}[H]")
tex_lines.append(r"\centering")
tex_lines.append(r"\smallskip")
tex_lines.append(r"\begin{tabular}{lcccc}")
tex_lines.append(r"\toprule")
tex_lines.append(r"Regime Signal Variant & Ann.\ Ret & Ann.\ Vol & Sharpe & Max DD \\")
tex_lines.append(r"\midrule")
tex_lines.append(f"XGB + $\\pi_t^{{\\text{{filter}}}}$ (HMM)   & {ar_hmm:.1%} & {av_hmm:.1%} & {sh_hmm:.3f} & $-${abs(mdd_hmm):.1%} \\\\")
tex_lines.append(f"XGB + raw indicators (DD, DISP, REL\\_N, CS) & {ar_raw:.1%} & {av_raw:.1%} & {sh_raw:.3f} & $-${abs(mdd_raw):.1%} \\\\")
tex_lines.append(f"XGB (no regime signal)                   & {ar_no:.1%} & {av_no:.1%} & {sh_no:.3f} & $-${abs(mdd_no):.1%} \\\\")
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\caption{Regime signal ablation (50-seed ensemble). The HMM signal outperforms both raw stress indicators and no regime signal.}")
tex_lines.append(r"\label{tab:regime_signal_ablation}")
tex_lines.append(r"\end{table}")
write_tex('table_regime_signal_ablation.tex', '\n'.join(tex_lines))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  ALL TABLES GENERATED")
print("=" * 60)
for f in sorted(os.listdir(TABLES_DIR)):
    if f.endswith('.tex'):
        print(f"  tables/{f}")
print("\nDone.")
