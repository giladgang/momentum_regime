"""
fundamentals_test.py
====================
Production-level fundamentals ablation for M2 (XGBoost, 50 seeds).

Retrains 5 XGB configurations on the full 1990-2010 training sample and
evaluates them out-of-sample on 2011-onwards. Reports the same metrics the
production pipeline reports for M2 (Ann.Ret, Ann.Vol, Sharpe, MaxDD, beta,
Newey-West t, Final $), plus turnover, regime-conditional Sharpe, sub-period
Sharpe, factor alphas (CAPM/FF3/Carhart/FF5/FF6), and a block-bootstrap
Sharpe confidence interval for the baseline and full variants.

Outputs
-------
results/fundamentals_test_results.csv          — scalar metrics, one row per variant
results/fundamentals_returns.pkl               — dict[name] -> returns Series
tables/table_performance_fund_row.tex          — LaTeX fragment, \input{}-able
                                                 into table_performance.tex
tables/table_fund_alphas.tex                   — factor alphas for the fund variant

Usage:
    python -u scripts/fundamentals_test.py
"""

import numpy as np
import pandas as pd
import pickle, sys, os, time
from xgboost import XGBRegressor
import statsmodels.api as sm
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
                    COLSAMPLE, XGB_SEEDS, TRADING_FEE, TRAIN_END,
                    SUB_PERIODS)

np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
stocks = pd.read_parquet('data/crsp_msf_raw.parquet')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks = stocks.sort_values(['permno', 'date']).reset_index(drop=True)
stocks = stocks[stocks['shrcd'].isin([10, 11])]
stocks = stocks[stocks['exchcd'].isin([1, 2, 3])]
stocks = stocks[stocks['prc'].abs() > 1.0]
stocks = stocks.reset_index(drop=True)

regimes = pd.read_parquet('data/panel_with_regimes.parquet')[['date', 'pi_filter']]
regimes['date'] = pd.to_datetime(regimes['date'])
stocks = stocks.merge(regimes, on='date', how='left')
stocks['pi_filter'] = stocks['pi_filter'].ffill()

# Momentum
print("Computing momentum ...")
stocks['_log_ret'] = np.log1p(stocks['ret_adj'].clip(lower=-0.999))
stocks['_log_ret_s1'] = stocks.groupby('permno')['_log_ret'].shift(1)
for lb in range(1, 13):
    roll_sum = (
        stocks.groupby('permno', sort=False)['_log_ret_s1']
        .rolling(lb, min_periods=lb).sum()
        .reset_index(level='permno', drop=True).sort_index()
    )
    stocks[f'mom_{lb}'] = np.expm1(roll_sum)
stocks.drop(columns=['_log_ret', '_log_ret_s1'], inplace=True)

# Size
stocks['log_me'] = np.log(
    stocks.groupby('permno')['me'].transform(lambda x: x.shift(1)).replace(0, np.nan)
)

# Target
stocks['ret_fwd'] = stocks.groupby('permno')['ret_adj'].transform(lambda x: x.shift(-1))

# Feature sets
MOM_FEATURES = [f'mom_{lb}' for lb in range(1, 13)]
FUND_FEATURES = ['bm', 'roe', 'earnings_growth', 'leverage', 'asset_growth',
                 'gross_profit_a', 'log_me',
                 'cfo_a', 'fcf_a', 'accruals']

FEATURES_BASE = MOM_FEATURES + ['pi_filter']
FEATURES_FUND = MOM_FEATURES + ['pi_filter'] + FUND_FEATURES

# Drop rows missing core features (XGB handles NaN fundamentals natively)
CORE = MOM_FEATURES + ['pi_filter', 'log_me']
df = stocks.dropna(subset=['ret_fwd'] + CORE).copy().reset_index(drop=True)

train = df[df['date'] < TRAIN_END].copy()
test = df[df['date'] >= TRAIN_END].copy()

print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

# Market return — prefer production r_mkt (CRSP VW, in cs_artefacts_data.pkl)
# so that beta / NW t / factor alphas are directly comparable to the main
# performance table. Fall back to value-weighting the filtered test-month
# cross-section if the artefacts pickle is not available.
_ART = 'artefacts/cs_artefacts_data.pkl'
if os.path.exists(_ART):
    with open(_ART, 'rb') as _f:
        r_mkt = pickle.load(_f)['r_mkt']
    print(f"  r_mkt loaded from production artefacts ({len(r_mkt)} months)")
else:
    r_mkt = (test.groupby('date')
                 .apply(lambda g: (g['ret_fwd'] * g['me']).sum() / g['me'].sum())
                 .rename('r_mkt'))
    print(f"  r_mkt computed from test cross-section ({len(r_mkt)} months)")

# Monthly pi_filter at the test-period month level (for regime split)
pi_monthly = test.groupby('date')['pi_filter'].first()

# Fama-French factors for factor alpha regressions
ff = pd.read_parquet('data/ff_factors.parquet')


# ── Portfolio + metric helpers ─────────────────────────────────────────────

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    """Returns (monthly_return_series, monthly_turnover_series)."""
    monthly, turnover = [], []
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
        lme = longs['me'].sum()
        new_lw = (longs.set_index('permno')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('permno')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(p, 0) - prev_lw.get(p, 0))
                 for p in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(p, 0) - prev_sw.get(p, 0))
                 for p in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        turnover.append({'date': date, 'turnover': (tl + ts) / 2})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    r = pd.DataFrame(monthly).set_index('date')['ret']
    t = pd.DataFrame(turnover).set_index('date')['turnover']
    return r, t


def full_metrics(r, r_mkt, pi_series):
    """Sharpe/Ret/Vol/MDD/beta/NW t/Final $ + regime + sub-period Sharpe."""
    r = pd.Series(r).dropna()
    n = len(r)
    ann_ret = (1 + r).prod() ** (12 / n) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0.0
    cum = (1 + r).cumprod()
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    final = float(cum.iloc[-1])

    # Beta vs market
    common = r.index.intersection(r_mkt.index)
    if len(common) > 12:
        a, b = r.loc[common].values, r_mkt.loc[common].values
        valid = ~(np.isnan(a) | np.isnan(b))
        cov = np.cov(a[valid], b[valid])
        beta = float(cov[0, 1] / cov[1, 1])
    else:
        beta = np.nan

    # Newey-West t on raw return (H0: mean = 0). Matches the thesis
    # convention used in table_performance.tex (main_results_analysis.py).
    # For market-neutral L/S strategies the standard test is mean = 0; the
    # market exposure is controlled separately in the factor-alpha table.
    nw = sm.OLS(r.values, np.ones((n, 1))).fit(cov_type='HAC',
                                                 cov_kwds={'maxlags': 6})
    nw_t, nw_p = float(nw.tvalues[0]), float(nw.pvalues[0])

    # Mean-excess-vs-market t-stat retained for diagnostic comparison only;
    # NOT used in any published table. Test C/D (factor regression) is the
    # right way to control for market exposure (see compute_factor_alphas).
    common = r.index.intersection(r_mkt.index)
    if len(common) > 12:
        excess = r.loc[common].values - r_mkt.loc[common].values
        valid = ~np.isnan(excess)
        nw_ex = sm.OLS(excess[valid], np.ones((valid.sum(), 1))).fit(
            cov_type='HAC', cov_kwds={'maxlags': 6})
        nw_t_excess = float(nw_ex.tvalues[0])
        nw_p_excess = float(nw_ex.pvalues[0])
    else:
        nw_t_excess, nw_p_excess = np.nan, np.nan

    # Regime-conditional Sharpe
    pi_aligned = pi_series.reindex(r.index)
    calm = r[pi_aligned < 0.5]
    panic = r[pi_aligned >= 0.5]
    sh_calm = calm.mean() / calm.std() * np.sqrt(12) if calm.std() > 0 else np.nan
    sh_panic = panic.mean() / panic.std() * np.sqrt(12) if panic.std() > 0 else np.nan

    # Sub-period Sharpe
    sub_sharpes = {}
    for name, start, end in SUB_PERIODS:
        if name == 'Full':
            continue
        sub = r[(r.index >= start) & (r.index < end)]
        sub_sharpes[f'sharpe_{name}'] = (
            sub.mean() / sub.std() * np.sqrt(12) if len(sub) > 0 and sub.std() > 0
            else np.nan
        )

    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        mdd=mdd, final=final, beta=beta,
        nw_t=nw_t, nw_p=nw_p,
        nw_t_excess=nw_t_excess, nw_p_excess=nw_p_excess,
        sharpe_calm=sh_calm, sharpe_panic=sh_panic,
        n_months=n, n_calm=int(calm.notna().sum()), n_panic=int(panic.notna().sum()),
        **sub_sharpes,
    )


def factor_alphas(r):
    """CAPM, FF3, Carhart, FF5, FF6 alphas and t-stats (Newey-West, 6 lags)."""
    r_df = r.to_frame('ret')
    r_df.index = r_df.index + pd.offsets.MonthEnd(0)
    merged = r_df.join(ff[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'UMD']],
                       how='inner')
    if len(merged) < 24:
        return {}
    y = merged['ret'].values
    results = {}
    for mname, cols in [
        ('CAPM', ['Mkt-RF']),
        ('FF3', ['Mkt-RF', 'SMB', 'HML']),
        ('Carhart', ['Mkt-RF', 'SMB', 'HML', 'UMD']),
        ('FF5', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']),
        ('FF6', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD']),
    ]:
        X = sm.add_constant(merged[cols].values)
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
        results[mname] = dict(
            alpha=float(res.params[0] * 12),
            t=float(res.tvalues[0]),
            p=float(res.pvalues[0]),
            betas={c: float(res.params[j + 1]) for j, c in enumerate(cols)},
        )
    return results


def block_bootstrap_sharpe_ci(r, n_boot=10_000, block=12, seed=42):
    """Stationary-block bootstrap (12-month blocks) 95% CI for annualised Sharpe."""
    r = pd.Series(r).dropna().values
    n = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        # Concatenate blocks and truncate to n
        idx = np.concatenate([np.arange(s, s + block) for s in starts[i]])[:n]
        sub = r[idx]
        sharpes[i] = sub.mean() / sub.std() * np.sqrt(12) if sub.std() > 0 else 0.0
    lo, hi = np.percentile(sharpes, [2.5, 97.5])
    return float(lo), float(hi)


# ── Train and evaluate one variant ─────────────────────────────────────────

def train_and_eval(name, features, train_df, test_df, want_alphas=False,
                    want_bootstrap=False):
    X_tr = train_df[features].values.astype(float)
    X_te = test_df[features].values.astype(float)
    y_tr = train_df['ret_fwd'].values.astype(float)

    print(f"\n  {name} ({len(features)} features, {len(XGB_SEEDS)} seeds) ...")
    t0 = time.time()
    preds = np.zeros(len(X_te))
    last_model = None
    for xs in XGB_SEEDS:
        model = XGBRegressor(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE, tree_method='hist',
            random_state=xs, verbosity=0,
        )
        model.fit(X_tr, y_tr)
        preds += model.predict(X_te)
        last_model = model
    preds /= len(XGB_SEEDS)

    score_col = f'score_{name}'
    test_df[score_col] = preds
    r, turnover_s = long_short_port(test_df, score_col)

    m = full_metrics(r, r_mkt, pi_monthly)

    # SHAP
    explainer = shap.TreeExplainer(last_model)
    sv = explainer.shap_values(X_te)
    abs_shap = np.abs(sv).mean(axis=0)
    total = abs_shap.sum()
    pi_share = (abs_shap[features.index('pi_filter')] / total * 100
                if 'pi_filter' in features else 0.0)
    mom_share = sum(abs_shap[features.index(f)] for f in MOM_FEATURES
                    if f in features) / total * 100
    fund_share = sum(abs_shap[features.index(f)] for f in FUND_FEATURES
                     if f in features) / total * 100
    feat_shap = {features[i]: abs_shap[i] / total * 100 for i in range(len(features))}

    # Turnover
    avg_turnover = float(turnover_s.mean()) if len(turnover_s) else np.nan

    # Factor alphas (optional — heavy variants)
    alphas = factor_alphas(r) if want_alphas else {}

    # Bootstrap CI (optional — expensive)
    if want_bootstrap:
        ci_lo, ci_hi = block_bootstrap_sharpe_ci(r)
    else:
        ci_lo, ci_hi = np.nan, np.nan

    elapsed = time.time() - t0
    print(f"    Sharpe={m['sharpe']:.2f}  Ret={m['ann_ret']:.1%}  "
          f"Vol={m['ann_vol']:.1%}  MDD={m['mdd']:.1%}  "
          f"beta={m['beta']:.2f}  NWt={m['nw_t']:.2f}")
    print(f"    Regime: calm={m['sharpe_calm']:.2f}  panic={m['sharpe_panic']:.2f}")
    print(f"    SHAP: Mom={mom_share:.0f}%  Pi={pi_share:.0f}%  Fund={fund_share:.0f}%")
    print(f"    Turnover: {avg_turnover:.1%}/mo  Time: {elapsed:.0f}s")
    if alphas:
        for mn, d in alphas.items():
            stars = ('***' if d['p'] < 0.01 else '**' if d['p'] < 0.05
                     else '*' if d['p'] < 0.10 else '')
            print(f"    {mn:<8s}: alpha={d['alpha']:>6.1%}  t={d['t']:>5.2f}{stars}")
    if want_bootstrap:
        print(f"    Bootstrap Sharpe 95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")

    return {
        'name': name, 'n_features': len(features),
        'returns': r, 'turnover_series': turnover_s,
        'turnover_avg': round(avg_turnover, 4) if not np.isnan(avg_turnover) else np.nan,
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        'ci_lo': round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
        'ci_hi': round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
        'mom_share': round(mom_share, 1), 'pi_share': round(pi_share, 1),
        'fund_share': round(fund_share, 1),
        'alphas': alphas,
        **{f'shap_{k}': round(v, 2) for k, v in feat_shap.items()},
    }


# ── Run comparisons ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  FUNDAMENTALS ABLATION TEST (cfo_a, fcf_a, accruals added)")
print("=" * 72)

os.makedirs('results', exist_ok=True)
os.makedirs('tables', exist_ok=True)


def persist(results):
    """Idempotent save: called after every variant to avoid losing work.
    Handles heterogeneous feature sets (not every variant has every shap_ key)."""
    if not results:
        return
    # Scalar CSV — use pd.DataFrame on list of dicts (auto-NaN for missing keys)
    rows = [{k: v for k, v in r.items()
             if k not in ('returns', 'turnover_series', 'alphas')}
            for r in results]
    pd.DataFrame(rows).to_csv('results/fundamentals_test_results.csv', index=False)

    # Returns + turnover + alphas pickle
    artifact = {
        r['name']: {
            'returns': r['returns'],
            'turnover': r['turnover_series'],
            'alphas': r['alphas'],
        }
        for r in results
    }
    artifact['r_mkt'] = r_mkt
    artifact['pi_monthly'] = pi_monthly
    with open('results/fundamentals_returns.pkl', 'wb') as f:
        pickle.dump(artifact, f)


results = []

for name, feats, want_alphas, want_boot in [
    ('baseline_mom_pi',  FEATURES_BASE,                    True,  True),
    ('full_mom_pi_fund', FEATURES_FUND,                    True,  True),
    ('fund_only',        FUND_FEATURES,                    False, False),
    ('fund_pi',          FUND_FEATURES + ['pi_filter'],    False, False),
    ('mom_fund_no_pi',   MOM_FEATURES + FUND_FEATURES,     False, False),
]:
    results.append(train_and_eval(name, feats, train, test,
                                   want_alphas=want_alphas,
                                   want_bootstrap=want_boot))
    persist(results)  # checkpoint after every variant
    print(f"    [checkpoint] saved {len(results)} variant(s) to disk")

print(f"\nSaved: results/fundamentals_test_results.csv")
print(f"Saved: results/fundamentals_returns.pkl  ({len(results)} variants + r_mkt + pi)")

# 3. LaTeX fragment for table_performance.tex
#    Writes the mom+pi+fund row in the exact same column schema.
def sig_stars(p):
    if np.isnan(p): return ''
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

def fmt_beta(b):
    if np.isnan(b): return '--'
    return f'$-${abs(b):.2f}' if b < 0 else f'{b:.2f}'

fund_row = next(r for r in results if r['name'] == 'full_mom_pi_fund')
# Thesis convention: raw mean t-stat (H0: mean = 0), matching
# table_performance.tex. Market exposure is controlled in the factor-alpha
# table, not by ad-hoc subtraction.
nw_str = f"{fund_row['nw_t']:.2f}{sig_stars(fund_row['nw_p'])}"
def fmt_pct(x):
    return f"{x*100:.1f}\\%"
row = (f"M2: XGB (mom+$\\pi$+fund) & "
       f"{fmt_pct(fund_row['ann_ret'])} & "
       f"{fmt_pct(fund_row['ann_vol'])} & "
       f"{fund_row['sharpe']:.2f} & "
       f"{fmt_pct(fund_row['mdd'])} & "
       f"{fmt_beta(fund_row['beta'])} & "
       f"{nw_str} & "
       f"{fund_row['final']:.1f} \\\\")
with open('tables/table_performance_fund_row.tex', 'w') as f:
    f.write(row + '\n')
print(f"Saved: tables/table_performance_fund_row.tex")

# 4. LaTeX factor alpha table for the fund variant
tex = [r'\begin{table}[H]', r'\centering', r'\small',
       r'\begin{tabular}{l r r r r r r r r}', r'\toprule',
       r'Model & $\alpha$ (\%) & $t(\alpha)$ & Mkt-RF & SMB & HML & UMD & RMW & CMA \\',
       r'\midrule']
for mn, d in fund_row['alphas'].items():
    cells = [mn, f"{d['alpha']*100:.1f}", f"{d['t']:.2f}"]
    for fc in ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RMW', 'CMA']:
        cells.append(f"{d['betas'][fc]:+.2f}" if fc in d['betas'] else '')
    tex.append(' & '.join(cells) + r' \\')
tex.extend([r'\bottomrule', r'\end{tabular}',
            r'\caption{Factor model regressions for M2 (XGBoost, mom+$\pi$+fund). '
            r'$\alpha$ is annualised. Newey--West $t$-statistics (6 lags).}',
            r'\label{tab:fund_alphas}', r'\end{table}'])
with open('tables/table_fund_alphas.tex', 'w') as f:
    f.write('\n'.join(tex) + '\n')
print(f"Saved: tables/table_fund_alphas.tex")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"\n{'Config':<22s} {'Feats':>5s} {'Sharpe':>7s} {'Ret':>7s} {'Vol':>6s} "
      f"{'MDD':>7s} {'NWt_raw':>7s} {'NWt_exc':>7s} {'Calm':>6s} {'Panic':>6s} "
      f"{'Mom%':>5s} {'Pi%':>5s} {'Fund%':>5s}")
print("-" * 118)
for r in results:
    nw_raw = f"{r['nw_t']:.2f}{sig_stars(r['nw_p'])}"
    nw_exc = (f"{r['nw_t_excess']:.2f}{sig_stars(r['nw_p_excess'])}"
              if not np.isnan(r.get('nw_t_excess', np.nan)) else '--')
    print(f"{r['name']:<22s} {r['n_features']:>5d} {r['sharpe']:>7.2f} "
          f"{r['ann_ret']:>6.1%} {r['ann_vol']:>5.1%} {r['mdd']:>6.1%} "
          f"{nw_raw:>7s} {nw_exc:>7s} {r['sharpe_calm']:>6.2f} {r['sharpe_panic']:>6.2f} "
          f"{r['mom_share']:>5.1f} {r['pi_share']:>5.1f} {r['fund_share']:>5.1f}")
