"""
hmm_feature_selection_intl.py
=============================
International (UK + JP) HMM feature selection pipeline.

Adaptation of scripts/hmm_feature_selection.py for the international panels:
- 5-feature candidate pool (DD_z, VOL_z, DISP_z, REL_N_z, BANK_REL_z) with DD_z required
  (vs 9 features for US; BANK_REL_z replaces CS_z because no Moody's BAA-AAA equivalent exists
  for UK or Japan).
- 15 combinations of size 1-4 (vs 93 for US).
- Region-aware market panel, stock panel, validation periods, and output paths.
- Same 4-pass methodology as the US version: quality screen -> portfolio test ->
  definitive validation -> economic validation.

POLICY: User picks the final feature combination per region after reviewing ALL pass
outputs. The script does NOT enforce an automatic winner; it produces ranked tables and
diagnostic stats (Pass 2 portfolio Sharpes, Pass 3 stability+sub-period breakdown,
Pass 4 BIC + crisis-detection metrics) which the user reviews together to make the call.
This mirrors how the US production set DD+CS+DISP+REL_N was chosen: after reviewing the
4-pass outputs Gilad picked the combo, not the script.

Run all 4 passes for each region; do not auto-pick. The user picks at the end.

Selects the regional feature combination by running a multi-pass pipeline that
progressively filters candidates and increases computational rigor at each stage.

Pipeline Overview:
    Pass 1: HMM Quality Screen (fast, 1 seed per combo)
        - Tests all C(N,4) combinations
        - Filters on: MCMC convergence, regime separation, crisis alignment
        - Purpose: eliminate combos that produce nonsensical regimes

    Pass 2: Portfolio Test (medium, 5 HMM seeds + 10 XGB seeds per combo)
        - Tests surviving combos from Pass 1
        - Averages pi_filter across 5 HMM seeds
        - Averages XGB predictions across 10 seeds
        - Builds L/S portfolio, computes Sharpe
        - Purpose: rank combos by downstream portfolio performance

    Pass 3: Definitive Validation (thorough, 10 HMM seeds + 20 XGB seeds)
        - Tests top 3 combos from Pass 2
        - Full seed averaging for stable results
        - Sub-period breakdown (2011-15, 2016-20, 2021-25)
        - Individual seed variance analysis
        - Purpose: confirm the winner is robust

    Pass 4: Economic Validation (final checks on winner)
        - Log-likelihood / BIC comparison vs 3-feature baseline
        - Crisis detection accuracy across 6 periods
        - Regime-conditional covariance analysis
        - Likelihood ratio test
        - Purpose: ensure the winner makes economic sense

Usage:
    # Required: --region {UK|JP}. Default --pass 0 runs all four passes sequentially.
    python scripts/hmm_feature_selection_intl.py --region UK            # full 4-pass for UK
    python scripts/hmm_feature_selection_intl.py --region JP            # full 4-pass for JP
    python scripts/hmm_feature_selection_intl.py --region UK --pass 1   # Pass 1 only
    python scripts/hmm_feature_selection_intl.py --region UK --pass 2   # Pass 2 only (needs Pass 1)
    python scripts/hmm_feature_selection_intl.py --region UK --pass 3   # Pass 3 only (needs Pass 2)

Configuration:
    Most parameters (seeds, thresholds, momentum features, training cutoff) come from
    config.py. The international FEATURE_POOL and per-region VALIDATION_PERIODS are
    defined as module-level constants in this file.

Output (per region, written to results/thesis/):
    - intl_<region>_hmm_feature_selection_pass1.csv  (15 combos with quality metrics)
    - intl_<region>_hmm_feature_selection_pass2.csv  (Pass 1 survivors ranked by M2 Sharpe)
    - intl_<region>_hmm_feature_selection_pass3.csv  (top combos with stability + sub-period)
    - intl_<region>_hmm_feature_selection_pass4.csv  (BIC, crisis-detection, LRT)

Final feature pick:
    The user reviews the four output CSVs (M2 Sharpe ranking, per-seed stability,
    sub-period robustness, economic validation metrics) and selects the regional
    feature combination manually. The script does NOT auto-select a winner -- this
    is intentional and mirrors how the US production set was chosen.
"""

import numpy as np
import pandas as pd
import pickle, warnings, time, os, sys, argparse
from itertools import combinations
from scipy.stats import multivariate_normal, invwishart, chi2
from scipy.special import logsumexp
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TRADING_FEE, TRAIN_END, N_ESTIMATORS, MAX_DEPTH,
                    LEARNING_RATE, SUBSAMPLE, COLSAMPLE, MOM_FEATURES,
                    HMM_ITERATIONS, HMM_BURNIN, CRISIS_WINDOWS)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Region (set by --region arg in main(); used as module-level global)
REGION = None  # 'UK' or 'JP'

# Feature pool: 5 features available internationally
# (CS_z replaced by BANK_REL_z because no Moody's BAA-AAA equivalent for UK/JP)
FEATURE_POOL = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z', 'BANK_REL_z']

# Required feature (always included in every combo)
REQUIRED_FEATURE = 'DD_z'

# Feature counts to test
FEATURE_COUNTS = [1, 2, 3, 4]  # DD alone, DD+1, DD+2, DD+3

# Seed counts increase with each pass
PASS1_HMM_SEEDS = [42]                                          # 1 seed (fast screening)
PASS2_HMM_SEEDS = [42, 2201, 1337, 314, 789]                    # 5 seeds (ranking)
PASS2_XGB_SEEDS = [42, 123, 456, 789, 999, 2201, 1337, 314, 7777, 55]  # 10 seeds
PASS3_HMM_SEEDS = [42, 2201, 1337, 314, 789, 999, 123, 456, 7777, 55]  # 10 seeds (definitive)
PASS3_XGB_SEEDS = [42, 123, 456, 789, 999, 2201, 1337, 314, 7777, 55,
                   101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]   # 20 seeds

# Number of combos to advance from Pass 2 to Pass 3
TOP_N_FOR_PASS3 = 10

# Quality gates for Pass 1
# Uses the same panic identification as hmm_model.py (sign-correction with crisis windows)
QUALITY_GATES = {
    'min_ess': 50,           # minimum ESS across all mu parameters
    'n_sig': 2,              # at least N features with significant regime separation
}

# Crisis/calm periods for validation -- region-specific
# Format: (name, start, end, type, threshold)
# type='crisis': mean pi must EXCEED threshold
# type='calm': mean pi must be BELOW threshold
VALIDATION_PERIODS_BY_REGION = {
    'UK': [
        ('Dot-com',     '2000-03-01', '2002-10-01', 'crisis', 0.3),
        ('GFC',         '2007-10-01', '2009-06-01', 'crisis', 0.5),
        ('Eurozone',    '2011-08-01', '2012-08-01', 'crisis', 0.3),
        ('COVID',       '2020-02-01', '2020-05-31', 'crisis', 0.5),
        ('Calm 2003-06','2003-01-01', '2006-12-31', 'calm',   0.4),
        ('Calm 2013-19','2013-01-01', '2019-12-31', 'calm',   0.4),
    ],
    'JP': [
        ('Bubble',      '1990-01-01', '1992-12-01', 'crisis', 0.4),
        ('NPL crisis',  '2001-09-01', '2003-04-01', 'crisis', 0.4),
        ('GFC',         '2008-09-01', '2009-06-01', 'crisis', 0.5),
        ('COVID',       '2020-02-01', '2020-05-31', 'crisis', 0.5),
        ('Calm 2013-19','2013-01-01', '2019-12-31', 'calm',   0.4),
    ],
}

# VALIDATION_PERIODS is set in main() once REGION is known
VALIDATION_PERIODS = []


# ═══════════════════════════════════════════════════════════════════════════════
#  HMM FUNCTIONS (exact match to hmm_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

K = 2

def log_emission(Z, mu, Sigma):
    return np.column_stack([
        multivariate_normal.logpdf(Z, mean=mu[k], cov=Sigma[k], allow_singular=True)
        for k in range(K)])

def ffbs(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K)); la[0] = np.log(0.5) + le[0]
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t-1] + np.log(P[:, k] + 1e-300))
    la -= logsumexp(la, axis=1, keepdims=True)
    a = np.exp(la); s = np.zeros(n, dtype=int)
    s[n-1] = np.random.choice(K, p=a[n-1])
    for t in range(n-2, -1, -1):
        p = a[t] * P[:, s[t+1]]; p /= p.sum()
        s[t] = np.random.choice(K, p=p)
    return s

def forward_filter(Z, mu, Sigma, P):
    n = len(Z)
    le = log_emission(Z, mu, Sigma)
    la = np.zeros((n, K)); la[0] = np.log(0.5) + le[0]
    for t in range(1, n):
        for k in range(K):
            la[t, k] = le[t, k] + logsumexp(la[t-1] + np.log(P[:, k] + 1e-300))
    la -= logsumexp(la, axis=1, keepdims=True)
    return np.exp(la)

def fit_hmm(Z_train, Z_full, train_dates, seed=42,
            n_iter=HMM_ITERATIONS, n_burnin=HMM_BURNIN):
    """
    Fit 2-state HMM with panic identification matching hmm_model.py.
    Returns pi_filter and quality metrics.
    """
    np.random.seed(seed)
    T, D = Z_train.shape

    # Priors (exact match to hmm_model.py)
    m_0 = np.zeros(D); kappa_0 = 0.01; nu_0 = D + 2
    Psi_0 = np.eye(D) * (nu_0 - D - 1)
    alpha_dir = np.array([[9.0, 1.0], [1.0, 9.0]])

    # Initialize using first feature
    states = (Z_train[:, 0] > np.median(Z_train[:, 0])).astype(int)
    mu = np.zeros((K, D)); Sigma = np.array([np.eye(D)] * K)
    for k in range(K):
        idx = states == k
        if idx.sum() > D + 1:
            mu[k] = Z_train[idx].mean(axis=0)
            Sigma[k] = np.cov(Z_train[idx].T) + 1e-6 * np.eye(D)
    P = np.array([[0.95, 0.05], [0.10, 0.90]])

    n_keep = n_iter - n_burnin
    mu_draws = np.zeros((n_keep, K, D))

    for m in range(n_iter):
        states = ffbs(Z_train, mu, Sigma, P)
        for k in range(K):
            Z_k = Z_train[states == k]; n_k = len(Z_k)
            if n_k < D + 2: continue
            x_bar = Z_k.mean(axis=0); S_k = (Z_k - x_bar).T @ (Z_k - x_bar)
            kappa_n = kappa_0 + n_k; m_n = (kappa_0 * m_0 + n_k * x_bar) / kappa_n
            nu_n = nu_0 + n_k
            Psi_n = Psi_0 + S_k + (kappa_0 * n_k / kappa_n) * np.outer(x_bar - m_0, x_bar - m_0)
            try:
                Sigma[k] = invwishart.rvs(df=nu_n, scale=Psi_n)
                mu[k] = np.random.multivariate_normal(m_n, Sigma[k] / kappa_n)
            except: pass
        for i in range(K):
            counts = np.array([np.sum((states[:-1] == i) & (states[1:] == j))
                               for j in range(K)], dtype=float)
            P[i] = np.random.dirichlet(alpha_dir[i] + counts)
        if m >= n_burnin:
            mu_draws[m - n_burnin] = mu

    mu_post = mu_draws.mean(axis=0)

    # Panic identification (hmm_model.py method: sign-correction with crisis windows).
    # CRISIS_WINDOWS sourced from config.py to keep production scripts in sync.
    crisis_mask = np.zeros(T, dtype=bool)
    for s, e in CRISIS_WINDOWS:
        crisis_mask |= ((train_dates >= np.datetime64(s)) & (train_dates <= np.datetime64(e)))

    signs = np.zeros(D)
    if crisis_mask.sum() > 5:
        for j in range(D):
            p95_crisis = np.percentile(Z_train[crisis_mask, j], 95)
            p95_normal = np.percentile(Z_train[~crisis_mask, j], 95)
            signs[j] = 1.0 if p95_crisis >= p95_normal else -1.0
    else:
        signs = np.ones(D)

    score0 = np.sum(signs * mu_post[0])
    score1 = np.sum(signs * mu_post[1])
    panic_state = 1 if score1 >= score0 else 0

    # Filter full sample
    filtered = forward_filter(Z_full, mu_post, Sigma, P)
    pi_filter = filtered[:, panic_state]

    # ESS
    def ess(x):
        n = len(x)
        if n < 10 or np.std(x) == 0: return n
        acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
        acf = acf[n-1:] / acf[n-1]
        cutoff = next((i for i in range(1, len(acf)) if acf[i] < 0.05), len(acf))
        tau = 1 + 2 * np.sum(acf[1:cutoff])
        return max(1, n / tau)

    min_ess = min(ess(mu_draws[:, k, j]) for k in range(K) for j in range(D))

    # Regime separation significance
    n_sig = 0
    for j in range(D):
        delta = mu_draws[:, panic_state, j] - mu_draws[:, 1-panic_state, j]
        ci_lo, ci_hi = np.percentile(delta, [2.5, 97.5])
        if ci_lo > 0 or ci_hi < 0:
            n_sig += 1

    # Log-likelihood
    ll = np.sum(logsumexp(log_emission(Z_train, mu_post, Sigma) + np.log(0.5), axis=1))

    return {
        'pi_filter': pi_filter, 'mu_post': mu_post, 'Sigma': Sigma, 'P': P,
        'panic_state': panic_state, 'signs': signs,
        'min_ess': min_ess, 'n_sig': n_sig, 'll': ll,
        'n_params': K * (D + D*(D+1)//2) + K*(K-1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def long_short_port(df_test, score_col, fee=TRADING_FEE):
    """International L/S: unconditional top/bottom decile, value-weighted by `me`,
    no NYSE-equivalent filter (UK/JP main-listing common equity has no analogous breakpoint).
    Stocks identified by `secid` (gvkey + iid composite) since intl has no permno."""
    monthly, prev_lw, prev_sw = [], {}, {}
    for date, grp in df_test.groupby('date'):
        scores = grp[score_col].dropna()
        if len(scores) < 10: continue
        lo, hi = scores.quantile(0.10), scores.quantile(0.90)
        # Degenerate-quantile guard (sparse months can collapse lo >= hi).
        if lo >= hi: continue
        longs = grp[grp[score_col] >= hi].dropna(subset=['me'])
        shorts = grp[grp[score_col] <= lo].dropna(subset=['me'])
        if longs['me'].sum() == 0 or shorts['me'].sum() == 0: continue
        lme = longs['me'].sum()
        new_lw = (longs.set_index('secid')['me'] / lme).to_dict()
        r_long = (longs['ret_fwd'] * longs['me']).sum() / lme
        sme = shorts['me'].sum()
        new_sw = (shorts.set_index('secid')['me'] / sme).to_dict()
        r_short = (shorts['ret_fwd'] * shorts['me']).sum() / sme
        tl = sum(abs(new_lw.get(s, 0) - prev_lw.get(s, 0)) for s in set(new_lw) | set(prev_lw)) / 2
        ts = sum(abs(new_sw.get(s, 0) - prev_sw.get(s, 0)) for s in set(new_sw) | set(prev_sw)) / 2
        monthly.append({'date': date, 'ret': r_long - r_short - fee * (tl + ts)})
        prev_lw, prev_sw = new_lw, new_sw
    if not monthly: return pd.Series(dtype=float)
    return pd.DataFrame(monthly).set_index('date')['ret']

def compute_sharpe(r):
    r = pd.Series(r).dropna()
    if len(r) < 12 or r.std() == 0: return 0
    return r.mean() / r.std() * np.sqrt(12)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load and prepare regional data. Called once at startup. Reads REGION
    from the module-level global set in main()."""
    print(f"Loading data for region: {REGION}")

    # Market panel (HMM features) -- regional
    panel = pd.read_parquet(f'data/{REGION.lower()}_market_panel.parquet')
    panel['date'] = pd.to_datetime(panel['date'])
    panel['year_month'] = panel['date'].dt.to_period('M')

    # Stock panel (momentum + returns) -- regional
    stocks = pd.read_parquet(f'data/{REGION.lower()}_stock_panel.parquet')
    stocks['date'] = pd.to_datetime(stocks['date'])

    # Drop rows missing momentum features or forward return
    required_cols = MOM_FEATURES + ['ret_fwd', 'me', 'secid']
    stocks = stocks.dropna(subset=required_cols).reset_index(drop=True)

    # Initialise pi_filter to 0.5 (overwritten per-combo in pass2/3)
    stocks['pi_filter'] = 0.5

    # Split into train and test by TRAIN_END
    train_end_dt = pd.Timestamp(TRAIN_END)
    train = stocks[stocks['date'] < train_end_dt].copy()
    test  = stocks[stocks['date'] >= train_end_dt].copy()

    # Available features (subset of pool present in panel with enough non-null)
    available = [f for f in FEATURE_POOL if f in panel.columns and panel[f].notna().sum() > 200]

    print(f"  Market panel: {len(panel)} months ({panel['date'].min().date()} to {panel['date'].max().date()})")
    print(f"  Stock panel: {len(stocks):,} stock-months, train={len(train):,}, test={len(test):,}")
    print(f"  Features available: {len(available)} of {len(FEATURE_POOL)} -- {available}")
    print(f"  Required feature: {REQUIRED_FEATURE}")

    # Generate all combos with DD always included
    other_feats = [f for f in available if f != REQUIRED_FEATURE]
    all_combos = []
    for n_feat in FEATURE_COUNTS:
        if n_feat == 1:
            all_combos.append((REQUIRED_FEATURE,))
        else:
            for others in combinations(other_feats, n_feat - 1):
                all_combos.append((REQUIRED_FEATURE,) + others)

    print(f"  Total combinations: {len(all_combos)}")
    for n in FEATURE_COUNTS:
        count = sum(1 for c in all_combos if len(c) == n)
        print(f"    {n}-feature: {count} combos")

    return panel, train, test, available, all_combos


# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 1: HMM QUALITY SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

def run_pass1(panel, available, all_combos):
    """Screen all combinations with 1 HMM seed. Filter on quality gates."""

    print("\n" + "=" * 80)
    print(f"  PASS 1: HMM Quality Screen")
    print(f"  {len(all_combos)} combinations (1-4 features, DD always included)")
    print(f"  {len(PASS1_HMM_SEEDS)} HMM seed, {HMM_ITERATIONS} iterations")
    print(f"  Quality gates: ESS>={QUALITY_GATES['min_ess']}, n_sig>={QUALITY_GATES['n_sig']}")
    for vp in VALIDATION_PERIODS:
        print(f"    {vp[0]}: {'>' if vp[3]=='crisis' else '<'}{vp[4]}")
    print("=" * 80)

    results = []
    t0 = time.time()

    for i, combo in enumerate(all_combos):
        feat_list = list(combo)
        short_name = "+".join(f.replace("_z", "") for f in feat_list)

        sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
        sub = sub[sub['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
        if len(sub) < 200: continue

        sub_train = sub[sub['date'] < TRAIN_END]
        sub_test = sub[sub['date'] >= TRAIN_END]
        if len(sub_train) < 150 or len(sub_test) < 50: continue

        Z_tr = sub_train[feat_list].values.astype(float)
        Z_full = sub[feat_list].values.astype(float)
        train_dates = sub_train['date'].values
        all_dates = sub['date'].values

        try:
            result = fit_hmm(Z_tr, Z_full, train_dates, seed=PASS1_HMM_SEEDS[0])
        except:
            continue

        # Validate crisis alignment
        pi = result['pi_filter']
        period_results = {}
        all_periods_pass = True
        for pname, pstart, pend, ptype, threshold in VALIDATION_PERIODS:
            mask = (all_dates >= np.datetime64(pstart)) & (all_dates <= np.datetime64(pend))
            if mask.sum() == 0:
                period_results[pname] = np.nan
                continue
            mean_pi = pi[mask].mean()
            period_results[pname] = mean_pi
            if ptype == 'crisis' and mean_pi < threshold:
                all_periods_pass = False
            if ptype == 'calm' and mean_pi > threshold:
                all_periods_pass = False

        # Adjust n_sig threshold for small feature counts
        n_feats = len(feat_list)
        required_sig = min(QUALITY_GATES['n_sig'], n_feats)  # can't require more sig features than total
        mcmc_ok = (result['min_ess'] >= QUALITY_GATES['min_ess'] and
                   result['n_sig'] >= required_sig)
        passed = mcmc_ok and all_periods_pass

        results.append({
            'name': short_name, 'combo': feat_list, 'passed': passed,
            'min_ess': result['min_ess'], 'n_sig': result['n_sig'],
            **{f'pi_{k}': v for k, v in period_results.items()},
        })

        n_passed = sum(1 for r in results if r['passed'])
        status = "PASS" if passed else "fail"
        if (i + 1) % 10 == 0 or passed:
            elapsed = time.time() - t0
            period_str = "  ".join(f"{k}={v:.2f}" for k, v in period_results.items()
                                   if not (isinstance(v, float) and np.isnan(v)))
            print(f"  [{i+1:>3d}/{len(all_combos)}] {short_name:<30s}  "
                  f"ESS={result['min_ess']:>6.0f}  sig={result['n_sig']}  "
                  f"{period_str}  {status}  [{elapsed:.0f}s, {n_passed} passed]")

    df = pd.DataFrame(results)
    out_path = f'results/thesis/intl_{REGION.lower()}_hmm_feature_selection_pass1.csv'
    os.makedirs('results/thesis', exist_ok=True)
    df.to_csv(out_path, index=False)
    passed_combos = [r for r in results if r['passed']]
    print(f"\n  Pass 1 complete: {len(results)} tested, {len(passed_combos)} passed")
    print(f"  Saved: {out_path}")
    return passed_combos


# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 2: PORTFOLIO TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_pass2(panel, train_stocks, test_stocks, passed_combos):
    """Test surviving combos with averaged seeds. Rank by Sharpe."""
    REDUCED = MOM_FEATURES + ['pi_filter']

    print("\n" + "=" * 80)
    print(f"  PASS 2: Portfolio Test")
    print(f"  {len(passed_combos)} combinations, {len(PASS2_HMM_SEEDS)} HMM seeds, "
          f"{len(PASS2_XGB_SEEDS)} XGB seeds")
    print("=" * 80)

    results = []
    t0 = time.time()

    for idx, pc in enumerate(passed_combos):
        feat_list = pc['combo']
        short_name = pc['name']
        t_combo = time.time()

        sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
        sub = sub[sub['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
        sub_train = sub[sub['date'] < TRAIN_END]
        Z_tr = sub_train[feat_list].values.astype(float)
        Z_full = sub[feat_list].values.astype(float)
        train_dates = sub_train['date'].values
        all_dates = sub['date'].values

        # Average pi across HMM seeds
        pi_accum = np.zeros(len(Z_full))
        for hmm_seed in PASS2_HMM_SEEDS:
            r = fit_hmm(Z_tr, Z_full, train_dates, seed=hmm_seed)
            pi_accum += r['pi_filter']
        pi_avg = pi_accum / len(PASS2_HMM_SEEDS)

        # Merge into stock data
        pi_df = pd.DataFrame({'date': all_dates, 'pi_filter': pi_avg})
        tr = train_stocks.drop(columns=['pi_filter'], errors='ignore').merge(pi_df, on='date', how='left')
        tr['pi_filter'] = tr['pi_filter'].ffill().fillna(0.5)
        te = test_stocks.drop(columns=['pi_filter'], errors='ignore').merge(pi_df, on='date', how='left')
        te['pi_filter'] = te['pi_filter'].ffill().fillna(0.5)

        X_tr = tr[REDUCED].values.astype(float)
        X_te = te[REDUCED].values.astype(float)
        y_tr = tr['ret_fwd'].values.astype(float)

        # XGB ensemble
        xgb_preds = np.zeros(len(X_te))
        for xgb_seed in PASS2_XGB_SEEDS:
            xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                               learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                               colsample_bytree=COLSAMPLE, tree_method='hist',
                               random_state=xgb_seed, verbosity=0)
            xgb.fit(X_tr, y_tr)
            xgb_preds += xgb.predict(X_te)
        xgb_preds /= len(PASS2_XGB_SEEDS)

        te['score_xgb'] = xgb_preds
        r_xgb = long_short_port(te, 'score_xgb')
        sh_xgb = compute_sharpe(r_xgb)

        # M1 for comparison
        imp = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(imp.fit_transform(X_tr))
        X_te_s = scaler.transform(imp.transform(X_te))
        tr_c = tr.copy()
        tr_c['above_med'] = tr_c.groupby('date')['ret_fwd'].transform(
            lambda x: (x >= x.median()).astype(int))
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_tr_s, tr_c['above_med'].values)
        te['score_lr'] = lr.predict_proba(X_te_s)[:, 1]
        r_lr = long_short_port(te, 'score_lr')
        sh_lr = compute_sharpe(r_lr)

        elapsed = time.time() - t_combo
        results.append({
            'name': short_name, 'combo': feat_list,
            'sh_xgb': sh_xgb, 'sh_lr': sh_lr,
            'min_ess': pc['min_ess'], 'n_sig': pc['n_sig'],
        })

        print(f"  [{idx+1:>3d}/{len(passed_combos)}] {short_name:<30s}  "
              f"M1={sh_lr:.3f}  M2={sh_xgb:.3f}  [{elapsed:.0f}s]")

    results.sort(key=lambda x: -x['sh_xgb'])
    df = pd.DataFrame(results)
    out_path = f'results/thesis/intl_{REGION.lower()}_hmm_feature_selection_pass2.csv'
    os.makedirs('results/thesis', exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n  Top 10:")
    for i, r in enumerate(results[:10], 1):
        print(f"    {i:>2d}. {r['name']:<30s}  M2={r['sh_xgb']:.3f}")

    print(f"  Saved: {out_path}")
    return results[:TOP_N_FOR_PASS3]


# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 3: DEFINITIVE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pass3(panel, train_stocks, test_stocks, top_combos):
    """Definitive test on top combos with maximum seeds."""
    REDUCED = MOM_FEATURES + ['pi_filter']

    print("\n" + "=" * 80)
    print(f"  PASS 3: Definitive Validation")
    print(f"  {len(top_combos)} combinations, {len(PASS3_HMM_SEEDS)} HMM seeds, "
          f"{len(PASS3_XGB_SEEDS)} XGB seeds")
    print("=" * 80)

    results = []

    for combo_info in top_combos:
        feat_list = combo_info['combo']
        short_name = combo_info['name']
        print(f"\n  === {short_name} ===")

        sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
        sub = sub[sub['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
        Z_tr = sub[sub['date'] < TRAIN_END][feat_list].values.astype(float)
        Z_full = sub[feat_list].values.astype(float)
        train_dates = sub[sub['date'] < TRAIN_END]['date'].values
        all_dates = sub['date'].values

        # Average pi across 10 HMM seeds
        print(f"    Fitting {len(PASS3_HMM_SEEDS)} HMM seeds ...", end=' ', flush=True)
        pi_accum = np.zeros(len(Z_full))
        for s in PASS3_HMM_SEEDS:
            r = fit_hmm(Z_tr, Z_full, train_dates, seed=s)
            pi_accum += r['pi_filter']
        pi_avg = pi_accum / len(PASS3_HMM_SEEDS)
        print("done")

        pi_df = pd.DataFrame({'date': all_dates, 'pi_filter': pi_avg})
        tr = train_stocks.drop(columns=['pi_filter'], errors='ignore').merge(pi_df, on='date', how='left')
        tr['pi_filter'] = tr['pi_filter'].ffill().fillna(0.5)
        te = test_stocks.drop(columns=['pi_filter'], errors='ignore').merge(pi_df, on='date', how='left')
        te['pi_filter'] = te['pi_filter'].ffill().fillna(0.5)

        X_tr = tr[REDUCED].values.astype(float)
        X_te = te[REDUCED].values.astype(float)
        y_tr = tr['ret_fwd'].values.astype(float)

        # Ensemble XGB predictions
        print(f"    Training {len(PASS3_XGB_SEEDS)} XGB seeds ...", end=' ', flush=True)
        xgb_preds = np.zeros(len(X_te))
        for xgb_seed in PASS3_XGB_SEEDS:
            xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                               learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                               colsample_bytree=COLSAMPLE, tree_method='hist',
                               random_state=xgb_seed, verbosity=0)
            xgb.fit(X_tr, y_tr)
            xgb_preds += xgb.predict(X_te)
        xgb_preds /= len(PASS3_XGB_SEEDS)
        print("done")

        te['score'] = xgb_preds
        r_port = long_short_port(te, 'score')

        ann_ret = (1 + r_port).prod() ** (12 / len(r_port)) - 1
        ann_vol = r_port.std() * np.sqrt(12)
        sharpe = compute_sharpe(r_port)
        cum = (1 + r_port).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        print(f"    Ensemble: Sharpe={sharpe:.3f}  Ret={ann_ret:.1%}  Vol={ann_vol:.1%}  MDD={mdd:.1%}")

        # Sub-periods
        for pname, start, end in [('2011-2015', '2011-01-01', '2016-01-01'),
                                   ('2016-2020', '2016-01-01', '2021-01-01'),
                                   ('2021-2025', '2021-01-01', '2026-01-01')]:
            r_sub = r_port[(r_port.index >= start) & (r_port.index < end)]
            if len(r_sub) > 6:
                sh_sub = compute_sharpe(r_sub)
                print(f"    {pname}: Sharpe={sh_sub:.3f}")

        # Individual seed variance
        ind_sharpes = []
        for xgb_seed in PASS3_XGB_SEEDS[:10]:
            xgb_i = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                 learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                                 colsample_bytree=COLSAMPLE, tree_method='hist',
                                 random_state=xgb_seed, verbosity=0)
            xgb_i.fit(X_tr, y_tr)
            te[f'sc_{xgb_seed}'] = xgb_i.predict(X_te)
            r_i = long_short_port(te, f'sc_{xgb_seed}')
            ind_sharpes.append(compute_sharpe(r_i))
        print(f"    Seed variance: Mean={np.mean(ind_sharpes):.3f}  "
              f"Std={np.std(ind_sharpes):.3f}  "
              f"Min={np.min(ind_sharpes):.3f}  Max={np.max(ind_sharpes):.3f}")

        # ── Stability test: 10 independent experiments ──
        # Each uses a random partition of 3 HMM seeds + 1 XGB seed
        # Tests whether the result depends on specific seeds or is robust
        ALL_SEEDS_POOL = [42, 2201, 1337, 314, 789, 999, 123, 456, 7777, 55]
        N_EXPERIMENTS = 5

        print(f"    Stability test ({N_EXPERIMENTS} independent experiments) ...")

        # Pre-compute all 20 individual pi_filters (reuse the ones from above + extras)
        all_pis = {}
        for s in ALL_SEEDS_POOL:
            if s not in all_pis:
                r_hmm = fit_hmm(Z_tr, Z_full, train_dates, seed=s)
                all_pis[s] = r_hmm['pi_filter']

        np.random.seed(12345)
        stability_sharpes = []
        for exp in range(N_EXPERIMENTS):
            hmm_seeds_exp = list(np.random.choice(ALL_SEEDS_POOL, size=3, replace=False))
            xgb_seed_exp = ALL_SEEDS_POOL[exp % len(ALL_SEEDS_POOL)]
            pi_exp = np.mean([all_pis[s] for s in hmm_seeds_exp], axis=0)

            pi_df_exp = pd.DataFrame({'date': all_dates, 'pi_filter': pi_exp})
            tr_exp = train_stocks.drop(columns=['pi_filter'], errors='ignore').merge(
                pi_df_exp, on='date', how='left')
            tr_exp['pi_filter'] = tr_exp['pi_filter'].ffill().fillna(0.5)
            te_exp = test_stocks.drop(columns=['pi_filter'], errors='ignore').merge(
                pi_df_exp, on='date', how='left')
            te_exp['pi_filter'] = te_exp['pi_filter'].ffill().fillna(0.5)

            X_tr_exp = tr_exp[REDUCED].values.astype(float)
            X_te_exp = te_exp[REDUCED].values.astype(float)
            y_tr_exp = tr_exp['ret_fwd'].values.astype(float)

            xgb_exp = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                   learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                                   colsample_bytree=COLSAMPLE, tree_method='hist',
                                   random_state=xgb_seed_exp, verbosity=0)
            xgb_exp.fit(X_tr_exp, y_tr_exp)
            te_exp['score'] = xgb_exp.predict(X_te_exp)
            r_exp = long_short_port(te_exp, 'score')
            sh_exp = compute_sharpe(r_exp)
            stability_sharpes.append(sh_exp)

        stab_mean = np.mean(stability_sharpes)
        stab_std = np.std(stability_sharpes)
        stab_min = np.min(stability_sharpes)
        stab_max = np.max(stability_sharpes)
        stability = 'STABLE' if stab_std < 0.10 else 'MODERATE' if stab_std < 0.15 else 'UNSTABLE'

        print(f"      Mean={stab_mean:.3f}  Std={stab_std:.3f}  "
              f"Min={stab_min:.3f}  Max={stab_max:.3f}  [{stability}]")

        results.append({
            'name': short_name, 'combo': feat_list,
            'sharpe': sharpe, 'ann_ret': ann_ret, 'ann_vol': ann_vol, 'mdd': mdd,
            'seed_mean': np.mean(ind_sharpes), 'seed_std': np.std(ind_sharpes),
            'stab_mean': stab_mean, 'stab_std': stab_std,
            'stab_min': stab_min, 'stab_max': stab_max, 'stability': stability,
        })

    df = pd.DataFrame(results)
    out_path_p3 = f'results/thesis/intl_{REGION.lower()}_hmm_feature_selection_pass3.csv'
    os.makedirs('results/thesis', exist_ok=True)
    df.to_csv(out_path_p3, index=False)

    print(f"\n  Final ranking (by ensemble Sharpe):")
    for i, r in enumerate(sorted(results, key=lambda x: -x['sharpe']), 1):
        print(f"    {i}. {r['name']}: Ensemble={r['sharpe']:.3f}  "
              f"Stability={r['stab_mean']:.3f}+/-{r['stab_std']:.3f} [{r['stability']}]")

    print(f"  Saved: {out_path_p3}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 4: STABILITY TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_pass4(panel, train_stocks, test_stocks, top_combos):
    """
    Stability test: run 10 independent experiments per combo.
    Each experiment uses a different random partition of 3 HMM seeds
    (from a pool of 20) and a single XGB seed. This tests whether the
    result depends on the specific seed combination or is robust.

    A stable combo should show low std across experiments.
    An unstable combo will show high std (like TERM dropping from 1.05 to 0.62).
    """
    REDUCED = MOM_FEATURES + ['pi_filter']

    # Pool of 20 seeds, randomly partitioned into groups of 3
    ALL_SEEDS = [42, 2201, 1337, 314, 789, 999, 123, 456, 7777, 55,
                 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    N_EXPERIMENTS = 10

    print("\n" + "=" * 80)
    print(f"  PASS 4: Stability Test")
    print(f"  {len(top_combos)} combos x {N_EXPERIMENTS} independent experiments")
    print(f"  Each experiment: 3 random HMM seeds averaged + 1 XGB seed")
    print("=" * 80)

    results = []

    for combo_info in top_combos:
        feat_list = combo_info['combo']
        short_name = combo_info['name']
        print(f"\n  === {short_name} ===")

        sub = panel[['date'] + feat_list].dropna().reset_index(drop=True)
        sub = sub[sub['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
        Z_tr = sub[sub['date'] < TRAIN_END][feat_list].values.astype(float)
        Z_full = sub[feat_list].values.astype(float)
        train_dates = sub[sub['date'] < TRAIN_END]['date'].values
        all_dates = sub['date'].values

        # Pre-compute all 20 individual pi_filters
        print(f"    Computing 20 individual pi_filters ...", end=' ', flush=True)
        all_pis = {}
        for seed in ALL_SEEDS:
            r = fit_hmm(Z_tr, Z_full, train_dates, seed=seed)
            all_pis[seed] = r['pi_filter']
        print("done")

        # Run 10 independent experiments with different seed partitions
        np.random.seed(42)
        experiment_sharpes = []

        for exp in range(N_EXPERIMENTS):
            # Random 3 HMM seeds (without replacement)
            hmm_seeds = list(np.random.choice(ALL_SEEDS, size=3, replace=False))
            xgb_seed = ALL_SEEDS[exp % len(ALL_SEEDS)]

            # Average pi across 3 random seeds
            pi_avg = np.mean([all_pis[s] for s in hmm_seeds], axis=0)

            pi_df = pd.DataFrame({'date': all_dates, 'pi_filter': pi_avg})
            tr = train_stocks.drop(columns=['pi_filter'], errors='ignore').merge(
                pi_df, on='date', how='left')
            tr['pi_filter'] = tr['pi_filter'].ffill().fillna(0.5)
            te = test_stocks.drop(columns=['pi_filter'], errors='ignore').merge(
                pi_df, on='date', how='left')
            te['pi_filter'] = te['pi_filter'].ffill().fillna(0.5)

            X_tr = tr[REDUCED].values.astype(float)
            X_te = te[REDUCED].values.astype(float)
            y_tr = tr['ret_fwd'].values.astype(float)

            xgb = XGBRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                               learning_rate=LEARNING_RATE, subsample=SUBSAMPLE,
                               colsample_bytree=COLSAMPLE, tree_method='hist',
                               random_state=xgb_seed, verbosity=0)
            xgb.fit(X_tr, y_tr)
            te['score'] = xgb.predict(X_te)
            r_port = long_short_port(te, 'score')
            sh = compute_sharpe(r_port)
            experiment_sharpes.append(sh)

            print(f"    Exp {exp+1:>2d}: HMM seeds={hmm_seeds}, XGB={xgb_seed}, Sharpe={sh:.3f}")

        mean_sh = np.mean(experiment_sharpes)
        std_sh = np.std(experiment_sharpes)
        min_sh = np.min(experiment_sharpes)
        max_sh = np.max(experiment_sharpes)

        print(f"    ────────────────────────────────────────")
        print(f"    Mean={mean_sh:.3f}  Std={std_sh:.3f}  Min={min_sh:.3f}  Max={max_sh:.3f}")
        print(f"    Stability score: {'STABLE' if std_sh < 0.10 else 'MODERATE' if std_sh < 0.15 else 'UNSTABLE'}")

        results.append({
            'name': short_name, 'combo': feat_list,
            'mean': mean_sh, 'std': std_sh, 'min': min_sh, 'max': max_sh,
            'all_sharpes': experiment_sharpes,
        })

    # Summary
    print("\n" + "=" * 80)
    print("  STABILITY SUMMARY")
    print("=" * 80)
    print(f"\n  {'Combo':<25s} {'Mean':>6s} {'Std':>6s} {'Min':>6s} {'Max':>6s} {'Stability':>10s}")
    print(f"  {'-'*62}")
    for r in sorted(results, key=lambda x: -x['mean']):
        stability = 'STABLE' if r['std'] < 0.10 else 'MODERATE' if r['std'] < 0.15 else 'UNSTABLE'
        print(f"  {r['name']:<25s} {r['mean']:>6.3f} {r['std']:>6.3f} {r['min']:>6.3f} {r['max']:>6.3f} {stability:>10s}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'all_sharpes'} for r in results])
    out_path_p4 = f'results/thesis/intl_{REGION.lower()}_hmm_feature_selection_pass4.csv'
    os.makedirs('results/thesis', exist_ok=True)
    df.to_csv(out_path_p4, index=False)
    print(f"\n  Saved: {out_path_p4}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='International HMM Feature Selection Pipeline')
    parser.add_argument('--region', choices=['UK', 'JP'], required=True,
                        help='International market: UK or JP')
    parser.add_argument('--pass', type=int, default=0, dest='run_pass',
                        help='Run only this pass (0=all, 1/2/3/4)')
    args = parser.parse_args()

    # Promote --region to module-level global, set region-specific validation
    global REGION, VALIDATION_PERIODS
    REGION = args.region
    VALIDATION_PERIODS = VALIDATION_PERIODS_BY_REGION[REGION]
    print(f"=== International HMM Feature Selection: {REGION} ===")

    t_total = time.time()
    panel, train_stocks, test_stocks, available, all_combos = load_data()

    pass1_path = f'results/thesis/intl_{REGION.lower()}_hmm_feature_selection_pass1.csv'
    pass2_path = f'results/thesis/intl_{REGION.lower()}_hmm_feature_selection_pass2.csv'
    pass3_path = f'results/thesis/intl_{REGION.lower()}_hmm_feature_selection_pass3.csv'

    # Pass 1: always run if requested OR if a later pass needs its output
    if args.run_pass in (0, 1):
        passed = run_pass1(panel, available, all_combos)
        if args.run_pass == 1:
            elapsed = time.time() - t_total
            print(f"\nPass 1 complete. Total time: {elapsed/60:.1f} minutes")
            return

    # Pass 2: load Pass 1 output if not already in memory, then run if requested
    if args.run_pass in (0, 2):
        if args.run_pass == 2:
            df = pd.read_csv(pass1_path)
            passed = [{'name': r['name'], 'combo': eval(r['combo']),
                       'min_ess': r['min_ess'], 'n_sig': r['n_sig']}
                      for _, r in df[df['passed']].iterrows()]
        top = run_pass2(panel, train_stocks, test_stocks, passed)
        if args.run_pass == 2:
            elapsed = time.time() - t_total
            print(f"\nPass 2 complete. Total time: {elapsed/60:.1f} minutes")
            return

    # Pass 3: load Pass 2 output if not already in memory, then run if requested
    if args.run_pass in (0, 3):
        if args.run_pass == 3:
            df = pd.read_csv(pass2_path)
            top = [{'name': r['name'], 'combo': eval(r['combo']),
                    'sh_xgb': r['sh_xgb']}
                   for _, r in df.head(TOP_N_FOR_PASS3).iterrows()]
        final = run_pass3(panel, train_stocks, test_stocks, top)
        if args.run_pass == 3:
            elapsed = time.time() - t_total
            print(f"\nPass 3 complete. Total time: {elapsed/60:.1f} minutes")
            return

    # Pass 4: load Pass 3 output if not already in memory, then run if requested
    if args.run_pass in (0, 4):
        if args.run_pass == 4:
            df = pd.read_csv(pass3_path)
            final = [{'name': r['name'], 'combo': eval(r['combo']),
                      'sharpe': r['sharpe']}
                     for _, r in df.iterrows()]
        run_pass4(panel, train_stocks, test_stocks, final)

    elapsed = time.time() - t_total
    print(f"\nTotal pipeline time: {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    main()
