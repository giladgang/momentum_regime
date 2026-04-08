"""
fit_emission_distributions.py
=============================
For each feature x regime, fit several candidate distributions and compare
via BIC / AIC / KS-test to find the best-fitting emission model.

Candidates:
  - Normal
  - Student-t (heavier tails)
  - Skew-Normal (asymmetry)
  - Generalized Hyperbolic (nests Normal, t, skew-t)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────────

draws = np.load('data/mcmc_draws.npz')
panel = pd.read_parquet('data/panel_with_regimes.parquet')

features = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']
feature_labels = ['Drawdown', 'Volatility', 'Dispersion', 'Rel. Participation']

panic_state = int(draws['panic_state'])
calm_state = int(draws['calm_state'])

panel['regime'] = (panel['pi_filter'] > 0.5).astype(int)
if panic_state == 0:
    panel['regime'] = 1 - panel['regime']

calm_mask = panel['regime'] == calm_state
panic_mask = panel['regime'] == panic_state


def fit_and_score(data, dist_name):
    """Fit a distribution and return log-likelihood, AIC, BIC, KS p-value."""
    n = len(data)

    if dist_name == 'Normal':
        mu, sigma = np.mean(data), np.std(data, ddof=1)
        ll = np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))
        k = 2

    elif dist_name == 'Student-t':
        # Fit location-scale t: 3 params (df, loc, scale)
        df, loc, scale = stats.t.fit(data)
        ll = np.sum(stats.t.logpdf(data, df, loc=loc, scale=scale))
        k = 3

    elif dist_name == 'Skew-Normal':
        # scipy skewnorm: 3 params (a=skewness, loc, scale)
        a, loc, scale = stats.skewnorm.fit(data)
        ll = np.sum(stats.skewnorm.logpdf(data, a, loc=loc, scale=scale))
        k = 3

    elif dist_name == 'Laplace':
        loc, scale = stats.laplace.fit(data)
        ll = np.sum(stats.laplace.logpdf(data, loc=loc, scale=scale))
        k = 2

    elif dist_name == 'NIG':
        # Normal-Inverse Gaussian: 4 params (a, b, loc, scale)
        # Flexible: nests Normal-like and heavy-tailed skewed distributions
        try:
            a, b, loc, scale = stats.norminvgauss.fit(data)
            ll = np.sum(stats.norminvgauss.logpdf(data, a, b, loc=loc, scale=scale))
            k = 4
        except Exception:
            return None

    elif dist_name == 'Logistic':
        loc, scale = stats.logistic.fit(data)
        ll = np.sum(stats.logistic.logpdf(data, loc=loc, scale=scale))
        k = 2

    else:
        return None

    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    # KS test
    if dist_name == 'Normal':
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, sigma))
    elif dist_name == 'Student-t':
        ks_stat, ks_p = stats.kstest(data, 't', args=(df, loc, scale))
    elif dist_name == 'Skew-Normal':
        ks_stat, ks_p = stats.kstest(data, 'skewnorm', args=(a, loc, scale))
    elif dist_name == 'Laplace':
        ks_stat, ks_p = stats.kstest(data, 'laplace', args=(loc, scale))
    elif dist_name == 'NIG':
        ks_stat, ks_p = stats.kstest(data, 'norminvgauss', args=(a, b, loc, scale))
    elif dist_name == 'Logistic':
        ks_stat, ks_p = stats.kstest(data, 'logistic', args=(loc, scale))

    return {
        'k': k,
        'LogLik': ll,
        'AIC': aic,
        'BIC': bic,
        'KS_stat': ks_stat,
        'KS_p': ks_p,
    }


# ── Run fits ─────────────────────────────────────────────────────────────────

distributions = ['Normal', 'Student-t', 'Skew-Normal', 'Laplace', 'NIG', 'Logistic']

all_results = []

print("=" * 85)
print("DISTRIBUTION FIT COMPARISON (per feature x regime)")
print("=" * 85)

for i, (feat, label) in enumerate(zip(features, feature_labels)):
    for regime_name, mask in [('Calm', calm_mask), ('Panic', panic_mask)]:
        data = panel.loc[mask, feat].dropna().values
        n = len(data)

        print(f"\n{'─' * 85}")
        print(f"  {label} | {regime_name} (n={n})")
        print(f"  {'Distribution':<15} {'Params':>6} {'LogLik':>10} {'AIC':>10} {'BIC':>10} {'KS p':>8}  Note")
        print(f"  {'─'*80}")

        best_bic = np.inf
        best_dist = None
        row_results = []

        for dist_name in distributions:
            result = fit_and_score(data, dist_name)
            if result is None:
                print(f"  {dist_name:<15} {'—':>6} {'failed':>10}")
                continue

            note = ''
            if result['KS_p'] > 0.05:
                note = 'PASS KS'
            if result['BIC'] < best_bic:
                best_bic = result['BIC']
                best_dist = dist_name

            print(f"  {dist_name:<15} {result['k']:>6} {result['LogLik']:>10.1f} "
                  f"{result['AIC']:>10.1f} {result['BIC']:>10.1f} {result['KS_p']:>8.4f}  {note}")

            all_results.append({
                'Feature': label,
                'Regime': regime_name,
                'Distribution': dist_name,
                **result,
            })

        print(f"  >>> Best by BIC: {best_dist}")

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'=' * 85}")
print("SUMMARY: Best distribution per feature x regime (by BIC)")
print(f"{'=' * 85}")

df = pd.DataFrame(all_results)
for (feat, regime), group in df.groupby(['Feature', 'Regime']):
    best = group.loc[group['BIC'].idxmin()]
    normal = group.loc[group['Distribution'] == 'Normal'].iloc[0]
    delta_bic = normal['BIC'] - best['BIC']
    marker = '' if best['Distribution'] == 'Normal' else f'  (BIC improvement over Normal: {delta_bic:.1f})'
    print(f"  {feat:<20} {regime:<6} → {best['Distribution']:<15}{marker}")

print(f"\n{'=' * 85}")
print("BIC improvement interpretation:")
print("  < 2:  No meaningful difference (Normal is fine)")
print("  2-6:  Positive evidence for alternative")
print("  6-10: Strong evidence")
print("  > 10: Very strong evidence")
print(f"{'=' * 85}")
