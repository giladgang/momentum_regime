"""
check_residual_normality.py
===========================
Diagnostic: are the within-regime residuals (x_t - mu_k) approximately Normal?

For each HMM feature (DD_z, VOL_z, DISP_z, REL_N_z) and each regime (calm/panic),
computes:
  1. Shapiro-Wilk test (p-value)
  2. Skewness and kurtosis
  3. QQ plot

If residuals are Normal, the conjugate Normal-Inverse-Wishart prior is well-specified.
If not, Student-t or other heavy-tailed emissions might be more appropriate.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Load data ────────────────────────────────────────────────────────────────

draws = np.load('data/mcmc_draws.npz')
panel = pd.read_parquet('data/panel_with_regimes.parquet')

features = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']
feature_labels = ['Drawdown', 'Volatility', 'Dispersion', 'Rel. Participation']

# Posterior mean of mu for each regime (average over MCMC draws)
mu_post = draws['mu_draws'].mean(axis=0)  # shape (2, 4)

# Assign regimes using the most likely state (pi_filter > 0.5 = panic)
panic_state = int(draws['panic_state'])
calm_state = int(draws['calm_state'])

panel['regime'] = (panel['pi_filter'] > 0.5).astype(int)
# regime=1 means panic if panic_state=1, otherwise flip
if panic_state == 0:
    panel['regime'] = 1 - panel['regime']

calm_mask = panel['regime'] == calm_state
panic_mask = panel['regime'] == panic_state

# ── Compute residuals and test normality ─────────────────────────────────────

print("=" * 75)
print("WITHIN-REGIME RESIDUAL NORMALITY DIAGNOSTICS")
print("=" * 75)

results = []

for i, (feat, label) in enumerate(zip(features, feature_labels)):
    for regime_name, mask, state_idx in [('Calm', calm_mask, calm_state),
                                          ('Panic', panic_mask, panic_state)]:
        x = panel.loc[mask, feat].dropna().values
        mu_k = mu_post[state_idx, i]
        residuals = x - mu_k

        n = len(residuals)
        skew = stats.skew(residuals)
        kurt = stats.kurtosis(residuals)  # excess kurtosis (0 = Normal)

        # Shapiro-Wilk (max 5000 obs)
        sw_stat, sw_p = stats.shapiro(residuals[:5000])

        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(residuals)

        results.append({
            'Feature': label,
            'Regime': regime_name,
            'N': n,
            'Skewness': skew,
            'Excess Kurtosis': kurt,
            'Shapiro-Wilk p': sw_p,
            'Jarque-Bera p': jb_p,
        })

        print(f"\n{label} | {regime_name} (n={n})")
        print(f"  Skewness:        {skew:+.3f}  (Normal = 0)")
        print(f"  Excess Kurtosis: {kurt:+.3f}  (Normal = 0)")
        print(f"  Shapiro-Wilk p:  {sw_p:.4f}  {'OK' if sw_p > 0.05 else 'REJECT normality'}")
        print(f"  Jarque-Bera p:   {jb_p:.4f}  {'OK' if jb_p > 0.05 else 'REJECT normality'}")

# ── Summary table ────────────────────────────────────────────────────────────

print("\n" + "=" * 75)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# ── QQ Plots ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 2, figsize=(10, 14))
fig.suptitle('Within-Regime Residual QQ Plots\n(Normal assumption check)',
             fontsize=14, fontweight='bold')

for i, (feat, label) in enumerate(zip(features, feature_labels)):
    for j, (regime_name, mask, state_idx) in enumerate([
            ('Calm', calm_mask, calm_state),
            ('Panic', panic_mask, panic_state)]):
        ax = axes[i, j]
        x = panel.loc[mask, feat].dropna().values
        mu_k = mu_post[state_idx, i]
        residuals = x - mu_k

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'{label} - {regime_name} (n={len(residuals)})')
        ax.get_lines()[0].set_markersize(3)
        ax.get_lines()[0].set_alpha(0.5)

plt.tight_layout()
plt.savefig('residual_qq_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nQQ plots saved to: residual_qq_plots.png")

# ── Verdict ──────────────────────────────────────────────────────────────────

n_reject = sum(1 for r in results if r['Shapiro-Wilk p'] < 0.05)
print(f"\n{'=' * 75}")
print(f"VERDICT: {n_reject}/{len(results)} feature-regime pairs reject normality at 5%")
if n_reject == 0:
    print("Normal emissions appear well-specified.")
elif n_reject <= 2:
    print("Mild departures. Normal emissions are a reasonable approximation.")
else:
    print("Substantial departures. Consider Student-t emissions or other heavy-tailed alternatives.")
print("=" * 75)
