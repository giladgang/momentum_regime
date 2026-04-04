"""
regime_predictor.py
===================
Predict the next month's regime using a Ridge logistic regression on
current feature levels, feature changes (Δz), and pi_filter.

Motivation
----------
The HMM's pi_next uses only the transition matrix:
    pi_next = P(calm→panic)*(1-π) + P(panic→panic)*π
This treats transitions as fixed-probability coin flips, ignoring
whether features are actively trending toward panic.

This model conditions on observable feature dynamics:
    P(panic at t+1 | z_t, Δz_t, pi_filter_t)
so it can detect transitions *before* the filter catches up.

Features (9 total)
------------------
  pi_filter         -- current HMM filtered panic probability
  DD_z ... REL_N_z  -- 4 feature levels (z-scored)
  delta_DD_z ... delta_REL_N_z -- 4 feature changes (month-over-month)

Target
------
  realized regime next month: pi_smooth[t+1] > 0.5

Train: 1990-2010  |  Test: 2011-2025
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                             classification_report, log_loss)

# ── Section 1: Load data ────────────────────────────────────────────────────

panel = pd.read_parquet('data/panel_with_regimes.parquet')
panel['date'] = pd.to_datetime(panel['date'])

FEATURES_Z = ['DD_z', 'VOL_z', 'DISP_z', 'REL_N_z']

# ── Section 2: Build features ───────────────────────────────────────────────

# Feature changes (month-over-month)
for f in FEATURES_Z:
    panel[f'Δ{f}'] = panel[f].diff()

DELTA_FEATURES = [f'Δ{f}' for f in FEATURES_Z]

# Target: realized regime NEXT month (shift smooth probability back by 1)
panel['regime_next'] = (panel['pi_smooth'].shift(-1) > 0.5).astype(float)

# All predictor columns
PRED_FEATURES = ['pi_filter'] + FEATURES_Z + DELTA_FEATURES

# Drop rows with NaN (first row has NaN deltas, last row has NaN target)
df = panel.dropna(subset=PRED_FEATURES + ['regime_next']).copy()
df = df.reset_index(drop=True)

print(f"Usable rows: {len(df)}  |  {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Features: {len(PRED_FEATURES)}  |  Target: regime_next (pi_smooth[t+1] > 0.5)")

# ── Section 3: Train / test split ───────────────────────────────────────────

train = df[df['date'] < '2011-01-01'].copy()
test  = df[df['date'] >= '2011-01-01'].copy()

X_train = train[PRED_FEATURES].values
y_train = train['regime_next'].values
X_test  = test[PRED_FEATURES].values
y_test  = test['regime_next'].values

print(f"Train: {len(train)} months  ({y_train.sum():.0f} panic, "
      f"{len(train) - y_train.sum():.0f} calm)")
print(f"Test:  {len(test)} months  ({y_test.sum():.0f} panic, "
      f"{len(test) - y_test.sum():.0f} calm)")

# ── Section 4: Standardize ──────────────────────────────────────────────────

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Section 5: Fit Ridge Logistic Regression ─────────────────────────────────

# C = inverse regularization strength. Lower C = stronger Ridge penalty.
# Use class_weight='balanced' to handle class imbalance (few panic months).
lr = LogisticRegression(
    penalty='l2',
    C=0.1,              # strong regularization for small sample
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    random_state=42,
)
lr.fit(X_train_s, y_train)

# Predicted probabilities
p_train = lr.predict_proba(X_train_s)[:, 1]
p_test  = lr.predict_proba(X_test_s)[:, 1]

# ── Section 6: Comparison with HMM pi_next ──────────────────────────────────

pi_next_train = train['pi_next'].values
pi_next_test  = test['pi_next'].values

print("\n" + "=" * 60)
print("  MODEL COMPARISON: Ridge LR vs HMM pi_next")
print("=" * 60)

def eval_model(name, probs, y):
    auc   = roc_auc_score(y, probs)
    brier = brier_score_loss(y, probs)
    ll    = log_loss(y, probs)
    preds = (probs > 0.5).astype(int)
    acc   = (preds == y).mean()
    return auc, brier, ll, acc

for label, y, p_lr, p_hmm in [
    ('TRAIN', y_train, p_train, pi_next_train),
    ('TEST',  y_test,  p_test,  pi_next_test),
]:
    auc_lr, brier_lr, ll_lr, acc_lr       = eval_model('LR', p_lr, y)
    auc_hmm, brier_hmm, ll_hmm, acc_hmm   = eval_model('HMM', p_hmm, y)

    print(f"\n  {label}")
    print(f"  {'Metric':<20} {'Ridge LR':>10} {'HMM pi_next':>12} {'Δ (LR−HMM)':>12}")
    print("  " + "-" * 56)
    print(f"  {'AUC-ROC':<20} {auc_lr:>10.3f} {auc_hmm:>12.3f} {auc_lr - auc_hmm:>+12.3f}")
    print(f"  {'Brier score':<20} {brier_lr:>10.3f} {brier_hmm:>12.3f} {brier_lr - brier_hmm:>+12.3f}")
    print(f"  {'Log loss':<20} {ll_lr:>10.3f} {ll_hmm:>12.3f} {ll_lr - ll_hmm:>+12.3f}")
    print(f"  {'Accuracy (0.5)':<20} {acc_lr:>10.3f} {acc_hmm:>12.3f} {acc_lr - acc_hmm:>+12.3f}")

# ── Section 7: Test-period classification report ────────────────────────────

print(f"\n  Classification report (TEST, threshold=0.5):")
print(classification_report(y_test, (p_test > 0.5).astype(int),
                            target_names=['Calm', 'Panic'], digits=3))

# ── Section 8: Feature weights ──────────────────────────────────────────────

coef = pd.Series(lr.coef_[0], index=PRED_FEATURES).sort_values()

print("\n  Ridge LR coefficients (standardized):")
print("  " + "-" * 40)
for feat, w in coef.items():
    print(f"  {feat:<16} {w:>+8.3f}")

# ── Section 9: Save enhanced pi_next to panel ───────────────────────────────

# Compute LR predictions for the full panel (train + test)
X_full = df[PRED_FEATURES].values
X_full_s = scaler.transform(X_full)
p_full = lr.predict_proba(X_full_s)[:, 1]

# Merge back into panel
df['pi_next_lr'] = p_full
panel = panel.merge(
    df[['date', 'pi_next_lr']],
    on='date', how='left'
)
panel.to_parquet('data/panel_with_regimes.parquet', index=False)
print("\nSaved pi_next_lr to data/panel_with_regimes.parquet")

# ── Section 10: Plots ───────────────────────────────────────────────────────

test_dates = test['date'].values

# Plot 1: LR coefficients
fig, ax = plt.subplots(figsize=(8, 6))
colors_coef = ['crimson' if c < 0 else 'steelblue' for c in coef]
coef.plot(kind='barh', ax=ax, color=colors_coef, alpha=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (standardized)', fontsize=9)
ax.set_title('Ridge LR: what predicts next-month regime?\n'
             '(positive = predicts panic)', fontsize=10)
plt.tight_layout()
fig.savefig('regime_predictor_weights.png', dpi=150)
plt.close(fig)

# Plot 2: Test-period predictions vs realized regime
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Panel A: LR prediction vs realized
ax = axes[0]
ax.plot(test_dates, p_test, color='steelblue', linewidth=1.2, label='Ridge LR prediction')
ax.fill_between(test_dates, y_test, alpha=0.2, color='crimson', label='Realized panic')
ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':')
ax.set_ylabel('P(panic next month)', fontsize=9)
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.set_title('Test period: Ridge LR vs realized regime', fontsize=10)

# Panel B: HMM pi_next vs realized (for comparison)
ax = axes[1]
ax.plot(test_dates, pi_next_test, color='grey', linewidth=1.2, label='HMM pi_next')
ax.fill_between(test_dates, y_test, alpha=0.2, color='crimson', label='Realized panic')
ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':')
ax.set_ylabel('P(panic next month)', fontsize=9)
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.set_title('Test period: HMM pi_next vs realized regime', fontsize=10)

# Panel C: Difference (LR - HMM) — where does LR add value?
ax = axes[2]
diff = p_test - pi_next_test
ax.bar(test_dates, diff, width=25, color=np.where(diff > 0, 'crimson', 'steelblue'),
       alpha=0.6)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('LR − HMM', fontsize=9)
ax.set_xlabel('Date')
ax.set_title('Where does the LR disagree with the HMM?\n'
             '(red = LR more panicky, blue = LR more calm)', fontsize=10)

plt.tight_layout()
fig.savefig('regime_predictor_comparison.png', dpi=150)
plt.close(fig)

# Plot 3: Transition detection — zoom into actual transitions
# Find months where regime actually changed
regime_realized = (test['pi_smooth'] > 0.5).astype(int).values
transitions = np.where(np.diff(regime_realized) != 0)[0]

if len(transitions) > 0:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_dates, p_test, color='steelblue', linewidth=1.5,
            label='Ridge LR prediction')
    ax.plot(test_dates, pi_next_test, color='grey', linewidth=1.5,
            linestyle='--', label='HMM pi_next')
    ax.fill_between(test_dates, y_test, alpha=0.15, color='crimson',
                    label='Realized panic')
    ax.axhline(0.5, color='black', linewidth=0.5, linestyle=':')

    # Mark actual transitions
    for t_idx in transitions:
        ax.axvline(test_dates[t_idx], color='red', linewidth=1, linestyle='-',
                   alpha=0.5)

    ax.set_ylim(0, 1)
    ax.set_ylabel('P(panic next month)', fontsize=9)
    ax.set_xlabel('Date')
    ax.legend(fontsize=8)
    ax.set_title(f'Transition detection — red verticals = actual regime changes '
                 f'({len(transitions)} transitions in test period)\n'
                 f'Does the blue line (LR) rise before the grey line (HMM) at transitions?',
                 fontsize=10)
    plt.tight_layout()
    fig.savefig('regime_predictor_transitions.png', dpi=150)
    plt.close(fig)
    print("Saved: regime_predictor_transitions.png")

print("Saved: regime_predictor_weights.png  regime_predictor_comparison.png")
