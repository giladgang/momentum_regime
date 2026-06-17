"""
2026-06-16-nn-stock-picker.py
=============================
EXPLORATORY (experiments/, throwaway). Feedforward neural net (MLP) stock-picker
vs the production XGBoost in Stage 2, with an ablation:

  A  MLP, raw target, 13 features (mom_1..12 + pi_filter)        -- plain net
  B  MLP, winsorized target, 13 features                          -- outlier fix
  C  MLP, winsorized target, 13 + memory features                -- the swing

Memory features (month-level, point-in-time; computed only from past+current pi):
  pi_lag1, pi_lag3, pi_lag6   lagged regime probability
  pi_delta1                   pi_t - pi_{t-1}  (rate of change)
  panic_dur                   consecutive months with pi>=0.5 (0 when calm)

Apples-to-apples: same artefact (artefacts/cs_artefacts_data.pkl), same pre-2011
train / 2011-2025 test split, same L/S portfolio (src.utils.long_short_port),
same metrics. Scaler is refit per variant on TRAIN only (leakage-safe). Writes
nothing to thesis paths.

Env: NN_SEEDS (default 5), NN_MAX_ITER (default 200), NN_HIDDEN (default 32,16,8)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_artefacts, long_short_port, metrics
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

N_SEEDS  = int(os.environ.get('NN_SEEDS', 5))
MAX_ITER = int(os.environ.get('NN_MAX_ITER', 200))
HIDDEN   = tuple(int(x) for x in os.environ.get('NN_HIDDEN', '32,16,8').split(','))
print(f"[cfg] seeds={N_SEEDS} max_iter={MAX_ITER} hidden={HIDDEN}")

# ── load identical preprocessed data (UNSCALED matrices + dfs) ───────────────────
print("[1/4] Loading artefact ...")
d = load_artefacts()
X_train, X_test = d['X_train'], d['X_test']        # unscaled, FEATURES order
y_train_raw     = d['y_train']
train, test     = d['train'], d['test']
FEATURES        = d['FEATURES']
r_xgb           = d['strategies_lo']['Method 2: XGB']
print(f"  train={X_train.shape} test={X_test.shape} base features={len(FEATURES)}")

# ── month-level memory features from the contiguous pi timeline ──────────────────
pim = (pd.concat([train[['date', 'pi_filter']], test[['date', 'pi_filter']]])
       .drop_duplicates('date').sort_values('date').reset_index(drop=True))
for k in (1, 3, 6):
    pim[f'pi_lag{k}'] = pim['pi_filter'].shift(k).fillna(pim['pi_filter'])
pim['pi_delta1'] = (pim['pi_filter'] - pim['pi_filter'].shift(1)).fillna(0.0)
_panic = (pim['pi_filter'] >= 0.5).astype(int)
_run   = (_panic != _panic.shift()).cumsum()
pim['panic_dur'] = _panic.groupby(_run).cumsum().astype(float)
MEM = ['pi_lag1', 'pi_lag3', 'pi_lag6', 'pi_delta1', 'panic_dur']
mmap = pim.set_index('date')[MEM]
mem_train_df = mmap.reindex(train['date']).reset_index(drop=True)  # rows aligned w/ X_train
mem_test_df  = mmap.reindex(test['date']).reset_index(drop=True)
assert not mem_train_df.isna().any().any() and not mem_test_df.isna().any().any()

def winsor(y, p=0.01):
    lo, hi = np.percentile(y, [100 * p, 100 * (1 - p)])
    return np.clip(y, lo, hi)

def run_variant(winsorize, mem_cols, label):
    if mem_cols:
        Xtr = np.hstack([X_train, mem_train_df[mem_cols].values])
        Xte = np.hstack([X_test,  mem_test_df[mem_cols].values])
    else:
        Xtr, Xte = X_train, X_test
    sc  = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    ytr = winsor(y_train_raw) if winsorize else y_train_raw
    preds = np.zeros(Xte_s.shape[0])
    for s in range(N_SEEDS):
        m = MLPRegressor(hidden_layer_sizes=HIDDEN, activation='relu', solver='adam',
                         alpha=1e-3, batch_size=1024, learning_rate_init=1e-3,
                         early_stopping=True, n_iter_no_change=10, max_iter=MAX_ITER,
                         random_state=s)
        m.fit(Xtr_s, ytr)
        preds += m.predict(Xte_s)
    preds /= N_SEEDS
    t = test.copy(); t['score'] = preds
    print(f"  [{label}] done ({Xtr.shape[1]} feat, winsor={winsorize}, mem={mem_cols})")
    return long_short_port(t, 'score')

# ── run variants: isolate which memory feature degrades the MLP ──────────────────
# All built on B (winsorized base). C = all 5 memory feats (known bad). D/E/F
# isolate lags / delta / panic_dur to attribute the degradation.
print("[2/4] Training variants ...")
results = {'XGB (thesis)': r_xgb}
results['B: winsor base']    = run_variant(True, [],                                  'B')
results['C: +all memory']    = run_variant(True, MEM,                                 'C')
results['D: +lags only']     = run_variant(True, ['pi_lag1', 'pi_lag3', 'pi_lag6'],   'D')
results['E: +delta only']    = run_variant(True, ['pi_delta1'],                       'E')
results['F: +panic_dur only']= run_variant(True, ['panic_dur'],                       'F')

# ── metrics ──────────────────────────────────────────────────────────────────────
print("[3/4] Metrics\n")
pi_by_month = test[['date', 'pi_filter']].drop_duplicates('date').set_index('date')['pi_filter']
def regime_sharpe(r):
    r = pd.Series(r).dropna(); pi = pi_by_month.reindex(r.index)
    sr = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else float('nan')
    return sr(r), sr(r[pi < 0.5]), sr(r[pi >= 0.5])

print(f"{'Strategy':<16} {'Ann.Ret':>8} {'Ann.Vol':>8} {'Sharpe':>7} {'MaxDD':>8} "
      f"{'SR-calm':>8} {'SR-panic':>9}")
print("-" * 74)
for name, r in results.items():
    ar, av, sh, mdd = metrics(r)
    full, calm, panic = regime_sharpe(r)
    print(f"{name:<16} {ar:>7.1%} {av:>7.1%} {sh:>7.2f} {mdd:>7.1%} {calm:>8.2f} {panic:>9.2f}")

print("\n[4/4] Deltas vs XGB (Sharpe):")
sh_xgb = metrics(r_xgb)[2]
for name, r in results.items():
    if name == 'XGB (thesis)':
        continue
    print(f"  {name:<16} {metrics(r)[2] - sh_xgb:+.2f}")
