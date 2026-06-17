"""
2026-06-16-nn-perstock-lags.py
==============================
EXPLORATORY gate (Probe A): does PER-STOCK idiosyncratic sequential history help
the MLP? This is the feedforward stand-in for a per-stock LSTM. If it can't beat
the plain winsorized MLP (Sharpe ~0.71), a per-stock LSTM won't either -> skip it.

Per-stock lags (each stock's OWN past, via groupby('permno').shift()):
  mom_1 at t-1,t-2,t-3   stock's recent monthly returns (the disaggregated path)
  pi at t-1,t-2,t-3      (redundant with macro lags, included per the prescription)

Lags are computed on train+test CONCATENATED and sorted by (permno,date) so test
rows get their true t-1 even when it lives in the train period (no boundary hole),
then re-aligned to the original X_train/X_test row order via the DataFrame index
(df was reset_index(drop=True) upstream, so labels are unique).

Compares: XGB (1.11) | B winsor base (0.71) | G base + per-stock lags.
DECISION: pursue LSTM only if G >= ~0.95 Sharpe AND MaxDD no worse than ~-25%.

Env: NN_SEEDS (default 5), NN_MAX_ITER (default 200)
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
HIDDEN   = (32, 16, 8)
print(f"[cfg] seeds={N_SEEDS} max_iter={MAX_ITER} hidden={HIDDEN}")

print("[1/4] Loading artefact ...")
d = load_artefacts()
X_train, X_test = d['X_train'], d['X_test']
y_train_raw     = d['y_train']
train, test     = d['train'], d['test']
r_xgb           = d['strategies_lo']['Method 2: XGB']
assert train.index.is_unique and test.index.is_unique
assert len(train) == X_train.shape[0] and len(test) == X_test.shape[0]

# ── per-stock lags across the train/test boundary, re-aligned to X row order ─────
print("[2/4] Building per-stock lags ...")
tr = train[['permno', 'date', 'mom_1', 'pi_filter']].copy()
te = test[['permno', 'date', 'mom_1', 'pi_filter']].copy()
both = pd.concat([tr, te]).sort_values(['permno', 'date'])
lagcols = []
for col in ['mom_1', 'pi_filter']:
    for k in (1, 2, 3):
        name = f'{col}_slag{k}'
        both[name] = both.groupby('permno')[col].shift(k)
        # new-entrant NaN: pi-lag -> current pi (no change); return-lag -> 0
        both[name] = both[name].fillna(both['pi_filter'] if col == 'pi_filter' else 0.0)
        lagcols.append(name)
slag_train = both.loc[train.index, lagcols].values   # back to X_train order
slag_test  = both.loc[test.index,  lagcols].values
assert slag_train.shape == (X_train.shape[0], 6) and not np.isnan(slag_train).any()
assert slag_test.shape  == (X_test.shape[0],  6) and not np.isnan(slag_test).any()
print(f"  added {len(lagcols)} per-stock lag features: {lagcols}")

def winsor(y, p=0.01):
    lo, hi = np.percentile(y, [100 * p, 100 * (1 - p)])
    return np.clip(y, lo, hi)

def run(add_lags, label):
    Xtr = np.hstack([X_train, slag_train]) if add_lags else X_train
    Xte = np.hstack([X_test,  slag_test])  if add_lags else X_test
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    ytr = winsor(y_train_raw)
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
    print(f"  [{label}] done ({Xtr.shape[1]} feat)")
    return long_short_port(t, 'score')

print("[3/4] Training ...")
results = {'XGB (thesis)': r_xgb,
           'B: winsor base':        run(False, 'B'),
           'G: + per-stock lags':   run(True,  'G')}

print("[4/4] Results\n")
pi_by_month = test[['date', 'pi_filter']].drop_duplicates('date').set_index('date')['pi_filter']
def regime_sharpe(r):
    r = pd.Series(r).dropna(); pi = pi_by_month.reindex(r.index)
    sr = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 1 and x.std() > 0 else float('nan')
    return sr(r), sr(r[pi < 0.5]), sr(r[pi >= 0.5])

print(f"{'Strategy':<22} {'Sharpe':>7} {'MaxDD':>8} {'SR-calm':>8} {'SR-panic':>9}")
print("-" * 60)
for name, r in results.items():
    _, _, sh, mdd = metrics(r)
    _, calm, panic = regime_sharpe(r)
    print(f"{name:<22} {sh:>7.2f} {mdd:>7.1%} {calm:>8.2f} {panic:>9.2f}")

sh_b = metrics(results['B: winsor base'])[2]
sh_g = metrics(results['G: + per-stock lags'])[2]
mdd_g = metrics(results['G: + per-stock lags'])[3]
print(f"\nGate: G={sh_g:.2f} vs B={sh_b:.2f} (delta {sh_g - sh_b:+.2f}), G MaxDD={mdd_g:.1%}")
verdict = "PURSUE LSTM" if (sh_g >= 0.95 and mdd_g >= -0.25) else "SKIP LSTM"
print(f"DECISION RULE -> {verdict}  (pursue iff G>=0.95 and MaxDD>=-25%)")
