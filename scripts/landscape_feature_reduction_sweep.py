"""
landscape_feature_reduction_sweep.py
=====================================
Tests whether reducing feature count produces a more sensible unified
clustering (no regime split). Compares four configurations:

  A. baseline      : 12 means + 12 stds + pi          (25 features)
  B. no_std        : 12 means              + pi       (13 features)
  C. grouped_means : 3 mean tertiles       + pi       ( 4 features)
  D. grouped_both  : 3 mean tertiles + 3 std tertiles + pi  ( 7 features)

For each config, sweeps k=2..6 with KMeans and reports silhouette and
cluster sizes. If reducing features uncovers a clean regime separation,
it will show up as a sharp silhouette at k=2 with sizes near (108, 59).

Output:
- results/thesis/landscape_feature_reduction_sweep.csv
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RES_DIR = 'results/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'
RNG_SEED = 42
N_INIT = 20
K_RANGE = list(range(2, 7))
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

# Tertile groups
SHORT = [1, 2, 3, 4]
MID   = [5, 6, 7, 8]
LONG  = [9, 10, 11, 12]

# ----------------------------------------------------------------------
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])

agg_mean = test.groupby('date')[MOM_COLS].mean()
agg_std = test.groupby('date')[MOM_COLS].std()
pi = test.groupby('date')['pi_filter'].first()


def tertile(df, group):
    cols = [f'mom_{h}' for h in group]
    return df[cols].mean(axis=1)


# Pre-compute tertile aggregates
mean_short = tertile(agg_mean, SHORT)
mean_mid = tertile(agg_mean, MID)
mean_long = tertile(agg_mean, LONG)
std_short = tertile(agg_std, SHORT)
std_mid = tertile(agg_std, MID)
std_long = tertile(agg_std, LONG)


def build(config):
    if config == 'baseline':  # 25
        return pd.concat([
            agg_mean.rename(columns=lambda c: f'{c}_mean'),
            agg_std.rename(columns=lambda c: f'{c}_std'),
            pi.to_frame(),
        ], axis=1)
    if config == 'no_std':  # 13
        return pd.concat([
            agg_mean.rename(columns=lambda c: f'{c}_mean'),
            pi.to_frame(),
        ], axis=1)
    if config == 'grouped_means':  # 4
        return pd.DataFrame({
            'mean_short': mean_short,
            'mean_mid':   mean_mid,
            'mean_long':  mean_long,
            'pi_filter':  pi,
        })
    if config == 'grouped_both':  # 7
        return pd.DataFrame({
            'mean_short': mean_short,
            'mean_mid':   mean_mid,
            'mean_long':  mean_long,
            'std_short':  std_short,
            'std_mid':    std_mid,
            'std_long':   std_long,
            'pi_filter':  pi,
        })
    raise ValueError(config)


CONFIGS = ['baseline', 'no_std', 'grouped_means', 'grouped_both']

# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
rows = []
for cfg in CONFIGS:
    feats = build(cfg)
    X = StandardScaler().fit_transform(feats.values)
    print(f'\n=== {cfg} ({feats.shape[1]} features) ===')
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RNG_SEED,
                    n_init=N_INIT).fit(X)
        sil = float(silhouette_score(X, km.labels_))
        sizes = tuple(sorted(np.bincount(km.labels_, minlength=k).tolist()))

        # For k=2, also compute composition by π regime to flag whether the
        # split is calm-vs-panic or outlier-vs-rest.
        regime_purity = ''
        if k == 2:
            df = pd.DataFrame({
                'cluster': km.labels_,
                'is_panic': (pi.values > 0.5).astype(int),
            })
            ct = pd.crosstab(df['cluster'], df['is_panic'])
            # Higher purity = better calm/panic separation
            purity = max(
                ct.iloc[0, 0] + ct.iloc[1, 1],
                ct.iloc[0, 1] + ct.iloc[1, 0],
            ) / len(df)
            regime_purity = f'{purity:.2f}'

        rows.append({
            'config': cfg,
            'n_features': feats.shape[1],
            'k': k,
            'silhouette': round(sil, 3),
            'cluster_sizes': str(sizes),
            'k2_calm_panic_purity': regime_purity,
        })
        print(f'  k={k}: silhouette {sil:.3f}, sizes {sizes}'
              + (f', calm/panic purity {regime_purity}' if regime_purity else ''))

df = pd.DataFrame(rows)
out_path = f'{RES_DIR}/landscape_feature_reduction_sweep.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')

# Headline: which config gives the best k=2 calm/panic separation?
print('\n--- Calm/panic purity at k=2 (1.00 = perfect regime split) ---')
print(df[df['k'] == 2][['config', 'n_features', 'silhouette',
                        'k2_calm_panic_purity']].to_string(index=False))
