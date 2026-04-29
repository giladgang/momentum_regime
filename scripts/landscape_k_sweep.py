"""
landscape_k_sweep.py
=====================
Sweeps cluster count k for both input-landscape and output-selection
clusterings, within each regime. Uses the best feature config from the
ablation (means + std). For each k, reports:

  - input silhouette
  - output silhouette
  - ARI between input-k and output-k labels

Tells us whether k=3 is the natural cluster count or whether finer/coarser
splits change the input->output alignment.

Output:
- results/thesis/landscape_k_sweep.csv

Usage:
    python scripts/landscape_k_sweep.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RES_DIR = 'results/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'

RNG_SEED = 42
N_INIT = 20
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

# k_max chosen so each cluster has at least ~10 months on average.
K_RANGES = {'calm': range(2, 9), 'panic': range(2, 7)}

# ----------------------------------------------------------------------
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])

agg_mean = test.groupby('date')[MOM_COLS].mean()
agg_std = test.groupby('date')[MOM_COLS].std()
pi = test.groupby('date')['pi_filter'].first()

long_z = (pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv',
                      parse_dates=['date'])
          .set_index('date'))

calm_dates = pi.index[pi <= 0.5]
panic_dates = pi.index[pi > 0.5]

# ----------------------------------------------------------------------
def input_features(dates):
    feats = pd.concat([
        agg_mean.loc[dates].rename(columns=lambda c: f'{c}_mean'),
        agg_std.loc[dates].rename(columns=lambda c: f'{c}_std'),
    ], axis=1)
    return StandardScaler().fit_transform(feats.values)


def output_features(dates):
    return long_z.loc[dates, MOM_COLS].values


def sweep(regime_name, dates, k_range):
    X_in = input_features(dates)
    Y_out = output_features(dates)
    rows = []
    for k in k_range:
        km_in = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=N_INIT).fit(X_in)
        km_out = KMeans(n_clusters=k, random_state=RNG_SEED, n_init=N_INIT).fit(Y_out)
        sil_in = float(silhouette_score(X_in, km_in.labels_))
        sil_out = float(silhouette_score(Y_out, km_out.labels_))
        ari = float(adjusted_rand_score(km_in.labels_, km_out.labels_))
        rows.append({
            'regime': regime_name,
            'k': k,
            'input_silhouette': round(sil_in, 3),
            'output_silhouette': round(sil_out, 3),
            'ari': round(ari, 3),
            'min_cluster_size': int(min(np.bincount(km_in.labels_, minlength=k))),
        })
    return pd.DataFrame(rows)


calm_df = sweep('calm', calm_dates, K_RANGES['calm'])
panic_df = sweep('panic', panic_dates, K_RANGES['panic'])

print('\n=== CALM (n=108) ===')
print(calm_df.drop(columns='regime').to_string(index=False))
print('\n=== PANIC (n=59) ===')
print(panic_df.drop(columns='regime').to_string(index=False))

out_path = f'{RES_DIR}/landscape_k_sweep.csv'
pd.concat([calm_df, panic_df], ignore_index=True).to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
