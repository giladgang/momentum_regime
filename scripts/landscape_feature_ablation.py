"""
landscape_feature_ablation.py
==============================
Tests three hypotheses about why the cross-sectional MEAN of mom_h alone
doesn't predict the model's selection shape (low ARI in
landscape_subregime_analysis.py):

  (1) Higher cross-sectional moments — add per-horizon std and skew
      across stocks. Captures dispersion and asymmetry.
  (2) Stock-level routing — partly captured by std/skew, since stock
      heterogeneity shows up as cross-sectional dispersion.
  (3) pi_filter within regime — variation in pi_filter even after
      conditioning on the regime label.

Six feature configurations per regime, each clustered with KMeans(k=3)
and compared to the output-side k=3 clustering via ARI:

    means              (12-d)     baseline
    means + pi         (13-d)     tests (3)
    means + std        (24-d)     tests (1)
    means + std + pi   (25-d)     tests (1) + (3)
    means + std + skew (36-d)     tests (1) more fully
    means + std + skew + pi (37)  full

If ARI rises monotonically through the configs, we're capturing more of
what the model actually responds to. If it plateaus below the output
silhouette ceiling, what remains must be true stock-level routing that
no cross-sectional aggregate can recover.

Output:
- results/thesis/landscape_feature_ablation.csv

Usage:
    python scripts/landscape_feature_ablation.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
RES_DIR = 'results/thesis'
ARTEFACTS_PATH = 'artefacts/cs_artefacts_data.pkl'

RNG_SEED = 42
N_INIT = 20
K = 3
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]

# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------
with open(ARTEFACTS_PATH, 'rb') as f:
    art = pickle.load(f)
test = art['test'].copy()
test['date'] = pd.to_datetime(test['date'])

# Per-month aggregates
agg_mean = test.groupby('date')[MOM_COLS].mean()
agg_std = test.groupby('date')[MOM_COLS].std()
agg_skew = test.groupby('date')[MOM_COLS].skew()
pi = test.groupby('date')['pi_filter'].first()

# Output-side selection shape per month (z-scored, no further scaling)
long_z = pd.read_csv(f'{RES_DIR}/zscore_long_by_month.csv', parse_dates=['date'])
long_z = long_z.set_index('date')

# Align everything by date
all_dates = agg_mean.index
assert (agg_std.index == all_dates).all()
assert (agg_skew.index == all_dates).all()
assert (pi.index == all_dates).all()
assert (long_z.index == all_dates).all()

print(f'Total OOS months: {len(all_dates)}')

# ----------------------------------------------------------------------
# Feature configs
# ----------------------------------------------------------------------
def build_features(dates, include_std, include_skew, include_pi):
    pieces = [agg_mean.loc[dates].rename(columns=lambda c: f'{c}_mean')]
    if include_std:
        pieces.append(agg_std.loc[dates].rename(columns=lambda c: f'{c}_std'))
    if include_skew:
        pieces.append(agg_skew.loc[dates].rename(columns=lambda c: f'{c}_skew'))
    if include_pi:
        pieces.append(pi.loc[dates].to_frame())
    return pd.concat(pieces, axis=1)


CONFIGS = [
    ('means',                 dict(include_std=False, include_skew=False, include_pi=False)),
    ('means+pi',              dict(include_std=False, include_skew=False, include_pi=True)),
    ('means+std',             dict(include_std=True,  include_skew=False, include_pi=False)),
    ('means+std+pi',          dict(include_std=True,  include_skew=False, include_pi=True)),
    ('means+std+skew',        dict(include_std=True,  include_skew=True,  include_pi=False)),
    ('means+std+skew+pi',     dict(include_std=True,  include_skew=True,  include_pi=True)),
]

# ----------------------------------------------------------------------
# Per-regime evaluation
# ----------------------------------------------------------------------
def output_labels(dates):
    """Recompute the output-side k=3 clustering on the long-leg z-profile
    for these dates."""
    Z = long_z.loc[dates, MOM_COLS].values
    km = KMeans(n_clusters=K, random_state=RNG_SEED, n_init=N_INIT).fit(Z)
    return km.labels_, float(silhouette_score(Z, km.labels_))


def knn_r2_to_output(X, Y, k=5):
    """Honest kNN R^2 from input features to 12-d output shape via LOO CV.
    Y is the multivariate output (long-leg z-profile, 12 columns).
    Returns the uniform-averaged R^2 across output dimensions.
    """
    knn = KNeighborsRegressor(n_neighbors=k)
    Y_pred = cross_val_predict(knn, X, Y, cv=LeaveOneOut(), n_jobs=-1)
    return float(r2_score(Y, Y_pred, multioutput='uniform_average'))


def evaluate(regime_name, dates):
    print(f'\n=== {regime_name} (n={len(dates)}) ===')
    out_labels, out_sil = output_labels(dates)
    Y = long_z.loc[dates, MOM_COLS].values  # 12-d output target for kNN R^2
    print(f'output-side silhouette: {out_sil:.3f}')

    rows = []
    for cfg_name, cfg_kwargs in CONFIGS:
        feats = build_features(dates, **cfg_kwargs)
        X = StandardScaler().fit_transform(feats.values)
        km = KMeans(n_clusters=K, random_state=RNG_SEED, n_init=N_INIT).fit(X)
        sil = float(silhouette_score(X, km.labels_))
        ari = float(adjusted_rand_score(km.labels_, out_labels))
        r2 = knn_r2_to_output(X, Y, k=5)
        sizes = tuple(sorted(np.bincount(km.labels_, minlength=K).tolist()))
        rows.append({
            'regime': regime_name,
            'config': cfg_name,
            'n_features': feats.shape[1],
            'input_silhouette': round(sil, 3),
            'output_silhouette': round(out_sil, 3),
            'ari_vs_output': round(ari, 3),
            'knn_r2_to_output': round(r2, 3),
            'cluster_sizes': str(sizes),
        })
    df = pd.DataFrame(rows)
    print(df[['config', 'n_features', 'input_silhouette',
              'output_silhouette', 'ari_vs_output',
              'knn_r2_to_output', 'cluster_sizes']].to_string(index=False))
    return df


calm_dates = pi.index[pi <= 0.5]
panic_dates = pi.index[pi > 0.5]

calm_df = evaluate('calm', calm_dates)
panic_df = evaluate('panic', panic_dates)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
out_path = f'{RES_DIR}/landscape_feature_ablation.csv'
all_df = pd.concat([calm_df, panic_df], ignore_index=True)
all_df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')

print('\n--- Headline ---')
for regime in ['calm', 'panic']:
    sub = all_df[all_df['regime'] == regime]
    base_ari = sub.iloc[0]['ari_vs_output']
    best_ari = sub['ari_vs_output'].max()
    best_cfg = sub.loc[sub['ari_vs_output'].idxmax(), 'config']
    print(f'{regime}: ARI {base_ari:.3f} (means) -> {best_ari:.3f} ({best_cfg})')
