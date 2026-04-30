# scripts/picked_stock_cluster_search.py
"""Picked-stock cluster + feature search.

Stage 1 (this task): cluster months on the 15-d picked-stock fingerprint;
sweep K=2..5 with 6-seed stability and silhouette; pick smallest stable K
with sane size distribution; save labels + centroids + sweep CSV.

Stages 2-3 (later tasks) extend this script with predictor features,
ANOVA, multinomial logistic, decision tree, and per-cluster summary.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_feature_search_helpers import (
    picked_stock_fingerprint,
    pi_panic_freq,
    past_strategy_sharpe,
    cross_section_skew,
    SHORT_H, MID_H, LONG_H,
)

RES_DIR = 'results/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]
K_GRID = [2, 3, 4, 5]
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
N_INIT = 20
ARI_THRESHOLD = 0.90
MIN_CLUSTER_SIZE_FRAC = 0.05  # smallest cluster must be >=5% of months


def tertile_mean(df, group):
    """Mean of mom_h columns for h in `group` (a list of horizon ints)."""
    cols = [f'mom_{h}' for h in group]
    return df[cols].mean(axis=1)


def build_feature_panel(panel, m2_returns, dates):
    """Build the 14-feature predictor panel for the given dates.

    Parameters
    ----------
    panel : DataFrame full cross-section (date, permno, mom_1..mom_12, pi_filter)
    m2_returns : pd.Series M2 monthly returns indexed by date
    dates : iterable of pd.Timestamp -- months to include
    """
    pi = panel.groupby('date')['pi_filter'].first().sort_index()
    agg_mean = panel.groupby('date')[MOM_COLS].mean()
    agg_std = panel.groupby('date')[MOM_COLS].std()
    skew_df = cross_section_skew(panel, MOM_COLS)

    rows = []
    for d in dates:
        rows.append({
            'date': d,
            # Group A -- context
            'pi_panic': int(pi.loc[d] > 0.5),
            'cs_mom_short': float(tertile_mean(agg_mean.loc[[d]], SHORT_H).iloc[0]),
            'cs_mom_mid':   float(tertile_mean(agg_mean.loc[[d]], MID_H).iloc[0]),
            'cs_mom_long':  float(tertile_mean(agg_mean.loc[[d]], LONG_H).iloc[0]),
            'mom_overall':  float(agg_mean.loc[d, MOM_COLS].mean()),
            'cs_disp_short': float(tertile_mean(agg_std.loc[[d]], SHORT_H).iloc[0]),
            'cs_disp_mid':   float(tertile_mean(agg_std.loc[[d]], MID_H).iloc[0]),
            'cs_disp_long':  float(tertile_mean(agg_std.loc[[d]], LONG_H).iloc[0]),
            # Group B -- time-series
            'pi_panic_freq_6mo':  pi_panic_freq(pi, d, n_months=6),
            'pi_panic_freq_12mo': pi_panic_freq(pi, d, n_months=12),
            'past_sharpe_12mo':   past_strategy_sharpe(m2_returns, d, n_months=12),
            # Group C -- cross-section skew
            'cs_skew_short': float(tertile_mean(skew_df.loc[[d]], SHORT_H).iloc[0]),
            'cs_skew_mid':   float(tertile_mean(skew_df.loc[[d]], MID_H).iloc[0]),
            'cs_skew_long':  float(tertile_mean(skew_df.loc[[d]], LONG_H).iloc[0]),
        })
    return pd.DataFrame(rows)


def cluster_sweep(X_std):
    """Run KMeans for each K in K_GRID, with 6-seed stability + silhouette."""
    rows = []
    labels_by_k = {}
    for k in K_GRID:
        fits = []
        sils = []
        for s in STABILITY_SEEDS:
            km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(X_std)
            fits.append(km.labels_)
            sils.append(silhouette_score(X_std, km.labels_))
        pair_aris = []
        for i in range(len(STABILITY_SEEDS)):
            for j in range(i + 1, len(STABILITY_SEEDS)):
                pair_aris.append(adjusted_rand_score(fits[i], fits[j]))
        sizes = sorted(np.bincount(fits[0], minlength=k).tolist())
        rows.append({
            'k': k,
            'mean_silhouette': round(float(np.mean(sils)), 4),
            'mean_ari': round(float(np.mean(pair_aris)), 4),
            'min_size_frac': round(min(sizes) / len(X_std), 4),
            'sizes_seed42': str(tuple(sizes)),
            'stable': bool(np.mean(pair_aris) >= ARI_THRESHOLD),
        })
        labels_by_k[k] = fits[0]
        print(f'  k={k}: silhouette={rows[-1]["mean_silhouette"]:.3f}, '
              f'ari={rows[-1]["mean_ari"]:.3f}, sizes={sizes}, '
              f'stable={rows[-1]["stable"]}', flush=True)
    return pd.DataFrame(rows), labels_by_k


def pick_k(sweep_df):
    """Smallest K that is stable AND whose smallest cluster is >=5%."""
    eligible = sweep_df[
        sweep_df['stable']
        & (sweep_df['min_size_frac'] >= MIN_CLUSTER_SIZE_FRAC)
    ].sort_values('k')
    if len(eligible) == 0:
        # Fall back to most stable K
        return int(sweep_df.sort_values('mean_ari', ascending=False).iloc[0]['k'])
    return int(eligible.iloc[0]['k'])


def main():
    print('Loading data...', flush=True)
    picks = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv',
                        parse_dates=['date'])[['date', 'permno']]
    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    panel = art['test'].copy()
    panel['date'] = pd.to_datetime(panel['date'])

    print('Computing fingerprint...', flush=True)
    fp = picked_stock_fingerprint(picks, panel, MOM_COLS)
    fp = fp.dropna()
    print(f'  fingerprint shape: {fp.shape}', flush=True)

    X = fp.values.astype(np.float64)
    X_std = StandardScaler().fit_transform(X)

    print('\n=== Cluster sweep K=2..5 ===', flush=True)
    sweep_df, labels_by_k = cluster_sweep(X_std)
    sweep_df.to_csv(f'{RES_DIR}/picked_stock_cluster_sweep.csv', index=False)
    print(f'\nSaved: {RES_DIR}/picked_stock_cluster_sweep.csv', flush=True)
    print(sweep_df.to_string(index=False))

    chosen_k = pick_k(sweep_df)
    print(f'\n>>> CHOSEN K = {chosen_k}', flush=True)

    labels = labels_by_k[chosen_k]
    label_df = pd.DataFrame({
        'date': fp.index,
        'cluster': labels,
    })
    label_df.to_csv(f'{RES_DIR}/picked_stock_cluster_labels.csv', index=False)
    print(f'Saved: {RES_DIR}/picked_stock_cluster_labels.csv', flush=True)

    # Centroids on raw (un-standardised) fingerprint
    centroids = fp.assign(cluster=labels).groupby('cluster').mean()
    centroids['n_months'] = pd.Series(labels).value_counts().sort_index().values
    centroids.to_csv(f'{RES_DIR}/picked_stock_cluster_centroids.csv')
    print(f'Saved: {RES_DIR}/picked_stock_cluster_centroids.csv', flush=True)
    print('\nCentroids (raw mom) per cluster:')
    print(centroids.round(3).to_string())

    # ---- Predictor feature panel ----
    print('\nLoading M2 returns...', flush=True)
    with open(f'{RES_DIR}/fundamentals_returns.pkl', 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    print('Building predictor feature panel...', flush=True)
    feat = build_feature_panel(panel, m2_ret, list(fp.index))
    feat = feat.merge(label_df, on='date', how='left')
    feat.to_csv(f'{RES_DIR}/picked_stock_feature_panel.csv', index=False)
    print(f'Saved: {RES_DIR}/picked_stock_feature_panel.csv', flush=True)
    print(f'\nFeature panel shape: {feat.shape}')
    print('\nMissingness per feature:')
    print(feat.isna().sum())


if __name__ == '__main__':
    main()
