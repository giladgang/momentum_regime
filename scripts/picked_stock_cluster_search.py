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
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, silhouette_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

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
    m2_returns : pd.Series XGB monthly returns indexed by date
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


PRED_COLS = [
    'pi_panic',
    'cs_mom_short', 'cs_mom_mid', 'cs_mom_long', 'mom_overall',
    'cs_disp_short', 'cs_disp_mid', 'cs_disp_long',
    'pi_panic_freq_6mo', 'pi_panic_freq_12mo', 'past_sharpe_12mo',
    'cs_skew_short', 'cs_skew_mid', 'cs_skew_long',
]


def univariate_ranking(feat_df):
    """ANOVA F-statistic per feature against cluster labels."""
    valid = feat_df[PRED_COLS + ['cluster']].dropna()
    F, p = f_classif(valid[PRED_COLS].values, valid['cluster'].values)
    rank = pd.DataFrame({
        'feature': PRED_COLS,
        'F_statistic': F,
        'p_value': p,
    }).sort_values('F_statistic', ascending=False)
    return rank


def multinomial_l1_loo(feat_df, C_grid=(0.05, 0.1, 0.3, 1.0, 3.0)):
    """Multinomial L1 logistic regression, LOO-CV across C grid."""
    valid = feat_df[PRED_COLS + ['cluster']].dropna()
    X = valid[PRED_COLS].values
    y = valid['cluster'].values
    X_std = StandardScaler().fit_transform(X)

    print('\n  LOO-CV across C grid:')
    best_acc = -1.0
    best_C = None
    for C in C_grid:
        lr = LogisticRegression(
            solver='saga', l1_ratio=1, C=C,
            max_iter=10000,
        )
        y_pred = cross_val_predict(lr, X_std, y, cv=LeaveOneOut(), n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        print(f'    C={C:>5.2f}: acc={acc:.3f}', flush=True)
        if acc > best_acc:
            best_acc = acc
            best_C = C

    lr = LogisticRegression(
        solver='saga', l1_ratio=1, C=best_C,
        max_iter=10000,
    ).fit(X_std, y)
    # Binary case: coef_ is shape (1, n_features); use positive-class label as column.
    # Multinomial case: coef_ is shape (n_classes, n_features); use all class labels.
    if lr.coef_.shape[0] == 1:
        col_labels = [lr.classes_[1]]
    else:
        col_labels = list(lr.classes_)
    coef = pd.DataFrame(lr.coef_.T, index=PRED_COLS, columns=col_labels)
    coef.index.name = 'feature'
    baseline = float(pd.Series(y).value_counts(normalize=True).max())
    return coef, best_acc, baseline, best_C


def decision_tree_readout(feat_df, max_depth=3):
    """Fit a depth-3 tree, return text rule + train + LOO accuracy."""
    valid = feat_df[PRED_COLS + ['cluster']].dropna()
    X = valid[PRED_COLS].values
    y = valid['cluster'].values
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42).fit(X, y)
    text = export_text(tree, feature_names=PRED_COLS)
    train_acc = accuracy_score(y, tree.predict(X))
    y_loo = cross_val_predict(
        DecisionTreeClassifier(max_depth=max_depth, random_state=42),
        X, y, cv=LeaveOneOut(), n_jobs=-1,
    )
    loo_acc = accuracy_score(y, y_loo)
    return text, train_acc, loo_acc


def per_cluster_summary(feat_df, fp_df):
    """Mean +/- std of every fingerprint dim AND every predictor feature, per cluster."""
    fp_with_cluster = fp_df.merge(
        feat_df[['date', 'cluster']], left_index=True, right_on='date',
    ).drop(columns='date')
    fp_summary = fp_with_cluster.groupby('cluster').agg(['mean', 'std'])
    feat_summary = (
        feat_df[PRED_COLS + ['cluster']]
        .groupby('cluster').agg(['mean', 'std'])
    )
    full = pd.concat([fp_summary, feat_summary], axis=1)
    return full


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
    print('\nLoading XGB returns...', flush=True)
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

    # ---- Univariate ranking ----
    print('\n=== Univariate ANOVA ranking ===', flush=True)
    rank = univariate_ranking(feat)
    rank.to_csv(f'{RES_DIR}/picked_stock_feature_ranking.csv', index=False)
    print(rank.to_string(index=False, float_format='%.3f'))
    print(f'Saved: {RES_DIR}/picked_stock_feature_ranking.csv')

    # ---- Multinomial L1 logistic with LOO-CV ----
    print('\n=== Multinomial L1 logistic (LOO-CV) ===', flush=True)
    coef, loo_acc, baseline, best_C = multinomial_l1_loo(feat)
    coef.to_csv(f'{RES_DIR}/picked_stock_logistic_coefficients.csv')
    print(f'\n  best C = {best_C}, LOO-CV acc = {loo_acc:.3f}, baseline (max class) = {baseline:.3f}')
    print('  Coefficients:')
    print(coef.round(3).to_string())
    print(f'Saved: {RES_DIR}/picked_stock_logistic_coefficients.csv')

    # ---- Decision tree ----
    print('\n=== Depth-3 decision tree ===', flush=True)
    tree_text, tree_train_acc, tree_loo_acc = decision_tree_readout(feat, max_depth=3)
    with open(f'{RES_DIR}/picked_stock_decision_tree.txt', 'w') as f:
        f.write(tree_text)
    print(tree_text)
    print(f'  train acc = {tree_train_acc:.3f}, LOO acc = {tree_loo_acc:.3f}')
    print(f'Saved: {RES_DIR}/picked_stock_decision_tree.txt')

    # ---- Per-cluster summary ----
    print('\n=== Per-cluster summary ===', flush=True)
    summary = per_cluster_summary(feat, fp)
    summary.to_csv(f'{RES_DIR}/picked_stock_cluster_summary.csv')
    print(f'Saved: {RES_DIR}/picked_stock_cluster_summary.csv')

    # ---- Headline verdict ----
    print('\n' + '=' * 60)
    print('  HEADLINE')
    print('=' * 60)
    print(f'  Chosen K:                     {chosen_k}')
    print(f'  Logistic LOO-CV accuracy:     {loo_acc:.3f}')
    print(f'  Decision tree LOO-CV accuracy: {tree_loo_acc:.3f}')
    print(f'  Class-imbalance baseline:     {baseline:.3f}')
    if loo_acc >= 0.70:
        print('\n  STRONG: features cleanly recover cluster structure.')
        print('  -> Use top features as the natural group-by variables for §5.2.')
    elif loo_acc >= 0.60:
        print('\n  WEAK: marginal predictability; check decision tree for partial split.')
    else:
        print('\n  NEGATIVE: no month-level feature predicts cluster membership.')
        print('  -> Picks are determined below the monthly aggregate (stock-level).')
        print('  -> Establishes a boundary for the §5.2 narrative.')


if __name__ == '__main__':
    main()
