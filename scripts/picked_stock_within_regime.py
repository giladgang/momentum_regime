# scripts/picked_stock_within_regime.py
"""Within-regime sub-cluster analysis of picked-stock fingerprint.

Conditions on regime (calm / panic) first, then runs K=2 KMeans within each
subset of months.  Isolates within-regime structure that the aggregate K=2
analysis collapsed because mom_overall and pi are correlated.

Outputs (all in results/thesis/):
  picked_stock_within_calm_centroids.csv
  picked_stock_within_panic_centroids.csv
  picked_stock_within_regime_labels.csv
  picked_stock_within_calm_feature_ranking.csv
  picked_stock_within_panic_feature_ranking.csv
  picked_stock_within_calm_logistic_coefficients.csv
  picked_stock_within_panic_logistic_coefficients.csv
  picked_stock_within_calm_decision_tree.txt
  picked_stock_within_panic_decision_tree.txt
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
from bootstrap_helpers import block_bootstrap_sharpe

RES_DIR = 'results/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
N_INIT = 20
ARI_THRESHOLD = 0.85  # spec: require >=0.85 stability

# 13 predictor features (pi_panic excluded -- constant within each subset)
PRED_COLS = [
    'cs_mom_short', 'cs_mom_mid', 'cs_mom_long', 'mom_overall',
    'cs_disp_short', 'cs_disp_mid', 'cs_disp_long',
    'pi_panic_freq_6mo', 'pi_panic_freq_12mo', 'past_sharpe_12mo',
    'cs_skew_short', 'cs_skew_mid', 'cs_skew_long',
]


# ── helpers ──────────────────────────────────────────────────────────────────

def tertile_mean(df, group):
    """Mean of mom_h columns for h in `group` (a list of horizon ints)."""
    cols = [f'mom_{h}' for h in group]
    return df[cols].mean(axis=1)


def build_feature_panel(panel, m2_returns, dates):
    """Build the 13-feature predictor panel (no pi_panic) for given dates."""
    pi = panel.groupby('date')['pi_filter'].first().sort_index()
    agg_mean = panel.groupby('date')[MOM_COLS].mean()
    agg_std = panel.groupby('date')[MOM_COLS].std()
    skew_df = cross_section_skew(panel, MOM_COLS)

    rows = []
    for d in dates:
        rows.append({
            'date': d,
            'cs_mom_short': float(tertile_mean(agg_mean.loc[[d]], SHORT_H).iloc[0]),
            'cs_mom_mid':   float(tertile_mean(agg_mean.loc[[d]], MID_H).iloc[0]),
            'cs_mom_long':  float(tertile_mean(agg_mean.loc[[d]], LONG_H).iloc[0]),
            'mom_overall':  float(agg_mean.loc[d, MOM_COLS].mean()),
            'cs_disp_short': float(tertile_mean(agg_std.loc[[d]], SHORT_H).iloc[0]),
            'cs_disp_mid':   float(tertile_mean(agg_std.loc[[d]], MID_H).iloc[0]),
            'cs_disp_long':  float(tertile_mean(agg_std.loc[[d]], LONG_H).iloc[0]),
            'pi_panic_freq_6mo':  pi_panic_freq(pi, d, n_months=6),
            'pi_panic_freq_12mo': pi_panic_freq(pi, d, n_months=12),
            'past_sharpe_12mo':   past_strategy_sharpe(m2_returns, d, n_months=12),
            'cs_skew_short': float(tertile_mean(skew_df.loc[[d]], SHORT_H).iloc[0]),
            'cs_skew_mid':   float(tertile_mean(skew_df.loc[[d]], MID_H).iloc[0]),
            'cs_skew_long':  float(tertile_mean(skew_df.loc[[d]], LONG_H).iloc[0]),
        })
    return pd.DataFrame(rows)


def run_kmeans_stability(X_std, k=2):
    """Run KMeans K=2 across STABILITY_SEEDS; return labels from seed-42 fit,
    mean pairwise ARI, mean silhouette."""
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
    mean_ari = float(np.mean(pair_aris))
    mean_sil = float(np.mean(sils))
    sizes = sorted(np.bincount(fits[0], minlength=k).tolist())
    return fits[0], mean_ari, mean_sil, sizes


def order_by_mom_overall(labels, feat_df):
    """Re-map labels so that cluster 0 = higher mean mom_overall (within-regime).

    Returns re-labelled integer array same length as `labels`.
    """
    df = feat_df[['mom_overall']].copy()
    df['cluster'] = labels
    means = df.groupby('cluster')['mom_overall'].mean()
    # cluster with higher mean mom_overall becomes 0
    ordered = means.sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(ordered)}
    return np.array([remap[l] for l in labels])


def compute_regime_centroids(fp_subset, labels, m2_ret, dates_subset):
    """Compute centroid row for each sub-cluster including Sharpe + CI."""
    fp_subset = fp_subset.copy()
    fp_subset['cluster'] = labels
    centroids = fp_subset.groupby('cluster').mean()
    centroids['n_months'] = pd.Series(labels).value_counts().sort_index().values

    m2_ret_idx = pd.to_datetime(m2_ret.index)
    date_arr = pd.to_datetime(dates_subset)

    for cl in [0, 1]:
        cl_dates = date_arr[labels == cl]
        mask = m2_ret_idx.isin(cl_dates)
        cl_rets = m2_ret.values[mask]
        if len(cl_rets) == 0:
            centroids.loc[cl, 'sharpe'] = np.nan
            centroids.loc[cl, 'sharpe_lo95'] = np.nan
            centroids.loc[cl, 'sharpe_hi95'] = np.nan
        else:
            bbs = block_bootstrap_sharpe(cl_rets, block_size=6, n_reps=5000, seed=42)
            centroids.loc[cl, 'sharpe'] = bbs['sharpe_point']
            centroids.loc[cl, 'sharpe_lo95'] = bbs['sharpe_lo95']
            centroids.loc[cl, 'sharpe_hi95'] = bbs['sharpe_hi95']
            # Sanity check
            s = bbs['sharpe_point']
            if np.isfinite(s) and (s < -2 or s > 3):
                print(f'  WARNING: cluster {cl} Sharpe={s:.3f} outside [-2, 3]')
    return centroids


def univariate_ranking(feat_df, labels):
    """ANOVA F-statistic per feature vs sub-cluster labels."""
    df = feat_df[PRED_COLS].copy()
    df['cluster'] = labels
    valid = df.dropna()
    F, p = f_classif(valid[PRED_COLS].values, valid['cluster'].values)
    rank = pd.DataFrame({
        'feature': PRED_COLS,
        'F_statistic': F,
        'p_value': p,
    }).sort_values('F_statistic', ascending=False)
    return rank


def multinomial_l1_loo(feat_df, labels, C_grid=(0.05, 0.1, 0.3, 1.0, 3.0)):
    """Binary L1 logistic regression (saga, l1_ratio=1), LOO-CV across C grid.

    Uses the sklearn 1.8+ API: l1_ratio=1.0 with solver='saga' for pure
    L1 penalty; avoids deprecated penalty= and solver= kwargs.
    """
    df = feat_df[PRED_COLS].copy()
    df['cluster'] = labels
    valid = df.dropna()
    X = valid[PRED_COLS].values
    y = valid['cluster'].values
    X_std = StandardScaler().fit_transform(X)

    print('  LOO-CV across C grid (saga l1_ratio=1):', flush=True)
    best_acc = -1.0
    best_C = None
    for C in C_grid:
        lr = LogisticRegression(
            solver='saga', l1_ratio=1.0, C=C, max_iter=10000,
        )
        y_pred = cross_val_predict(lr, X_std, y, cv=LeaveOneOut(), n_jobs=-1)
        acc = accuracy_score(y, y_pred)
        print(f'    C={C:>5.2f}: acc={acc:.3f}', flush=True)
        if acc > best_acc:
            best_acc = acc
            best_C = C

    lr = LogisticRegression(
        solver='saga', l1_ratio=1.0, C=best_C, max_iter=10000,
    ).fit(X_std, y)
    # Binary: coef_ shape (1, n_features); report as column labelled with positive class
    if lr.coef_.shape[0] == 1:
        col_labels = [int(lr.classes_[1])]
    else:
        col_labels = list(lr.classes_)
    coef = pd.DataFrame(lr.coef_.T, index=PRED_COLS, columns=col_labels)
    coef.index.name = 'feature'
    baseline = float(pd.Series(y).value_counts(normalize=True).max())
    return coef, best_acc, baseline, best_C


def decision_tree_readout(feat_df, labels, max_depth=3):
    """Fit a depth-3 tree; return text rule + train + LOO accuracy."""
    df = feat_df[PRED_COLS].copy()
    df['cluster'] = labels
    valid = df.dropna()
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading data...', flush=True)
    picks = pd.read_csv(f'{RES_DIR}/rule_path_labels.csv',
                        parse_dates=['date'])[['date', 'permno']]
    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    panel = art['test'].copy()
    panel['date'] = pd.to_datetime(panel['date'])

    with open(f'{RES_DIR}/fundamentals_returns.pkl', 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    # ── fingerprint ──────────────────────────────────────────────────────────
    print('Computing picked-stock fingerprint...', flush=True)
    fp = picked_stock_fingerprint(picks, panel, MOM_COLS)
    fp = fp.dropna()
    print(f'  fingerprint shape: {fp.shape}', flush=True)

    # ── per-month pi_panic ────────────────────────────────────────────────────
    pi = panel.groupby('date')['pi_filter'].first().sort_index()
    fp_dates = fp.index  # pd.DatetimeIndex
    pi_at_dates = pi.reindex(fp_dates)
    pi_panic_flag = (pi_at_dates > 0.5).astype(int)

    # ── sanity: calm + panic = total ─────────────────────────────────────────
    n_calm  = int((pi_panic_flag == 0).sum())
    n_panic = int((pi_panic_flag == 1).sum())
    n_total = len(fp_dates)
    print(f'\n  Total months: {n_total} (calm={n_calm}, panic={n_panic})', flush=True)
    assert n_calm + n_panic == n_total, 'SANITY FAIL: calm + panic != total'

    # ── feature panel (full set; we subset later) ─────────────────────────────
    print('Building predictor feature panel...', flush=True)
    feat_full = build_feature_panel(panel, m2_ret, list(fp_dates))
    feat_full = feat_full.set_index('date')

    # ── process each regime ───────────────────────────────────────────────────
    results = {}  # regime -> dict of artefacts

    for regime_name, mask_flag in [('calm', 0), ('panic', 1)]:
        regime_mask = (pi_panic_flag == mask_flag).values
        fp_sub    = fp.iloc[regime_mask]          # fingerprint rows for this regime
        feat_sub  = feat_full.iloc[regime_mask]   # feature rows for this regime
        dates_sub = fp_dates[regime_mask]
        n_sub     = int(regime_mask.sum())

        print(f'\n{"="*60}', flush=True)
        print(f'  REGIME: {regime_name.upper()}  (n={n_sub})', flush=True)
        print(f'{"="*60}', flush=True)

        # Standardise fingerprint
        X = fp_sub.values.astype(np.float64)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # K=2 stability
        labels_raw, mean_ari, mean_sil, sizes = run_kmeans_stability(X_std, k=2)
        print(f'\n  KMeans K=2:', flush=True)
        print(f'    silhouette = {mean_sil:.3f}', flush=True)
        print(f'    mean ARI   = {mean_ari:.3f}', flush=True)
        print(f'    sizes (seed 42) = {sizes}', flush=True)
        if mean_ari < ARI_THRESHOLD:
            print(f'  WARNING: mean ARI {mean_ari:.3f} < {ARI_THRESHOLD} threshold; '
                  f'sub-clusters not stable.', flush=True)

        # Sanity: each cluster >= 5 months
        for sz in sizes:
            if sz < 5:
                print(f'  WARNING: sub-cluster has only {sz} months (<5); '
                      f'analysis may be unreliable.', flush=True)

        # Order by mom_overall desc: cluster 0 = momentum-up basket
        labels = order_by_mom_overall(labels_raw, feat_sub.reset_index())

        # Centroids + Sharpe
        centroids = compute_regime_centroids(fp_sub, labels, m2_ret, dates_sub)
        centroids_path = f'{RES_DIR}/picked_stock_within_{regime_name}_centroids.csv'
        centroids.to_csv(centroids_path)
        print(f'\n  Centroids saved: {centroids_path}', flush=True)
        print(centroids.round(3).to_string())

        # Feature ranking (ANOVA)
        rank = univariate_ranking(feat_sub.reset_index(), labels)
        rank_path = f'{RES_DIR}/picked_stock_within_{regime_name}_feature_ranking.csv'
        rank.to_csv(rank_path, index=False)
        print(f'\n  Feature ranking saved: {rank_path}', flush=True)
        print(rank.to_string(index=False, float_format='%.3f'))

        # Logistic regression
        print(f'\n  === Logistic L1 LOO-CV ({regime_name}) ===', flush=True)
        coef, loo_acc_lr, baseline_lr, best_C = multinomial_l1_loo(
            feat_sub.reset_index(), labels,
        )
        coef_path = f'{RES_DIR}/picked_stock_within_{regime_name}_logistic_coefficients.csv'
        coef.to_csv(coef_path)
        print(f'\n    best C={best_C}, LOO acc={loo_acc_lr:.3f}, '
              f'baseline={baseline_lr:.3f}', flush=True)
        print(coef.round(3).to_string())
        print(f'  Logistic coefficients saved: {coef_path}', flush=True)

        # Decision tree
        print(f'\n  === Decision tree (depth 3, {regime_name}) ===', flush=True)
        tree_text, tree_train_acc, tree_loo_acc = decision_tree_readout(
            feat_sub.reset_index(), labels,
        )
        tree_path = f'{RES_DIR}/picked_stock_within_{regime_name}_decision_tree.txt'
        with open(tree_path, 'w') as fh:
            fh.write(tree_text)
        print(tree_text, flush=True)
        print(f'    train acc={tree_train_acc:.3f}, LOO acc={tree_loo_acc:.3f}', flush=True)
        print(f'  Decision tree saved: {tree_path}', flush=True)

        results[regime_name] = {
            'n': n_sub,
            'labels': labels,
            'dates': dates_sub,
            'pi_panic_val': mask_flag,
            'centroids': centroids,
            'rank': rank,
            'loo_acc_lr': loo_acc_lr,
            'tree_loo_acc': tree_loo_acc,
            'baseline': baseline_lr,
            'best_C': best_C,
            'mean_ari': mean_ari,
            'mean_sil': mean_sil,
            'sizes': sizes,
        }

    # ── per-month labels CSV ──────────────────────────────────────────────────
    print('\nBuilding per-month label CSV...', flush=True)
    label_rows = []
    for regime_name, mask_flag in [('calm', 0), ('panic', 1)]:
        r = results[regime_name]
        for date, sub_cl in zip(r['dates'], r['labels']):
            label_rows.append({
                'date': date,
                'pi_panic': mask_flag,
                'sub_cluster': int(sub_cl),
                'combined_label': f'{regime_name}.{int(sub_cl)}',
            })
    label_df = pd.DataFrame(label_rows).sort_values('date').reset_index(drop=True)
    labels_path = f'{RES_DIR}/picked_stock_within_regime_labels.csv'
    label_df.to_csv(labels_path, index=False)
    print(f'  Saved: {labels_path}', flush=True)
    print(f'  Label distribution:\n{label_df["combined_label"].value_counts().to_string()}')

    # ── HEADLINE ──────────────────────────────────────────────────────────────
    print('\n' + '=' * 70, flush=True)
    print('  HEADLINE SUMMARY', flush=True)
    print('=' * 70, flush=True)

    for regime_name in ['calm', 'panic']:
        r = results[regime_name]
        print(f'\n  REGIME: {regime_name.upper()}  (n={r["n"]}, '
              f'ARI={r["mean_ari"]:.3f}, sil={r["mean_sil"]:.3f})', flush=True)
        print(f'    Logistic LOO-CV:  {r["loo_acc_lr"]:.3f}  (baseline={r["baseline"]:.3f})', flush=True)
        print(f'    Tree LOO-CV:      {r["tree_loo_acc"]:.3f}', flush=True)
        top3 = r['rank'].head(3)
        print(f'    Top-3 ANOVA features:', flush=True)
        for _, row in top3.iterrows():
            print(f'      {row["feature"]:25s}  F={row["F_statistic"]:.2f}  p={row["p_value"]:.4f}', flush=True)
        print(f'    Sub-cluster sizes: {r["sizes"]}', flush=True)
        c = r['centroids']
        for cl in [0, 1]:
            n_cl = int(c.loc[cl, 'n_months'])
            mo = c.loc[cl, 'pick_mom_1':'pick_mom_12'].mean() if 'pick_mom_1' in c.columns else float('nan')
            sh = c.loc[cl, 'sharpe']
            lo = c.loc[cl, 'sharpe_lo95']
            hi = c.loc[cl, 'sharpe_hi95']
            print(f'    Sub-cluster {cl}: n={n_cl}, '
                  f'mean_mom_overall=n/a (centroid mean across horizons={mo:.3f}), '
                  f'Sharpe={sh:.3f} [{lo:.3f}, {hi:.3f}]', flush=True)

    print('\n' + '=' * 70, flush=True)
    print('  OUTPUT FILES', flush=True)
    print('=' * 70, flush=True)
    output_files = [
        'picked_stock_within_calm_centroids.csv',
        'picked_stock_within_panic_centroids.csv',
        'picked_stock_within_regime_labels.csv',
        'picked_stock_within_calm_feature_ranking.csv',
        'picked_stock_within_panic_feature_ranking.csv',
        'picked_stock_within_calm_logistic_coefficients.csv',
        'picked_stock_within_panic_logistic_coefficients.csv',
        'picked_stock_within_calm_decision_tree.txt',
        'picked_stock_within_panic_decision_tree.txt',
    ]
    for fn in output_files:
        path = f'{RES_DIR}/{fn}'
        exists = os.path.exists(path)
        print(f'  {"OK" if exists else "MISSING":7s} {path}', flush=True)


if __name__ == '__main__':
    main()
