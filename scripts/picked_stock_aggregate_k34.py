# scripts/picked_stock_aggregate_k34.py
"""Picked-stock aggregate K=3 and K=4 clustering on the full 167-month set.

No calm/panic pre-split.  Data-driven K=3 and K=4 are compared to the
within-regime 4 cells (calm.0, calm.1, panic.0, panic.1) via confusion matrix.

Outputs (results/thesis/):
  picked_stock_aggregate_k3_labels.csv
  picked_stock_aggregate_k3_centroids.csv
  picked_stock_aggregate_k4_labels.csv
  picked_stock_aggregate_k4_centroids.csv
  picked_stock_aggregate_k3_vs_within_regime_confusion.csv
  picked_stock_aggregate_k4_vs_within_regime_confusion.csv

Plot (plots/thesis/):
  picked_stock_aggregate_k4_dispersion.png
  picked_stock_aggregate_k4_dispersion.pdf
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_feature_search_helpers import picked_stock_fingerprint
from bootstrap_helpers import block_bootstrap_sharpe

# ── constants ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "results/thesis"
PLT_DIR = ROOT / "plots/thesis"
PLT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS        = list(range(1, 13))
MOM_COLS        = [f'mom_{h}' for h in HORIZONS]
STABILITY_SEEDS = [42, 123, 456, 789, 1011, 1213]
N_INIT          = 20
ARI_THRESHOLD   = 0.85   # sanity floor

# Cluster aesthetics (K=4, ordered by mom_overall desc)
K4_COLORS = {
    0: "#1f77b4",   # light blue  -- most trending up
    1: "#9ecae1",   # gray-blue
    2: "#ff7f0e",   # orange
    3: "#d62728",   # red         -- most trending down
}


# ── helpers ──────────────────────────────────────────────────────────────────

def run_kmeans_stability(X_std, k):
    """Fit KMeans at each stability seed; return seed-42 labels + mean ARI + sil."""
    fits = []
    sils = []
    for s in STABILITY_SEEDS:
        km = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit(X_std)
        fits.append(km.labels_)
        sils.append(silhouette_score(X_std, km.labels_))
    pair_aris = [
        adjusted_rand_score(fits[i], fits[j])
        for i in range(len(STABILITY_SEEDS))
        for j in range(i + 1, len(STABILITY_SEEDS))
    ]
    # seed-42 is index 0 in STABILITY_SEEDS
    return fits[0], float(np.mean(pair_aris)), float(np.mean(sils))


def order_by_mom_overall(labels_raw, fp_df, pi_series, fp_dates):
    """Re-label so cluster 0 = highest mean mom_overall (avg across pick_mom_1..12)."""
    # Compute per-month mom_overall from the fingerprint
    pick_cols = [f'pick_mom_{h}' for h in HORIZONS]
    mom_overall = fp_df[pick_cols].mean(axis=1)   # Series indexed by date
    tmp = pd.DataFrame({'mom_overall': mom_overall.values, 'cluster': labels_raw})
    means = tmp.groupby('cluster')['mom_overall'].mean()
    ordered = means.sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(ordered)}
    labels_new = np.array([remap[l] for l in labels_raw])
    return labels_new, remap


def compute_centroids(fp_df, labels, m2_ret, fp_dates, k, pi_series):
    """Per-cluster raw centroid + n_months + mean_pi + Sharpe block-bootstrap."""
    fp_c = fp_df.copy()
    fp_c['cluster'] = labels
    centroids = fp_c.groupby('cluster').mean()
    counts = pd.Series(labels).value_counts().sort_index()
    centroids['n_months'] = counts.values

    # mean_pi per cluster
    pi_vals = pi_series.reindex(fp_dates).values
    for cl in range(k):
        mask = (labels == cl)
        centroids.loc[cl, 'mean_pi'] = float(pi_vals[mask].mean())

    # pick_cols mean -> mean_mom_overall per cluster
    pick_cols = [f'pick_mom_{h}' for h in HORIZONS]
    centroids['mean_mom_overall'] = centroids[pick_cols].mean(axis=1)

    # Sharpe block-bootstrap
    m2_dates = pd.to_datetime(m2_ret.index)
    fp_dates_arr = pd.to_datetime(fp_dates)
    for cl in range(k):
        cl_dates = fp_dates_arr[labels == cl]
        mask = m2_dates.isin(cl_dates)
        cl_rets = m2_ret.values[mask]
        if len(cl_rets) < 2:
            centroids.loc[cl, 'sharpe']      = np.nan
            centroids.loc[cl, 'sharpe_lo95'] = np.nan
            centroids.loc[cl, 'sharpe_hi95'] = np.nan
        else:
            bbs = block_bootstrap_sharpe(cl_rets, block_size=6, n_reps=5000, seed=42)
            centroids.loc[cl, 'sharpe']      = bbs['sharpe_point']
            centroids.loc[cl, 'sharpe_lo95'] = bbs['sharpe_lo95']
            centroids.loc[cl, 'sharpe_hi95'] = bbs['sharpe_hi95']
            s = bbs['sharpe_point']
            if np.isfinite(s) and (s < -2 or s > 3):
                print(f'  WARNING: K={k} cluster {cl} Sharpe={s:.3f} outside [-2,3]')

    return centroids


def confusion_matrix_vs_within(labels, fp_dates, within_df, k):
    """Build K x 4 confusion matrix: data-driven clusters vs within-regime cells."""
    within_cells = ['calm.0', 'calm.1', 'panic.0', 'panic.1']
    agg = pd.DataFrame({'date': pd.to_datetime(fp_dates), 'agg_cluster': labels})
    merged = agg.merge(within_df[['date', 'combined_label']], on='date', how='left')
    rows = []
    for cl in range(k):
        row = {}
        for cell in within_cells:
            row[cell] = int(((merged['agg_cluster'] == cl) &
                             (merged['combined_label'] == cell)).sum())
        rows.append(row)
    cf = pd.DataFrame(rows, index=[f'cluster_{i}' for i in range(k)])
    cf.index.name = 'agg_cluster'
    return cf


def plot_k4(fp_df, labels, pi_series, fp_dates, centroids_df, zscore_csv, m2_ret):
    """4-panel dispersion plot for K=4 aggregate clusters."""
    Z = pd.read_csv(zscore_csv, parse_dates=['date']).set_index('date')
    fp_dates_arr = pd.to_datetime(fp_dates)

    # All-calm reference (pi <= 0.5)
    pi_vals = pi_series.reindex(fp_dates_arr)
    calm_dates = fp_dates_arr[pi_vals.values <= 0.5]
    Z_calm = Z.reindex(calm_dates).dropna()
    ref_centroid = Z_calm.mean(axis=0).values

    k = 4
    cell_info = {}
    for cl in range(k):
        cl_mask  = (labels == cl)
        cl_dates = fp_dates_arr[cl_mask]
        Z_cl     = Z.reindex(cl_dates).dropna()
        centroid = Z_cl.mean(axis=0).values
        pi_bar   = float(pi_vals.values[cl_mask].mean())
        n_cl     = int(cl_mask.sum())
        sharpe   = float(centroids_df.loc[cl, 'sharpe'])
        lo       = float(centroids_df.loc[cl, 'sharpe_lo95'])
        hi       = float(centroids_df.loc[cl, 'sharpe_hi95'])
        cell_info[cl] = dict(
            Z_cl=Z_cl, centroid=centroid, pi_bar=pi_bar,
            n=n_cl, sharpe=sharpe, lo=lo, hi=hi,
        )

    # Compute y-limits
    all_vals = list(ref_centroid)
    for cl in range(k):
        all_vals.extend(cell_info[cl]['centroid'].tolist())
    y_min = round((min(all_vals) - 0.15) * 4) / 4
    y_max = round((max(all_vals) + 0.15) * 4) / 4
    span  = y_max - y_min
    if span < 1.5:
        mid   = (y_max + y_min) / 2
        y_min = mid - 0.75
        y_max = mid + 0.75

    fig, axes = plt.subplots(1, k, figsize=(4.5 * k, 5.5), sharey=True,
                              constrained_layout=True)
    fig.suptitle(
        "Picked-stock z-curves by data-driven aggregate cluster (K=4, ordered by mom_overall)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for cl, ax in enumerate(axes):
        info  = cell_info[cl]
        color = K4_COLORS[cl]

        # Thin per-month lines
        for _, row in info['Z_cl'].iterrows():
            ax.plot(HORIZONS, row.values, color=color, lw=0.8, alpha=0.4)

        # Bold centroid
        ax.plot(
            HORIZONS, info['centroid'],
            color=color, lw=2.5, marker='o', ms=6,
            label=f'centroid ({info["n"]} mo)',
            zorder=5,
        )

        # Dashed all-calm reference
        ax.plot(
            HORIZONS, ref_centroid,
            color='#333333', lw=1.5, ls='--',
            label='all-calm ref',
            zorder=4,
        )

        # y=0 guideline
        ax.axhline(0, color='#888888', lw=0.6, zorder=1)

        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel('Horizon (months)', fontsize=9)
        if cl == 0:
            ax.set_ylabel('Cross-sectional z-score', fontsize=9)

        sharpe_str = f'{info["sharpe"]:.2f}' if np.isfinite(info['sharpe']) else 'n/a'
        lo_str     = f'{info["lo"]:.2f}'     if np.isfinite(info['lo'])     else 'n/a'
        hi_str     = f'{info["hi"]:.2f}'     if np.isfinite(info['hi'])     else 'n/a'
        ax.set_title(
            f'Cluster {cl}\n'
            f'(n={info["n"]}, π̅={info["pi_bar"]:.2f})\n'
            f'Sharpe={sharpe_str} [{lo_str}, {hi_str}]',
            fontsize=9.5, fontweight='bold', loc='center',
        )

        ax.legend(fontsize=7.5, frameon=True, loc='upper left')
        ax.tick_params(axis='both', labelsize=8)

    for ext in ('png', 'pdf'):
        out = PLT_DIR / f'picked_stock_aggregate_k4_dispersion.{ext}'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved plot: {out}', flush=True)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading data...', flush=True)
    picks = pd.read_csv(
        str(RES_DIR / 'rule_path_labels.csv'), parse_dates=['date']
    )[['date', 'permno']]

    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    panel = art['test'].copy()
    panel['date'] = pd.to_datetime(panel['date'])

    with open(str(RES_DIR / 'fundamentals_returns.pkl'), 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    within_df = pd.read_csv(
        str(RES_DIR / 'picked_stock_within_regime_labels.csv'), parse_dates=['date']
    )

    # ── fingerprint ──────────────────────────────────────────────────────────
    print('Computing picked-stock fingerprint...', flush=True)
    fp = picked_stock_fingerprint(picks, panel, MOM_COLS)
    fp = fp.dropna()
    fp_dates = fp.index
    print(f'  fingerprint shape: {fp.shape}  ({len(fp_dates)} months)', flush=True)

    # per-month pi
    pi_series = panel.groupby('date')['pi_filter'].first().sort_index()

    # Standardise
    X = fp.values.astype(np.float64)
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    # ── K=3 and K=4 ──────────────────────────────────────────────────────────
    results = {}
    for k in (3, 4):
        print(f'\n=== K={k} ===', flush=True)
        labels_raw, mean_ari, mean_sil = run_kmeans_stability(X_std, k)
        print(f'  seed-42 raw label counts: {np.bincount(labels_raw, minlength=k).tolist()}')
        print(f'  mean pairwise ARI = {mean_ari:.4f}', flush=True)
        print(f'  mean silhouette   = {mean_sil:.4f}', flush=True)
        if mean_ari < ARI_THRESHOLD:
            print(f'  WARNING: ARI {mean_ari:.4f} < {ARI_THRESHOLD} threshold', flush=True)

        # Relabel by mom_overall desc
        labels, remap = order_by_mom_overall(labels_raw, fp, pi_series, fp_dates)
        print(f'  relabel mapping (raw->new): {remap}', flush=True)
        sizes = [int((labels == cl).sum()) for cl in range(k)]
        print(f'  cluster sizes (ordered): {sizes}', flush=True)
        for sz in sizes:
            if sz < 5:
                print(f'  WARNING: cluster has only {sz} months (<5)', flush=True)

        # Centroids
        centroids = compute_centroids(fp, labels, m2_ret, fp_dates, k, pi_series)

        # Save labels
        label_df = pd.DataFrame({'date': fp_dates, 'cluster': labels})
        label_path = RES_DIR / f'picked_stock_aggregate_k{k}_labels.csv'
        label_df.to_csv(str(label_path), index=False)
        print(f'  Saved: {label_path}', flush=True)

        # Save centroids
        cent_path = RES_DIR / f'picked_stock_aggregate_k{k}_centroids.csv'
        centroids.to_csv(str(cent_path))
        print(f'  Saved: {cent_path}', flush=True)

        # Confusion matrix
        cf = confusion_matrix_vs_within(labels, fp_dates, within_df, k)
        cf_path = RES_DIR / f'picked_stock_aggregate_k{k}_vs_within_regime_confusion.csv'
        cf.to_csv(str(cf_path))
        print(f'  Saved: {cf_path}', flush=True)

        # Per-cluster summary
        print(f'\n  Per-cluster summary (K={k}):')
        for cl in range(k):
            n_cl   = int(centroids.loc[cl, 'n_months'])
            pi_bar = float(centroids.loc[cl, 'mean_pi'])
            mo     = float(centroids.loc[cl, 'mean_mom_overall'])
            sh     = float(centroids.loc[cl, 'sharpe'])
            lo     = float(centroids.loc[cl, 'sharpe_lo95'])
            hi     = float(centroids.loc[cl, 'sharpe_hi95'])
            print(f'    Cluster {cl}: n={n_cl:3d}, pi_bar={pi_bar:.3f}, '
                  f'mom_overall={mo:.4f}, '
                  f'Sharpe={sh:.3f} [{lo:.3f}, {hi:.3f}]', flush=True)

        results[k] = dict(
            labels=labels, centroids=centroids, mean_ari=mean_ari,
            mean_sil=mean_sil, cf=cf,
        )

    # ── K=4 dispersion plot ──────────────────────────────────────────────────
    print('\nGenerating K=4 dispersion plot...', flush=True)
    zscore_csv = str(RES_DIR / 'zscore_long_by_month.csv')
    plot_k4(
        fp_df=fp,
        labels=results[4]['labels'],
        pi_series=pi_series,
        fp_dates=fp_dates,
        centroids_df=results[4]['centroids'],
        zscore_csv=zscore_csv,
        m2_ret=m2_ret,
    )

    # ── HEADLINE ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 70, flush=True)
    print('  HEADLINE', flush=True)
    print('=' * 70, flush=True)

    for k in (3, 4):
        r = results[k]
        print(f'\nK={k}:  mean ARI={r["mean_ari"]:.4f}  '
              f'({"STABLE" if r["mean_ari"] >= ARI_THRESHOLD else "UNSTABLE"})', flush=True)
        cen = r['centroids']
        for cl in range(k):
            n_cl   = int(cen.loc[cl, 'n_months'])
            pi_bar = float(cen.loc[cl, 'mean_pi'])
            mo     = float(cen.loc[cl, 'mean_mom_overall'])
            sh     = float(cen.loc[cl, 'sharpe'])
            lo     = float(cen.loc[cl, 'sharpe_lo95'])
            hi     = float(cen.loc[cl, 'sharpe_hi95'])
            print(f'  Cluster {cl}: n={n_cl:3d}, pi_bar={pi_bar:.3f}, '
                  f'mom_overall={mo:.4f}, '
                  f'Sharpe={sh:.3f} [{lo:.3f}, {hi:.3f}]', flush=True)

    # K=4 confusion matrix
    print('\nK=4 confusion matrix (data-driven clusters vs within-regime cells):')
    print(results[4]['cf'].to_string())

    # Interpretation: does each within-regime cell have its plurality in a distinct K=4 cluster?
    cf4 = results[4]['cf']
    within_cells = ['calm.0', 'calm.1', 'panic.0', 'panic.1']
    pluralities = {cell: cf4[cell].idxmax() for cell in within_cells}
    distinct_pluralities = len(set(pluralities.values())) == len(within_cells)
    print(f'\n  Plurality mapping (within-regime cell -> dominant K=4 cluster):')
    for cell, pl in pluralities.items():
        pct = cf4.loc[pl, cell] / cf4[cell].sum() * 100
        print(f'    {cell:8s} -> {pl}  ({cf4.loc[pl, cell]} / {cf4[cell].sum()} = {pct:.0f}%)')
    if distinct_pluralities:
        print('\n  INTERPRETATION: K=4 data-driven clusters CLEANLY MAP to the 4 within-regime '
              'cells (each cell has its plurality in a distinct cluster).')
    else:
        print('\n  INTERPRETATION: K=4 clusters do NOT perfectly replicate the within-regime '
              'partition -- at least two within-regime cells share the same dominant cluster.')

    print('\nK=3 confusion matrix (data-driven clusters vs within-regime cells):')
    print(results[3]['cf'].to_string())

    # Output file checklist
    print('\n' + '=' * 70)
    print('  OUTPUT FILES')
    print('=' * 70)
    expected = [
        RES_DIR / 'picked_stock_aggregate_k3_labels.csv',
        RES_DIR / 'picked_stock_aggregate_k3_centroids.csv',
        RES_DIR / 'picked_stock_aggregate_k4_labels.csv',
        RES_DIR / 'picked_stock_aggregate_k4_centroids.csv',
        RES_DIR / 'picked_stock_aggregate_k3_vs_within_regime_confusion.csv',
        RES_DIR / 'picked_stock_aggregate_k4_vs_within_regime_confusion.csv',
        PLT_DIR / 'picked_stock_aggregate_k4_dispersion.png',
        PLT_DIR / 'picked_stock_aggregate_k4_dispersion.pdf',
    ]
    all_ok = True
    for p in expected:
        exists = p.exists()
        print(f'  {"OK" if exists else "MISSING":7s} {p}')
        if not exists:
            all_ok = False
    print(f'\n  All outputs present: {all_ok}')


if __name__ == '__main__':
    main()
