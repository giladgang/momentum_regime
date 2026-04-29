"""Phase 3+5 unified: characterise + cross-tabs for all rule-label variants.

Variants:
  unified_k2     — current primary (rule_path_labels.csv)
  unified_k3     — panic split into two sub-rules (rule_path_k3_labels.csv)
  per_regime_k2  — 4 rules: 2 within calm + 2 within panic

For each variant:
  - Per-rule centroid (12 mom + pi) + Sharpe with bootstrap 95% CI
  - Rule x regime cross-tab
  - Rule x landscape-tier cross-tab (tertiles of typical-stock long-mom)
  - Per-month rule-firing HHI

Plus appendix:
  - SHAP-based clustering ARI vs leaf-based (sanity check)

Outputs everything to results/thesis/ with prefixes per variant.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_helpers import block_bootstrap_sharpe

RES_DIR = 'results/thesis'
HORIZONS = list(range(1, 13))
MOM_COLS = [f'mom_{h}' for h in HORIZONS]


def characterise(labels_df, test_panel, m2_ret, pi):
    """Per-rule centroid + bootstrap Sharpe CI on plurality months."""
    feature_cols = MOM_COLS + ['pi_filter']
    merged = labels_df.merge(
        test_panel[['date', 'permno'] + feature_cols],
        on=['date', 'permno'], how='left',
    )

    # Plurality month per rule
    counts = (labels_df.groupby(['date', 'rule_id']).size()
              .reset_index(name='n'))
    totals = labels_df.groupby('date').size().reset_index(name='total')
    counts = counts.merge(totals, on='date')
    counts['share'] = counts['n'] / counts['total']
    plurality = counts[counts['share'] > 0.5][['date', 'rule_id']]

    rules = sorted(labels_df['rule_id'].unique())
    rows = []
    for r in rules:
        sub = merged[merged['rule_id'] == r]
        n_sm = len(sub)
        n_panic = int((sub['pi_filter'] > 0.5).sum())
        rule_dates = plurality[plurality['rule_id'] == r]['date']
        rs = m2_ret.reindex(rule_dates).dropna().values
        if len(rs) > 0:
            ci = block_bootstrap_sharpe(rs, block_size=6, n_reps=5000, seed=42)
        else:
            ci = {'sharpe_point': np.nan, 'sharpe_lo95': np.nan,
                  'sharpe_hi95': np.nan, 'n_reps_valid': 0}
        row = {
            'rule_id': int(r),
            'n_stockmonths': n_sm,
            'n_plurality_months': len(rule_dates),
            'mean_pi': round(float(sub['pi_filter'].mean()), 3),
            'panic_share': round(n_panic / n_sm, 3),
            'sharpe': round(ci['sharpe_point'], 3) if not np.isnan(ci['sharpe_point']) else None,
            'sharpe_lo95': round(ci['sharpe_lo95'], 3) if not np.isnan(ci['sharpe_lo95']) else None,
            'sharpe_hi95': round(ci['sharpe_hi95'], 3) if not np.isnan(ci['sharpe_hi95']) else None,
            'ci_excludes_zero': bool(ci['sharpe_lo95'] > 0) if not np.isnan(ci['sharpe_lo95']) else False,
        }
        for c in feature_cols:
            row[f'centroid_{c}'] = round(float(sub[c].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def crosstabs(labels_df, pi):
    """Rule x regime, rule x landscape-tier."""
    df = labels_df.copy()
    df['regime'] = np.where(df['date'].map(pi) > 0.5, 'panic', 'calm')

    ct_regime = pd.crosstab(df['rule_id'], df['regime'])
    ct_regime['total'] = ct_regime.sum(axis=1)
    if 'panic' in ct_regime.columns and 'calm' in ct_regime.columns:
        ct_regime['panic_share'] = (ct_regime['panic'] / ct_regime['total']).round(3)
    elif 'panic' in ct_regime.columns:
        ct_regime['panic_share'] = 1.0
    else:
        ct_regime['panic_share'] = 0.0
    return ct_regime


def hhi_per_month(labels_df):
    """Per month: HHI of rule firing (1.0 = single rule dominates,
    1/k = uniformly across rules)."""
    rows = []
    for d, sub in labels_df.groupby('date'):
        counts = sub['rule_id'].value_counts()
        p = counts / counts.sum()
        rows.append({
            'date': d,
            'hhi': float((p ** 2).sum()),
            'dominant_rule': int(counts.index[0]),
            'dominant_share': float(p.iloc[0]),
        })
    return pd.DataFrame(rows)


def run_variant(name, labels_csv, test_panel, m2_ret, pi):
    print(f'\n========== {name} ==========', flush=True)
    labels_df = pd.read_csv(labels_csv, parse_dates=['date'])

    # Characterise
    char = characterise(labels_df, test_panel, m2_ret, pi)
    char.to_csv(f'{RES_DIR}/{name}_centroids.csv', index=False)
    print(f'\nPer-rule centroids:')
    print(char[['rule_id', 'n_stockmonths', 'n_plurality_months',
                'mean_pi', 'panic_share',
                'sharpe', 'sharpe_lo95', 'sharpe_hi95',
                'ci_excludes_zero']].to_string(index=False))

    # Cross-tab: rule x regime
    ct = crosstabs(labels_df, pi)
    ct.to_csv(f'{RES_DIR}/{name}_crosstab_regime.csv')
    print(f'\nRule x regime crosstab:')
    print(ct.to_string())

    # HHI
    hhi = hhi_per_month(labels_df)
    hhi.to_csv(f'{RES_DIR}/{name}_hhi.csv', index=False)
    print(f'\nMonthly HHI summary:')
    print(f'  mean={hhi["hhi"].mean():.3f}, '
          f'median={hhi["hhi"].median():.3f}, '
          f'min={hhi["hhi"].min():.3f}, '
          f'max={hhi["hhi"].max():.3f}')
    print(f'  months where dominant_share > 0.95: '
          f'{(hhi["dominant_share"] > 0.95).sum()} / {len(hhi)}')

    return char, ct, hhi


def shap_vs_leaf_ari(test_panel, leaf_labels):
    """Sanity: cluster long-leg stock-months on SHAP profile, ARI vs leaf labels."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler

    print(f'\n========== SHAP-vs-leaf ARI sanity check ==========',
          flush=True)
    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    shap = art['shap_values']

    # Identify long-leg stock-months (mirrors Phase 1)
    df = test_panel.copy()
    df['leg'] = 'middle'
    for date, grp in df.groupby('date'):
        nyse = grp[grp['exchcd'] == 1]['score_xgb'].dropna()
        if len(nyse) < 10:
            continue
        hi = nyse.quantile(0.90)
        df.loc[grp[grp['score_xgb'] >= hi].index, 'leg'] = 'long'
    long_mask = df['leg'].values == 'long'
    shap_long = shap[long_mask]
    print(f'  long-leg shap shape: {shap_long.shape}', flush=True)

    X = StandardScaler().fit_transform(shap_long)
    k = int(len(np.unique(leaf_labels)))
    print(f'  clustering SHAP at k={k}...', flush=True)
    km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
    ari = float(adjusted_rand_score(km.labels_, leaf_labels))
    print(f'  ARI(SHAP cluster vs leaf cluster, k={k}) = {ari:.4f}', flush=True)

    out = pd.DataFrame([{
        'k': k,
        'shap_vs_leaf_ari': round(ari, 4),
    }])
    out.to_csv(f'{RES_DIR}/rule_path_shap_vs_leaf_ari.csv', index=False)
    print(f'  Saved: {RES_DIR}/rule_path_shap_vs_leaf_ari.csv', flush=True)
    return ari


def main():
    # Load shared data
    print('Loading shared panel + returns + pi...', flush=True)
    with open('artefacts/cs_artefacts_data.pkl', 'rb') as f:
        art = pickle.load(f)
    test = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])

    with open('results/thesis/fundamentals_returns.pkl', 'rb') as f:
        rets = pickle.load(f)
    m2_ret = rets['baseline_mom_pi']['returns']
    m2_ret.index = pd.to_datetime(m2_ret.index)

    pi = test.groupby('date')['pi_filter'].first()

    # Run each variant
    variants = [
        ('rule_path_unified_k2', f'{RES_DIR}/rule_path_labels.csv'),
        ('rule_path_unified_k3', f'{RES_DIR}/rule_path_k3_labels.csv'),
        ('rule_path_per_regime_k2',
         f'{RES_DIR}/rule_path_per_regime_labels.csv'),
    ]
    for name, csv in variants:
        if os.path.exists(csv):
            run_variant(name, csv, test, m2_ret, pi)
        else:
            print(f'\n[SKIP] {name}: {csv} not found', flush=True)

    # SHAP sanity check on the unified k=2 labels
    unified_csv = f'{RES_DIR}/rule_path_labels.csv'
    if os.path.exists(unified_csv):
        leaf_labels = pd.read_csv(unified_csv)['rule_id'].values
        shap_vs_leaf_ari(test, leaf_labels)


if __name__ == '__main__':
    main()
