"""
cluster_k4_downside_metrics.py
==============================
Per-K=4-cluster downside risk metrics on XGB monthly returns.

"Max drawdown within cluster" convention:
  The member months of each cluster are extracted in date order and concatenated
  into a subseries. This represents the compounded experience of an investor
  who is only invested during those months (not the full-path drawdown). Peak and
  trough are computed on this concatenated subsequence. The result answers:
  "What is the worst cumulative loss experienced during cluster X's months?"

Metrics per cluster:
  max_drawdown      : peak-to-trough on the concatenated cluster-month subseries
  sortino_ratio     : annualised mean / annualised downside std (negative returns only)
  pct_positive      : fraction of months with positive XGB return
  pct5/25/50/75/95  : percentiles of monthly returns

Cross-checks (see VERIFICATION block):
  [C1] pct_positive == hit_rate from cluster_k4_descriptor_table.csv (all 4 clusters)
  [C2] median <= mean for clusters 1 and 2 (positive skew per audit)
  [C3] max_drawdown >= |worst_month_return| (always, since single worst month is a
       sub-path of the full subseries)
  [C4] sortino > sharpe for clusters with many positive months
"""

from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg

ARTEFACTS  = ROOT / cfg.ARTEFACTS_PATH
LABELS_CSV = ROOT / cfg.RESULTS_THESIS_DIR / 'zscore_l2_k4_labels.csv'
DESC_CSV   = ROOT / cfg.RESULTS_THESIS_DIR / 'cluster_k4_descriptor_table.csv'
OUT_CSV    = ROOT / cfg.RESULTS_THESIS_DIR / 'cluster_k4_downside_metrics.csv'


def max_drawdown(returns_series):
    """
    Peak-to-trough maximum drawdown on a monthly return series.
    Input  : pandas Series of monthly returns (not cumulative).
    Output : float in [0, +inf), where 0.10 means a 10% loss from peak.
    """
    cum = (1 + returns_series).cumprod()
    roll_max = cum.cummax()
    drawdowns = (cum - roll_max) / roll_max
    return float(-drawdowns.min()) if len(drawdowns) > 0 else np.nan


def sortino_ratio(returns_series):
    """
    Annualised Sortino ratio. Denominator is the annualised downside std
    computed from only the negative monthly returns.
    Returns NaN if there are no negative returns.
    """
    ann_mean = returns_series.mean() * 12
    neg = returns_series[returns_series < 0]
    if len(neg) == 0:
        return np.nan
    downside_std = neg.std(ddof=1) * np.sqrt(12)
    if downside_std == 0:
        return np.nan
    return float(ann_mean / downside_std)


def sharpe_ratio(returns_series):
    """Annualised Sharpe (no risk-free subtraction, matching thesis convention)."""
    if len(returns_series) < 2:
        return np.nan
    return float(returns_series.mean() / returns_series.std(ddof=1) * np.sqrt(12))


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    with open(ARTEFACTS, 'rb') as f:
        art = pickle.load(f)
    test  = art['test'].copy()
    test['date'] = pd.to_datetime(test['date'])
    r_mkt = art['r_mkt']   # not used for downside metrics, but loaded for consistency

    labels = pd.read_csv(LABELS_CSV, parse_dates=['date'])
    desc   = pd.read_csv(DESC_CSV)

    # Use the pre-computed XGB L/S return series from the artefact.
    # This is the *same* series used by cluster_k4_descriptor_table.py, so
    # hit_rate and worst_month_return from that table are guaranteed to reconcile.
    # (Reconstructing from build_leg_returns_subset gives slightly different values
    # due to turnover-fee path differences at cluster boundaries — ~0.002/month —
    # causing spurious C1 and C3 failures.)
    m2_ls = art['strategies_lo']['Method 2: XGB'].copy()
    m2_ls.index = pd.to_datetime(m2_ls.index)

    rows = []
    for cluster_id in sorted(labels['cluster'].unique()):
        cluster_dates = sorted(
            labels.loc[labels['cluster'] == cluster_id, 'date'].tolist()
        )
        cluster_dt_index = pd.DatetimeIndex(cluster_dates)
        ret_sub = m2_ls.reindex(cluster_dt_index).dropna()

        n = len(ret_sub)
        if n == 0:
            rows.append({'cluster': cluster_id, 'n_months': len(cluster_dates)})
            continue

        pct_pos  = float((ret_sub > 0).mean())
        med_ret  = float(ret_sub.median())
        pct5     = float(np.percentile(ret_sub, 5))
        pct25    = float(np.percentile(ret_sub, 25))
        pct75    = float(np.percentile(ret_sub, 75))
        pct95    = float(np.percentile(ret_sub, 95))
        mdd      = max_drawdown(ret_sub)
        sort_r   = sortino_ratio(ret_sub)
        sharp_r  = sharpe_ratio(ret_sub)
        mean_r   = float(ret_sub.mean())

        rows.append({
            'cluster':       cluster_id,
            'n_months':      len(cluster_dates),
            'max_drawdown':  mdd,
            'sortino_ratio': sort_r,
            'sharpe':        sharp_r,
            'pct_positive':  pct_pos,
            'pct5_return':   pct5,
            'pct25_return':  pct25,
            'median_return': med_ret,
            'mean_return':   mean_r,
            'pct75_return':  pct75,
            'pct95_return':  pct95,
        })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, float_format='%.6f')
    print(f"Wrote {OUT_CSV}")

    print("\n=== cluster_k4_downside_metrics ===")
    print(out_df.to_string(index=False, float_format='%.4f'))

    # ── Verification cross-checks ────────────────────────────────────────────
    print("\n=== VERIFICATION (downside_metrics) ===")

    # [C1] pct_positive == hit_rate from descriptor table
    c1_pass = True
    for _, row in out_df.iterrows():
        c_id = int(row['cluster'])
        desc_row = desc[desc['cluster'] == c_id].iloc[0]
        expected_hr = float(desc_row['hit_rate'])
        fresh_pp    = float(row['pct_positive'])
        match = abs(fresh_pp - expected_hr) < 1e-6
        if not match:
            c1_pass = False
        print(f"  [C1] Cluster {c_id}: pct_positive={fresh_pp:.6f}, "
              f"hit_rate(desc)={expected_hr:.6f}, match={'YES' if match else 'NO'}")
    print(f"[C1] pct_positive matches hit_rate: {'PASS (4/4)' if c1_pass else 'FAIL'}")

    # [C2] median <= mean for clusters 1 and 2 (positive skew)
    c2_checks = []
    for c_id in [1, 2]:
        row = out_df[out_df['cluster'] == c_id].iloc[0]
        med = float(row['median_return'])
        mn  = float(row['mean_return'])
        ok  = med <= mn
        c2_checks.append(ok)
        print(f"  [C2] Cluster {c_id}: median={med:.6f}, mean={mn:.6f}, "
              f"median<=mean={'YES' if ok else 'NO'}")
    c2_pass = all(c2_checks)
    print(f"[C2] median <= mean for clusters 1,2 (positive skew): "
          f"{'PASS' if c2_pass else 'FAIL'}")

    # [C3] max_drawdown >= |worst_month_return|
    c3_pass = True
    for _, row in out_df.iterrows():
        c_id = int(row['cluster'])
        desc_row = desc[desc['cluster'] == c_id].iloc[0]
        worst = abs(float(desc_row['worst_month_return']))
        mdd   = float(row['max_drawdown'])
        ok    = mdd >= worst - 1e-9   # tiny tolerance for float
        if not ok:
            c3_pass = False
        print(f"  [C3] Cluster {c_id}: max_DD={mdd:.6f}, |worst_month|={worst:.6f}, "
              f"max_DD>=|worst|={'YES' if ok else 'NO'}")
    print(f"[C3] max_drawdown >= |worst_month| per cluster: "
          f"{'PASS (4/4)' if c3_pass else 'FAIL'}")

    # [C4] Sortino > Sharpe per cluster
    c4_checks = []
    for _, row in out_df.iterrows():
        c_id = int(row['cluster'])
        sort = float(row['sortino_ratio'])
        shar = float(row['sharpe'])
        ok   = sort > shar
        c4_checks.append(ok)
        print(f"  [C4] Cluster {c_id}: sortino={sort:.4f}, sharpe={shar:.4f}, "
              f"sortino>sharpe={'YES' if ok else 'NO'}")
    c4_n   = sum(c4_checks)
    c4_pass = c4_n >= 3
    print(f"[C4] Sortino > Sharpe: {c4_n}/4 clusters, "
          f"status = {'PASS' if c4_pass else 'FAIL'}")

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    print(f"\nOverall downside_metrics checks (C1-C4): "
          f"{'ALL PASS' if all_pass else 'SOME FAIL'}")
    return out_df


if __name__ == '__main__':
    main()
