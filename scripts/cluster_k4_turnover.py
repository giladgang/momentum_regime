"""
cluster_k4_turnover.py
======================
Per-K=4-cluster basket turnover (Jaccard distance) for M2 long-leg picks.

Source: results/thesis/rule_path_labels.csv
  Each row (date, permno, rule_id) is a long-leg pick.

Definition:
  For consecutive test months (t, t+1), let P_t = set of picked permnos at t,
  P_{t+1} = set of picked permnos at t+1.
  Jaccard distance = 1 - |P_t ∩ P_{t+1}| / |P_t ∪ P_{t+1}|.

Per-cluster metrics:
  within_cluster_avg_turnover  : avg Jaccard for pairs where BOTH t and t+1 are in cluster X
  entering_cluster_avg_turnover: avg Jaccard for pairs where t is NOT in X and t+1 IS in X
  n_within_pairs               : number of within-cluster consecutive pairs
  n_entering_pairs             : number of entering-cluster consecutive pairs

Cross-checks:
  [E1] All Jaccard distances in [0, 1]
  [E2] entering_cluster_avg_turnover >= within_cluster_avg_turnover for >=3 of 4 clusters
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg

LABELS_CSV = ROOT / cfg.RESULTS_THESIS_DIR / 'zscore_l2_k4_labels.csv'
RPL_CSV    = ROOT / cfg.RESULTS_THESIS_DIR / 'rule_path_labels.csv'
OUT_CSV    = ROOT / cfg.RESULTS_THESIS_DIR / 'cluster_k4_turnover.csv'


def jaccard_distance(set_a, set_b):
    """Jaccard distance = 1 - |A ∩ B| / |A ∪ B|. Returns 1.0 if both empty."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    intersection = len(set_a & set_b)
    return 1.0 - intersection / union


def main():
    labels = pd.read_csv(LABELS_CSV, parse_dates=['date'])
    rpl    = pd.read_csv(RPL_CSV,    parse_dates=['date'])

    # Build date -> cluster mapping
    date_to_cluster = labels.set_index('date')['cluster'].to_dict()

    # Build date -> set of permnos
    date_to_picks = {
        date: set(grp['permno'].tolist())
        for date, grp in rpl.groupby('date')
    }

    # All dates in chronological order (test period)
    all_dates = sorted(date_to_cluster.keys())

    # Compute all consecutive (t, t+1) Jaccard distances
    pair_records = []
    for i in range(len(all_dates) - 1):
        t0 = all_dates[i]
        t1 = all_dates[i + 1]
        c0 = date_to_cluster.get(t0)
        c1 = date_to_cluster.get(t1)
        p0 = date_to_picks.get(t0, set())
        p1 = date_to_picks.get(t1, set())
        jd = jaccard_distance(p0, p1)
        pair_records.append({
            'date_t':    t0,
            'date_t1':   t1,
            'cluster_t': c0,
            'cluster_t1': c1,
            'jaccard_distance': jd,
        })

    pairs = pd.DataFrame(pair_records)

    # ── Per-cluster aggregation ───────────────────────────────────────────────
    rows = []
    for cluster_id in sorted(labels['cluster'].unique()):
        # Within: both t and t+1 in this cluster
        within = pairs[
            (pairs['cluster_t']  == cluster_id) &
            (pairs['cluster_t1'] == cluster_id)
        ]['jaccard_distance']

        # Entering: t NOT in this cluster, t+1 IS in this cluster
        entering = pairs[
            (pairs['cluster_t']  != cluster_id) &
            (pairs['cluster_t1'] == cluster_id)
        ]['jaccard_distance']

        rows.append({
            'cluster':                      cluster_id,
            'n_months':                     int((labels['cluster'] == cluster_id).sum()),
            'within_cluster_avg_turnover':  float(within.mean())  if len(within)  > 0 else np.nan,
            'within_cluster_std_turnover':  float(within.std(ddof=1)) if len(within) > 1 else np.nan,
            'entering_cluster_avg_turnover':float(entering.mean()) if len(entering) > 0 else np.nan,
            'entering_cluster_std_turnover':float(entering.std(ddof=1)) if len(entering) > 1 else np.nan,
            'n_within_pairs':               len(within),
            'n_entering_pairs':             len(entering),
        })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    print(f"Wrote {OUT_CSV}")

    print("\n=== cluster_k4_turnover ===")
    print(out_df.to_string(index=False, float_format='%.4f'))

    # ── Verification cross-checks ────────────────────────────────────────────
    print("\n=== VERIFICATION (turnover) ===")

    # [E1] All Jaccard distances in [0, 1]
    all_jd = pairs['jaccard_distance'].values
    e1_pass = bool((all_jd >= 0).all() and (all_jd <= 1).all())
    print(f"[E1] All Jaccard distances in [0, 1]: "
          f"min={all_jd.min():.4f}, max={all_jd.max():.4f}, "
          f"status = {'PASS' if e1_pass else 'FAIL'}")

    # [E2] Entering >= within for >= 3/4 clusters
    e2_checks = []
    for _, row in out_df.iterrows():
        c_id = int(row['cluster'])
        wi   = row['within_cluster_avg_turnover']
        en   = row['entering_cluster_avg_turnover']
        if np.isnan(wi) or np.isnan(en):
            ok = True   # no data => no violation
        else:
            ok = en >= wi
        e2_checks.append(ok)
        print(f"  [E2] Cluster {c_id}: within={wi:.4f}, entering={en:.4f}, "
              f"entering>=within={'YES' if ok else 'NO'}")
    e2_n    = sum(e2_checks)
    e2_pass = e2_n >= 3
    print(f"[E2] Entering >= within: {e2_n}/4 clusters, "
          f"status = {'PASS' if e2_pass else 'FAIL'}")

    all_pass = e1_pass and e2_pass
    print(f"\nOverall turnover checks (E1-E2): {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return out_df


if __name__ == '__main__':
    main()
