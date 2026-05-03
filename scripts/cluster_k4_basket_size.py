"""
cluster_k4_basket_size.py
=========================
Per-K=4-cluster average number of long-leg picks per month.

Source: results/thesis/rule_path_labels.csv
  Each row is a (date, permno, rule_id) triple representing a stock picked
  in the long leg in a given month. The number of rows per date is the number
  of picks in that month.

Metrics per cluster:
  mean_picks_per_month
  min_picks_per_month
  max_picks_per_month
  std_picks_per_month

Cross-checks:
  [D1] sum(n_months * mean_picks_per_month) == total rows in rule_path_labels.csv (85,599 ± 1)
  [D2] All values > 0
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
OUT_CSV    = ROOT / cfg.RESULTS_THESIS_DIR / 'cluster_k4_basket_size.csv'


def main():
    labels = pd.read_csv(LABELS_CSV, parse_dates=['date'])
    rpl    = pd.read_csv(RPL_CSV,    parse_dates=['date'])

    total_rows = len(rpl)
    print(f"Total rows in rule_path_labels.csv: {total_rows}")

    # Picks per month
    picks_per_month = rpl.groupby('date').size().rename('n_picks')
    # Merge cluster assignment
    picks_pm = picks_per_month.reset_index().merge(
        labels[['date', 'cluster']], on='date', how='inner'
    )

    rows = []
    for cluster_id in sorted(labels['cluster'].unique()):
        sub = picks_pm[picks_pm['cluster'] == cluster_id]['n_picks']
        n_months = int(len(labels[labels['cluster'] == cluster_id]))
        rows.append({
            'cluster':              cluster_id,
            'n_months':             n_months,
            'mean_picks_per_month': float(sub.mean()),
            'min_picks_per_month':  float(sub.min()),
            'max_picks_per_month':  float(sub.max()),
            'std_picks_per_month':  float(sub.std(ddof=1)),
        })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, float_format='%.4f')
    print(f"Wrote {OUT_CSV}")

    print("\n=== cluster_k4_basket_size ===")
    print(out_df.to_string(index=False, float_format='%.2f'))

    # ── Verification cross-checks ────────────────────────────────────────────
    print("\n=== VERIFICATION (basket_size) ===")

    # [D1] Sum of (n_months * mean_picks) == total_rows (85,599)
    implied_total = (out_df['n_months'] * out_df['mean_picks_per_month']).sum()
    d1_pass = abs(implied_total - total_rows) <= 1
    print(f"[D1] sum(n_months * mean_picks) = {implied_total:.2f}, "
          f"expected {total_rows}, diff={abs(implied_total - total_rows):.2f}, "
          f"status = {'PASS' if d1_pass else 'FAIL'}")

    # [D2] All values > 0
    num_cols = ['mean_picks_per_month', 'min_picks_per_month',
                'max_picks_per_month', 'std_picks_per_month']
    min_val = out_df[['mean_picks_per_month', 'min_picks_per_month']].min().min()
    d2_pass = bool(min_val > 0)
    print(f"[D2] All mean/min picks > 0: min_value={min_val:.2f}, "
          f"status = {'PASS' if d2_pass else 'FAIL'}")

    all_pass = d1_pass and d2_pass
    print(f"\nOverall basket_size checks (D1-D2): {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return out_df


if __name__ == '__main__':
    main()
