# Direct cost saved by widening the band in each THESIS cluster

Thesis k=4 labels, all clusters one at a time, 2011-01..2024-11. Cost is the outcome (DNMV half-spread). Baseline = no-op band (E=10) every month. cluster_minus_uniform = cluster saving MINUS the uniform band that removes the same turnover (efficiency test). Ceiling: detrended cross-cluster spread ratio 1.052 => cluster-timing capped at ~2.9%.

## Per-strategy summary

     strategy  cost_base_bpyr  best_cluster_saved_pct  max_cluster_minus_uniform_bpyr  max_cluster_minus_uniform_pct
   investment            5.72                    5.92                           0.063                           1.10
       lowvol            7.94                   13.22                           0.097                           1.22
     momentum           15.51                   14.51                           0.230                           1.48
profitability            2.40                    3.26                           0.031                           1.29
     reversal           38.24                   23.09                           0.229                           0.60
        value            3.35                    3.40                           0.012                           0.36
          xgb           33.08                   21.60                           0.168                           0.51

If max_cluster_minus_uniform_pct <= ~2.9 for all strategies, cluster timing adds nothing beyond plain banding at equal turnover, exactly as the spread-timing ceiling predicts. If any exceeds it materially, the ceiling is wrong (falsification).

## Full grid: paper/results/banding_study/cluster_cost.csv (7 strat x 4 clusters x 8 levels)
