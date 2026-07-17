# Jointly OPTIMAL per-cluster band vs uniform (cost, matched turnover)

Brute-force (E_0..E_3) in {10,20,35,60,100}^4 per strategy; each combo compared to the uniform band at the SAME turnover. Thesis k=4 labels, 2011-2024, in-sample (perfect-hindsight) UPPER BOUND -- the strongest possible case for cluster-conditioning.

     strategy  cost_base_bpyr opt_vec_E0_E1_E2_E3  max_opt_minus_uniform_bpyr  max_opt_minus_uniform_pct  at_turnover_yr  opt_cost_bpyr  uniform_cost_bpyr
     momentum           15.51 (100, 10, 100, 100)                      0.5521                       3.56          37.662         10.271             10.823
     reversal           38.24    (10, 10, 60, 35)                      0.5153                       1.35         117.999         33.409             33.924
       lowvol            7.94  (10, 10, 100, 100)                      0.1518                       1.91          31.054          6.694              6.846
          xgb           33.08   (10, 20, 100, 60)                      0.5191                       1.57          87.488         24.576             25.095
        value            3.35   (60, 100, 20, 10)                      0.0481                       1.43           8.330          2.899              2.947
profitability            2.40 (100, 10, 100, 100)                      0.0670                       2.79           9.104          2.315              2.382
   investment            5.72   (60, 60, 10, 100)                      0.1203                       2.11          16.289          4.615              4.735

max_opt_minus_uniform_pct = the MOST an optimally-fit per-cluster band beats a uniform band at equal turnover, as % of cost. <= ~2.9% for all => optimizing the per-cluster band adds nothing beyond uniform banding even with perfect hindsight; the spread-timing ceiling binds regardless of how the band is fit. Any material exceedance => ceiling falsified.

opt_vec = the winning (E_C0, E_C1, E_C2, E_C3).
