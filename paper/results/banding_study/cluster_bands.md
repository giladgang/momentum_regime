# Cluster-conditioned banding vs static/regime (honest WF)

One global scale per arm, walk-forward-selected annually (matched to static). PIT expanding-refit clusters. econ = hold deep-losers wider; gp = (spread/var)^(1/3) per cluster.

strategy                arm  oos_ir  minus_static          ci_x12  excl0
momentum            monthly   0.043         0.106 [-0.007,+0.036]  False
momentum             static  -0.063         0.000 [+0.000,+0.000]  False
momentum             regime   0.003         0.066 [+0.001,+0.022]   True
momentum cluster_month_econ   0.073         0.136 [-0.002,+0.035]  False
momentum   cluster_month_gp   0.041         0.104 [-0.001,+0.028]  False
momentum cluster_stock_econ  -0.031         0.032 [-0.001,+0.021]  False
momentum   cluster_stock_gp  -0.055         0.008 [-0.005,+0.011]  False
     xgb            monthly   0.422        -0.162 [-0.030,+0.015]  False
     xgb             static   0.585         0.000 [+0.000,+0.000]  False
     xgb             regime   0.710         0.126 [-0.007,+0.038]  False
     xgb cluster_month_econ   0.434        -0.151 [-0.030,-0.002]   True
     xgb   cluster_month_gp   0.585         0.000 [+0.000,+0.000]  False
     xgb cluster_stock_econ   0.562        -0.023 [-0.023,+0.013]  False
     xgb   cluster_stock_gp   0.585         0.000 [+0.000,+0.000]  False

momentum: best arm vs static = cluster_month_econ (+0.136, excl0 False).
xgb: best arm vs static = regime (+0.126, excl0 False).
