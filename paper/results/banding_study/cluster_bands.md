# Cluster-conditioned banding, all 7 strategies (honest WF)

One global scale per arm, walk-forward-selected annually. PIT expanding-refit clusters (stock labels computed once). econ = hold deep-losers wider; gp = (spread/var)^(1/3). BH-FDR q=0.05 across the 5 conditioning arms x 7 strategies.

     strategy                arm  oos_ir  minus_static  excl0  p_boot  bh_sig
     momentum             regime   0.003         0.066   True   0.203   False
     momentum cluster_month_econ   0.073         0.136  False   0.109   False
     momentum   cluster_month_gp   0.041         0.104  False   0.098   False
     momentum cluster_stock_econ  -0.092        -0.030  False   0.617   False
     momentum   cluster_stock_gp  -0.055         0.008  False   0.814   False
     reversal             regime  -0.004        -0.006  False   0.816   False
     reversal cluster_month_econ  -0.027        -0.029  False   0.356   False
     reversal   cluster_month_gp  -0.093        -0.095   True   0.006   False
     reversal cluster_stock_econ  -0.048        -0.050  False   0.221   False
     reversal   cluster_stock_gp  -0.020        -0.022  False   0.627   False
       lowvol             regime   0.069        -0.027  False   0.294   False
       lowvol cluster_month_econ   0.045        -0.051  False   0.152   False
       lowvol   cluster_month_gp   0.047        -0.049   True   0.034   False
       lowvol cluster_stock_econ   0.050        -0.046  False   0.190   False
       lowvol   cluster_stock_gp   0.035        -0.061  False   0.094   False
          xgb             regime   0.710         0.126  False   0.190   False
          xgb cluster_month_econ   0.434        -0.151   True   0.010   False
          xgb   cluster_month_gp   0.585         0.000  False   0.000    True
          xgb cluster_stock_econ   0.492        -0.093   True   0.061   False
          xgb   cluster_stock_gp   0.585         0.000  False   0.000    True
        value             regime   0.071        -0.015  False   0.767   False
        value cluster_month_econ   0.083        -0.003  False   0.954   False
        value   cluster_month_gp   0.066        -0.020  False   0.072   False
        value cluster_stock_econ   0.070        -0.016  False   0.672   False
        value   cluster_stock_gp   0.076        -0.010  False   0.173   False
profitability             regime   0.175        -0.030  False   0.864   False
profitability cluster_month_econ   0.264         0.060  False   0.536   False
profitability   cluster_month_gp   0.295         0.090  False   0.077   False
profitability cluster_stock_econ   0.161        -0.044  False   0.745   False
profitability   cluster_stock_gp   0.257         0.053  False   0.713   False
   investment             regime   0.298        -0.015  False   0.494   False
   investment cluster_month_econ   0.270        -0.043  False   0.269   False
   investment   cluster_month_gp   0.323         0.011  False   0.777   False
   investment cluster_stock_econ   0.279        -0.034  False   0.370   False
   investment   cluster_stock_gp   0.309        -0.004  False   0.814   False

Positive vs static: 9/35 arm-tests. CI excludes 0: 5/35. Survive BH-FDR (q=0.05): 2/35.
