# Garleanu-Pedersen / Constantinides analytic regime bands (no grid)

Panic/calm band ratio derived from prior-data (spread, variance); only ONE scale E_base fit, matched to the static band. Walk-forward, expanding, OOS from 2001 (xgb 2013).

     strategy  panic_ratio  oos_months  monthly_ir  static_ir  theory_ir  theory_minus_static          ci_x12  excl0
     momentum        0.678         299       0.043     -0.063      0.003                0.066 [+0.001,+0.022]   True
     reversal        0.678         299      -0.060      0.002     -0.004               -0.006 [-0.009,+0.007]  False
       lowvol        0.678         288       0.036      0.096      0.069               -0.027 [-0.005,+0.002]  False
          xgb        0.460         119       0.422      0.585      0.710                0.126 [-0.007,+0.038]  False
        value        0.685         299       0.042      0.086      0.071               -0.015 [-0.008,+0.006]  False
profitability        0.678         299       0.331      0.205      0.175               -0.030 [-0.013,+0.008]  False
   investment        0.678         299       0.239      0.313      0.298               -0.015 [-0.005,+0.003]  False

Theory-regime beats static OOS for 2/7 strategies; 1/7 with bootstrap CI excluding 0. ratio<1 => theory says trade MORE in panic (variance spike beats the modest spread spike); ratio>1 => trade less.
