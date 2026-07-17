# Trained per-stock band on (momentum score, regime score), walk-forward

E_it = clip(E_base*exp(a*mom_z_it + b*(pi_t-.5)),5,100); (E_base,a,b) fit expanding-window 1992-> on TRAIN net-of-cost active IR. reg = 1-SE regularized toward static (honest trained band); argmax = unregularized (shows overfit inflation). OOS from 2001 (xgb 2013). CI on the IR DIFFERENCE (correct statistic).

     strategy  ir_static  ir_reg_trained  reg_minus_static  ci_lo  ci_hi  p_boot  excl0  ir_argmax_overfit  argmax_minus_static
     momentum     -0.037          -0.037               0.0    0.0    0.0     2.0  False              0.056                0.093
     reversal      0.002           0.002               0.0    0.0    0.0     2.0  False              0.033                0.031
       lowvol      0.096           0.096               0.0    0.0    0.0     2.0  False              0.044               -0.053
          xgb      0.585           0.585               0.0    0.0    0.0     2.0  False              0.651                0.067
        value      0.086           0.086               0.0    0.0    0.0     2.0  False              0.073               -0.013
profitability      0.244           0.244               0.0    0.0    0.0     2.0  False              0.260                0.016
   investment      0.313           0.313               0.0    0.0    0.0     2.0  False              0.222               -0.091

Trained(reg) beats static: 0/7. Positive AND CI excludes 0: 0/7. The argmax-minus-static column vs reg-minus-static shows how much the raw (unregularized) trained band overfits.
