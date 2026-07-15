# Walk-forward (rolling annual re-selection) banding results

Expanding window, band re-selected EVERY YEAR on all prior months (non-XGB OOS from 2001; XGB from 2013). Stitched true-OOS net-of-measured-cost active return; supersedes the single 2011 split. Two fits: regime_argmax (raw argmax over 120 bands) vs regime_reg (1-SE complexity-regularized -- adopt conditioning only when it clearly beats the simpler band).

     strategy  oos_start  oos_months  monthly_ir  static_ir  regime_argmax_ir  regime_reg_ir  argmax_minus_static  reg_minus_static  argmax_excl0  reg_excl0      reg_ci_x12  reg_pct_3state  n_reselections
     momentum       2001         299       0.043     -0.063             0.099         -0.063                0.162               0.0          True      False [+0.000,+0.000]             0.0              25
     reversal       2001         299      -0.060      0.002             0.002          0.002                0.000               0.0         False      False [+0.000,+0.000]             0.0              25
       lowvol       2001         288       0.036      0.096             0.112          0.096                0.016               0.0         False      False [+0.000,+0.000]             0.0              25
          xgb       2013         119       0.422      0.585             0.675          0.585                0.090               0.0         False      False [+0.000,+0.000]             0.0              10
        value       2001         299       0.042      0.086             0.076          0.086               -0.010               0.0         False      False [+0.000,+0.000]             0.0              25
profitability       2001         299       0.331      0.205             0.206          0.205                0.001               0.0         False      False [+0.000,+0.000]             0.0              25
   investment       2001         299       0.239      0.313             0.307          0.313               -0.005               0.0         False      False [+0.000,+0.000]             0.0              25

Regime beats static (stitched-OOS IR): argmax fit 5/7, REGULARIZED fit 0/7 (0/7 with bootstrap CI excluding 0). The regularized column is the honest number: it neutralizes the 24x search asymmetry between the 120-cell regime grid and the 5-cell static grid.
