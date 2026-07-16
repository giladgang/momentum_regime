# Beaten-down-hold cost rules vs static banding (honest WF)

Parameter-free freeze rules (panic_no_sell, panic_hold_losers) and a 1-param thr-selected variant, vs walk-forward-selected static. OOS from 2001 (xgb 2013). BH-FDR q=0.05.

     strategy                  arm  oos_ir  minus_static  excl0  p_boot  bh_sig
     momentum        panic_no_sell  -0.024         0.039  False   0.899   False
     momentum    panic_hold_losers  -0.033         0.030  False   0.907   False
     momentum panic_hold_losers_wf  -0.055         0.008  False   0.984   False
     reversal        panic_no_sell  -0.244        -0.246   True   0.026   False
     reversal    panic_hold_losers  -0.207        -0.209  False   0.053   False
     reversal panic_hold_losers_wf  -0.327        -0.329   True   0.002    True
       lowvol        panic_no_sell  -0.045        -0.141  False   0.106   False
       lowvol    panic_hold_losers   0.007        -0.090  False   0.501   False
       lowvol panic_hold_losers_wf  -0.019        -0.115  False   0.311   False
          xgb        panic_no_sell   0.581        -0.003  False   0.885   False
          xgb    panic_hold_losers   0.262        -0.323  False   0.130   False
          xgb panic_hold_losers_wf   0.442        -0.143  False   0.701   False
        value        panic_no_sell   0.109         0.023  False   0.646   False
        value    panic_hold_losers   0.009        -0.077  False   0.498   False
        value panic_hold_losers_wf   0.060        -0.026  False   0.894   False
profitability        panic_no_sell   0.262         0.057  False   0.545   False
profitability    panic_hold_losers   0.262         0.058  False   0.428   False
profitability panic_hold_losers_wf   0.308         0.103  False   0.297   False
   investment        panic_no_sell   0.223        -0.090  False   0.452   False
   investment    panic_hold_losers   0.250        -0.063  False   0.557   False
   investment panic_hold_losers_wf   0.280        -0.032  False   0.726   False

Positive vs static: 7/21. CI excludes 0: 2/21. Survive BH-FDR: 1/21 (positive+sig: 0).
