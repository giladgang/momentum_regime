# Trained-per-cluster band adjustment vs static (honest WF)

Per-cluster band LEARNED from prior data (argmax over 81 vectors, clusters in {10,20,40}), applied OOS. trained_reg = 1-SE complexity-regularized (matches static search). PIT month clusters. BH-FDR q=0.05.

     strategy            arm  oos_ir  minus_static  excl0  p_boot  bh_sig
     momentum trained_argmax   0.016         0.079  False   0.248   False
     momentum    trained_reg  -0.052         0.011  False   0.917   False
     reversal trained_argmax  -0.039        -0.041  False   0.108   False
     reversal    trained_reg   0.002         0.000  False   0.000    True
       lowvol trained_argmax   0.089        -0.007  False   0.138   False
       lowvol    trained_reg   0.096         0.000  False   0.000    True
          xgb trained_argmax   0.669         0.084  False   0.320   False
          xgb    trained_reg   0.599         0.015  False   0.823   False
        value trained_argmax   0.086         0.000  False   0.000    True
        value    trained_reg   0.086         0.000  False   0.000    True
profitability trained_argmax   0.382         0.177  False   0.043   False
profitability    trained_reg   0.341         0.137   True   0.015    True
   investment trained_argmax   0.322         0.010  False   0.944   False
   investment    trained_reg   0.313         0.000  False   0.000    True

trained beats static: argmax 4/7, reg 3/7. CI excl 0: 1/14. positive+BH-sig: 1/14.
