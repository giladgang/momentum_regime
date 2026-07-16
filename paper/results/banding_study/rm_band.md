# Band = f(regime x past-momentum), honest WF, all 7

econ = panic low-mom wider (one scale); argmax/reg = learn the 4-cell band (81 vectors). vs WF-selected static. BH-FDR q=0.05.

     strategy            arm  oos_ir  minus_static  excl0  p_boot  bh_sig
     momentum           econ  -0.047         0.016  False   0.699   False
     momentum trained_argmax   0.061         0.124  False   0.173   False
     momentum    trained_reg  -0.052         0.011  False   0.917   False
     reversal           econ  -0.088        -0.090  False   0.121   False
     reversal trained_argmax  -0.003        -0.005  False   0.814   False
     reversal    trained_reg   0.002         0.000  False   0.000    True
       lowvol           econ   0.105         0.009  False   0.785   False
       lowvol trained_argmax   0.091        -0.006  False   0.814   False
       lowvol    trained_reg   0.096         0.000  False   0.000    True
          xgb           econ   0.470        -0.114  False   0.211   False
          xgb trained_argmax   0.669         0.084  False   0.404   False
          xgb    trained_reg   0.599         0.015  False   0.823   False
        value           econ   0.068        -0.018   True   0.025   False
        value trained_argmax   0.050        -0.036  False   0.243   False
        value    trained_reg   0.086         0.000  False   0.000    True
profitability           econ   0.249         0.044  False   0.550   False
profitability trained_argmax   0.230         0.026  False   0.765   False
profitability    trained_reg   0.341         0.137   True   0.015   False
   investment           econ   0.346         0.033  False   0.067   False
   investment trained_argmax   0.307        -0.005  False   0.596   False
   investment    trained_reg   0.313         0.000  False   0.000    True

Positive vs static: 10/21. CI excl 0: 2/21. positive+BH-sig: 0/21.
