# Continuous band E_t = E_base*exp(beta*stress) — HMM + market

stress = hmm (2*(pi-0.5)) or multi (mean DD_z/VOL_z/DISP_z/CS_z). beta=0 = rigid static. WF-selected. BH-FDR q=0.05.

     strategy stress    arm  minus_static  excl0  p_boot  bh_sig
     momentum    hmm argmax         0.150  False   0.279   False
     momentum    hmm    reg         0.000  False   0.000    True
     momentum  multi argmax         0.091  False   0.356   False
     momentum  multi    reg         0.000  False   0.000    True
     reversal    hmm argmax         0.113  False   0.273   False
     reversal    hmm    reg         0.000  False   0.000    True
     reversal  multi argmax        -0.028  False   0.460   False
     reversal  multi    reg         0.000  False   0.000    True
       lowvol    hmm argmax        -0.055  False   0.096   False
       lowvol    hmm    reg         0.000  False   0.000    True
       lowvol  multi argmax         0.046  False   0.265   False
       lowvol  multi    reg         0.000  False   0.000    True
          xgb    hmm argmax         0.407  False   0.058   False
          xgb    hmm    reg         0.156  False   0.583   False
          xgb  multi argmax        -0.023  False   0.777   False
          xgb  multi    reg         0.000  False   0.000    True
        value    hmm argmax         0.000  False   0.000    True
        value    hmm    reg         0.000  False   0.000    True
        value  multi argmax         0.000  False   0.000    True
        value  multi    reg         0.000  False   0.000    True
profitability    hmm argmax        -0.104  False   0.241   False
profitability    hmm    reg         0.000  False   0.000    True
profitability  multi argmax        -0.154  False   0.150   False
profitability  multi    reg         0.000  False   0.000    True
   investment    hmm argmax        -0.016  False   0.554   False
   investment    hmm    reg         0.000  False   0.000    True
   investment  multi argmax        -0.032  False   0.255   False
   investment  multi    reg         0.000  False   0.000    True

Positive vs static: 6/28. CI excl 0: 0/28. positive+BH-sig: 0/28.
